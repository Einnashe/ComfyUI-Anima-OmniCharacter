from contextlib import contextmanager
from contextvars import ContextVar
import os

import torch

import comfy.conds
import comfy.ldm.common_dit
import comfy.lora
import comfy.lora_convert
import comfy.model_base
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter
import folder_paths
from node_helpers import conditioning_set_values


REFERENCE_CONDITION = "anima_omnicharacter_reference"
REFERENCE_CACHE = "anima_omnicharacter_reference_cache"
REFERENCE_CACHE_KEY = "anima_omnicharacter_reference_cache_key"
_CONTROL_LORA_ACTIVE = ContextVar("anima_omnicharacter_lora_active", default=False)


@contextmanager
def use_control_lora():
    token = _CONTROL_LORA_ACTIVE.set(True)
    try:
        yield
    finally:
        _CONTROL_LORA_ACTIVE.reset(token)


class RoutedLoRAAdapter(comfy.weight_adapter.LoRAAdapter):
    def bypass_forward(self, org_forward, x, *args, **kwargs):
        if not _CONTROL_LORA_ACTIVE.get():
            return org_forward(x, *args, **kwargs)
        return super().bypass_forward(org_forward, x, *args, **kwargs)


def _rope_with_offset(model, shape, offset, device):
    _, frames, height, width, _ = shape
    position = model.pos_embedder
    h_freqs = 1.0 / ((10000.0 * position.h_ntk_factor) ** position.dim_spatial_range.to(device=device))
    w_freqs = 1.0 / ((10000.0 * position.w_ntk_factor) ** position.dim_spatial_range.to(device=device))
    t_freqs = 1.0 / ((10000.0 * position.t_ntk_factor) ** position.dim_temporal_range.to(device=device))

    temporal = torch.outer(torch.arange(offset, offset + frames, device=device, dtype=torch.float32), t_freqs)
    vertical = torch.outer(torch.arange(height, device=device, dtype=torch.float32), h_freqs)
    horizontal = torch.outer(torch.arange(width, device=device, dtype=torch.float32), w_freqs)

    temporal = torch.stack((temporal.cos(), -temporal.sin(), temporal.sin(), temporal.cos()), dim=-1).reshape(frames, -1, 2, 2)
    vertical = torch.stack((vertical.cos(), -vertical.sin(), vertical.sin(), vertical.cos()), dim=-1).reshape(height, -1, 2, 2)
    horizontal = torch.stack((horizontal.cos(), -horizontal.sin(), horizontal.sin(), horizontal.cos()), dim=-1).reshape(width, -1, 2, 2)

    temporal = temporal[:, None, None].expand(-1, height, width, -1, -1, -1)
    vertical = vertical[None, :, None].expand(frames, -1, width, -1, -1, -1)
    horizontal = horizontal[None, None, :].expand(frames, height, -1, -1, -1, -1)
    return torch.cat((temporal, vertical, horizontal), dim=3).reshape(frames * height * width, -1, 2, 2)


def _modulate(block, kind, state, embedding, adaln_embedding):
    modulation = getattr(block, f"adaln_modulation_{kind}")(embedding)
    if block.use_adaln_lora:
        modulation = modulation + adaln_embedding
    shift, scale, gate = modulation.chunk(3, dim=-1)
    shift = shift[:, :, None, None]
    scale = scale[:, :, None, None]
    gate = gate[:, :, None, None]
    normalized = getattr(block, f"layer_norm_{kind}")(state) * (1.0 + scale) + shift
    return normalized, gate


def _project_qkv(module, hidden, rope, transformer_options, control_lora=False):
    if rope.ndim == 4:
        rope = rope.unsqueeze(0).unsqueeze(2)
    if control_lora:
        with use_control_lora():
            return module.compute_qkv(hidden, rope_emb=rope, transformer_options=transformer_options)
    return module.compute_qkv(hidden, rope_emb=rope, transformer_options=transformer_options)


def _branch_attention(module, query, keys, values, transformer_options, control_lora=False):
    batch_size = query.shape[0]
    keys = [key.expand(batch_size, *key.shape[1:]) if key.shape[0] == 1 else comfy.utils.repeat_to_batch_size(key, batch_size) for key in keys]
    values = [value.expand(batch_size, *value.shape[1:]) if value.shape[0] == 1 else comfy.utils.repeat_to_batch_size(value, batch_size) for value in values]
    key = torch.cat(keys, dim=1)
    value = torch.cat(values, dim=1)
    if query.dtype != value.dtype:
        query = query.to(value.dtype)
        key = key.to(value.dtype)
    if control_lora:
        with use_control_lora():
            return module.compute_attention(query, key, value, transformer_options=transformer_options)
    return module.compute_attention(query, key, value, transformer_options=transformer_options)


def _run_block(block, states, embeddings, adaln_embeddings, ropes, context, transformer_options):
    normalized_self = []
    self_gates = []
    for state, embedding, adaln_embedding in zip(states, embeddings, adaln_embeddings):
        normalized, gate = _modulate(block, "self_attn", state, embedding, adaln_embedding)
        normalized_self.append(normalized.to(embedding.dtype))
        self_gates.append(gate)

    projections = []
    for branch_index, normalized in enumerate(normalized_self):
        projections.append(_project_qkv(block.self_attn, normalized.flatten(1, 3), ropes[branch_index], transformer_options, control_lora=branch_index > 0))

    updated = []
    for branch_index, normalized in enumerate(normalized_self):
        key_value_indices = range(len(states)) if branch_index == 0 else (branch_index,)
        result = _branch_attention(
            block.self_attn,
            projections[branch_index][0],
            [projections[index][1] for index in key_value_indices],
            [projections[index][2] for index in key_value_indices],
            transformer_options,
            control_lora=branch_index > 0,
        ).reshape_as(normalized)
        updated.append(torch.addcmul(states[branch_index], self_gates[branch_index].to(states[branch_index].dtype), result.to(states[branch_index].dtype)))
    states = updated

    normalized_target, target_gate = _modulate(block, "cross_attn", states[0], embeddings[0], adaln_embeddings[0])
    normalized_target = normalized_target.to(embeddings[0].dtype)
    target_shape = normalized_target.shape
    cross = block.cross_attn(normalized_target.flatten(1, 3), context, transformer_options=transformer_options)
    states[0] = torch.addcmul(states[0], target_gate.to(states[0].dtype), cross.reshape(target_shape).to(states[0].dtype))

    patches = transformer_options.get("patches", {})
    next_states = []
    for branch_index, state in enumerate(states):
        normalized, gate = _modulate(block, "mlp", state, embeddings[branch_index], adaln_embeddings[branch_index])
        normalized = normalized.to(embeddings[branch_index].dtype)
        if "mlp_patch" in patches:
            patch_args = {"x": normalized, "transformer_options": transformer_options}
            for patch in patches["mlp_patch"]:
                patch_args = patch(patch_args)
            normalized = patch_args["x"]
        if branch_index == 0:
            result = block.mlp(normalized)
        else:
            with use_control_lora():
                result = block.mlp(normalized)
        next_states.append(torch.addcmul(state, gate.to(state.dtype), result.to(state.dtype)))
    return next_states, (projections[1][1], projections[1][2])


def _run_target_block(block, state, embedding, adaln_embedding, rope, context, reference_key, reference_value, transformer_options):
    normalized, gate = _modulate(block, "self_attn", state, embedding, adaln_embedding)
    normalized = normalized.to(embedding.dtype)
    query, key, value = _project_qkv(block.self_attn, normalized.flatten(1, 3), rope, transformer_options)
    result = _branch_attention(block.self_attn, query, [key, reference_key], [value, reference_value], transformer_options).reshape_as(normalized)
    state = torch.addcmul(state, gate.to(state.dtype), result.to(state.dtype))

    normalized, gate = _modulate(block, "cross_attn", state, embedding, adaln_embedding)
    normalized = normalized.to(embedding.dtype)
    shape = normalized.shape
    result = block.cross_attn(normalized.flatten(1, 3), context, transformer_options=transformer_options)
    state = torch.addcmul(state, gate.to(state.dtype), result.reshape(shape).to(state.dtype))

    normalized, gate = _modulate(block, "mlp", state, embedding, adaln_embedding)
    normalized = normalized.to(embedding.dtype)
    patches = transformer_options.get("patches", {})
    if "mlp_patch" in patches:
        patch_args = {"x": normalized, "transformer_options": transformer_options}
        for patch in patches["mlp_patch"]:
            patch_args = patch(patch_args)
        normalized = patch_args["x"]
    result = block.mlp(normalized)
    return torch.addcmul(state, gate.to(state.dtype), result.to(state.dtype))


def _omnicharacter_forward(model, x, timesteps, context, fps, padding_mask, reference, reference_cache_key, transformer_options):
    original_shape = x.shape
    if x.ndim == 4:
        x = x.unsqueeze(2)
    if reference.ndim == 4:
        reference = reference.unsqueeze(2)
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (model.patch_temporal, model.patch_spatial, model.patch_spatial))
    reference = comfy.ldm.common_dit.pad_to_patch_size(reference, (model.patch_temporal, model.patch_spatial, model.patch_spatial))

    target, target_rope, extra_position = model.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)
    if extra_position is not None:
        raise RuntimeError("Anima OmniCharacter does not support per-block absolute position embeddings")

    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(1)
    target_embedding, target_adaln = model.t_embedder[1](model.t_embedder[0](timesteps).to(target.dtype))
    target_embedding = model.t_embedding_norm(target_embedding)
    target = target.float() if target.dtype == torch.float16 else target

    cache = transformer_options.get(REFERENCE_CACHE)
    cache_entry_key = None
    reference_kvs = None
    if cache is not None and reference_cache_key is not None and "mlp_patch" not in transformer_options.get("patches", {}):
        cache_entry_key = reference_cache_key
        reference_kvs = cache.get(cache_entry_key)

    if reference_kvs is None:
        control, _, _ = model.prepare_embedded_sequence(reference, padding_mask=None)
        control_rope = _rope_with_offset(model, control.shape, 1, control.device)
        zero_timestep = torch.zeros((control.shape[0], control.shape[1]), device=timesteps.device, dtype=timesteps.dtype)
        control_embedding, control_adaln = model.t_embedder[1](model.t_embedder[0](zero_timestep).to(control.dtype))
        control_embedding = model.t_embedding_norm(control_embedding)
        control = control.float() if control.dtype == torch.float16 else control

        states = [target, control]
        embeddings = [target_embedding, control_embedding]
        adaln_embeddings = [target_adaln, control_adaln]
        ropes = [target_rope, control_rope]
        generated_reference_kvs = [] if cache_entry_key is not None else None
        for block_index, block in enumerate(model.blocks):
            transformer_options["block_index"] = block_index
            states, reference_kv = _run_block(block, states, embeddings, adaln_embeddings, ropes, context, transformer_options)
            if generated_reference_kvs is not None:
                generated_reference_kvs.append(reference_kv)
        target = states[0]
        if cache_entry_key is not None:
            cache[cache_entry_key] = tuple(generated_reference_kvs)
    else:
        for block_index, (block, reference_kv) in enumerate(zip(model.blocks, reference_kvs)):
            transformer_options["block_index"] = block_index
            target = _run_target_block(block, target, target_embedding, target_adaln, target_rope, context, reference_kv[0], reference_kv[1], transformer_options)

    output = model.final_layer(target.to(context.dtype), target_embedding, adaln_lora_B_T_3D=target_adaln)
    output = model.unpatchify(output)
    if len(original_shape) == 4:
        return output[:, :, 0, :original_shape[-2], :original_shape[-1]]
    return output[:, :, :original_shape[-3], :original_shape[-2], :original_shape[-1]]


def _omnicharacter_wrapper(executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
    reference = kwargs.pop(REFERENCE_CONDITION, None)
    reference_cache_key = kwargs.pop(REFERENCE_CACHE_KEY, None)
    if reference is None:
        return executor(x, timesteps, context, fps, padding_mask, **kwargs)
    transformer_options = kwargs.get("transformer_options", {})
    return _omnicharacter_forward(executor.class_obj, x, timesteps, context, fps, padding_mask, reference, reference_cache_key, transformer_options)


def _reference_cache_wrapper(executor, guider, sigmas, extra_args, callback, noise, *args, **kwargs):
    transformer_options = extra_args["model_options"].setdefault("transformer_options", {})
    cache = {}
    transformer_options[REFERENCE_CACHE] = cache
    try:
        return executor(guider, sigmas, extra_args, callback, noise, *args, **kwargs)
    finally:
        cache.clear()
        transformer_options.pop(REFERENCE_CACHE)


def _patch_extra_conds(model):
    original = model.model.extra_conds

    def extra_conds(**kwargs):
        out = original(**kwargs)
        reference = kwargs.get(REFERENCE_CONDITION)
        if reference is not None:
            out[REFERENCE_CONDITION] = comfy.conds.CONDConstant(model.model.process_latent_in(reference))
        reference_cache_key = kwargs.get(REFERENCE_CACHE_KEY)
        if reference_cache_key is not None:
            out[REFERENCE_CACHE_KEY] = comfy.conds.CONDConstant(reference_cache_key)
        return out

    model.add_object_patch("extra_conds", extra_conds)


def _load_lora(model, lora_name, strength):
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    converted = comfy.lora_convert.convert_lora(state_dict)
    loaded = comfy.lora.load_lora(converted, key_map, log_missing=False)
    adapters = {key: adapter for key, adapter in loaded.items() if isinstance(adapter, comfy.weight_adapter.WeightAdapterBase)}
    if not adapters:
        raise RuntimeError(f"No Anima LoRA weights were loaded from {lora_name}")

    loaded_lora_keys = set()
    manager = comfy.weight_adapter.BypassInjectionManager()
    for key, adapter in adapters.items():
        adapter = RoutedLoRAAdapter(adapter.loaded_keys, adapter.weights)
        manager.add_adapter(key, adapter, strength=strength)
        loaded_lora_keys.update(adapter.loaded_keys)

    missing = set(converted) - loaded_lora_keys
    if missing:
        sample = sorted(missing)[:4]
        raise RuntimeError(f"Failed to load {len(missing)} Anima OmniCharacter LoRA keys: {sample}")

    injections = manager.create_injections(model.model)
    if manager.get_hook_count() != len(adapters):
        raise RuntimeError(f"Loaded {manager.get_hook_count()} of {len(adapters)} Anima OmniCharacter adapters")
    model.set_injections("anima_omnicharacter_lora", injections)


def _omnicharacter_loras():
    loras = []
    for name in folder_paths.get_filename_list("loras"):
        basename = os.path.basename(name).lower()
        if "anima_omnicharacter" in basename:
            loras.append(name)
    return loras


class AnimaOmniCharacterApply:
    @classmethod
    def INPUT_TYPES(cls):
        loras = _omnicharacter_loras()
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE",),
                "lora_name": (loras or ["Put Anima OmniCharacter LoRAs in models/loras"],),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "cache_reference_kv": ("BOOLEAN", {
                    "default": True,
                    "label_on": "cached",
                    "label_off": "uncached",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "apply"
    CATEGORY = "conditioning/anima"

    def apply(self, model, positive, negative, vae, reference_image, lora_name, strength, cache_reference_kv):
        if not isinstance(model.model, comfy.model_base.Anima):
            raise RuntimeError("Anima OmniCharacter requires an Anima base model")

        height = max(16, reference_image.shape[1] // 16 * 16)
        width = max(16, reference_image.shape[2] // 16 * 16)
        image = reference_image[:, :, :, :3]
        if height != reference_image.shape[1] or width != reference_image.shape[2]:
            image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "area", "disabled").movedim(1, -1)
        reference = vae.encode(image)

        patched = model.clone()
        _load_lora(patched, lora_name, strength)
        _patch_extra_conds(patched)
        patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "anima_omnicharacter")
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "anima_omnicharacter", _omnicharacter_wrapper)
        patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, REFERENCE_CACHE)
        if cache_reference_kv:
            patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, REFERENCE_CACHE, _reference_cache_wrapper)

        reference_values = {REFERENCE_CONDITION: reference}
        if cache_reference_kv:
            reference_values[REFERENCE_CACHE_KEY] = object()
        positive = conditioning_set_values(positive, reference_values)
        negative = conditioning_set_values(negative, reference_values)
        return patched, positive, negative


NODE_CLASS_MAPPINGS = {
    "AnimaOmniCharacterApply": AnimaOmniCharacterApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaOmniCharacterApply": "Apply Anima OmniCharacter",
}
