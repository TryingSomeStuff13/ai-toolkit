import os
"""Turbo registry for MiniMax-H3 graphs: files, sampler settings and the workflow patch of every candidate turbo.

Imported by h3_recipe_eval.py (the eval driver) and h3_record/record_chunk.py (the trajectory recorder). No import-time
side effects: no tunnel, no sys.argv, no dashboard import.

Every turbo but Larry is built from the dashboard builder's Larry graph (h3_attention "turbo_8step"):
  1 UNETLoader -> [h3_lora_N LoRA stack, optional] -> 30 MiniMaxH3TurboLoRA -> 31 MiniMaxH3SigmaShift(12, 3) -> 7 BasicGuider
  8 MiniMaxH3TurboSampler, 9 BasicScheduler(model 1, simple, 8 steps) -> 10 SamplerCustomAdvanced
Larry's sampler is a plain Euler step on this engine (ModelSamplingAV carries the audio clock) and BasicGuider means no
CFG, so a turbo differs only in its adapter node, shifts and sigma grid. Files are per DiT partition, read from node 1:
the fl2va DiT (text-to-video clip 1; t2va shares it) or the ref2va DiT (continuations). Sizes/checksums: turbo_fetch.sh.

patch(wf, name) returns a patched copy (Larry: an unchanged copy). For every turbo, node 7 still reads ["31", 0]
(anything spliced between the model chain and the guider goes there) and node 10 keeps its noise, guider and latent
inputs. What a patch changes: node 30 (the turbo), node 31's shifts, node 8 (KSamplerSelect), node 9 (BasicScheduler
on the shifted model for "simple" grids, KJNodes CustomSigmas for a fixed grid, or removed for the HyperFlow node,
whose SIGMAS are ["30", 1]), and, for the HyperFlow node, node 31's model input and the LoRA stack's head (ApplyHyperFlow
must see the checkpoint unmodified, so the stack moves after it) and node 1's DiT for hyperflow_full. A checkpoint turbo
(merged_turbo: the turbo is baked into the DiT) replaces node 1's DiT on every clip and removes node 30, so node 31 reads the
LoRA stack (or node 1) directly.
"""
import copy

FL2VA_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"    # the dashboard's app.MINIMAX_H3_UNET
REF2VA_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"  # app.MINIMAX_H3_REFERENCE_UNET
LARRY_LORA = "minimax_h3_turbo_v4_step600_ema.safetensors"          # app.MINIMAX_H3_TURBO_LORA
# the upstream ComfyUI line ComfyUI-Hyperflow's block patch needs; turbo_fetch.sh --install adds it to a pod's engine
ENGINE_LAYOUT_LINE = 'transformer_options["minimax_h3_layout"] = layout'


def _dit_partition(wf):
    return {FL2VA_UNET: "fl2va", REF2VA_UNET: "ref2va"}[wf["1"]["inputs"]["unet_name"]]


def _check_larry_graph(wf):
    """The builder's turbo_8step nodes that a turbo patch rewires: fail loudly if the dashboard ever changes them."""
    assert wf["30"]["class_type"] == "MiniMaxH3TurboLoRA" and wf["30"]["inputs"]["lora_name"] == LARRY_LORA, wf["30"]
    assert wf["31"]["class_type"] == "MiniMaxH3SigmaShift" and wf["31"]["inputs"]["model"] == ["30", 0], wf["31"]
    assert wf["7"]["class_type"] == "BasicGuider" and wf["7"]["inputs"]["model"] == ["31", 0], wf["7"]
    assert wf["8"] == {"class_type": "MiniMaxH3TurboSampler", "inputs": {}}, wf["8"]
    assert wf["9"]["class_type"] == "BasicScheduler" and wf["9"]["inputs"]["model"] == ["1", 0], wf["9"]
    assert wf["10"]["inputs"]["sampler"] == ["8", 0] and wf["10"]["inputs"]["sigmas"] == ["9", 0], wf["10"]
    users = sorted(k for k, node in wf.items() if ["1", 0] in node["inputs"].values())
    stack = [k for k in users if wf[k]["class_type"] == "LoraLoaderModelOnly"]
    assert len(stack) <= 1 and users == sorted(stack + ["9"] + (["30"] if wf["30"]["inputs"]["model"] == ["1", 0] else [])), users
    return stack  # the head of the optional LoRA stack, the only other node reading the checkpoint


def _set_sampling(wf, turbo):
    t = TURBOS[turbo]
    wf["31"]["inputs"].update(shift_video=t["shift"][0], shift_audio=t["shift"][1])
    wf["8"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": t["sampler"]}}
    if t["sigmas"] == "simple":  # the turbo's own shifts drive the schedule, as in the official lightx2v graphs
        wf["9"]["inputs"].update(model=["31", 0], steps=t["steps"])
    else:  # a fixed grid (KJNodes Custom Sigmas: ", "-separated, no interpolation when steps + 1 values)
        assert len(t["sigmas"]) == t["steps"] + 1 and t["sigmas"][-1] == 0.0
        wf["9"] = {"class_type": "CustomSigmas", "inputs": {
            "sigmas_string": ", ".join(repr(s) for s in t["sigmas"]), "interpolate_to_steps": t["steps"]}}


def patch_lora_turbo(wf, turbo):
    """A plain LoRA turbo (backbone only: no adaLN, no time embedder) in Larry's place, as LoraLoaderModelOnly. It
    merges into the pruned int8 DiT like the dashboard's native lightx2v profiles; the anti-burn stack stays before it."""
    t = TURBOS[turbo]
    _check_larry_graph(wf)
    wf["30"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
        "model": wf["30"]["inputs"]["model"], "lora_name": t["files"][_dit_partition(wf)], "strength_model": t["strength"]}}
    if os.environ.get("TURBO_APPLY", "merge") == "runtime":  # 2026-09-25: the turbo applied live (never merged into int8),
        n = wf["30"]["inputs"]                                # as the trainer applies an assistant LoRA; needs H3RuntimeLoRA
        wf["30"] = {"class_type": "H3RuntimeLoRA", "inputs": {"model": n["model"], "lora_name": n["lora_name"], "strength": n["strength_model"]}}
    _set_sampling(wf, turbo)
    return wf


def patch_hyperflow(wf, turbo):
    """ApplyHyperFlow (ComfyUI-Hyperflow 1.3.1) in Larry's place; its SIGMAS output (the file's 9-point grid, video
    shift 12) replaces the scheduler. It must see the checkpoint unmodified (the curve refit falls back to backbone
    only on a patched MODEL), so it reads node 1 and the anti-burn stack moves after it."""
    t = TURBOS[turbo]
    stack = _check_larry_graph(wf)
    part = _dit_partition(wf)
    stack_out = wf["30"]["inputs"]["model"]
    if "dit" in t:  # the full-base build: the unpruned int8 DiTs with the time embedder and full adaLN weights
        wf["1"]["inputs"]["unet_name"] = t["dit"][part]
    wf["30"] = {"class_type": "ApplyHyperFlow", "inputs": {
        "model": ["1", 0], "hyperflow_file": t["files"][part], "strength": t["strength"], "lora_mode": "bypass",
        "variant": "auto", "download_if_missing": False, "verbose": False, "experimental_curve_refit": t["curve_refit"]}}
    for head in stack:
        wf[head]["inputs"]["model"] = ["30", 0]
    wf["31"]["inputs"].update(model=stack_out if stack else ["30", 0], shift_video=t["shift"][0], shift_audio=t["shift"][1])
    wf["8"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": t["sampler"]}}
    del wf["9"]
    wf["10"]["inputs"]["sigmas"] = ["30", 1]
    return wf


def patch_checkpoint_turbo(wf, turbo):
    """A DiT with the turbo baked in (a merged-turbo checkpoint): that checkpoint in node 1 for every partition,
    no turbo LoRA (node 30 removed, node 31 reads the LoRA stack or node 1), the profile's shifts on node 31, Euler on
    the simple schedule of the shifted model. The dashboard's graph puts its shift node (41) before the LoRA stack; a
    shift node and weight-patch LoRAs commute, so node 31 stays after the stack and node 7 keeps reading ["31", 0]."""
    t = TURBOS[turbo]
    _check_larry_graph(wf)
    wf["1"]["inputs"]["unet_name"] = t["dit"][_dit_partition(wf)]
    wf["31"]["inputs"]["model"] = wf.pop("30")["inputs"]["model"]
    _set_sampling(wf, turbo)
    if wf.get("14", {}).get("class_type") == "SaveVideo" and "filename_prefix" in t:  # clip 1's output name, as the dashboard
        wf["14"]["inputs"]["filename_prefix"] = t["filename_prefix"]
    return wf


MERGED_TURBO = "your_merged_turbo_checkpoint.safetensors"          # any H3 DiT with a turbo merged in
LIGHTX2V_FL2V = "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
LIGHTX2V_REF2V = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
HYPERFLOW_SIGMAS = (1.0, 0.9939097762107849, 0.9842875003814697, 0.9660636782646179, 0.9230769276618958,
                    0.8349423408508301, 0.6968520283699036, 0.4687533378601074, 0.0)  # the file's grid shifted 12, float32
TURBOS = {
    # today's recipe: backbone LoRA bypassed, int8-fused fc2 merged, and the 51 adaLN pieces re-injected at run time on the
    # pruned DiTs from the node's silu(t_emb) grid (1025 x 2688) matched to adaln_t_table
    "larry": {"source": "larryvrh/MiniMax-H3-Turbo-Lora",
              "files": dict.fromkeys(("t2va", "fl2va", "ref2va"), LARRY_LORA),
              "steps": 8, "shift": (12.0, 3.0), "sigmas": "simple", "sampler": "MiniMaxH3TurboSampler", "strength": 1.0,
              "patch": None},
    # 8-step v1.0 768p, backbone only; ModelTC's table: fl2v 768p trained at shifts 6/3, 8 NFE, euler + simple, LoRA 1.0.
    # The ref2v 8-step file is not in that table: shifts 6/3 are assumed from its 768p fl2v sibling.
    "lightx2v8": {"source": "lightx2v/Minimax-h3-Turbo",
                  "files": {"t2va": LIGHTX2V_FL2V, "fl2va": LIGHTX2V_FL2V, "ref2va": LIGHTX2V_REF2V},
                  "steps": 8, "shift": (6.0, 3.0), "sigmas": "simple", "sampler": "euler", "strength": 1.0,
                  "patch": patch_lora_turbo},
    # HyperFlow's backbone LoRA only, as a stock LoRA on the pruned DiTs: no time-embedder LoRA and no (t, r) two-time
    # conditioning, so off-recipe (the node author measured fl2va video latent cosine 0.81 against the released model)
    "hyperflow_lora": {"source": "drbaph/MiniMax-H3-Turbo-Lora-ComfyUI",
                       "files": dict.fromkeys(("t2va", "fl2va", "ref2va"),
                                              "minimax_h3_hyperflow_8step_v1.0_comfyui_pruned_bf16.safetensors"),
                       "steps": 8, "shift": (12.0, 3.0), "sigmas": HYPERFLOW_SIGMAS, "sampler": "euler", "strength": 1.0,
                       "patch": patch_lora_turbo},
    # the node's pruned build + its experimental curve refit, fitted to exactly our two pruned int8 DiTs (sha256 bound):
    # the two-time signal re-expressed in the 8-dim adaln_t_table basis (cosine 0.88 fl2va video / 0.997 ref2va audio)
    "hyperflow_refit": {"source": "drbaph/Hyperflow-Comfyui",
                        "files": dict.fromkeys(("t2va", "fl2va", "ref2va"),
                                               "custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors"),
                        "steps": 8, "shift": (12.0, 3.0), "sigmas": "file", "sampler": "euler", "strength": 1.0,
                        "patch": patch_hyperflow, "curve_refit": True},
    # the released recipe: full node build on the unpruned int8 DiTs (34 GB each), which keep the time embedder and the
    # 2688-wide adaLN weights the two-time embedding feeds
    "hyperflow_full": {"source": "drbaph/Hyperflow-Comfyui",
                       "files": dict.fromkeys(("t2va", "fl2va", "ref2va"), "custom_node_hyperflow_8step_v1.0_comfyui.safetensors"),
                       "dit": {"fl2va": "minimax_h3_fl2va_int8_convrot.safetensors",
                               "ref2va": "minimax_h3_ref2va_int8_convrot.safetensors"},
                       "steps": 8, "shift": (12.0, 3.0), "sigmas": "file", "sampler": "euler", "strength": 1.0,
                       "patch": patch_hyperflow, "curve_refit": False},
    # a checkpoint turbo: one int8 H3 DiT with a turbo delta merged in (we tested a hybrid fl2va + ref2va one, grafted on
    # the fl2va pruned base); 8 steps, euler, its own shifts (12/7 for the one we tested), no LoRA. It runs our ref2va
    # graph on every partition. Point MERGED_TURBO at your file. Strength 1.0 = the baked-in turbo.
    "merged_turbo": {"source": "an H3 checkpoint with a turbo merged in",
                     "files": dict.fromkeys(("t2va", "fl2va", "ref2va"), MERGED_TURBO),
                     "dit": dict.fromkeys(("fl2va", "ref2va"), MERGED_TURBO),
                     "steps": 8, "shift": (12.0, 7.0), "sigmas": "simple", "sampler": "euler", "strength": 1.0,
                     "patch": patch_checkpoint_turbo},
    # Parallel Decoding Distillation: 32-interval grid in blocks of 4 = 8 Euler steps on the simple shift-12 grid (Larry's
    # sigmas), CFG 1. Not a plain LoRA, and no ComfyUI loader exists.
    "pai_pdd": {"source": "alibaba-pai/MiniMax-H3-Acc-LoRAs",
                "files": {"t2va": "MiniMax-H3-FL2VA-Acc-8Step.safetensors", "fl2va": "MiniMax-H3-FL2VA-Acc-8Step.safetensors",
                          "ref2va": "MiniMax-H3-Ref2VA-Acc-8Step.safetensors"},
                "steps": 8, "shift": (12.0, 3.0), "sigmas": "simple", "sampler": "euler", "strength": 1.0, "patch": None,
                "blocked": "no ComfyUI loader. It needs a node that (1) maps its diffusers LoRA keys (lora_down/up; separate "
                           "to_q/k/v -> fused qkv_proj, ff.net.0.proj -> fc1 with value/gate rows swapped, ff.net.2 -> fc2), "
                           "(2) re-injects its 50 adaln_proj.linear LoRAs at run time on the pruned DiTs through the Larry "
                           "node's silu(t_emb) grid, and (3) replaces final_layer.video_out/audio_out with its 32 parallel "
                           "heads (proj_out [32,96,5376], audio_proj_out [32,32,5376]) fused per step by the PDD plan"},
}


def check(name):
    """The registry entry of a runnable turbo; ValueError for an unknown or blocked name."""
    if name not in TURBOS:
        raise ValueError(f"TURBO={name!r} is not one of: {', '.join(TURBOS)}")
    if TURBOS[name].get("blocked"):
        raise ValueError(f"TURBO={name} cannot run yet: {TURBOS[name]['blocked']}")
    return TURBOS[name]


def patch(wf, name):
    """A copy of the builder's Larry graph `wf` with turbo `name` in Larry's place (Larry: an unchanged copy)."""
    t = check(name)
    wf = copy.deepcopy(wf)
    return t["patch"](wf, name) if t["patch"] else wf


def preflight_needs(name):
    """[(node class, input field or None, [names that field must list])] the engine needs for this turbo's graph."""
    t = check(name)
    files = sorted({t["files"]["fl2va"], t["files"]["ref2va"]})
    if t["patch"] is None:
        needs = [("MiniMaxH3TurboLoRA", "lora_name", files), ("MiniMaxH3TurboSampler", None, [])]
    elif t["patch"] is patch_hyperflow:
        needs = [("ApplyHyperFlow", "hyperflow_file", files)] + (
            [("UNETLoader", "unet_name", sorted(t["dit"].values()))] if "dit" in t else [])
    elif t["patch"] is patch_checkpoint_turbo:
        needs = [("UNETLoader", "unet_name", sorted(set(t["dit"].values())))]
    else:
        needs = [("LoraLoaderModelOnly", "lora_name", files)] + ([("CustomSigmas", None, [])] if t["sigmas"] != "simple" else [])
    if t["patch"] is not None:
        needs.append(("KSamplerSelect", "sampler_name", [t["sampler"]]))
    return needs + [("MiniMaxH3SigmaShift", None, [])]


def needs_engine_layout(name):
    """True when the engine must publish transformer_options["minimax_h3_layout"] (not visible in object_info)."""
    return check(name)["patch"] is patch_hyperflow


def combo_options(spec):
    """A combo input's choices from an object_info input spec, in either format."""
    return spec[1]["options"] if len(spec) > 1 and isinstance(spec[1], dict) and "options" in spec[1] else spec[0]
