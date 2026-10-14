"""ComfyUI node: record MiniMax H3 sampling trajectories for v6 anti-burn LoRA training.

H3RecordTrajectory (MODEL -> MODEL) wraps the model so that every sampling run made with it is written to
<output_dir>/<output_name>.safetensors (+ a .json sidecar): for every model call the DiT's input state (video and
audio latents, audio in its own flow space), its timestep and its output, plus the schedule, seed, initial noise,
final latent and the positive conditioning. Put it last in the model chain (after the turbo LoRA, the sigma shift and
any attention patch), right before the guider. Conventions and file layout: h3_record_core.py and ../README.md.
The pack also carries the R+ graph nodes (rplus_nodes.py).
"""
import os

from . import rplus_nodes
from .h3_record_core import (
    TrajectoryRecorder,
    conditioning_tensors,
    make_diffusion_model_wrapper,
    make_outer_sample_wrapper,
)

WRAPPER_KEY = "h3_record_trajectory"


class H3RecordTrajectory:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "output_name": ("STRING", {
                    "default": "h3traj_{seed}",
                    "tooltip": "File stem of the recording; {seed} and {run} are filled in. The training dataset's clip "
                               "and caption files must use the same stem."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off: the model passes through untouched."}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "", "tooltip": "Empty: <ComfyUI output>/h3_trajectories"}),
                "overwrite": ("BOOLEAN", {"default": False,
                                          "tooltip": "Off: a run whose file exists fails before sampling."}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "The prompt text, stored with its sha256 in the sidecar."}),
                "student_positive": ("CONDITIONING", {
                    "tooltip": "Optional conditioning the trainer uses for continuation samples (e.g. the "
                               "continuation wording + prompt). Recorded only, never sampled with."}),
                "save_noise": ("BOOLEAN", {"default": True, "tooltip": "Also store the initial noise (fp32)."}),
                "student_context_path": ("STRING", {
                    "default": "",
                    "tooltip": "Masked continuation runs: absolute path of a safetensors file (video, audio) holding "
                               "the context the trainer pins instead of the sampled one (run.context.student)."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MiniMaxH3Turbo/record"
    DESCRIPTION = ("Record every model call of MiniMax H3 sampling runs (state, timestep, output) for trajectory "
                   "self-distillation training. Place it last in the model chain, right before the guider.")

    def patch(self, model, output_name, enabled, output_dir="", overwrite=False, prompt="", student_positive=None,
              save_noise=True, student_context_path=""):
        if not enabled:
            return (model,)
        import comfy.patcher_extension

        directory = (output_dir or "").strip()
        if not directory:
            import folder_paths

            directory = os.path.join(folder_paths.get_output_directory(), "h3_trajectories")
        recorder = TrajectoryRecorder(output_name, directory, overwrite=overwrite, prompt=prompt,
                                      student=conditioning_tensors(student_positive), save_noise=save_noise,
                                      student_context=student_context_path)
        m = model.clone()
        wrappers = comfy.patcher_extension.WrappersMP
        for kind in (wrappers.OUTER_SAMPLE, wrappers.DIFFUSION_MODEL):
            if hasattr(m, "remove_wrappers_with_key"):
                m.remove_wrappers_with_key(kind, WRAPPER_KEY)  # a re-patched model records once
        m.add_wrapper_with_key(wrappers.OUTER_SAMPLE, WRAPPER_KEY, make_outer_sample_wrapper(recorder))
        m.add_wrapper_with_key(wrappers.DIFFUSION_MODEL, WRAPPER_KEY, make_diffusion_model_wrapper(recorder))
        return (m,)


NODE_CLASS_MAPPINGS = {"H3RecordTrajectory": H3RecordTrajectory, **rplus_nodes.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {"H3RecordTrajectory": "MiniMax H3 Record Trajectory",
                              **getattr(rplus_nodes, "NODE_DISPLAY_NAME_MAPPINGS", {})}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
