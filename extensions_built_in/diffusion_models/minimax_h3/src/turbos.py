"""Turbos for MiniMax-H3 training, by the names of the turbo registry the recording and eval tooling use.

The registry is a Python file (``model_kwargs.turbo_registry``, a copy of the rental tooling's ``turbos.py``) with a
``TURBOS`` dict: per name the files per DiT partition (``files["fl2va"|"ref2va"]``), the sampler shifts
``(video, audio)``, the step count and sigma grid ("simple" or a fixed tuple), the strength, and a ``patch`` function
whose name says what kind of turbo it is. Kinds the trainer runs:

- ``lora``: a plain backbone LoRA (``patch_lora_turbo``: lightx2v8, hyperflow_lora; the engine runs them live with the
  H3RuntimeLoRA node) and Larry (``patch`` None: the MiniMax-H3-Turbo node, a backbone LoRA plus 51 adaLN pieces
  re-injected at run time through the node's silu(t_emb) grid). The trainer applies them live as the frozen assistant
  (src/runtime_lora.py), never merged; several share one base DiT and are switched per sample.
- ``checkpoint``: a DiT with the turbo baked in (``patch_checkpoint_turbo``: merged_turbo). That file is the DiT; no assistant.

Refused: the HyperFlow node (``patch_hyperflow``: ApplyHyperFlow rewrites the time conditioning; not a plain LoRA) and
entries marked ``blocked``.

This file has no package imports (stdlib only), so tooling can load it by path, without the trainer, to check a config
before a run: ``check_model_kwargs(model_kwargs, partition, dataset_folder)`` or ``python turbos.py config.yaml``.
"""
import importlib.util
import json
import math
import os
import re
import struct
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

PARTITIONS = ("fl2va", "ref2va")
LORA_PATCHES = ("patch_lora_turbo", None)  # None: Larry's own node (MiniMaxH3TurboLoRA)
CHECKPOINT_PATCHES = ("patch_checkpoint_turbo",)
DEFAULT_SHIFTS = (12.0, 3.0)  # a recording without a schedule block was sampled at the released shifts

# LoRA file layout: what the H3RuntimeLoRA node reads (its _pairs) and what the MiniMax-H3-Turbo node adds (adaLN)
PREFIX = "diffusion_model."
SUFFIXES = ((".lora_A.weight", "A"), (".lora_down.weight", "A"), (".lora_B.weight", "B"), (".lora_up.weight", "B"),
            (".alpha", "alpha"))
BACKBONE = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
ADALN = "adaln_proj.linear"
ADALN_WIDTH = 2688  # the full model's silu(t_emb) width the adaLN pieces act on (the turbo node's grid columns)
GRID_KEY = "silu_t_emb_grid"

# the pruned H3 DiTs (Comfy-Org fl2va/ref2va pruned int8, merged-turbo checkpoints): (out, in) per linear, for header-only checks
H3_BLOCKS, H3_REFINER_BLOCKS = 50, 2
H3_LINEARS = {"attn.qkv_proj": (21504, 5376), "attn.out_proj": (5376, 7168), "mlp.fc1": (28672, 5376),
              "mlp.fc2": (5376, 14336)}
H3_ADALN_OUT = {"blocks": 96768, "final_layer": 10752}
H3_TABLE_ROWS = 1025


@dataclass(frozen=True)
class Turbo:
    name: str
    kind: str  # "lora" | "checkpoint"
    file: str  # registry file name: the LoRA (lora) or the DiT (checkpoint) of this partition
    strength: float
    shift_video: float
    shift_audio: float
    steps: int
    sigmas: Optional[Tuple[float, ...]]  # None: the "simple" grid of the shifted schedule
    adaln: bool  # the engine re-injects adaLN pieces (Larry's node): the file must carry them

    @property
    def shifts(self) -> Tuple[float, float]:
        return self.shift_video, self.shift_audio


def partition_of(partition: str) -> str:
    """model_kwargs.partition (fl2va, fl2va_pruned, ref2va, ref2va_pruned) -> the registry's file key."""
    base = str(partition).lower().split("_")[0]
    if base not in PARTITIONS:
        raise ValueError(f"no turbo files for partition {partition!r} (fl2va or ref2va)")
    return base


def names_of(value) -> Tuple[str, ...]:
    """model_kwargs.turbo: one registry name or a list of them."""
    if value is None:
        return ()
    names = (value,) if isinstance(value, str) else tuple(value)
    names = tuple(str(n).strip() for n in names)
    if not names or any(not n for n in names):
        raise ValueError(f"model_kwargs.turbo must name registry turbos, got {value!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"model_kwargs.turbo lists a turbo twice: {list(names)}")
    return names


def load_registry(path: str) -> Dict[str, dict]:
    """The registry's TURBOS dict. The file has no import-time side effects (the eval driver imports it too)."""
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"model_kwargs.turbo_registry: no registry file at {path!r}")
    spec = importlib.util.spec_from_file_location("h3_turbo_registry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    turbos = getattr(module, "TURBOS", None)
    if not isinstance(turbos, dict):
        raise ValueError(f"{path} has no TURBOS dict")
    return turbos


def resolve(registry: Mapping[str, dict], name: str, partition: str) -> Turbo:
    if name not in registry:
        raise ValueError(f"turbo {name!r} is not in the registry ({', '.join(registry)})")
    entry = registry[name]
    if entry.get("blocked"):
        raise ValueError(f"turbo {name} cannot run: {entry['blocked']}")
    patch = entry.get("patch")
    patch = getattr(patch, "__name__", patch)
    if patch in LORA_PATCHES:
        kind = "lora"
    elif patch in CHECKPOINT_PATCHES:
        kind = "checkpoint"
    else:
        raise ValueError(f"turbo {name} ({patch}) is neither a plain LoRA nor a checkpoint turbo; the trainer cannot "
                         "run it")
    part = partition_of(partition)
    file = (entry.get("dit") or {}).get(part) if kind == "checkpoint" else (entry.get("files") or {}).get(part)
    if not file:
        raise ValueError(f"turbo {name} has no {part} file")
    steps = int(entry["steps"])
    sigmas = entry.get("sigmas", "simple")
    if sigmas == "simple":
        sigmas = None
    elif isinstance(sigmas, (tuple, list)):
        sigmas = tuple(float(s) for s in sigmas)
        if len(sigmas) != steps + 1 or sigmas[0] != 1.0 or sigmas[-1] != 0.0:
            raise ValueError(f"turbo {name}: its sigma grid must run from 1.0 to 0.0 in steps + 1 values")
    else:
        raise ValueError(f"turbo {name}: sigmas {sigmas!r} are neither 'simple' nor a grid")
    shift = entry["shift"]
    return Turbo(name=name, kind=kind, file=str(file), strength=float(entry.get("strength", 1.0)),
                 shift_video=float(shift[0]), shift_audio=float(shift[1]), steps=steps, sigmas=sigmas,
                 adaln=kind == "lora" and patch is None)


def resolve_all(registry: Mapping[str, dict], names: Iterable[str], partition: str) -> Dict[str, Turbo]:
    """The training mix, in config order: LoRA turbos (any number), or one checkpoint turbo on its own."""
    turbos: Dict[str, Turbo] = {}
    for name in names:
        turbos[name] = resolve(registry, name, partition)
    if not turbos:
        raise ValueError("model_kwargs.turbo names no turbo")
    if any(t.kind == "checkpoint" for t in turbos.values()) and len(turbos) > 1:
        raise ValueError(f"a checkpoint turbo is its own DiT; it cannot share a run with other turbos ({', '.join(turbos)})")
    return turbos


def file_path(turbo: Turbo, kw: Mapping) -> str:
    """Where a turbo's file is: ``turbo_paths[name]``, else ``turbo_lora_dir`` (LoRA) / ``turbo_dit_dir`` (checkpoint)
    joined with the registry file name."""
    override = (kw.get("turbo_paths") or {}).get(turbo.name)
    if override:
        return str(override)
    key = "turbo_lora_dir" if turbo.kind == "lora" else "turbo_dit_dir"
    folder = kw.get(key)
    if not folder:
        raise ValueError(f"turbo {turbo.name}: set model_kwargs.{key} (the folder holding {turbo.file}) or "
                         f"turbo_paths.{turbo.name}")
    return os.path.join(str(folder), turbo.file)


# ----------------------------------------------------------------------------------------------------------------
# LoRA file -> trainer modules
# ----------------------------------------------------------------------------------------------------------------
def kind_of(module: str) -> str:
    """"blocks.3.attn.qkv_proj" -> "attn.qkv_proj", "blocks.3.adaln_proj.linear" -> "adaln_proj.linear"."""
    parent, _, leaf = module.rpartition(".")
    return parent.rpartition(".")[2] + "." + leaf


def group_keys(keys: Iterable[str], name: str = "") -> Dict[str, Dict[str, str]]:
    """{module path: {"A"|"B"|"alpha": key}}, the way the H3RuntimeLoRA node reads a file ("diffusion_model." prefix
    optional; lora_A/lora_down, lora_B/lora_up, alpha). Any other key is an error: nothing is silently skipped."""
    parts: Dict[str, Dict[str, str]] = {}
    for key in keys:
        stripped = key[len(PREFIX):] if key.startswith(PREFIX) else key
        for suffix, slot in SUFFIXES:
            if stripped.endswith(suffix):
                slots = parts.setdefault(stripped[: -len(suffix)], {})
                if slot in slots:
                    raise ValueError(f"turbo {name}: {key} and {slots[slot]} are the same LoRA tensor")
                slots[slot] = key
                break
        else:
            raise ValueError(f"turbo {name}: unexpected key {key} (not a lora_A/lora_B/alpha tensor)")
    for module, slots in parts.items():
        if "A" not in slots or "B" not in slots:
            raise ValueError(f"turbo {name}: {module} lacks lora_A or lora_B")
    return parts


@dataclass(frozen=True)
class Plan:
    backbone: Dict[str, Tuple[str, str, Optional[str]]]  # module path -> (A key, B key, alpha key or None)
    adaln: Dict[str, Tuple[str, str]]  # adaln_proj path ("blocks.0.adaln_proj") -> (A key, B key)

    @property
    def tensors(self) -> int:
        return sum(3 if al else 2 for _, _, al in self.backbone.values()) + 2 * len(self.adaln)


def plan(shapes: Mapping[str, Sequence[int]], expected: Callable[[str], Optional[Tuple[int, int]]], name: str = "",
         adaln_width: int = ADALN_WIDTH) -> Plan:
    """Map every tensor of a turbo LoRA onto the model: backbone pairs (qkv_proj, out_proj, fc1, fc2 of the blocks and
    the token refiner) and adaLN pieces (``*.adaln_proj.linear``, Larry). ``expected(module)`` gives a linear's
    (out, in) or None when the model has no such linear. Raises on any key that does not map, any shape that does not
    fit, and alpha on an adaLN piece (the node applies those as plain B @ A)."""
    backbone, adaln = {}, {}
    for module, slots in sorted(group_keys(shapes, name).items()):
        a, b = tuple(shapes[slots["A"]]), tuple(shapes[slots["B"]])
        kind = kind_of(module)
        fits = expected(module)
        if fits is None:
            raise ValueError(f"turbo {name}: {module} is not a linear of this model")
        if len(a) != 2 or len(b) != 2 or a[0] != b[1]:
            raise ValueError(f"turbo {name}: {module} A{a} B{b} are not a LoRA pair")
        if "alpha" in slots and tuple(shapes[slots["alpha"]]) not in ((), (1,)):
            raise ValueError(f"turbo {name}: {module} alpha is not a scalar")
        if kind in BACKBONE:
            if (b[0], a[1]) != tuple(fits):
                raise ValueError(f"turbo {name}: {module} A{a} B{b} does not fit [{fits[0]}, {fits[1]}]")
            backbone[module] = (slots["A"], slots["B"], slots.get("alpha"))
        elif kind == ADALN:
            if "alpha" in slots:
                raise ValueError(f"turbo {name}: {module} carries an alpha; adaLN pieces are applied as plain B @ A")
            if b[0] != fits[0] or a[1] != adaln_width:
                raise ValueError(f"turbo {name}: {module} A{a} B{b} does not fit the grid width {adaln_width} and "
                                 f"the projection's {fits[0]} outputs")
            adaln[module[: -len(".linear")]] = (slots["A"], slots["B"])
        else:
            raise ValueError(f"turbo {name}: {module} is not a backbone linear ({', '.join(BACKBONE)}) or an adaLN "
                             "projection")
    return Plan(backbone=backbone, adaln=adaln)


def check_adaln(turbo: Turbo, p: Plan) -> None:
    """Larry's node always re-injects its adaLN pieces; H3RuntimeLoRA refuses any. The file must say the same."""
    if turbo.adaln and not p.adaln:
        raise ValueError(f"turbo {turbo.name}: its node re-injects adaLN pieces at run time, but the file has none "
                         f"(use the full {turbo.file}, not a stripped copy)")
    if not turbo.adaln and p.adaln:
        raise ValueError(f"turbo {turbo.name}: the file carries {len(p.adaln)} adaLN pieces, which its engine node "
                         "(H3RuntimeLoRA) refuses")


def h3_expected(module: str) -> Optional[Tuple[int, int]]:
    """(out, in) of a pruned H3 DiT linear by module path; None when there is no such linear."""
    m = re.fullmatch(r"(?:blocks\.(\d+)|token_refiner\.blocks\.(\d+))\.(attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)",
                     module)
    if m:
        block, refiner, kind = m.groups()
        ok = int(block) < H3_BLOCKS if block is not None else int(refiner) < H3_REFINER_BLOCKS
        return H3_LINEARS[kind] if ok else None
    m = re.fullmatch(r"(?:blocks\.(\d+)|final_layer)\.adaln_proj\.linear", module)
    if m:
        if m.group(1) is None:
            return H3_ADALN_OUT["final_layer"], 8
        return (H3_ADALN_OUT["blocks"], 8) if int(m.group(1)) < H3_BLOCKS else None
    return None


def read_header(path: str) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """A safetensors file's tensor table and metadata, without reading any tensor (stdlib only). A file that is not
    safetensors, or whose size is not exactly what its header describes (a truncated copy), is a ValueError."""
    total = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        size = struct.unpack("<Q", head)[0] if len(head) == 8 else None
        if size is None or size > total - 8:
            raise ValueError(f"{path} is not a safetensors file, or is truncated ({total} bytes on disk)")
        try:
            header = json.loads(f.read(size))
            meta = header.pop("__metadata__", None) or {}
            end = max((int(v["data_offsets"][1]) for v in header.values()), default=0)
        except (ValueError, AttributeError, KeyError, TypeError, IndexError) as e:
            raise ValueError(f"{path} is not a safetensors file (its header does not parse: {e!r:.200})") from None
    if 8 + size + end != total:
        raise ValueError(f"{path} is truncated or damaged: its header describes {8 + size + end} bytes, "
                         f"{total} are on disk")
    return header, meta


def check_lora_file(turbo: Turbo, path: str, grid_path: Optional[str] = None) -> Plan:
    """Header-only check of a LoRA turbo file against the pruned H3 layout: every tensor maps, adaLN as its node."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"turbo {turbo.name}: no file at {path}")
    header, _ = read_header(path)
    p = plan({k: v["shape"] for k, v in header.items()}, h3_expected, turbo.name)
    check_adaln(turbo, p)
    if p.adaln:
        if not grid_path or not os.path.isfile(grid_path):
            raise FileNotFoundError(f"turbo {turbo.name} carries adaLN pieces: set model_kwargs.turbo_adaln_grid_path "
                                    f"to the turbo node's h3_silu_temb_grid.safetensors (got {grid_path!r})")
        grid = read_header(grid_path)[0].get(GRID_KEY)
        if grid is None or list(grid["shape"]) != [H3_TABLE_ROWS, ADALN_WIDTH]:
            raise ValueError(f"{grid_path}: no {GRID_KEY} [{H3_TABLE_ROWS}, {ADALN_WIDTH}] tensor")
    if p.tensors != len(header):
        raise ValueError(f"turbo {turbo.name}: {len(header) - p.tensors} tensors of {path} would be unused")
    return p


# ----------------------------------------------------------------------------------------------------------------
# recordings
# ----------------------------------------------------------------------------------------------------------------
def recording_turbo(path: str, info: Mapping, known: Iterable[str]) -> Optional[str]:
    """The turbo a recording was sampled with: its info's "turbo" (a name, or a block with "name"), else the
    recorder's <stem>.json sidecar, else a known turbo name as a whole segment of the file stem (A_<turbo>_<idx>).
    None when nothing says."""
    def name_in(block):
        if isinstance(block, Mapping):
            block = block.get("name")
        return str(block) if block else None

    found = name_in(info.get("turbo"))
    if found:
        return found
    stem = os.path.splitext(path)[0]
    try:
        with open(stem + ".json") as f:
            found = name_in(json.load(f).get("turbo"))
        if found:
            return found
    except (OSError, ValueError, AttributeError):
        pass
    base = "_" + os.path.basename(stem) + "_"
    for name in sorted(known, key=len, reverse=True):
        if f"_{name}_" in base:
            return name
    return None


def _close(a, b) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-6)


def check_recording(path: str, info: Mapping, turbos: Mapping[str, Turbo]) -> str:
    """The configured turbo a recording belongs to, checked against the registry: its schedule's shifts and, when the
    recording carries a turbo block, the file, strength, step count and sigma grid. An untagged recording counts as
    the only configured turbo; with several configured it is refused."""
    name = recording_turbo(path, info, turbos)
    if name is None:
        if len(turbos) != 1:
            raise ValueError(f"{path} does not say which turbo sampled it (info['turbo']), and the run mixes "
                             f"{', '.join(turbos)}")
        name = next(iter(turbos))
    if name not in turbos:
        raise ValueError(f"{path} was sampled with turbo {name}, which model_kwargs.turbo does not list "
                         f"({', '.join(turbos)})")
    t = turbos[name]
    schedule = info.get("schedule") or {}
    shifts = (schedule.get("shift_video", DEFAULT_SHIFTS[0]), schedule.get("shift_audio", DEFAULT_SHIFTS[1]))
    if not (_close(shifts[0], t.shift_video) and _close(shifts[1], t.shift_audio)):
        raise ValueError(f"{path}: sampled at shifts {tuple(shifts)}, turbo {name} runs at {t.shifts}")
    block = info.get("turbo")
    if isinstance(block, Mapping):
        problems = []
        recorded = block.get("lora_file") or block.get("dit_file")
        if recorded and os.path.basename(str(recorded)) != t.file:
            problems.append(f"file {recorded} (this partition's registry file is {t.file})")
        if "strength" in block and not _close(block["strength"], t.strength):
            problems.append(f"strength {block['strength']} (registry {t.strength})")
        if "steps" in block and int(block["steps"]) != t.steps:
            problems.append(f"{block['steps']} steps (registry {t.steps})")
        if "shift" in block and not all(_close(a, b) for a, b in zip(block["shift"], t.shifts)):
            problems.append(f"shift {block['shift']} (registry {t.shifts})")
        sigmas = block.get("sigmas")
        if isinstance(sigmas, (list, tuple)) and (t.sigmas is None or len(sigmas) != len(t.sigmas)
                                                   or not all(_close(a, b) for a, b in zip(sigmas, t.sigmas))):
            problems.append("a sigma grid the registry does not give")
        if sigmas == "simple" and t.sigmas is not None:
            problems.append("the simple grid (the registry gives a fixed grid)")
        if problems:
            raise ValueError(f"{path} (turbo {name}): " + "; ".join(problems))
    return name



# ----------------------------------------------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------------------------------------------
LEGACY_ASSISTANT_KEYS = ("assistant_adaln_lora_path", "assistant_adaln_grid_path", "assistant_adaln_strength")
MEDIA_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".flv", ".wmv", ".png", ".jpg", ".jpeg", ".webp")


def configured(kw: Mapping, partition: str, assistant_lora_path=None) -> Optional[Dict[str, Turbo]]:
    """The run's turbos from model_kwargs (None when ``turbo`` is not set). Refuses the single-assistant keys next to
    them: in turbo mode the registry decides the assistant."""
    names = names_of(kw.get("turbo"))
    if not names:
        for key in ("turbo_registry", "turbo_lora_dir", "turbo_dit_dir", "turbo_paths", "turbo_adaln_grid_path",
                    "turbo_sample", "turbo_weights"):
            if kw.get(key):
                raise ValueError(f"model_kwargs.{key} is set but model_kwargs.turbo names no turbo")
        return None
    stale = [k for k in LEGACY_ASSISTANT_KEYS if kw.get(k) is not None]
    if assistant_lora_path:
        stale.insert(0, "model.assistant_lora_path")
    if stale:
        raise ValueError(f"model_kwargs.turbo replaces the single assistant; remove {stale}")
    turbos = resolve_all(load_registry(str(kw.get("turbo_registry") or "")), names, partition)
    unknown = [n for n in (kw.get("turbo_paths") or {}) if n not in turbos]
    unknown += [n for n in (kw.get("turbo_weights") or {}) if n not in turbos]
    if kw.get("turbo_sample") and kw["turbo_sample"] not in turbos:
        unknown.append(kw["turbo_sample"])
    if unknown:
        raise ValueError(f"model_kwargs names turbos the run does not use: {unknown} (turbo: {list(turbos)})")
    weights = kw.get("turbo_weights") or {}
    if any(float(w) < 0 for w in weights.values()) or (weights and sum(float(weights.get(n, 1.0)) for n in turbos) <= 0):
        raise ValueError(f"model_kwargs.turbo_weights must be non-negative with a positive sum, got {weights}")
    return turbos


def check_model_kwargs(kw: Mapping, partition: str, dataset_folder: Optional[str] = None,
                       assistant_lora_path=None) -> str:
    """Everything a turbo run needs, checked without the trainer: registry names, files (header-only key and shape
    check of every LoRA, the adaLN grid for Larry), and with ``dataset_folder`` every item's recording and its turbo.
    Returns a summary; raises on the first problem."""
    turbos = configured(kw, partition, assistant_lora_path)
    if turbos is None:
        return "no turbo configured"
    lines = []
    for t in turbos.values():
        path = file_path(t, kw)
        if t.kind == "lora":
            p = check_lora_file(t, path, kw.get("turbo_adaln_grid_path"))
            lines.append(f"{t.name}: {os.path.basename(path)} ({len(p.backbone)} live linears"
                         f"{f', {len(p.adaln)} adaLN pieces' if p.adaln else ''}), shifts {t.shifts}, "
                         f"{'fixed grid' if t.sigmas else 'simple'} {t.steps} steps")
        else:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"turbo {t.name}: no DiT at {path}")
            header, meta = read_header(path)
            if "_quantization_metadata" not in meta and not any(k.endswith(".comfy_quant") for k in header):
                raise ValueError(f"turbo {t.name}: {path} is not a pre-quantized (comfy_quant) DiT")
            lines.append(f"{t.name}: DiT {os.path.basename(path)}, shifts {t.shifts}")
    if dataset_folder and kw.get("continuation_trajectory_dir"):
        directory = str(kw["continuation_trajectory_dir"])
        stems = sorted({os.path.splitext(n)[0] for n in os.listdir(dataset_folder)
                        if os.path.splitext(n)[1].lower() in MEDIA_EXTS})
        counts = dict.fromkeys(turbos, 0)
        for stem in stems:
            path = os.path.join(directory, stem + ".safetensors")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"dataset item {stem} has no recording at {path}")
            meta = read_header(path)[1]
            counts[check_recording(path, json.loads(meta.get("info") or "{}"), turbos)] += 1
        lines.append(f"{len(stems)} recordings: " + ", ".join(f"{n} {c}" for n, c in counts.items()))
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    import yaml  # the trainer's configs are YAML; only the CLI needs it

    if len(argv) != 1:
        print("usage: turbos.py <ai-toolkit config.yaml>", file=sys.stderr)
        return 2
    with open(argv[0]) as f:
        config = yaml.safe_load(f)
    process = config["config"]["process"][0]
    model = process["model"]
    kw = model.get("model_kwargs") or {}
    datasets = [d.get("folder_path") for d in process.get("datasets") or []]
    summary = [check_model_kwargs(kw, str(kw.get("partition", "fl2va_pruned")), d, model.get("assistant_lora_path"))
               for d in (datasets or [None])]
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
