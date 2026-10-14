"""Assemble the ai-toolkit dataset folder for v6 trajectory training from recordings and their decoded clips.

    python tools/make_dataset.py RECORDINGS_DIR VIDEOS_DIR DATASET_DIR [--copy] [--wording]

For every RECORDINGS_DIR/<name>.safetensors, the clip VIDEOS_DIR/<name>_*.mp4 (what SaveVideo writes for the
filename_prefix <name>) becomes DATASET_DIR/<name>.mp4 (a symlink, or a copy with --copy), and DATASET_DIR/<name>.txt
gets the recorded prompt. Trajectory training reads the state, timestep and target from the recording; the dataset
only has to name the recordings (same file stem), and its captions are used only with
continuation_trajectory_text: dataset. --wording prepends the continuation wording for 124-frame clips to each
caption (the text continuation clips are generated with). Refuses to overwrite files that differ.
"""
import argparse
import glob
import json
import os
import shutil
import sys

WORDING = ("Continue the existing scene from the previous generated clip with no cut, reset, or re-establishment. "
           "The incoming protected audiovisual opening is authoritative for current pose, motion, camera trajectory, "
           "facial state, lighting, colour, environment, object state, voice, ambience and timing. "
           "The visible characters retain their identity and facial appearance. "
           "After this protected 1.625-second opening, the remaining {rest:.3f} seconds continue directly:\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings")
    p.add_argument("videos")
    p.add_argument("dataset")
    p.add_argument("--copy", action="store_true")
    p.add_argument("--wording", action="store_true")
    args = p.parse_args()
    os.makedirs(args.dataset, exist_ok=True)
    made, problems = 0, []
    for rec in sorted(glob.glob(os.path.join(args.recordings, "*.safetensors"))):
        name = os.path.basename(rec)[:-len(".safetensors")]
        sidecar = rec[:-len(".safetensors")] + ".json"
        if not os.path.exists(sidecar):
            problems.append(f"{name}: no sidecar json")
            continue
        with open(sidecar) as f:
            info = json.load(f)
        clips = sorted(glob.glob(os.path.join(args.videos, glob.escape(name) + "_*.mp4")))
        if len(clips) != 1:
            problems.append(f"{name}: {len(clips)} clips match {name}_*.mp4")
            continue
        video = os.path.join(args.dataset, name + ".mp4")
        if os.path.lexists(video):
            if os.path.realpath(video) != os.path.realpath(clips[0]):
                problems.append(f"{name}: {video} exists and is another file")
                continue
        elif args.copy:
            shutil.copy2(clips[0], video)
        else:
            os.symlink(os.path.abspath(clips[0]), video)
        caption = info.get("prompt") or ""
        if args.wording:
            frames = 17 * ((info["shapes"]["video"][2] - 2) // 5) + 5
            caption = WORDING.format(rest=(frames - 39) / 24) + caption
        text = os.path.join(args.dataset, name + ".txt")
        if os.path.exists(text) and open(text).read() != caption:
            problems.append(f"{name}: {text} exists with another caption")
            continue
        with open(text, "w") as f:
            f.write(caption)
        made += 1
    print(f"{made} dataset items in {args.dataset}")
    for problem in problems:
        print("SKIPPED", problem)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
