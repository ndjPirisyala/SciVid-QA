#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import cv2


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def sample_indices(num_frames, num_samples):
    if num_frames <= 0:
        return []
    if num_samples <= 1:
        return [num_frames // 2]
    return sorted(set(round(i * (num_frames - 1) / (num_samples - 1)) for i in range(num_samples)))


def sample_video(video_path, out_dir, num_samples):
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_indices(num_frames, num_samples)
    wanted = set(indices)

    saved = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in wanted:
            out_path = out_dir / f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved.append({
                "frame_index": frame_idx,
                "path": str(out_path)
            })

        frame_idx += 1

    cap.release()

    if len(saved) == 0:
        raise RuntimeError(f"No frames saved for video: {video_path}")

    return {
        "video_path": str(video_path),
        "num_frames_reported": num_frames,
        "num_frames_read": frame_idx,
        "sampled_frames": saved
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True)
    parser.add_argument("--out_dir", default="outputs/frames")
    parser.add_argument("--num_frames", type=int, default=12)
    parser.add_argument("--manifest_out", default="outputs/frames/frame_manifest.json")
    args = parser.parse_args()

    video_sets = load_json(args.videos)
    out_root = Path(args.out_dir)
    all_records = []

    for condition in video_sets["conditions"]:
        condition_id = condition["condition_id"]

        for video in condition["videos"]:
            prompt_id = video["prompt_id"]
            video_path = Path(video["video_path"])

            out_dir = out_root / condition_id / prompt_id
            print(f"Sampling {condition_id} / {prompt_id}: {video_path}")

            record = sample_video(video_path, out_dir, args.num_frames)
            record["condition_id"] = condition_id
            record["prompt_id"] = prompt_id
            all_records.append(record)

    manifest = {
        "source_video_manifest": args.videos,
        "num_sampled_frames_per_video": args.num_frames,
        "records": all_records
    }

    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved frame manifest to: {manifest_out}")


if __name__ == "__main__":
    main()