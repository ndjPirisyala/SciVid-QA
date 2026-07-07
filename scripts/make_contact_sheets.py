#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def make_contact_sheet(frame_paths, out_path, cols=4, thumb_width=320, label_height=24):
    images = []

    for frame_path in frame_paths:
        img = Image.open(frame_path).convert("RGB")
        w, h = img.size
        thumb_height = int(h * (thumb_width / w))
        img = img.resize((thumb_width, thumb_height))
        images.append((frame_path, img))

    if not images:
        raise RuntimeError("No images provided.")

    thumb_height = images[0][1].size[1]
    rows = (len(images) + cols - 1) // cols

    sheet_width = cols * thumb_width
    sheet_height = rows * (thumb_height + label_height)

    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for idx, (frame_path, img) in enumerate(images):
        row = idx // cols
        col = idx % cols

        x = col * thumb_width
        y = row * (thumb_height + label_height)

        sheet.paste(img, (x, y + label_height))

        label = Path(frame_path).stem
        draw.text((x + 5, y + 5), label, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_manifest", required=True)
    parser.add_argument("--out_dir", default="outputs/contact_sheets")
    parser.add_argument("--manifest_out", default="outputs/contact_sheets/contact_sheet_manifest.json")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--thumb_width", type=int, default=320)
    args = parser.parse_args()

    frame_manifest = load_json(args.frame_manifest)
    out_root = Path(args.out_dir)

    records = []

    for record in frame_manifest["records"]:
        condition_id = record["condition_id"]
        prompt_id = record["prompt_id"]

        frame_paths = [f["path"] for f in record["sampled_frames"]]

        out_path = out_root / condition_id / f"{prompt_id}.jpg"

        print(f"Making contact sheet: {condition_id} / {prompt_id}")
        make_contact_sheet(
            frame_paths=frame_paths,
            out_path=out_path,
            cols=args.cols,
            thumb_width=args.thumb_width
        )

        records.append({
            "condition_id": condition_id,
            "prompt_id": prompt_id,
            "contact_sheet_path": str(out_path),
            "source_frames": frame_paths
        })

    manifest = {
        "source_frame_manifest": args.frame_manifest,
        "records": records
    }

    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved contact sheet manifest to: {manifest_out}")


if __name__ == "__main__":
    main()