#!/usr/bin/env python3
"""
Prepare WLASL dataset for training.
Downloads and organizes WLASL dataset into the format needed for training.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import argparse
from tqdm import tqdm


def organize_wlasl_videos(
    wlasl_dir: Path,
    output_dir: Path,
    class_mapping_file: Optional[Path] = None,
) -> Optional[dict[str, int]]:
    """
    Organize WLASL videos into class directories.

    WLASL structure can be:
    - videos/ directory with all videos (after preprocessing)
    - start_kit/videos/ directory
    - start_kit/raw_videos/ directory
    - JSON file with annotations (WLASL_v0.3.json format)
    """
    possible_video_dirs = [
        wlasl_dir / "videos",
        wlasl_dir / "start_kit" / "videos",
        wlasl_dir / "start_kit" / "raw_videos",
    ]

    videos_dir = next((d for d in possible_video_dirs if d.exists()), None)

    if videos_dir is None:
        print(f"Error: No videos directory found in {wlasl_dir}")
        print("Expected one of:")
        for vdir in possible_video_dirs:
            print(f"  - {vdir}")
        print("\nTo download videos, see WLASL README.md or run:")
        print("  cd start_kit && python video_downloader.py && python preprocess.py")
        return None

    print(f"Found videos directory: {videos_dir}")

    annotation_files = [
        wlasl_dir / "start_kit" / "WLASL_v0.3_100.json",
        wlasl_dir / "start_kit" / "WLASL_v0.3_300.json",
        wlasl_dir / "start_kit" / "WLASL_v0.3.json",
        wlasl_dir / "WLASL_v0.3.json",
    ] + list(wlasl_dir.glob("**/WLASL_v*.json"))

    ann_file = next((af for af in annotation_files if af.exists()), None)

    if ann_file is None:
        print("Warning: No WLASL annotation JSON file found")
        print("Expected: start_kit/WLASL_v0.3.json")
        return None

    print(f"Found annotation file: {ann_file}")

    annotations: dict[str, str] = {}
    try:
        with open(ann_file) as f:
            data = json.load(f)

        if isinstance(data, list):
            for gloss_entry in data:
                gloss = gloss_entry.get("gloss", "")
                instances = gloss_entry.get("instances", [])
                for instance in instances:
                    video_id = instance.get("video_id", "")
                    if video_id and gloss:
                        annotations[video_id] = gloss
        print(
            f"Loaded {len(annotations)} video annotations "
            f"for {len(set(annotations.values()))} unique glosses"
        )
    except Exception as e:
        print(f"Error parsing annotation file: {e}")
        return None

    if not annotations:
        print("Error: No annotations found in JSON file")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = list(videos_dir.glob("*.mp4"))
    if not video_files:
        video_files = list(videos_dir.rglob("*.mp4"))

    organized = 0
    missing_annotations: list[str] = []

    print(f"Found {len(video_files)} video files")
    print(f"Organizing videos into {len(set(annotations.values()))} sign classes...")

    for video_file in tqdm(video_files, desc="Organizing videos"):
        video_id = video_file.stem

        if video_id in annotations:
            gloss = annotations[video_id]
            gloss_dir = output_dir / gloss.replace(" ", "_").replace("/", "_").replace(
                "\\", "_"
            )
            gloss_dir.mkdir(exist_ok=True)

            dest = gloss_dir / video_file.name
            if not dest.exists():
                shutil.copy2(video_file, dest)
            organized += 1
        else:
            missing_annotations.append(video_id)

    print(
        f"\nOrganized {organized} videos into {len(list(output_dir.iterdir()))} sign classes"
    )

    if missing_annotations:
        print(f"Warning: {len(missing_annotations)} videos had no annotations")

    class_dirs = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
    class_mapping = {name: idx for idx, name in enumerate(class_dirs)}

    if class_mapping_file is not None:
        with open(class_mapping_file, "w") as f:
            json.dump(class_mapping, f, indent=2)
        print(f"Saved class mapping to {class_mapping_file}")

    return class_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WLASL dataset for training")
    parser.add_argument(
        "--wlasl-dir",
        type=str,
        required=True,
        help="Directory containing WLASL dataset (should have videos/ subdirectory and JSON annotations)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/wlasl",
        help="Output directory for organized dataset",
    )
    parser.add_argument(
        "--class-mapping",
        type=str,
        default="models/wlasl_class_mapping.json",
        help="Output file for class mapping JSON",
    )

    args = parser.parse_args()

    wlasl_path = Path(args.wlasl_dir)
    output_path = Path(args.output_dir)
    mapping_path = Path(args.class_mapping)

    if not wlasl_path.exists():
        print(f"Error: WLASL directory not found: {wlasl_path}")
        print("\nTo download WLASL dataset:")
        print("1. Visit: https://github.com/dxli94/WLASL")
        print("2. Follow their download instructions")
        print("3. Extract the dataset to a directory")
        print("4. Run this script with --wlasl-dir pointing to that directory")
        return

    print(f"Preparing WLASL dataset from: {wlasl_path}")
    print(f"Output directory: {output_path}")

    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    class_mapping = organize_wlasl_videos(wlasl_path, output_path, mapping_path)

    if class_mapping:
        print("\nDataset prepared successfully!")
        print(f"  - Organized videos: {output_path}")
        print(f"  - Class mapping: {mapping_path}")
        print(f"  - Number of classes: {len(class_mapping)}")
        print("\nNext step: Train the model with:")
        print(
            f"  python src/train_asl_model.py --data-dir {output_path} --output-dir models/wlasl"
        )


if __name__ == "__main__":
    main()
