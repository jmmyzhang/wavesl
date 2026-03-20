#!/usr/bin/env python3
"""
Utility script to prepare ASL dataset for training.
Helps organize video files and create class mappings.
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def organize_dataset(
    source_dir: str,
    output_dir: str,
    class_names: Optional[list[str]] = None,
) -> None:
    """
    Organize video files into class directories.

    Args:
        source_dir: Directory containing all video files
        output_dir: Directory to create organized dataset
        class_names: Optional list of class names to create directories for
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if class_names:
        for class_name in class_names:
            (output_path / class_name).mkdir(parents=True, exist_ok=True)

    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files: list[Path] = []
    for ext in video_extensions:
        video_files.extend(source_path.rglob(f"*{ext}"))

    print(f"Found {len(video_files)} video files")

    if not class_names:
        print("\nTo organize your dataset:")
        print("1. Create subdirectories in the output directory for each sign class")
        print("2. Move video files into the appropriate class directories")
        print("3. Example structure:")
        print("   dataset/")
        print("     hello/")
        print("       video1.mp4")
        print("       video2.mp4")
        print("     thank_you/")
        print("       video1.mp4")
        print("       video2.mp4")
        return

    print(f"\nCreated directories for {len(class_names)} classes")
    print("Please organize your video files into these directories")


def create_class_mapping(
    data_dir: str,
    output_file: Optional[str] = None,
) -> dict[str, int]:
    """
    Create class mapping from directory structure.

    Args:
        data_dir: Directory containing class subdirectories
        output_file: Optional file to save mapping JSON
    """
    data_path = Path(data_dir)

    class_names = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    class_mapping = {name: idx for idx, name in enumerate(class_names)}

    print(f"Found {len(class_names)} classes:")
    for name, idx in class_mapping.items():
        video_count = len(list((data_path / name).glob("*.mp4")))
        print(f"  {idx}: {name} ({video_count} videos)")

    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(class_mapping, f, indent=2)
        print(f"\nSaved class mapping to {output_file}")

    return class_mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ASL dataset for training")
    parser.add_argument(
        "command", choices=["organize", "mapping"], help="Command to run"
    )
    parser.add_argument(
        "--source-dir", type=str, help="Source directory containing video files"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for organized dataset"
    )
    parser.add_argument(
        "--data-dir", type=str, help="Directory with class subdirectories"
    )
    parser.add_argument("--classes", type=str, nargs="+", help="List of class names")
    parser.add_argument(
        "--output-file",
        type=str,
        default="class_mapping.json",
        help="Output file for class mapping",
    )

    args = parser.parse_args()

    if args.command == "organize":
        if not args.source_dir or not args.output_dir:
            print("Error: --source-dir and --output-dir required for organize command")
            return
        organize_dataset(args.source_dir, args.output_dir, args.classes)

    elif args.command == "mapping":
        if not args.data_dir:
            print("Error: --data-dir required for mapping command")
            return
        create_class_mapping(args.data_dir, args.output_file)


if __name__ == "__main__":
    main()
