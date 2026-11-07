#!/usr/bin/env python3
"""Create ChangeGCC-compatible LEVIR-CD patches without modifying source data."""

import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/xxj/Data/LEVIR_CD"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/xxj/Data/ChangeGCC_data/LEVIR_CD"),
    )
    parser.add_argument("--patch-size", type=int, default=256)
    return parser.parse_args()


def ensure_separate_paths(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError(f"Source and output must be separate directories: {source} / {output}")


def read_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def main():
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError("patch-size must be positive")

    source = args.source.resolve()
    output = args.output.resolve()
    ensure_separate_paths(source, output)

    total = 0
    for split in ("train", "val", "test"):
        label_dir = source / split / "label"
        image_a_dir = source / split / "A"
        image_b_dir = source / split / "B"
        for directory in (label_dir, image_a_dir, image_b_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"Required directory missing: {directory}")

        label_paths = sorted(path for path in label_dir.iterdir() if path.is_file())
        if not label_paths:
            raise ValueError(f"No labels found in {label_dir}")

        written = 0
        for label_path in label_paths:
            image_a_path = image_a_dir / label_path.name
            image_b_path = image_b_dir / label_path.name
            if not image_a_path.is_file() or not image_b_path.is_file():
                raise FileNotFoundError(f"Missing paired images for {label_path.name} in {split}")

            image_a = read_image(image_a_path)
            image_b = read_image(image_b_path)
            label = read_image(label_path)
            if image_a.shape[:2] != image_b.shape[:2] or image_a.shape[:2] != label.shape[:2]:
                raise ValueError(
                    f"Spatial dimensions differ for {split}/{label_path.name}: "
                    f"A={image_a.shape}, B={image_b.shape}, label={label.shape}"
                )

            height, width = label.shape[:2]
            size = args.patch_size
            if height % size or width % size:
                raise ValueError(
                    f"{split}/{label_path.name} has size {width}x{height}, "
                    f"not divisible by patch size {size}"
                )

            for top in range(0, height, size):
                for left in range(0, width, size):
                    bottom, right = top + size, left + size
                    patch_name = (
                        f"{label_path.stem}_{top}_{bottom}_{left}_{right}{label_path.suffix}"
                    )
                    patch_slices = (slice(top, bottom), slice(left, right))
                    for folder, image in (("A", image_a), ("B", image_b), ("label", label)):
                        target_dir = output / split / folder
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_path = target_dir / patch_name
                        if not cv2.imwrite(str(target_path), image[patch_slices]):
                            raise OSError(f"Could not write patch: {target_path}")
                    written += 1

        total += written
        print(f"{split}: {len(label_paths)} source images -> {written} aligned patches")

    print(f"Done. Wrote {total} patches per modality to {output}")
    print("Source images were not modified.")


if __name__ == "__main__":
    main()
