# 功能：根据去畸变后的 images 目录生成 AI3D 风格的多尺度图像目录 images_2、images_4、images_8。

import argparse
import os
from pathlib import Path

from PIL import Image


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".BMP"}


def resize_with_scale(input_path: Path, output_path: Path, scale: int):
    with Image.open(input_path) as image:
        image = image.convert("RGB")
        new_width = max(1, image.width // scale)
        new_height = max(1, image.height // scale)
        resized = image.resize((new_width, new_height), Image.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Build image pyramid for Gaussian pipeline")
    parser.add_argument("--images_dir", required=True, help="Base undistorted images directory")
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 4, 8], help="Downsample scales to generate")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    image_files = []
    for root, _, files in os.walk(images_dir):
        root_path = Path(root)
        for name in sorted(files):
            if Path(name).suffix in VALID_EXTENSIONS:
                image_files.append((root_path, name))

    if not image_files:
        raise RuntimeError(f"No images found under: {images_dir}")

    for scale in args.scales:
        out_dir = images_dir.parent / f"{images_dir.name}_{scale}"
        for root_path, name in image_files:
            rel_dir = root_path.relative_to(images_dir)
            resize_with_scale(root_path / name, out_dir / rel_dir / name, scale)
        print(f"Generated {out_dir}")


if __name__ == "__main__":
    main()
