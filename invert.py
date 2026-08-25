import argparse
from pathlib import Path

from PIL import Image


def invert_grayscale_folder(input_dir: Path, output_dir: Path | None = None) -> None:
    input_dir = input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = (output_dir or input_dir / "inverted").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(input_dir.rglob("*.png"))
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return

    for image_path in png_files:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            inverted = Image.eval(gray, lambda p: 255 - p)

            rel_path = image_path.relative_to(input_dir)
            target_path = output_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            inverted.save(target_path)
            print(f"Created {target_path}")

    print(f"Done. Inverted {len(png_files)} image(s).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invert grayscale PNG images in a folder."
    )
    parser.add_argument("input_dir", help="Folder containing grayscale PNG images")
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Optional output folder for inverted images (default: <input_dir>/inverted)",
    )
    args = parser.parse_args()

    invert_grayscale_folder(Path(args.input_dir), Path(args.output_dir) if args.output_dir else None)


if __name__ == "__main__":
    main()
