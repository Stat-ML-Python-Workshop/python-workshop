"""Week 10: check your Week 9 model file and a large pathology image before vibe coding.

From package root:
    uv run python test_code/week_10.py --image path/to/large_image.tif

Prints what the saved model contains and the image size. It does not scan.
Building the scan and the demo page is your Week 10 task.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MODEL = Path(__file__).resolve().parent / 'week_09_best_model.npz'


def describe_model(path):
    if not path.is_file():
        raise SystemExit(f'Model not found: {path}\nRun the Week 9 demo first: uv run python test_code/week_09.py --headless')
    with np.load(path, allow_pickle=False) as model:
        print(f'Model: {path}')
        for key in model.files:
            value = model[key]
            summary = value.tolist() if value.size <= 8 else f'shape={value.shape}, dtype={value.dtype}'
            print(f'  {key}: {summary}')


def describe_image(path):
    with Image.open(path) as image:
        print(f'Image: {path}')
        print(f'  size (width, height): {image.size}, mode: {image.mode}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='Week 9 week_09_best_model.npz')
    parser.add_argument('--image', type=Path, help='Large pathology image (for example a Kather 2016 TIF)')
    args = parser.parse_args(argv)
    describe_model(args.model)
    if args.image is not None:
        describe_image(args.image)


if __name__ == '__main__':
    main()
