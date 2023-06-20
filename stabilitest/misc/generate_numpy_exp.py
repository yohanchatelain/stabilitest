import numpy as np
import os
import argparse
import shutil


def generate_gaussian(args):
    size = (args.sample,) + tuple(args.shape)
    return np.random.normal(size=size)


def dump(args, data):
    if args.force and os.path.exists(args.output):
        shutil.rmtree(args.output)

    os.makedirs(args.output, exist_ok=args.force)
    for i in range(args.sample):
        path = os.path.join(args.output, f"rep_{i}")
        os.makedirs(path, exist_ok=args.force)
        path = os.path.join(path, "data.npy")
        np.save(path, data[i])


def parse_args():
    parser = argparse.ArgumentParser(description="experiments generator", prog="expgen")
    parser.add_argument("--sample", type=int, default=10, help="Sample size")
    parser.add_argument(
        "--shape",
        type=int,
        default=[10, 10],
        nargs="+",
        help="Shape of each repetition",
    )
    parser.add_argument(
        "--output", default="gaussian_experiment_result", help="Directory output name"
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existent directory"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = generate_gaussian(args)
    dump(args, data)


if __name__ == "__main__":
    main()
