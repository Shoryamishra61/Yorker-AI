from __future__ import annotations

import argparse
import json

from .pipeline import run_all, run_benchmark, run_build_dataset, run_download, run_predict_upcoming, run_report, run_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IPL ML pipeline")
    parser.add_argument("--force-download", action="store_true", help="Re-download remote source files.")
    parser.add_argument("--target-accuracy", type=float, default=0.97, help="Benchmark quality-gate target.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["download-data", "build-dataset", "train", "benchmark", "predict-upcoming", "report", "run-all"]:
        subparser = subparsers.add_parser(command)
        if command == "benchmark":
            subparser.add_argument(
                "--target-accuracy",
                type=float,
                default=argparse.SUPPRESS,
                help="Benchmark quality-gate target.",
            )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "download-data":
        result = run_download(force=args.force_download)
    elif args.command == "build-dataset":
        result = run_build_dataset(force_download=args.force_download)
    elif args.command == "train":
        result = run_train(force_download=args.force_download)
    elif args.command == "benchmark":
        result = run_benchmark(force_download=args.force_download, target_accuracy=args.target_accuracy)
    elif args.command == "predict-upcoming":
        result = run_predict_upcoming(force_download=args.force_download)
    elif args.command == "report":
        result = run_report(force_download=args.force_download)
    else:
        result = run_all(force_download=args.force_download)

    printable = {}
    for key, value in result.items():
        if hasattr(value, "shape"):
            printable[key] = list(value.shape)
        elif isinstance(value, list):
            printable[key] = f"list[{len(value)}]"
        elif isinstance(value, dict):
            printable[key] = f"dict[{len(value)}]"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            printable[key] = value
        else:
            printable[key] = f"<{value.__class__.__name__}>"
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
