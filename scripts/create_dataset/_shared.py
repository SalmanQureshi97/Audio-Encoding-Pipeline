import argparse
from pathlib import Path

import torch


DEFAULT_EXTENSIONS = ".mp3,.wav,.flac,.ogg,.m4a,.aac"


def build_common_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--db-path", type=Path, required=True, help="Root input audio directory")
    parser.add_argument("--out-db", type=Path, required=True, help="Root output directory")
    parser.add_argument(
        "--bitrate-path",
        type=Path,
        default=None,
        help="Optional bitrate metadata .npy file used by the original FMA pipeline",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda or cpu")
    parser.add_argument("--data-sr", type=int, default=44100, help="Original dataset sample rate")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--min-duration", type=int, default=3, help="Minimum duration in seconds")
    parser.add_argument(
        "--max-duration",
        type=int,
        default=40,
        help="Maximum duration in seconds; use 0 to disable clipping",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=DEFAULT_EXTENSIONS,
        help="Comma-separated audio extensions",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose pipeline timing")
    return parser


def normalize_extensions(extensions):
    values = [ext.strip().lower() for ext in extensions.split(",") if ext.strip()]
    out = []
    for ext in values:
        out.append(ext if ext.startswith(".") else f".{ext}")
    return out


def discover_audio_paths(db_path, extensions):
    paths = []
    for ext in normalize_extensions(extensions):
        paths.extend(db_path.rglob(f"*{ext}"))
    return sorted(set(paths))


def config_from_args(args):
    return {
        "DEVICE": args.device,
        "DB_PATH": str(args.db_path),
        "OUT_DB": str(args.out_db),
        "BR_PATH": str(args.bitrate_path) if args.bitrate_path else None,
        "DATA_SR": args.data_sr,
        "SR": args.sr,
        "MIN_DURATION": args.min_duration,
        "MAX_DURATION": None if args.max_duration == 0 else args.max_duration,
        "VERBOSE": args.verbose,
    }


def configure_torch_gpu(gpu):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
