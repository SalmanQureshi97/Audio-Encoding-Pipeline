import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ae_models.griffinmel import GriffinMel
from ae_models.pipeline import Pipeline
from scripts.create_dataset._shared import (
    build_common_parser,
    config_from_args,
    configure_torch_gpu,
    discover_audio_paths,
)


def parse_args():
    parser = build_common_parser(
        "Generate GriffinMel reconstructions for a directory of audio files"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    conf = config_from_args(args)
    paths = discover_audio_paths(args.db_path, args.extensions, args.max_files)
    print(f"Found {len(paths)} audio paths")

    configure_torch_gpu(args.gpu)

    griffin_params = {
        "n_fft": int(400 / 1000 * conf["SR"]),
        "hop_fft": int(10 / 1000 * conf["SR"]),
        "win_fft": int(100 / 1000 * conf["SR"]),
        "griffin_iter": 32,
        "n_mels": 512,
    }
    griffin_256 = dict(griffin_params)
    griffin_256["n_mels"] = 256

    ae_griffinmel_512 = GriffinMel(griffin_params, conf["SR"], conf["DEVICE"])
    ae_griffinmel_256 = GriffinMel(griffin_256, conf["SR"], conf["DEVICE"])

    pipeline = Pipeline(paths, conf)
    pipeline.run_loop(
        [ae_griffinmel_512, ae_griffinmel_256],
        ["griffin512", "griffin256"],
    )


if __name__ == "__main__":
    main()
