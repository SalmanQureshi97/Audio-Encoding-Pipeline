import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ae_models.dac import Lac_ae
from ae_models.pipeline import Pipeline
from scripts.create_dataset._shared import build_common_parser, config_from_args, configure_torch_gpu, discover_audio_paths


def parse_args():
    parser = build_common_parser("Generate LAC reconstructions for a directory of audio files")
    return parser.parse_args()


def main():
    args = parse_args()
    conf = config_from_args(args)
    paths = discover_audio_paths(args.db_path, args.extensions)
    print(f"Found {len(paths)} audio paths")

    configure_torch_gpu(args.gpu)

    lac_ae = Lac_ae(conf["SR"], device=conf["DEVICE"])
    pipeline = Pipeline(paths, conf)
    pipeline.run_loop([lac_ae], ["lac"], [2, 7, 14])


if __name__ == "__main__":
    main()
