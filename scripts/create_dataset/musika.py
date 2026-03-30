import sys
from pathlib import Path

import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ae_models.musika import Musika_ae
from ae_models.pipeline import Pipeline
from scripts.create_dataset._shared import build_common_parser, config_from_args, configure_torch_gpu, discover_audio_paths


def parse_args():
    parser = build_common_parser("Generate Musika reconstructions for a directory of audio files")
    return parser.parse_args()


def configure_tf_gpu(gpu):
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return
    tf.config.set_visible_devices(gpus[gpu], "GPU")
    tf.config.experimental.set_memory_growth(gpus[gpu], True)


def main():
    args = parse_args()
    conf = config_from_args(args)
    paths = discover_audio_paths(args.db_path, args.extensions)
    print(f"Found {len(paths)} audio paths")

    configure_torch_gpu(args.gpu)
    configure_tf_gpu(args.gpu)

    musika = Musika_ae()
    pipeline = Pipeline(paths, conf)
    pipeline.run_loop([musika], ["musika"])


if __name__ == "__main__":
    main()
