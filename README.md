# Audio Encoding Pipeline

Standalone dataset-creation pipeline for generating encoded audio using:

- `Musika`
- `Encodec`
- `LAC`


## What This Repo Does

It takes a directory of audio files and produces encoded variants under an output directory:

- `encodec3`, `encodec6`, `encodec24`
- `lac2`, `lac7`, `lac14`
- `musika`

The pipeline:

1. Recursively discovers input audio files.
2. Loads each file with `torchaudio`.
3. Resamples to a target sampling rate if needed.
4. Skips files shorter than the configured minimum duration.
5. Optionally clips very long tracks.
6. Runs the selected encoder(s).
7. Saves outputs while preserving the input directory structure.

## Repository Layout

```text
audio-encoding-pipeline/
├── ae_models/
│   ├── __init__.py
│   ├── ae.py
│   ├── dac.py
│   ├── encodec.py
│   ├── musika.py
│   └── pipeline.py
├── pretrained/
│   └── README.md
├── scripts/
│   └── create_dataset/
│       ├── _shared.py
│       ├── encodec.py
│       ├── lac.py
│       └── musika.py
└── requirements.txt
```

## Pretrained Dependencies

This repository does **not** bundle all pretrained artifacts required by `Musika` and `LAC`.

To use the autoencoders, clone the following repositories into a local `pretrained/` folder:

- [Git Musika!](https://github.com/marcoppasini/musika)
- [LAC](https://github.com/hugofloresgarcia/lac)

For `LAC`, download the pretrained codec weights provided by [VampNet](https://github.com/hugofloresgarcia/vampnet) and place them in:

```text
pretrained/vampnet/
```

The expected local structure is:

```text
audio-encoding-pipeline/
├── pretrained/
│   ├── musika/
│   ├── lac/
│   └── vampnet/
```

`Encodec` does not require local checkpoint files in this repository because it is loaded from Hugging Face at runtime.

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running From Repo Root

Run all commands from the repository root:

```bash
cd audio-encoding-pipeline
```

This matters because the scripts and local pretrained assets are resolved relative to the repository itself.

## Example: Encode a dataset

If your dataset lives in `/path/to/dataaset/`, you can generate encoded variants like this:

### Encodec

```bash
python scripts/create_dataset/encodec.py \
  --db-path /path/to/dataset/ \
  --out-db /path/to/dataset/out \
  --gpu 0 \
  --max-duration 180
```

### LAC

```bash
python scripts/create_dataset/lac.py \
  --db-path /path/to/dataset \
  --out-db /path/to/dataset/out \
  --gpu 0 \
  --max-duration 180
```

### Musika

```bash
python scripts/create_dataset/musika.py \
  --db-path /path/to/dataset \
  --out-db /path/to/dataset/out \
  --gpu 0 \
  --max-duration 180
```

This will create output folders such as:

```text
/path/to/dataset/out/
├── encodec3/
├── encodec6/
├── encodec24/
├── lac2/
├── lac7/
├── lac14/
└── musika/
```

The relative directory structure from the input directory is preserved.

## Example: Reproduce FMA-Style Encoding

If you want to use the same style of pipeline on an FMA collection:

```bash
python scripts/create_dataset/encodec.py \
  --db-path /path/to/fma_medium \
  --out-db /path/to/fma_rebuilt_medium \
  --bitrate-path /path/to/bitrates_ffmpeg_medium.npy \
  --gpu 0 \
  --max-duration 40
```

Use the same pattern for `lac.py` and `musika.py`.

## Notes on Bitrate Metadata

The original FMA pipeline expected a `bitrates_ffmpeg_medium.npy` file and used it when saving compressed outputs.

This adapted pipeline makes bitrate metadata optional:

- If `--bitrate-path` is provided, output compression follows the stored bitrate metadata.
- If it is omitted, the pipeline still works and saves files without requiring FMA-specific metadata.

## Important Configuration Notes

- `--max-duration`:
  - For FMA-style reconstruction, `40` seconds matches the original scripts.
  - For SONICS fake songs, a larger value such as `120` or `180` is safer if you want to preserve more of the song.
  - Use `--max-duration 0` to disable clipping entirely.
- `--min-duration`:
  - Default is `3` seconds.
- `--extensions`:
  - By default: `.mp3,.wav,.flac,.ogg,.m4a,.aac`
