import os
from pathlib import Path
from time import time

import numpy as np
import soxr
import torch
import torchaudio
from tqdm import tqdm


default_conf = {
    "DEVICE": "cpu",
    "DB_PATH": "/",
    "OUT_DB": "/",
    "BR_PATH": None,
    "DATA_SR": 44100,
    "SR": 44100,
    "TO_MONO": False,
    "AUGMENT": False,
    "MIN_DURATION": 3,
    "MAX_DURATION": 40,
    "VERBOSE": False,
}


class Pipeline:
    def __init__(self, paths, conf=None):
        self.paths = paths
        self.conf = dict(default_conf)
        self.conf.update(conf or {})

        br_path = self.conf.get("BR_PATH")
        if br_path and os.path.exists(br_path):
            self.bitrates = np.load(br_path, allow_pickle=True).item()
        else:
            self.bitrates = None

        self.global_t = time()

    def clock(self, ops_name):
        if self.conf["VERBOSE"]:
            current_time = time()
            print(f"{ops_name}: {current_time - self.global_t:.2f}s")
            self.global_t = current_time

    def _prepare_output_dir(self, model_name, codec_suffix, rel_dir):
        out_dir = Path(self.conf["OUT_DB"]) / f"{model_name}{codec_suffix}"
        if rel_dir != Path("."):
            out_dir = out_dir / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _compute_save_kwargs(self, fname):
        kwargs = {
            "sample_rate": self.conf["SR"],
            "channels_first": True,
        }

        suffix = Path(fname).suffix.lower()
    
        if suffix == ".mp3":
            kwargs["backend"] = "ffmpeg"
    
        if self.bitrates is not None and fname in self.bitrates:
            kwargs["backend"] = "ffmpeg"
            kwargs["compression"] = min(int(self.bitrates[fname]), 320)
    
        return kwargs

    def run_loop(self, models, model_names, multi_codec=None, has_cpu_preprocess=False):
        assert len(models) == len(model_names)

        multi_codec = multi_codec or []
        using_multi = len(multi_codec) > 0
        if not using_multi:
            multi_codec = [""]

        if using_multi and len(models) > 1:
            raise ValueError("Does not make sense to have several models and multi_codec on.")

        db_root = Path(self.conf["DB_PATH"]).resolve()

        for fpath in tqdm(self.paths):
            fpath = Path(fpath).resolve()
            fname = fpath.name
            rel_dir = fpath.parent.relative_to(db_root) if fpath.parent != db_root else Path(".")

            multi_codec_todo = []
            for m_name in model_names:
                for c in multi_codec:
                    out_dir = self._prepare_output_dir(m_name, str(c), rel_dir)
                    if not (out_dir / fname).exists():
                        multi_codec_todo.append(c)

            if len(multi_codec_todo) == 0:
                continue

            try:
                audio_raw, sr = torchaudio.load(str(fpath))
                if sr != self.conf["SR"]:
                    print(f"Resampling {fpath}: {sr} -> {self.conf['SR']}")
                    audio_raw_rs = soxr.resample(audio_raw.T, sr, self.conf["SR"]).T
                    audio_raw = torch.tensor(audio_raw_rs)

                audio_raw = torch.squeeze(audio_raw)
                if audio_raw.shape[-1] < self.conf["SR"] * self.conf["MIN_DURATION"]:
                    print(f"Track {fpath} < min duration, skipping")
                    continue

                if len(audio_raw.shape) == 1:
                    audio_raw = torch.stack([audio_raw, audio_raw], 0)

                max_duration = self.conf.get("MAX_DURATION")
                if max_duration is not None and max_duration > 0:
                    max_len = self.conf["SR"] * max_duration
                    if audio_raw.shape[-1] > max_len:
                        audio_raw = audio_raw[:, :max_len]
            except Exception as err:
                print(f"Track {fpath} failed: [{type(err)}] {err}")
                continue

            if not has_cpu_preprocess:
                audio_raw = audio_raw.to(self.conf["DEVICE"])
            self.clock("opening")

            for m, m_name in zip(models, model_names):
                with torch.no_grad():
                    if not using_multi:
                        audio_rebuilt = m.autoencode(audio_raw)
                        audios_rebuilt = [audio_rebuilt.to("cpu")]
                    else:
                        audios_rebuilt = m.autoencode_multi(audio_raw, multi_codec_todo)
                        audios_rebuilt = [audio.to("cpu") for audio in audios_rebuilt]

                self.clock("autoencode")

                for audio_rebuilt, c in zip(audios_rebuilt, multi_codec_todo):
                    out_dir = self._prepare_output_dir(m_name, str(c), rel_dir)
                    out_path = out_dir / fname
                    torchaudio.save(str(out_path), audio_rebuilt, **self._compute_save_kwargs(fname))
                    self.clock("saved")
