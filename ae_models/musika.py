import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

from ae_models.ae import AE


REPO_ROOT = Path(__file__).resolve().parents[1]
MUSIKA_ROOT = REPO_ROOT / "pretrained" / "musika"

sys.path.append(str(MUSIKA_ROOT))

from pretrained.musika.models import Models_functions  # noqa: E402
from pretrained.musika.parse.parse_decode import parse_args  # noqa: E402
from pretrained.musika.utils import Utils_functions  # noqa: E402


CHECKPOINT = MUSIKA_ROOT / "checkpoints" / "techno"
AE_PATH = MUSIKA_ROOT / "checkpoints" / "ae"


class Musika_ae(AE):
    def __init__(self):
        super().__init__("Musika")

        args = parse_args()
        self.args = args

        args.load_path = str(CHECKPOINT)
        args.dec_path = str(AE_PATH)

        models = Models_functions(args)
        self.U = Utils_functions(args)
        self.models_ls = models.get_networks()

    def encode(self, x):
        x = x.cpu()
        return self.encode_audio(x.T.numpy(), models_ls=self.models_ls)

    def decode(self, z):
        return torch.Tensor(
            self.U.decode_waveform(
                z[None, None, :, :],
                self.models_ls[3],
                self.models_ls[5],
                batch_size=64,
            )
        ).T

    def encode_audio(self, audio, models_ls=None):
        critic, gen, enc, dec, enc2, dec2, gen_ema, [opt_dec, opt_disc], switch = models_ls

        _ = critic, gen, dec, dec2, gen_ema, opt_dec, opt_disc, switch
        shape2 = self.args.shape
        wv = np.squeeze(audio)

        if wv.shape[0] <= self.args.hop * self.args.shape * 2 + 3 * self.args.hop:
            raise ValueError("Audio too short for Musika encoding")

        rem = (wv.shape[0] - (3 * self.args.hop)) % (self.args.shape * self.args.hop)
        if rem != 0:
            wv = tf.concat([wv, tf.zeros([rem, 2], dtype=tf.float32)], 0)

        channels = []
        for channel in range(2):
            x = wv[:, channel]
            x = tf.expand_dims(tf.transpose(self.U.wv2spec(x, hop_size=self.args.hop), (1, 0)), -1)
            ds = []
            num = x.shape[1] // self.args.shape
            for i in range(num):
                start = i * self.args.shape
                ds.append(x[:, start : start + self.args.shape, :])
            del x

            ds = tf.convert_to_tensor(ds, dtype=tf.float32)
            lat = self.U.distribute_enc(ds, enc)
            del ds
            lat = tf.split(lat, lat.shape[0], 0)
            lat = tf.concat(lat, -2)
            lat = tf.squeeze(lat)

            ds2 = []
            num2 = lat.shape[-2] // shape2
            for j in range(num2):
                start = j * shape2
                ds2.append(lat[start : start + shape2, :])
            ds2 = tf.convert_to_tensor(ds2, dtype=tf.float32)
            lat = self.U.distribute_enc(tf.expand_dims(ds2, -3), enc2)
            del ds2
            lat = tf.split(lat, lat.shape[0], 0)
            lat = tf.concat(lat, -2)
            lat = tf.squeeze(lat)
            channels.append(lat)

        return tf.concat(channels, -1)
