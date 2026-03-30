import sys
from pathlib import Path

import torch

from ae_models.ae import AE


REPO_ROOT = Path(__file__).resolve().parents[1]
LAC_ROOT = REPO_ROOT / "pretrained" / "lac"
VAMPNET_CODEC = REPO_ROOT / "pretrained" / "vampnet" / "codec.pth"

sys.path.append(str(LAC_ROOT))

from pretrained.lac.lac.model.lac import LAC  # noqa: E402


class Lac_ae(AE):
    def __init__(self, sr, device="cuda"):
        super().__init__("LAC")

        self.sr = sr
        self.device = device
        self.model = LAC.load(str(VAMPNET_CODEC))
        self.model.eval()
        self.model.to(self.device)

    def autoencode_multi(self, x, codec):
        preprocess, _ = self.model.preprocess(x, self.sr)
        z = self.model.encode(preprocess[:, None, :], self.sr)
        codes = z["codes"]

        decoded_audio = []
        for c in codec:
            z_red = self.model.quantizer.from_codes(codes[:, :c, :])[0]
            rebuilt = self.model.decode(z_red)["audio"]
            decoded_audio.append(torch.squeeze(rebuilt))

        return decoded_audio
