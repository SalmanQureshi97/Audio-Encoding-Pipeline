This directory stores local pretrained dependencies required by the encoding scripts.

To use the autoencoders, clone the following repositories into this `pretrained/` directory:

- [Git Musika!](https://github.com/marcoppasini/musika)
- [LAC](https://github.com/hugofloresgarcia/lac)

For `LAC`, also download the pretrained codec weights provided by [VampNet](https://github.com/hugofloresgarcia/vampnet) and place them at:

```text
pretrained/vampnet/codec.pth
```

Expected contents:

```text
pretrained/
├── musika/
│   ├── checkpoints/
│   │   ├── ae/
│   │   └── techno/
├── lac/
│   └── lac/
│       └── model/...
└── vampnet/
    └── codec.pth
```

These assets are required by:

- `scripts/create_dataset/musika.py`
- `scripts/create_dataset/lac.py`

`Encodec` does not require local assets in this directory because it is loaded from Hugging Face at runtime.
