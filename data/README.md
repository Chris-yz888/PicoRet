# Dataset preparation

Datasets are not redistributed in this repository. Download the public archives:

```bash
python scripts/download_datasets.py lolv1 lolv2
python scripts/download_datasets.py dicm mef npe lime
```

The paired code expects this normalized layout:

```text
data/
├── LOLv1/
│   ├── Train/input
│   ├── Train/target
│   ├── Test/input
│   └── Test/target
└── LOLv2/
    ├── Real_captured/{Train,Test}/{Low,Normal}
    └── Synthetic/{Train,Test}/{Low,Normal}
```

Google Drive archives occasionally contain an extra top-level directory. If so,
move its contents into the normalized paths above. The four no-reference sets are
evaluation-only and must not be used for training or checkpoint selection.

Sources:

- LOL-v1: <https://drive.google.com/file/d/1vhJg75hIpYvsmryyaxdygAWeHuiY_HWu/view>
- LOL-v2: <https://drive.google.com/file/d/1OMfP6Ks2QKJcru1wS2eP629PgvKqF2Tw/view>
- DICM/MEF/NPE/LIME: <https://github.com/baidut/BIMEF#readme>
