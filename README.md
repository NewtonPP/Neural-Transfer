# NeuralTransfer

A PyTorch implementation of **AdaIN** — *Arbitrary Style Transfer in Real-time with Adaptive
Instance Normalization* ([Huang & Belongie, ICCV 2017](https://arxiv.org/abs/1703.06868)) — with a
Flask web app for running transfers in the browser.

Unlike optimization-based neural style transfer, AdaIN needs no per-image optimization loop. A
frozen VGG encoder embeds the content and style images, a single adaptive instance normalization
layer aligns the content feature statistics to the style's, and a learned decoder inverts the
result back to pixels. One forward pass per stylization.

## How it works

```
content ─┐
         ├─► VGG encoder (frozen, → relu4_1) ─► AdaIN ─► decoder ─► stylized image
style  ──┘
```

- **Encoder** (`model.py`) — VGG-19 truncated at `relu4_1`, split into four blocks (`enc1`…`enc4`)
  so training can read off intermediate activations at `relu1_1`, `relu2_1`, `relu3_1` and
  `relu4_1`. All weights are frozen.
- **AdaIN** (`utils.py`) — normalizes the content features to zero mean / unit variance per channel,
  then rescales them by the style features' per-channel mean and standard deviation.
- **Decoder** (`model.py`) — mirrors the encoder with nearest-neighbour upsampling and reflection
  padding. This is the only trained component.
- **Loss** (`train.py`) — content loss is the MSE between the stylized image's `relu4_1` features
  and the AdaIN target; style loss is the MSE between mean and std of the stylized and style
  features across all four layers, weighted by `--style_weight`.

At inference the app interpolates with an **alpha** knob: `alpha * AdaIN(c, s) + (1 - alpha) * c`.
`alpha = 1.0` is full stylization, `alpha = 0.0` reconstructs the content image.

## Repository layout

| Path | What it is |
| --- | --- |
| `model.py` | `Encoder` (frozen VGG) and `Decoder` definitions |
| `utils.py` | `AdaIN`, `calc_mean_std`, the image-folder `Dataset`, transform builder |
| `train.py` | Decoder training loop with checkpointing and sample-grid output |
| `app.py` | Flask app: upload content + style, pick alpha, get a stylized result |
| `templates/index.html` | Single-page UI for the app |
| `experiment/experiment1/` | Trained decoder + optimizer checkpoints and training args |
| `static/uploads/` | Uploaded and generated images (created at runtime) |

## Local setup

### 1. Prerequisites

Python 3.12 (the checked-in checkpoints and `.venv` were built with 3.12.13). A GPU is optional —
inference runs fine on CPU, just slower.

### 2. Create a virtual environment

```bash
git clone <your-remote-url> NeuralTransfer
cd NeuralTransfer

python3.12 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision flask Flask-WTF Flask-Bootstrap pillow numpy tqdm
```

On Linux with an NVIDIA GPU, install torch from the CUDA index instead, e.g.
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.

### 4. Get the pretrained VGG weights

`vgg_normalised.pth` (~80 MB) is **not** in the repository — it is excluded by `.gitignore`. Download
the normalised VGG-19 weights used by the reference AdaIN implementation
([naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN)) and place the file in the
repository root:

```
NeuralTransfer/
├── vgg_normalised.pth   ← here
├── app.py
└── ...
```

The layer ordering in `Encoder` matches that checkpoint exactly, so a differently structured VGG
export will fail to load.

### 5. Point the app at the decoder checkpoint

`app.py:38` currently hardcodes an absolute path to the decoder weights. Change it to a relative
path so it works from any clone:

```python
decoder.load_state_dict(torch.load('experiment/experiment1/best_decoder.pth', map_location=device))
```

Adding `map_location=device` also lets a CUDA-trained checkpoint load on a CPU-only machine.

### 6. Run the app

```bash
python app.py
```

Open <http://localhost:8000>. Upload a content image and a style image (`.png`, `.jpg`, `.jpeg`),
set alpha, and hit **Transfer Style**. Inputs and results are written to `static/uploads/`.

## Training your own decoder

The decoder is trained on pairs drawn from two unlabelled image folders — conventionally
[MS-COCO](https://cocodataset.org/) for content and
[WikiArt](https://www.kaggle.com/datasets/steubk/wikiart) for style. Neither dataset is committed;
create the folders and fill them with images:

```
content_data/    # e.g. MS-COCO train images
style_data/      # e.g. WikiArt paintings
```

Then:

```bash
python train.py \
  --content_dir content_data \
  --style_dir style_data \
  --vgg vgg_normalised.pth \
  --experiment experiment2 \
  --batch_size 8 \
  --epochs 20
```

Checkpoints, a sample output grid, and the full argument list are written to
`experiment/<experiment>/`. Resume with `--resume --decoder_path ... --optimizer_path ...`.

Key arguments:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--final_size` | 256 | Crop/resize size fed to the network |
| `--content_size` / `--style_size` | 512 | Resize before cropping |
| `--batch_size` | 4 | Images per step |
| `--lr` / `--lr_decay` | 1e-4 / 5e-5 | Adam learning rate and per-epoch decay |
| `--content_weight` | 1.0 | Weight on the content loss |
| `--style_weight` | 5 | Weight on the style loss — raise for stronger stylization |
| `--crop` | True | Random-crop instead of plain resize |

The checked-in `experiment/experiment1/` was trained for 2 epochs at `final_size=256`,
`batch_size=4`, `style_weight=5` — see `experiment/experiment1/args.txt`.

**Note:** `train.py:82` selects `cuda` if available and otherwise falls back to `mps` (Apple
Silicon). On a CPU-only machine, change that line to fall back to `cpu`.

## Things to know before deploying

- `app.config['SECRET_KEY']` is the literal string `'supersecretkey'` — replace it (e.g. read it
  from an environment variable) before exposing the app to anyone else.
- Uploads are saved under their original filename, so two users uploading `photo.jpg` overwrite each
  other. `static/uploads/` also grows without bound.
- `app.py` runs Werkzeug's development server with the debugger enabled; use a real WSGI server for
  anything beyond local use.

## Reference

Xun Huang, Serge Belongie. *Arbitrary Style Transfer in Real-time with Adaptive Instance
Normalization.* ICCV 2017. [arXiv:1703.06868](https://arxiv.org/abs/1703.06868)
