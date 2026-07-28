# AudioMuse-AI-DCLAP-Multilingual

[![IIIA-3*](https://img.shields.io/badge/IIIA-3*-A865B3)](https://github.com/ErSanSan233/IIIA)

A lightweight projection layer that adds multilingual text search to [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI)'s existing DCLAP audio embeddings.

> Only the text encoder path is changed; the audio embeddings and index remain untouched — no reindexing required.

## How It Works

AudioMuse-AI already loads [gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) (768-dim) for lyrics search. This project trains a small MLP to project GTE text embeddings into DCLAP's 512-dim audio-aligned space:

```
Text query (any language)
    → gte-multilingual-base (768-dim, already loaded)
    → MLP projection (768 → 256 → 512, this repo)
    → L2 normalize
    → cosine similarity vs existing DCLAP audio embeddings
```

The projection ONNX is **1.3 MB**. No new model downloads, no new dependencies, no reindexing.

## Benchmarks

### Song Describer Dataset (100 samples, seed=42)

| Method | EN R@1 | EN R@5 | ZH R@1 | ZH R@5 | JA R@1 | JA R@5 |
|--------|--------|--------|--------|--------|--------|--------|
| DCLAP native (RoBERTa, EN-only) | 14% | 64% | 1% | 3% | 2% | 4% |
| **GTE + projection (this repo)** | **30%** | **66%** | **22%** | **63%** | **22%** | **60%** |

### LP-MusicCaps (400 tracks, MusicCaps eval split)

| Method | R@1 | R@5 | R@10 | MRR |
|--------|-----|-----|------|-----|
| DCLAP native | 15.8% | 41.0% | 57.2% | 0.280 |
| **GTE + projection** | **16.2%** | **45.8%** | **57.5%** | **0.297** |

### Cross-domain generalization (365 tracks with Chinese descriptions, unseen during training)

| Method | ZH R@1 | ZH R@5 | ZH MRR |
|--------|--------|--------|--------|
| DCLAP native | 0.8% | 4.1% | 0.036 |
| **GTE + projection** | **2.7%** | **13.7%** | **0.092** |

English performance is preserved or slightly improved. Chinese and Japanese go from near-random to functional.

## Quick Start

### Requirements

- AudioMuse-AI v2.0.0+ (GTE model already present)
- `onnxruntime` (already installed in AudioMuse)

### Download

Get `gte_to_dclap_proj.onnx` from the [Releases](https://github.com/raindropQwQ/AudioMuse-AI-DCLAP-Multilingual/releases) page and place it at `/app/model/gte_to_dclap_proj.onnx`.

### Usage

```python
import numpy as np
import onnxruntime as ort
from lyrics.gte_onnx import embed_text, load_gte_model

# Load GTE (already loaded if lyrics search is active)
load_gte_model()

# Load projection (1.3 MB, loads in <10ms)
proj_session = ort.InferenceSession('/app/model/gte_to_dclap_proj.onnx')

# Encode any-language query
gte_emb = embed_text("可爱的日系电子音乐")  # [768], L2-normalized
projected = proj_session.run(None, {'input': gte_emb.reshape(1, -1)})[0][0]
projected = projected / np.linalg.norm(projected)  # [512], L2-normalized

# Compare against existing DCLAP audio embeddings via cosine similarity
# similarity = projected @ audio_embedding
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  gte-multilingual-base (INT8 ONNX, 325 MB)             │
│  Already loaded in AudioMuse for lyrics search          │
│  Input: text (any of 75+ languages)                     │
│  Output: [768] L2-normalized                            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Projection MLP (this repo, 1.3 MB ONNX)               │
│  Linear(768, 256) → GELU → Linear(256, 512) → L2-norm  │
│  Trainable params: 328,448                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
              [512] DCLAP-compatible embedding
              (cosine vs existing audio index)
```

## Training

### Method

Knowledge distillation with InfoNCE loss:
- **Teacher**: DCLAP RoBERTa text encoder (512-dim, English-only)
- **Student**: GTE-multilingual-base (768-dim) + trainable MLP projection
- **Loss**: Symmetric InfoNCE (temperature=0.05) + multi-positive mask + auxiliary audio cosine (weight=0.2)
- **Data**: 94,005 multilingual caption pairs (6,267 English × 15 languages)
- **Audio pairs**: 746 Song Describer tracks with DCLAP audio embeddings for auxiliary loss

### Training Data

| Source | Samples | License |
|--------|---------|---------|
| MusicCaps captions (Google) | 5,521 | CC BY-SA 4.0 |
| Song Describer Dataset | 746 | CC BY 4.0 |
| Multilingual translations (14 langs) | 87,738 | Derived from above |

Translations generated via `gemini-3.1-pro-preview` into: zh, ja, ko, es, de, ru, ar, hi, th, vi, id, tr, fi, sw.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Architecture | Linear(768→256) + GELU + Linear(256→512) |
| Optimizer | AdamW (lr=2e-3, weight_decay=0.01) |
| Batch size | 1024 |
| Temperature | 0.05 (fixed) |
| Epochs | 200 (early stopping patience=20) |
| Best epoch | ~45 |
| Val MRR (teacher-space) | 0.75 |

### Reproduce

```bash
cd train/
pip install -r requirements.txt
python train_ablation.py --config mlp256_mp1_sd1
```

See `train/README.md` for full instructions.

## Deployment in AudioMuse-AI

The projection replaces the DCLAP RoBERTa text encoder (478 MB) in `tasks/clap_analyzer.py`:

| | Before | After |
|---|--------|-------|
| Text model | DCLAP RoBERTa (478 MB) | GTE (already loaded) + projection (1.3 MB) |
| Languages | English only | 75+ languages |
| Token limit | 77 | 512 |
| RAM delta | +478 MB | +0 MB (GTE shared with lyrics) |
| Audio index | Unchanged | Unchanged |

A patch for `get_text_embedding()` is provided in `deployment/patch_clap_text.py`.

## Acknowledgements

- [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI) and [AudioMuse-AI-DCLAP](https://github.com/NeptuneHub/AudioMuse-AI-DCLAP) by NeptuneHub
- [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) by Google (CC BY-SA 4.0)
- [Song Describer Dataset](https://huggingface.co/datasets/renumics/song-describer-dataset) (CC BY 4.0)
- [gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) by Alibaba (Apache 2.0)
- [M2M](https://github.com/piyushsinghpasi/M2M) (EACL 2026) — validated the multilingual projection approach
