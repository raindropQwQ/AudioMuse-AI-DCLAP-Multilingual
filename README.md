# AudioMuse-AI-DCLAP-Multilingual

[![IIIA-4*](https://img.shields.io/badge/IIIA-4*-A865B3)](https://github.com/ErSanSan233/IIIA)

A lightweight projection layer that adds multilingual text search to [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI)'s existing DCLAP audio embeddings.

> Only the text encoder path is changed; the audio embeddings and index remain untouched, so no reindexing is required.

## How It Works

AudioMuse-AI already loads [gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) (768-dim) for lyrics search. This project trains a small MLP to project GTE text embeddings into DCLAP's 512-dim audio-aligned space:

```
Text query (any language)
    -> gte-multilingual-base (768-dim, already loaded)
    -> MLP projection (768 -> 256 -> 512, this repo)
    -> L2 normalize
    -> cosine similarity vs existing DCLAP audio embeddings
```

The projection ONNX is **1.3 MB** (`1,315,189` bytes). No new model downloads, no new Python dependencies, and no audio reindexing.

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

- AudioMuse-AI **v3.7.1** or **v3.7.2** (patch verified against these versions as of Nov 2026)
- `onnxruntime` (already installed in AudioMuse-AI)

### Release & Model Download

Download `gte_to_dclap_proj.onnx` (`1,315,189` bytes) from release tag [`v1`](https://github.com/raindropQwQ/AudioMuse-AI-DCLAP-Multilingual/releases/tag/v1) and place it inside the container at `/app/model/gte_to_dclap_proj.onnx` (or configure a custom location with the `CLAP_GTE_PROJ_PATH` environment variable).

### Standalone Python Usage

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
│  gte-multilingual-base (INT8 ONNX, 325 MB)              │
│  Already loaded in AudioMuse for lyrics search          │
│  Input: text (any of 75+ languages)                     │
│  Output: [768] L2-normalized                            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Projection MLP (this repo, 1.3 MB ONNX)                │
│  Linear(768, 256) -> GELU -> Linear(256, 512) -> L2-norm│
│  Trainable params: 328,448                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
              [512] DCLAP-compatible embedding
              (cosine vs existing audio index)
```

## Deployment in AudioMuse-AI

Deployment uses a GTE-first path with automatic RoBERTa fallback. When a text query arrives, `get_text_embeddings_batch` tries GTE + the projection ONNX first. If the projection file is missing or the GTE path raises an error, it falls back to the native DCLAP RoBERTa text encoder (~478 MB), which still remains available.

| Aspect | Before Patch | After Patch |
|--------|--------------|-------------|
| Primary text encoder | DCLAP RoBERTa (~478 MB) | GTE (shared with lyrics) + projection ONNX (1.3 MB) |
| Fallback text encoder | None | DCLAP RoBERTa (~478 MB, used if GTE or projection is unavailable) |
| Query languages | English only | 75+ languages (English-only if falling back to RoBERTa) |
| Token limit | 77 | 512 (77 on fallback) |
| Audio index | Unchanged | Unchanged |

### Deployment Artifacts

- `deployment/audiomuse-gte-dclap.patch`: A unified diff (`-p1`) touching two files:
  - `config.py`: Adds `CLAP_GTE_PROJ_PATH` (defaults to `/app/model/gte_to_dclap_proj.onnx`, overridable via environment variable).
  - `tasks/clap_analyzer.py`: Adds module state `_gte_proj_session` and helpers `_load_gte_proj`, `_get_gte_proj_session`, `_gte_text_embedding`, and `_gte_text_embeddings_batch`; wires `get_text_embeddings_batch` to try the GTE projection first before falling back to RoBERTa; clears `_gte_proj_session` in `unload_clap_model`.
- `deployment/apply_patch.sh`: An idempotent installer script that checks whether the diff is already applied, applies it with `patch --forward -p1`, checks for `/app/model/gte_to_dclap_proj.onnx` (warns without failing if missing, since search falls back to RoBERTa), and restarts `flask` via `supervisorctl`.

### `apply_patch.sh` Flags

- `--container NAME`: Target Docker container name (default: `audiomuse-ai-flask-app`).
- `--dir PATH`: Apply the patch to a local host source tree instead of a running container.
- `--check`: Dry-run only; exits `0` if the patch applies cleanly or is already applied.
- `--no-restart`: Skip the automatic `supervisorctl -c /etc/supervisor/conf.d/supervisord.conf restart flask` step after patching a container.

### Example 1: Apply to a Running Container

```bash
docker cp gte_to_dclap_proj.onnx audiomuse-ai-flask-app:/app/model/gte_to_dclap_proj.onnx
./deployment/apply_patch.sh --check
./deployment/apply_patch.sh
```

### Example 2: Apply to a Host Source Tree

```bash
./deployment/apply_patch.sh --check --dir /path/to/AudioMuse-AI
./deployment/apply_patch.sh --dir /path/to/AudioMuse-AI
```

## Compatibility & Maintenance

The patch in `deployment/audiomuse-gte-dclap.patch` is verified against **AudioMuse-AI v3.7.1 and v3.7.2** (as of Nov 2026). Because it applies as a contextual unified diff, you must re-verify the anchors before every AudioMuse-AI upgrade:

```bash
patch --dry-run --forward -p1 < deployment/audiomuse-gte-dclap.patch
# or
./deployment/apply_patch.sh --check --dir /path/to/AudioMuse-AI
```

The patch anchors live in two files:
- `config.py`: Near `CLAP_TEXT_MODEL_PATH` and `CLAP_EMBEDDING_DIMENSION`.
- `tasks/clap_analyzer.py`: The module globals block (`_audio_session`, `_text_session`, `_tokenizer`), `unload_clap_audio_only` / `unload_clap_model`, and `get_text_embeddings_batch`.

If an upstream upgrade refactors those lines and the dry-run fails, update the hunks in `deployment/audiomuse-gte-dclap.patch` against the new source tree.

## Training

### Method

Knowledge distillation with InfoNCE loss:
- **Teacher**: DCLAP RoBERTa text encoder (512-dim, English-only)
- **Student**: GTE-multilingual-base (768-dim) + trainable projection head
- **Loss**: InfoNCE (temperature=0.05), optionally with a multi-positive mask (treating same-group captions across languages as positives) and optionally with an auxiliary Song Describer audio cosine loss (weight=0.2)
- **Data**: 94,005 multilingual caption pairs (6,267 English x 15 languages)
- **Audio pairs**: 746 Song Describer tracks with DCLAP audio embeddings for auxiliary loss

### Model Architectures in `train/train_ablation.py`

`train/train_ablation.py` implements three projection architectures alongside a 12-configuration ablation sweep (`arch` x `multi-positive` x `SD audio`):
- `LinearProj` (`linear`): `Linear(768, 512)` followed by L2 normalization.
- `MLPProj` (`mlp256`): `Linear(768, 256) -> GELU -> Linear(256, 512)` followed by L2 normalization (the released model).
- `ResidualProj` (`residual64`): `Linear(768, 512) + alpha * Linear(64, 512)(GELU(Linear(768, 64)))` with `alpha=0.1` and zero-initialized up-projection weights and bias, followed by L2 normalization.

### Training Data

| Source | Samples | License |
|--------|---------|---------|
| MusicCaps captions (Google) | 5,521 | CC BY-SA 4.0 (see `dataset_license/MUSICCAPS_LICENSE.md`) |
| Song Describer Dataset | 746 | CC BY 4.0 (see `dataset_license/SONG_DESCRIBER_LICENSE.md`) |
| Multilingual translations (14 langs) | 87,738 | Derived from above |

Translations were generated via `gemini-3.1-pro-preview` into: zh, ja, ko, es, de, ru, ar, hi, th, vi, id, tr, fi, sw.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Released architecture | `MLPProj`: Linear(768 -> 256) + GELU + Linear(256 -> 512) |
| Optimizer | AdamW (lr=2e-3, weight_decay=0.01) + CosineAnnealingLR |
| Batch size | 1024 |
| Temperature | 0.05 (fixed) |
| Epochs | 200 (early stopping patience=20, evaluated every 5 epochs) |
| Best epoch | ~45 |
| Val MRR (teacher-space) | 0.75 |

### Environment & Data Inputs

`train/requirements.txt` installs:
- `torch>=2.0.0` (a CUDA GPU is required in `train_ablation.py`, which calls `.cuda()` on models and tensors)
- `numpy`
- `onnxruntime`
- `tqdm`
- `transformers`

To run training, you need four local data and model inputs. All six path flags below are required:
1. **DCLAP teacher text embeddings** (`--dclap-embeds`): A `.npy` array of shape `[N_unique, 512]` containing one row per unique English caption group.
2. **Song Describer audio-embedding cache** (`--sd-cache`): A `.npz` archive containing key `"audio"` with DCLAP audio embeddings (required when running configs with `sd1`).
3. **Multilingual captions directory** (`--captions-dir`): A directory of `*.json` files (including `en.json` for `sd1`), each holding a JSON list of caption strings aligned by group index.
4. **GTE INT8 ONNX model and tokenizer** (`--gte-onnx`, `--gte-tokenizer`): The INT8 ONNX file and Hugging Face tokenizer directory from `Alibaba-NLP/gte-multilingual-base`.

### Running the Training CLI

Pass `--config <arch>_mp{0,1}_sd{0,1}` (where `<arch>` is `linear`, `mlp256`, or `residual64`) to train a single configuration, or omit `--config` to run the full 12-configuration ablation sweep:

```bash
pip install -r train/requirements.txt

# Train the single released configuration (MLP-256 + multi-positive + SD audio)
python train/train_ablation.py \
  --config mlp256_mp1_sd1 \
  --dclap-embeds /path/to/dclap_text_embeds.npy \
  --sd-cache /path/to/sd_full_cache.npz \
  --gte-onnx /path/to/gte-multilingual-base-int8.onnx \
  --gte-tokenizer /path/to/gte-multilingual-base \
  --captions-dir /path/to/multilingual_captions \
  --output-dir out/
```

Outputs written to `--output-dir`:
- `best_proj_abl_<arch>_mp{0,1}_sd{0,1}.pt`: PyTorch state dict for the checkpoint with highest validation MRR.
- `gte_to_dclap_proj_abl_<arch>_mp{0,1}_sd{0,1}.onnx`: Exported ONNX model (opset 17, dynamic batch axis on `input` and `output`).
- `ablation_results.json`: JSON summary of validation MRR for the run (when `--config` is omitted, the 12-config sweep also prints an MRR-sorted summary table to stdout).
- `gte_int8_multi.npy` and `gte_int8_en.npy`: Cached GTE caption embeddings created on the first run.

## Acknowledgements

- [AudioMuse-AI](https://github.com/NeptuneHub/AudioMuse-AI) and [AudioMuse-AI-DCLAP](https://github.com/NeptuneHub/AudioMuse-AI-DCLAP) by NeptuneHub
- [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) by Google (CC BY-SA 4.0)
- [Song Describer Dataset](https://huggingface.co/datasets/renumics/song-describer-dataset) (CC BY 4.0)
- [gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) by Alibaba (Apache 2.0)
- [M2M](https://github.com/piyushsinghpasi/M2M) (EACL 2026), which validated the multilingual projection approach
- [IIIA](https://github.com/ErSanSan233/IIIA)
