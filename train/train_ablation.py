"""12-config ablation: arch × multi-positive × SD audio."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import json, argparse
import onnxruntime as ort
from tqdm import tqdm

DCLAP_EMBEDS = Path("/mnt/ZFSData/Media/Dataload/models/dclap_gte_proj/dclap_text_embeds.npy")
SD_CACHE = Path("/mnt/ZFSData/Media/Dataload/models/dclap_gte_proj/sd_full_cache.npz")
GTE_INT8 = "/mnt/ZFSData/Media/Dataload/models/gte-multilingual-base/gte-multilingual-base-int8.onnx"
GTE_TOKENIZER = "/mnt/ZFSData/Media/Dataload/models/gte-multilingual-base"
CAPTIONS_DIR = Path("/mnt/ZFSData/Media/Dataload/dataset/multilingual_captions")
OUTPUT_DIR = Path("/mnt/ZFSData/Media/Dataload/models/dclap_gte_proj")
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


class LinearProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 512)

    def forward(self, x):
        return F.normalize(self.fc(x), p=2, dim=-1)


class MLPProj(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(768, hidden)
        self.fc2 = nn.Linear(hidden, 512)

    def forward(self, x):
        return F.normalize(self.fc2(F.gelu(self.fc1(x))), p=2, dim=-1)


class ResidualProj(nn.Module):
    def __init__(self, bottleneck=64, alpha=0.1):
        super().__init__()
        self.linear = nn.Linear(768, 512)
        self.down = nn.Linear(768, bottleneck)
        self.up = nn.Linear(bottleneck, 512)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.alpha = alpha

    def forward(self, x):
        base = self.linear(x)
        residual = self.up(F.gelu(self.down(x)))
        return F.normalize(base + self.alpha * residual, p=2, dim=-1)


def infonce_multi_positive(projected, targets, group_ids, temperature=0.05):
    sim = projected @ targets.T / temperature
    pos_mask = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1)).float()
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values)
    loss = -torch.log((exp_sim * pos_mask).sum(1) / exp_sim.sum(1) + 1e-8).mean()
    return loss


def infonce_diagonal(projected, targets, group_ids, temperature=0.05):
    sim = projected @ targets.T / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


def encode_int8_gte(captions, cache_name):
    cache_path = OUTPUT_DIR / f"{cache_name}.npy"
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(captions):
            print(f"  Cache hit: {cache_path.name} {cached.shape}")
            return cached

    print(f"  Encoding {len(captions)} captions with INT8 GTE...")
    sess = ort.InferenceSession(GTE_INT8, providers=["CPUExecutionProvider"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(GTE_TOKENIZER, trust_remote_code=True)
    embs = []
    for i in tqdm(range(0, len(captions), 32), desc="GTE"):
        batch = captions[i:i+32]
        enc = tokenizer(batch, max_length=512, padding=True, truncation=True,
                       return_tensors="np")
        out = sess.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64)
        })
        cls_embs = out[0][:, 0, :]
        norms = np.linalg.norm(cls_embs, axis=1, keepdims=True) + 1e-9
        embs.append((cls_embs / norms).astype(np.float32))
    embs = np.concatenate(embs, axis=0)
    np.save(cache_path, embs)
    return embs


def load_sd_audio_pairs():
    cache = np.load(str(SD_CACHE))
    audio_embs = cache["audio"].astype(np.float32)
    norms = np.linalg.norm(audio_embs, axis=1, keepdims=True) + 1e-9
    return audio_embs / norms


def train_one(arch, use_multipos, use_sd_audio, lr=2e-3, temperature=0.05,
              epochs=200, batch_size=1024, patience=20):
    dclap_targets = np.load(DCLAP_EMBEDS).astype(np.float32)
    dclap_targets /= (np.linalg.norm(dclap_targets, axis=1, keepdims=True) + 1e-9)
    n_unique = dclap_targets.shape[0]

    captions, group_ids = [], []
    for lf in sorted(CAPTIONS_DIR.glob("*.json")):
        caps = json.loads(lf.read_text())
        captions.extend(caps)
        group_ids.extend(range(len(caps)))
    group_ids = np.array(group_ids, dtype=np.int64)

    gte_embs = encode_int8_gte(captions, cache_name="gte_int8_multi")
    n_samples = len(captions)

    rng = np.random.default_rng(SEED)
    val_ids = set(rng.choice(n_unique, int(n_unique * 0.1), replace=False))
    train_mask = np.array([group_ids[i] % n_unique not in val_ids for i in range(n_samples)])
    val_mask = ~train_mask

    train_gte = torch.tensor(gte_embs[train_mask])
    val_gte = torch.tensor(gte_embs[val_mask])
    train_groups = torch.tensor(group_ids[train_mask])
    val_groups = torch.tensor(group_ids[val_mask])
    train_targets = torch.tensor(dclap_targets[train_groups.numpy() % n_unique])
    val_targets = torch.tensor(dclap_targets[val_groups.numpy() % n_unique])

    sd_audio = None
    sd_gte = None
    if use_sd_audio:
        sd_audio_np = load_sd_audio_pairs()
        en_captions = json.loads((CAPTIONS_DIR / "en.json").read_text())
        sd_en = en_captions[:sd_audio_np.shape[0]]
        sd_gte = torch.tensor(encode_int8_gte(sd_en, cache_name="gte_int8_en")[:sd_audio_np.shape[0]])
        sd_audio = torch.tensor(sd_audio_np)

    if arch == "linear":
        model = LinearProj().cuda()
    elif arch == "mlp256":
        model = MLPProj(hidden=256).cuda()
    else:
        model = ResidualProj(bottleneck=64, alpha=0.1).cuda()

    loss_fn = infonce_multi_positive if use_multipos else infonce_diagonal
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_mrr = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(train_gte.shape[0])
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            idx = perm[i:i+batch_size]
            if len(idx) < 16:
                continue
            batch_gte = train_gte[idx].cuda()
            batch_targets = train_targets[idx].cuda()
            batch_groups = train_groups[idx].cuda()

            projected = model(batch_gte)
            target_norm = F.normalize(batch_targets, p=2, dim=-1)
            loss = loss_fn(projected, target_norm, batch_groups, temperature=temperature)

            if use_sd_audio and sd_audio is not None:
                sd_idx = torch.randint(0, sd_gte.shape[0], (min(64, sd_gte.shape[0]),))
                sd_proj = model(sd_gte[sd_idx].cuda())
                sd_tgt = F.normalize(sd_audio[sd_idx].cuda(), p=2, dim=-1)
                sd_loss = 1.0 - F.cosine_similarity(sd_proj, sd_tgt, dim=-1).mean()
                loss = loss + 0.2 * sd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_proj = model(val_gte.cuda()).cpu().numpy()
                val_tgt_norm = val_targets.numpy()
                val_tgt_norm /= (np.linalg.norm(val_tgt_norm, axis=1, keepdims=True) + 1e-9)
                sim = val_proj @ val_tgt_norm.T
                ranks = np.array([int((sim[i] > sim[i, i]).sum()) + 1
                                  for i in range(sim.shape[0])])
                mrr = float((1.0 / ranks).mean())

            if mrr > best_mrr:
                best_mrr = mrr
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    return best_mrr, best_state, model


def run_sweep():
    archs = ["linear", "mlp256", "residual64"]
    multipos_opts = [False, True]
    sd_opts = [False, True]

    results = []
    for arch in archs:
        for mp in multipos_opts:
            for sd in sd_opts:
                label = f"{arch}|mp={'Y' if mp else 'N'}|sd={'Y' if sd else 'N'}"
                print(f"\n{'='*60}")
                print(f"  {label}")
                print(f"{'='*60}")
                mrr, state, model = train_one(arch, mp, sd)
                print(f"  -> val MRR = {mrr:.4f}")
                results.append({"label": label, "mrr": mrr, "arch": arch,
                                "multipos": mp, "sd_audio": sd})

                suffix = f"abl_{arch}_mp{int(mp)}_sd{int(sd)}"
                torch.save(state, OUTPUT_DIR / f"best_proj_{suffix}.pt")
                model.load_state_dict(state)
                model.eval().cpu()
                dummy = torch.randn(1, 768)
                onnx_path = OUTPUT_DIR / f"gte_to_dclap_proj_{suffix}.onnx"
                torch.onnx.export(model, dummy, str(onnx_path), opset_version=17,
                                  input_names=["input"],
                                  output_names=["output"],
                                  dynamic_axes={"input": {0: "batch"},
                                               "output": {0: "batch"}})

    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS (12 configs)")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: -x["mrr"]):
        print(f"  {r['mrr']:.4f}  {r['label']}")

    with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_sweep()
