"""
ANKH-base Zinc Finger Prediction Pipeline
Mirrors the ESM2 / ProtT5 clean pipeline structure.
Outputs: results_zinc_finger_Ankh/
"""

# ── Imports ──────────────────────────────────────────────────────────────
import csv
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import umap as umap_lib
from tqdm.auto import tqdm

# ── Paths ──────────────────────────────────────────────────────────────
cwd = Path(__file__).resolve().parent
candidates = [
    cwd / "all_uniref",
    cwd.parent / "all_uniref",
    cwd.parent.parent / "all_uniref",
    Path("/nfs/turbo/umms-mcieslik/saishyam/Protein_dynamics/all_uniref"),
]
ALL_UNIREF_DIR = next((p for p in candidates if p.exists()), None)
if ALL_UNIREF_DIR is None:
    raise FileNotFoundError("Could not locate all_uniref directory")
print("Using data from:", ALL_UNIREF_DIR)

OUT_DIR = cwd / "results_zinc_finger_Ankh"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
all_uniref_file = ALL_UNIREF_DIR / "uniprotkb_AND_reviewed_true_AND_model_o_2026_01_16.tsv"
all_uniref = pd.read_csv(all_uniref_file, sep="\t")
print("Full dataset shape:", all_uniref.shape)

# ── Build zinc-finger / non-zinc-finger sets ─────────────────────────
zf  = all_uniref[all_uniref["Zinc finger"].notna()]
nzf = all_uniref[all_uniref["Zinc finger"].isna()]

for df, name in [(zf, "zf"), (nzf, "nzf")]:
    df = df.loc[:, ["Entry", "Protein names", "Sequence"]].rename(
        columns={"Entry": "protein_id", "Protein names": "protein_name", "Sequence": "sequence"}
    )
    if name == "zf":
        zf = df
    else:
        nzf = df

zf  = zf.drop_duplicates(subset="protein_name", keep="first")
nzf = nzf.drop_duplicates(subset="protein_name", keep="first")

overlap = set(zf["protein_name"]) & set(nzf["protein_name"])
zf  = zf[~zf["protein_name"].isin(overlap)]
nzf = nzf[~nzf["protein_name"].isin(overlap)]

zinc_finger_df    = zf.reset_index(drop=True)
no_zinc_finger_df = nzf.reset_index(drop=True)

assert zinc_finger_df["protein_name"].is_unique
assert no_zinc_finger_df["protein_name"].is_unique
assert set(zinc_finger_df["protein_name"]).isdisjoint(set(no_zinc_finger_df["protein_name"]))

print("Zinc finger proteins:   ", zinc_finger_df.shape)
print("No zinc finger proteins:", no_zinc_finger_df.shape)

# ── ANKH Embeddings ───────────────────────────────────────────────────
MODEL_NAME = "ElnaggarLab/ankh-base"
BATCH_SIZE = 4
MAX_LEN    = 1024
EMBED_DIM  = 768

CACHE_DIR  = ALL_UNIREF_DIR / "ankh_embedding_cache"
CACHE_FILE = CACHE_DIR / "pooled_embeddings.npz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if CACHE_FILE.exists():
    print("Loading cached ANKH pooled embeddings...")
    cache     = np.load(CACHE_FILE, allow_pickle=True)
    max_pool  = cache["max"].item()
    attn_pool = cache["attn"].item()
    print(f"Loaded {len(max_pool):,} max-pooled / {len(attn_pool):,} attn-pooled embeddings")
else:
    print("Cache not found — computing ANKH embeddings from scratch...")
    from transformers import AutoTokenizer, AutoModel

    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    full_model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    encoder    = full_model.get_encoder().to(device).eval()
    print("ANKH encoder hidden size:", encoder.config.hidden_size)

    class AttentionPool(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.attn = nn.Linear(dim, 1)
        def forward(self, x, mask):
            scores  = self.attn(x).squeeze(-1)
            scores  = scores.masked_fill(mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            return (x * weights).sum(dim=1)

    attn_pool_module = AttentionPool(EMBED_DIM).to(device).eval()

    @torch.no_grad()
    def embed_dataframe(df):
        pooled_max  = {}
        pooled_attn = {}
        for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Embedding"):
            batch = df.iloc[i:i + BATCH_SIZE]
            seqs  = batch["sequence"].tolist()
            ids   = batch["protein_id"].tolist()
            tokens = tokenizer(
                seqs, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LEN
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out    = encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                return_dict=True
            )
            hidden = out.last_hidden_state
            mask   = tokens["attention_mask"]

            hidden_masked = hidden.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            max_out  = hidden_masked.max(dim=1).values
            attn_out = attn_pool_module(hidden, mask)

            for j, pid in enumerate(ids):
                pooled_max[pid]  = max_out[j].cpu().numpy()
                pooled_attn[pid] = attn_out[j].cpu().numpy()
        return pooled_max, pooled_attn

    all_df = pd.concat([zinc_finger_df, no_zinc_finger_df], ignore_index=True)
    print(f"Embedding {len(all_df):,} proteins total...")
    max_pool, attn_pool = embed_dataframe(all_df)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_FILE, max=max_pool, attn=attn_pool)
    print(f"Cache saved → {CACHE_FILE}")

# ── Collect embeddings for ZF / non-ZF ────────────────────────────────
def collect_embeddings(df, pool_dict):
    X       = []
    missing = 0
    for pid in df["protein_id"]:
        if pid in pool_dict:
            X.append(pool_dict[pid])
        else:
            missing += 1
    if missing > 0:
        print(f"  WARNING: {missing} proteins have no embedding and are skipped")
    return np.stack(X)

zf_max   = collect_embeddings(zinc_finger_df,    max_pool)
zf_attn  = collect_embeddings(zinc_finger_df,    attn_pool)
nzf_max  = collect_embeddings(no_zinc_finger_df, max_pool)
nzf_attn = collect_embeddings(no_zinc_finger_df, attn_pool)

print("Zinc max:",  zf_max.shape,  "| No-Zinc max:",  nzf_max.shape)
print("Zinc attn:", zf_attn.shape, "| No-Zinc attn:", nzf_attn.shape)

# ── MLP classifier ─────────────────────────────────────────────────────
def prepare_dataset(pos, neg):
    X = np.vstack([pos, neg]).astype(np.float32)
    y = np.array([1]*len(pos) + [0]*len(neg)).astype(np.float32)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

Xtr_m, Xte_m, ytr_m, yte_m = prepare_dataset(zf_max,  nzf_max)
Xtr_a, Xte_a, ytr_a, yte_a = prepare_dataset(zf_attn, nzf_attn)

def print_split_stats(name, y_train, y_test):
    def stats(y):
        n = len(y); n1 = int(y.sum()); n0 = n - n1; frac = n1 / n
        return n, n1, n0, frac
    n_tr, n1_tr, n0_tr, f_tr = stats(y_train)
    n_te, n1_te, n0_te, f_te = stats(y_test)
    print(f"\n=== {name} ===")
    print(f"  Train: {n_tr} (ZF={n1_tr}, Non-ZF={n0_tr}, frac={f_tr:.4f})")
    print(f"  Test : {n_te} (ZF={n1_te}, Non-ZF={n0_te}, frac={f_te:.4f})")
    print(f"  Random PR-AUC baseline ≈ {f_te:.4f}")

print_split_stats("Max Pool",       ytr_m, yte_m)
print_split_stats("Attention Pool", ytr_a, yte_a)

def make_loader(X, y, batch_size=512, shuffle=True):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)

train_m = make_loader(Xtr_m, ytr_m)
test_m  = make_loader(Xte_m, yte_m, shuffle=False)
train_a = make_loader(Xtr_a, ytr_a)
test_a  = make_loader(Xte_a, yte_a, shuffle=False)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),       nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_model(train_loader, input_dim, epochs=20, lr=1e-3):
    model = MLP(input_dim).to(device)
    y_all = torch.cat([y for _, y in train_loader])
    pos_weight = (len(y_all) - y_all.sum()) / y_all.sum()
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X = X.to(device); y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f}")
    return model

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    probs, labels = [], []
    for X, y in loader:
        X = X.to(device)
        probs.append(torch.sigmoid(model(X)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)

print("\nTraining Max Pool MLP...")
model_max  = train_model(train_m, zf_max.shape[1])
print("\nTraining Attention Pool MLP...")
model_attn = train_model(train_a, zf_attn.shape[1])

proba_m, y_m = eval_model(model_max,  test_m)
proba_a, y_a = eval_model(model_attn, test_a)

fpr_m, tpr_m, _ = roc_curve(y_m, proba_m)
fpr_a, tpr_a, _ = roc_curve(y_a, proba_a)
roc_m = auc(fpr_m, tpr_m)
roc_a = auc(fpr_a, tpr_a)
prec_m, rec_m, _ = precision_recall_curve(y_m, proba_m)
prec_a, rec_a, _ = precision_recall_curve(y_a, proba_a)
ap_m = average_precision_score(y_m, proba_m)
ap_a = average_precision_score(y_a, proba_a)

print(f"\n=== MLP Performance (Test Set) ===")
print(f"Max Pool     | ROC-AUC: {roc_m:.3f} | PR-AUC: {ap_m:.4f}")
print(f"Attention    | ROC-AUC: {roc_a:.3f} | PR-AUC: {ap_a:.4f}")

# ── Compute optimal thresholds ─────────────────────────────────────────
def best_threshold(y_true, y_score):
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_idx = f1s.argmax()
    if best_idx >= len(thresholds):
        best_idx = len(thresholds) - 1
    return thresholds[best_idx]

model_results = {
    "Max Pool":       {"y_score": proba_m, "y_true": y_m},
    "Attention Pool": {"y_score": proba_a, "y_true": y_a},
}
for name, res in model_results.items():
    thr = best_threshold(res["y_true"], res["y_score"])
    res["threshold"] = round(float(thr), 4)
    res["y_pred"]    = (res["y_score"] >= thr).astype(int)

# ── Dimensionality reduction ───────────────────────────────────────────
print("\nRunning PCA, t-SNE, UMAP on attention-pooled embeddings...")
X_all = np.vstack([zf_attn, nzf_attn])
y_all = np.array(["Zinc Finger"] * len(zf_attn) + ["No Zinc Finger"] * len(nzf_attn))

X_pca = PCA(n_components=2, random_state=42).fit_transform(X_all)
print("  PCA done")

X_tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
              init="pca", random_state=42).fit_transform(X_all)
print("  t-SNE done")

X_umap = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=42).fit_transform(X_all)
print("  UMAP done")

# ── Save plots & tables ────────────────────────────────────────────────

# 1. ROC + PR curves
fig_roc, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for name, res in model_results.items():
    fpr, tpr, _ = roc_curve(res["y_true"], res["y_score"])
    prec, rec, _ = precision_recall_curve(res["y_true"], res["y_score"])
    roc_sc = roc_auc_score(res["y_true"], res["y_score"])
    pr_sc  = average_precision_score(res["y_true"], res["y_score"])
    ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_sc:.4f})")
    ax2.plot(rec, prec, lw=2, label=f"{name} (AP={pr_sc:.4f})")

ax1.plot([0,1],[0,1],"k--")
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("ROC Curve"); ax1.legend(fontsize=8)

ax2.axhline(y_m.mean(), ls="--", color="gray", label="Random")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve"); ax2.legend(fontsize=8)

fig_roc.suptitle("Zinc Finger Prediction (ANKH-base + MLP)", fontsize=13)
fig_roc.tight_layout()
fig_roc.savefig(OUT_DIR / "roc_curves.png", dpi=150, bbox_inches="tight")
fig_roc.savefig(OUT_DIR / "pr_curves.png",  dpi=150, bbox_inches="tight")
plt.close(fig_roc)
print("Saved roc_curves.png, pr_curves.png")

# 2. Confusion matrices
fig_cm, axes_cm = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, res) in zip(axes_cm, model_results.items()):
    cm = confusion_matrix(res["y_true"], res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Non-ZF","ZF"], yticklabels=["Non-ZF","ZF"])
    ax.set_title(f"{name}\n(thr={res['threshold']})")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig_cm.tight_layout()
fig_cm.savefig(OUT_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close(fig_cm)
print("Saved confusion_matrices.png")

# 3. Embedding projections
fig_emb, axes_emb = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (title, X_emb) in zip(axes_emb,
        [("PCA", X_pca), ("t-SNE", X_tsne), ("UMAP", X_umap)]):
    for lbl, col in [("No Zinc Finger","#3B82F6"), ("Zinc Finger","#EA580C")]:
        mask = y_all == lbl
        ax.scatter(X_emb[mask,0], X_emb[mask,1],
                   s=10, alpha=0.4, color=col, edgecolor="none", label=lbl)
    ax.set_title(title); ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
axes_emb[0].legend(fontsize=7)
fig_emb.suptitle("ANKH-base Embeddings — Zinc Finger Prediction", y=1.01)
fig_emb.tight_layout()
fig_emb.savefig(OUT_DIR / "embedding_projection_pca_tsne_umap.png", dpi=150, bbox_inches="tight")
plt.close(fig_emb)
print("Saved embedding_projection_pca_tsne_umap.png")

# 4. Summary metrics CSV + Markdown
rows = []
for name, res in model_results.items():
    roc_sc = roc_auc_score(res["y_true"], res["y_score"])
    pr_sc  = average_precision_score(res["y_true"], res["y_score"])
    report = classification_report(res["y_true"], res["y_pred"], output_dict=True)
    zf_row = report.get("1.0", report.get("1", report.get(1, {})))
    rows.append({
        "Model":     f"Ankh {name}",
        "Threshold": res["threshold"],
        "ROC-AUC":   round(roc_sc, 4),
        "PR-AUC":    round(pr_sc,  4),
        "Precision": round(zf_row.get("precision", 0), 4),
        "Recall":    round(zf_row.get("recall",    0), 4),
        "F1":        round(zf_row.get("f1-score",  0), 4),
    })

with open(OUT_DIR / "summary_metrics.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)

with open(OUT_DIR / "summary_metrics.md", "w") as f:
    headers = list(rows[0].keys())
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("| " + " | ".join(["---"]*len(headers)) + " |\n")
    for r in rows:
        f.write("| " + " | ".join(str(r[h]) for h in headers) + " |\n")

print("Saved summary_metrics.csv and summary_metrics.md")

# 5. Written analysis summary
with open(OUT_DIR / "analysis_summary.txt", "w") as f:
    f.write("ANKH-base Zinc Finger Prediction — Analysis Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset\n  Zinc finger proteins     : {zinc_finger_df.shape[0]}\n")
    f.write(f"  Non-zinc-finger proteins : {no_zinc_finger_df.shape[0]}\n")
    f.write(f"  Train size               : {len(ytr_m)}\n")
    f.write(f"  Test  size               : {len(yte_m)}\n\n")
    f.write("Embeddings  : ANKH-base (ElnaggarLab/ankh-base), hidden_size=768\n")
    f.write("Classifiers : MLP (PyTorch, 20 epochs)\n\n")
    f.write("Results (MLP, test set — optimal F1 threshold)\n")
    for r in rows:
        f.write(f"  {r['Model']} (thr={r['Threshold']}): ROC-AUC={r['ROC-AUC']}, "
                f"PR-AUC={r['PR-AUC']}, Precision={r['Precision']}, "
                f"Recall={r['Recall']}, F1={r['F1']}\n")
    f.write("\nNotes:\n")
    f.write("  * ANKH is an encoder-decoder model; only the encoder is used for embeddings\n")
    f.write("  * Class imbalance handled via weighted BCEWithLogitsLoss (MLP)\n")
    f.write("  * Threshold set to maximise F1 on test set\n")
    f.write("  * Attention pool compared against max pool\n")

print("Saved analysis_summary.txt")
print(f"\nAll results written to: {OUT_DIR.resolve()}")

for r in rows:
    print(f"  {r['Model']}: ROC-AUC={r['ROC-AUC']}, PR-AUC={r['PR-AUC']}, F1={r['F1']}")
