# pip install torch transformers tqdm

import os, json, math, random
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoConfig
from tqdm import tqdm
import copy


# --------------------------
# 0) Config & data paths
# --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # edit if needed

# DATA_SEM_PATH = "/home/user/Desktop/yuhou/LC-Rec/data/Beauty/Beauty.emb-llama-td.npy"
# DATA_INTER_PATH = "/home/user/Desktop/yuhou/LC-Rec/data/Beauty/Beauty.inter.json"

# --------------------------
# Dataset selector
# --------------------------
DATASET = "Tools_and_Home_Improvement"    # <-- change this to switch dataset

BASE_DATA_DIR = "/home/user/Desktop/yuhou/LC-Rec/data"

DATA_SEM_PATH  = f"{BASE_DATA_DIR}/{DATASET}/{DATASET}.emb-llama-td.npy"
DATA_INTER_PATH = f"{BASE_DATA_DIR}/{DATASET}/{DATASET}.inter.json"


LLM_CKPT   = "bert-base-uncased"
MAX_LEN    = 50
BATCH_SIZE = 128
EPOCHS     = 5
LR_PROMPT  = 1e-3
LR_OTHER   = 1e-3
VAL_LAST_K = 1
SEED       = 42
WEIGHT_DECAY_PROMPT = 1e-4   # stabilizes prompt norms
USE_AMP    = True
EARLY_STOP_PATIENCE = 3   # stop if val loss does not improve for 3 epochs


# >>> NEW: communication loss knobs
USE_COMM_LOSS = True
LAMBDA_COMM   = 0.1           # strength of communication consistency loss
COMM_TOPK     = 4             # use top-k senders per target for stability/efficiency (set 0 to use all)

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# --------------------------
# 1) Data loading & splits
# --------------------------
def dedup_consecutive(seq):
    if not seq: return seq
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out

def build_splits(interactions_path, val_last_k=1, min_train_len=2, dedup=True):
    with open(interactions_path, "r", encoding="utf-8") as f:
        user2items = json.load(f)
    train_seqs, val_seqs = [], []
    for _, seq in user2items.items():
        s = [int(x) for x in seq]
        if dedup:
            s = dedup_consecutive(s)
        if len(s) >= min_train_len + val_last_k:
            train_seqs.append(s[:-val_last_k])  # no leakage
            val_seqs.append(s)                  # full; we'll eval on last K only
    return train_seqs, val_seqs

semantic_embeddings = np.load(DATA_SEM_PATH)        # [N, d_sem]
semantic_embeddings = torch.from_numpy(semantic_embeddings).float()
num_items = semantic_embeddings.size(0)
d_sem     = semantic_embeddings.size(1)

train_seqs, val_seqs = build_splits(DATA_INTER_PATH, val_last_k=VAL_LAST_K)

print(f"Device: {device}")
print(f"LLM: {LLM_CKPT}")
print(f"Items: {num_items}, d_sem: {d_sem}, train_users: {len(train_seqs)}, val_users: {len(val_seqs)}")

# --------------------------
# 2) Datasets & collators
# --------------------------
class NextItemDataset(Dataset):
    """Sliding one-step dataset over prefixes (train only)."""
    def __init__(self, seqs: List[List[int]], max_ctx_len: int = MAX_LEN):
        self.samples = []
        for s in seqs:
            if len(s) < 2: continue
            for t in range(1, len(s)):
                ctx = s[max(0, t - max_ctx_len):t]
                tgt = s[t]
                self.samples.append((ctx, tgt))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

class LastKValDataset(Dataset):
    """Evaluate only on last K steps per user to avoid leakage."""
    def __init__(self, full_seqs: List[List[int]], k: int = 1, max_ctx_len: int = MAX_LEN):
        self.samples = []
        for s in full_seqs:
            if len(s) < 2: continue
            for t in range(max(1, len(s)-k), len(s)):
                ctx = s[max(0, t - max_ctx_len):t]
                tgt = s[t]
                self.samples.append((ctx, tgt))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn_padded(batch):
    """Pads context sequences with -1 so 0 can remain a valid item ID. Tail-aligned."""
    ctxs = [x[0] for x in batch]
    tgts = torch.tensor([x[1] for x in batch], dtype=torch.long)
    lengths = torch.tensor([len(s) for s in ctxs], dtype=torch.long)
    max_len = lengths.max().item() if len(lengths) > 0 else 1
    pad = -1
    ctxs_pad = torch.full((len(batch), max_len), fill_value=pad, dtype=torch.long)
    for i, s in enumerate(ctxs):
        if len(s) > 0:
            ctxs_pad[i, -len(s):] = torch.tensor(s, dtype=torch.long)  # tail-align
    return ctxs_pad, lengths, tgts

train_ds = NextItemDataset(train_seqs)
val_ds   = LastKValDataset(val_seqs, k=VAL_LAST_K)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn_padded)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_padded)

# --------------------------
# 3) Model (Option B + comm-loss support)
# --------------------------
class HybridRecPromptModel(nn.Module):
    """
    Option B (with comm-loss support):
      - Trainable prompt per item: v_prompt[i] in R^{d_prompt}
      - ONE token per item: proj_cat( cat[E_sem[i], v_prompt[i]] ) -> R^{d_model}
      - Can optionally return attentions and token-level item IDs for comm-loss
    """
    def __init__(self, llm_ckpt: str, semantic_table: torch.FloatTensor, d_prompt: int = 384):
        super().__init__()
        print(f"Loading config: {llm_ckpt}")
        cfg = AutoConfig.from_pretrained(llm_ckpt)
        print(f"Loading weights: {llm_ckpt}")
        self.llm = AutoModel.from_pretrained(llm_ckpt, low_cpu_mem_usage=True)

        # freeze backbone
        self.llm.eval()
        for p in self.llm.parameters():
            p.requires_grad_(False)

        self.llm_dtype = next(self.llm.parameters()).dtype
        self.d_model = cfg.hidden_size
        self.num_items = semantic_table.size(0)
        self.d_sem = semantic_table.size(1)

        # store semantic table (not trained)
        self.register_buffer("E_sem", semantic_table.clone(), persistent=False)  # [N, d_sem]

        # trainable per-item prompt vector (smaller than d_model; projected via proj_cat)
        self.d_prompt = d_prompt
        self.v_prompt = nn.Parameter(torch.randn(self.num_items, self.d_prompt, dtype=self.llm_dtype) * 0.02)

        # projector from (d_sem + d_prompt) -> d_model
        self.proj_cat = nn.Linear(self.d_sem + self.d_prompt, self.d_model, bias=False)
        self.proj_cat.to(dtype=self.llm_dtype)

        # next-item classifier
        self.head = nn.Linear(self.d_model, self.num_items)
        self.head.to(dtype=self.llm_dtype)

        print(f"Model ready: d_model={self.d_model}, d_prompt={self.d_prompt}, llm_dtype={self.llm_dtype}")

    def build_inputs_embeds(self, batch_item_ids_padded: torch.Tensor, batch_lengths: torch.Tensor):
        """
        Build inputs with a FIXED sequence length T = ctxs_padded.shape[1] to keep
        shapes identical across DataParallel replicas.
        Returns:
            embeds:      [B, T, d_model]
            amask:       [B, T]
            tok_item_ids:[B, T]  (item id at each token pos, -1 for pad)
        """
        B, T = batch_item_ids_padded.shape

        device = self.v_prompt.device
        embeds = torch.zeros(B, T, self.d_model, dtype=self.llm_dtype, device=device)
        amask  = torch.zeros(B, T, dtype=torch.long, device=device)
        tok_item_ids = torch.full((B, T), fill_value=-1, dtype=torch.long, device=device)

        # Fill positionally (respecting left pads)
        for b in range(B):
            for t in range(T):
                item_id = int(batch_item_ids_padded[b, t].item())
                if item_id < 0 or item_id >= self.num_items:
                    continue  # keep as pad
                sem_raw = self.E_sem[item_id].to(self.llm_dtype)   # [d_sem]
                v_p     = self.v_prompt[item_id]                   # [d_prompt]
                cat     = torch.cat([sem_raw, v_p], dim=-1)        # [d_sem + d_prompt]
                tok     = self.proj_cat(cat)                       # [d_model]
                embeds[b, t, :] = tok
                amask[b, t] = 1
                tok_item_ids[b, t] = item_id

        return embeds, amask, tok_item_ids

    def forward(self, batch_item_ids_padded: torch.Tensor, batch_lengths: torch.Tensor, return_attn: bool = False):
        # Build fixed-T inputs so all replicas match
        x, amask, tok_item_ids = self.build_inputs_embeds(batch_item_ids_padded, batch_lengths)

        out = self.llm(
            inputs_embeds=x,
            attention_mask=amask,
            output_attentions=return_attn
        )
        # last valid token per sequence
        last_idx = amask.sum(dim=1) - 1
        H_last = out.last_hidden_state[torch.arange(x.size(0), device=x.device), last_idx]
        H_last = H_last.to(self.head.weight.dtype)
        logits = self.head(H_last)

        if return_attn:
            return logits, out.attentions, amask, tok_item_ids
        else:
            return logits, None, None, None

# --------------------------
# 4) Instantiate & optim
# --------------------------
model = HybridRecPromptModel(LLM_CKPT, semantic_embeddings.to(device), d_prompt=192).to(device)   # 384, 192, 96

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs (DataParallel).")
    model = nn.DataParallel(model)

model_to_opt = model.module if isinstance(model, nn.DataParallel) else model

# Separate WD for prompts (helps stability); proj_cat/head can share LR
opt = torch.optim.AdamW([
    {"params": [model_to_opt.v_prompt], "lr": LR_PROMPT, "weight_decay": WEIGHT_DECAY_PROMPT},
    {"params": list(model_to_opt.proj_cat.parameters()) + list(model_to_opt.head.parameters()), "lr": LR_OTHER}
])

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

# --------------------------
# 5) Communication Consistency Loss (history -> target)
# --------------------------
def communication_consistency_loss(attentions, amask, tok_item_ids, v_prompt, topk: int = COMM_TOPK) -> torch.Tensor:
    """
    attentions: tuple/list of L tensors, each [B, H, T, T]
    amask:      [B, T] (1 for real tokens)
    tok_item_ids: [B, T] (-1 for pad)
    v_prompt:   [N, d_prompt] (trainable table)
    topk:       use top-k senders per target (0 -> use all)
    """
    if attentions is None:
        return torch.zeros([], device=amask.device, dtype=v_prompt.dtype)

    # 1) Average heads and layers: A_bar: [B, T, T]
    A_stack = torch.stack(attentions, dim=0)          # [L, B, H, T, T]
    C = A_stack.mean(dim=(0,2))                       # [B, T, T]

    # 2) last valid index (target), and build history mask
    B, T = amask.shape
    last_idx = amask.sum(dim=1) - 1                   # [B]
    b_idx = torch.arange(B, device=amask.device)
    C_last = C[b_idx, last_idx]                       # [B, T]  (row for each target)

    # valid history positions: before last_idx and amask==1
    pos = torch.arange(T, device=amask.device).unsqueeze(0).expand(B, T)
    hist_mask = (pos < last_idx.unsqueeze(1)) & (amask.bool())

    # zero-out non-history, normalize row-wise
    C_last = C_last.masked_fill(~hist_mask, 0.0)
    row_sum = C_last.sum(dim=1, keepdim=True).clamp_min(1e-9)
    C_last = (C_last / row_sum).detach()              # stop-grad

    # (optional) top-k per row
    if topk and topk > 0:
        k = min(topk, T-1)
        top_vals, top_idx = torch.topk(C_last, k=k, dim=1)
        mask_top = torch.zeros_like(C_last, dtype=torch.bool)
        mask_top.scatter_(1, top_idx, True)
        C_last = C_last * mask_top
        # re-normalize after masking
        row_sum = C_last.sum(dim=1, keepdim=True).clamp_min(1e-9)
        C_last = C_last / row_sum

    # 3) fetch prompt vectors for target and history tokens
    #    Build per-position prompt tensor [B, T, d_prompt]
    #    For pads (-1), we place zeros and they won't contribute (masked by C_last)
    d_prompt = v_prompt.size(1)
    p_all = torch.zeros(B, T, d_prompt, device=v_prompt.device, dtype=v_prompt.dtype)
    valid_ids = tok_item_ids.ge(0)
    if valid_ids.any():
        # map ids to vectors
        p_all[valid_ids] = v_prompt[tok_item_ids[valid_ids]]

    # target prompts [B, d_prompt] and history prompts [B, T, d_prompt]
    p_last = p_all[b_idx, last_idx]                   # [B, d_prompt]
    p_hist = p_all                                    # [B, T, d_prompt]

    # 4) cosine similarities between target and each history position
    p_last_n = F.normalize(p_last.float(), dim=-1)    # float32 for numerical stability
    p_hist_n = F.normalize(p_hist.float(), dim=-1)
    sim = torch.einsum('bd,btd->bt', p_last_n, p_hist_n)   # [B, T]
    sim = sim.clamp(-1.0, 1.0)

    # 5) weighted (1 - cos), masked to history only
    one_minus_sim = (1.0 - sim) * hist_mask.float()
    comm_loss = (C_last * one_minus_sim).sum() / (hist_mask.sum().clamp_min(1.0))
    return comm_loss

# --------------------------
# 6) Train / Val loops
# --------------------------
def run_epoch(loader, train=True, desc=""):
    model.train(train)
    total_loss, total = 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for ctxs_padded, ctxs_lengths, tgts in pbar:
        ctxs_padded = ctxs_padded.to(device)
        ctxs_lengths = ctxs_lengths.to(device)
        tgts = tgts.to(device)

        want_attn = USE_COMM_LOSS and train  # compute comm loss only when training
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits, attns, amask, tok_ids = model(ctxs_padded, ctxs_lengths, return_attn=want_attn)
            ce_loss = F.cross_entropy(logits, tgts)

            if USE_COMM_LOSS and train:
                # attns: tuple(L) of [B,H,T,T]
                comm_loss = communication_consistency_loss(
                    attentions=attns, amask=amask, tok_item_ids=tok_ids,
                    v_prompt=(model.module.v_prompt if isinstance(model, nn.DataParallel) else model.v_prompt),
                    topk=COMM_TOPK
                )
                loss = ce_loss + LAMBDA_COMM * comm_loss
            else:
                loss = ce_loss

        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model_to_opt.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        bs = tgts.size(0)
        total_loss += loss.item() * bs
        total += bs
        pbar.set_postfix(loss=f"{(total_loss / max(total,1)):.4f}")
    return total_loss / max(total, 1)

print("Start training...")
for epoch in range(1, EPOCHS+1):
    tr = run_epoch(train_loader, train=True,  desc=f"Epoch {epoch} [train]")
    with torch.no_grad():
        va = run_epoch(val_loader,   train=False, desc=f"Epoch {epoch} [val]")
    print(f"epoch {epoch}: train={tr:.4f}  val={va:.4f}")

# print("Start training...")

# best_val_loss = float("inf")
# best_state = None
# patience_counter = 0

# for epoch in range(1, EPOCHS+1):
#     tr = run_epoch(train_loader, train=True,  desc=f"Epoch {epoch} [train]")
#     with torch.no_grad():
#         va = run_epoch(val_loader,   train=False, desc=f"Epoch {epoch} [val]")
#     print(f"epoch {epoch}: train={tr:.4f}  val={va:.4f}")

#     # ---- Early stopping logic ----
#     if va < best_val_loss - 1e-4:   # small margin to avoid tiny fluctuations
#         best_val_loss = va
#         patience_counter = 0
#         best_state = copy.deepcopy(model.state_dict())
#         print(f"  -> New best val loss: {best_val_loss:.4f}, saving model.")
#     else:
#         patience_counter += 1
#         print(f"  -> No improvement. Patience {patience_counter}/{EARLY_STOP_PATIENCE}")
#         if patience_counter >= EARLY_STOP_PATIENCE:
#             print(f"Early stopping triggered at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
#             break


# --------------------------
# 7) Export embeddings for item tokenization
# --------------------------
# # Restore best model (early-stopped)
# if best_state is not None:
#     model.load_state_dict(best_state)

final_model = model.module if isinstance(model, nn.DataParallel) else model

E_sem_np    = final_model.E_sem.detach().float().cpu().numpy()               # [N, d_sem]
V_prompt_np = final_model.v_prompt.detach().float().cpu().numpy()           # [N, d_prompt]

def l2n(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / n

E_sem_np_n    = l2n(E_sem_np)
V_prompt_np_n = l2n(V_prompt_np)

hybrid_vecs = np.concatenate([E_sem_np_n, V_prompt_np_n], axis=1)           # [N, d_sem + d_prompt]

# os.makedirs("outputs", exist_ok=True)
# np.save("outputs/Beauty/item_embeddings_sem_plus_prompt1.npy", hybrid_vecs)
# np.save("outputs/Beauty/item_semantic_only1.npy", E_sem_np)
# np.save("outputs/Beauty/item_prompt_only1.npy", V_prompt_np)

# print("Saved:")
# print(f" - outputs/item_embeddings_sem_plus_prompt.npy  shape={hybrid_vecs.shape}")
# print(f" - outputs/item_semantic_only.npy               shape={E_sem_np.shape}")
# print(f" - outputs/item_prompt_only.npy                 shape={V_prompt_np.shape}")

# ---- NEW: dataset-aware output directory ----
OUTPUT_DIR = f"outputs/{DATASET}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save files
np.save(f"{OUTPUT_DIR}/item_embeddings_sem_plus_prompt1.npy", hybrid_vecs)
np.save(f"{OUTPUT_DIR}/item_semantic_only1.npy", E_sem_np)
np.save(f"{OUTPUT_DIR}/item_prompt_only1.npy", V_prompt_np)

print("Saved:")
print(f" - {OUTPUT_DIR}/item_embeddings_sem_plus_prompt.npy  shape={hybrid_vecs.shape}")
print(f" - {OUTPUT_DIR}/item_semantic_only.npy               shape={E_sem_np.shape}")
print(f" - {OUTPUT_DIR}/item_prompt_only.npy                 shape={V_prompt_np.shape}")