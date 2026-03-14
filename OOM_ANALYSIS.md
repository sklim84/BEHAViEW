# CUDA OOM Analysis: GBT, GRACE, MVGRL on HOFINET (452K nodes)

## Root Cause Summary

The OOM occurs when the PyGCL loss functions create **N x N similarity/mask matrices** where N = number of nodes in the batch/graph. For 452K nodes with float32, a single N x N matrix = 452K^2 * 4 bytes = **762 GB** -- far beyond any GPU memory.

---

## Per-Model Analysis

### 1. GBT (`gbt_w_cen.py`) -- WithinEmbedContrast + BarlowTwins

**Training**: FULL-BATCH (line 61: `encoder_model(data.x, x_cen, data.edge_index)`)

**OOM location**: `WithinEmbedContrast.forward()` calls `BarlowTwins.compute()` which receives `pos_mask` and `neg_mask` from `SameScaleSampler`. However, BarlowTwins loss itself does **NOT** use the masks -- it computes `z1_norm.T @ z2_norm` which is **D x D** (feature_dim x feature_dim), NOT N x N.

**Actual OOM**: The OOM is NOT in the loss itself. It's in the **full-batch GNN forward pass** -- encoding all 452K nodes at once through GCN with `hidden_dim=256` on 2.56M edges. The intermediate activations during forward+backward are the bottleneck.

**Wait -- re-checking**: `WithinEmbedContrast` (line 117-120 of contrast_model.py) calls `self.loss(anchor=h1, sample=h2)` directly, bypassing the sampler entirely. BarlowTwins.compute receives anchor/sample but the `Loss` base class may add masks. Let me check... Actually `WithinEmbedContrast` passes directly to `self.loss()` which calls `BarlowTwins.compute()`. BarlowTwins ignores pos_mask/neg_mask (its compute signature accepts them but doesn't use them in `bt_loss`). So the cross-correlation matrix is D x D (e.g., 256 x 256), not N x N.

**Revised OOM cause for GBT**: Full-batch forward pass on 452K nodes. Memory needed: node features (452K x input_dim), edge_index (2.56M x 2), intermediate GCN layers (452K x 512 for first layer, 452K x 256 for second), plus backward pass activations. Estimated: ~3-5 GB for forward, doubled for backward. This may actually fit on a large GPU (24-48GB). If it OOMs, it's likely with `hidden_dim=256` + `2*hidden_dim=512` intermediate = tight on 16GB GPUs.

### 2. GRACE (`grace_w_cen.py`) -- DualBranchContrast L2L + InfoNCE

**Training**: MINI-BATCH via NeighborLoader (batch_size=4096, num_neighbors=[10,10])

**OOM location**: `DualBranchContrast` in L2L mode uses `SameScaleSampler` (line 51-52 of contrast_model.py):
```python
pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)  # N x N
neg_mask = 1. - pos_mask  # N x N
```
Then with `intraview_negs=True`, `add_intraview_negs` expands to N x 2N masks.

For batch_size=4096 with 2-hop [10,10] neighbors, the actual batch `num_nodes` can expand to ~4096 + 4096*10 + 4096*10*10 = **~450K nodes** (nearly the entire graph due to high connectivity). The similarity matrix in InfoNCE is then ~450K x 450K = **756 GB**.

**This is the primary OOM**: The NeighborLoader batch explodes because HOFINET is dense. The loss computes `_similarity(anchor, sample)` = `h1 @ h2.t()` producing the N x N matrix.

### 3. MVGRL (`mvgrl_w_cen.py`) -- DualBranchContrast G2L + BootstrapLatent

**Training**: FULL-BATCH (line 68: `encoder_model(data.x, x_cen, data.edge_index)`)

**OOM location**: `DualBranchContrast` in G2L mode with `neg_sample` provided (line 59-61 of contrast_model.py). The sampler creates:
- `pos_mask`: 1 x 2N (1 x 904K)
- `sample`: 2N x D (904K x 256)

Then `BootstrapLatent.compute()` does `anchor @ sample.t()` where anchor is 1 x D and sample is 2N x D, producing a 1 x 2N matrix -- only 3.6 MB. This is fine.

**Revised OOM cause for MVGRL**: Full-batch forward pass. The encoder runs **4 forward passes** through two separate GConv encoders (z1, z2, z1n, z2n), each processing 452K nodes. That's 4 x (452K x 256 x num_layers) activations + gradients. With `num_layers=3` and `hidden_dim=256`: approximately 4 passes x 3 layers x 452K x 256 x 4 bytes = **~5.3 GB** just for activations, doubled for gradients = ~10.6 GB. Plus edge_index storage, augmented copies, etc. Total likely 15-20 GB.

---

## Why Working Models Don't OOM

### DGI-Inductive (`dgi_inductive_w_cen.py`)
- Uses **NeighborLoader** (batch_size=4096)
- Uses **SingleBranchContrast G2L** -- the anchor `g` is 1 x D (graph-level summary), so the similarity matrix is 1 x N (tiny)
- SAGEConv is more memory-efficient than GCNConv
- Only 30 epochs

### BGRL (`bgrl_w_cen.py`)
- Full-batch BUT uses **custom `bootstrap_latent_loss()`** (line 108-112) that computes `F.cosine_similarity(h1_pred, h2_target, dim=-1)` -- element-wise across the batch dimension, producing an N-length vector, NOT an N x N matrix
- Bypasses PyGCL's contrast model entirely
- Only 100 epochs

---

## Solutions by Model

### Solution A: GBT -- Add NeighborLoader Mini-batching

GBT's BarlowTwins loss is already memory-safe (D x D cross-correlation). The fix is mini-batching the GNN forward pass.

**Changes needed:**
1. Add NeighborLoader to `main()` (like GRACE/DGI-IND)
2. Modify `train()` to iterate over batches, index `x_cen` by `batch.n_id`
3. BarlowTwins loss works per-batch naturally (cross-correlation over batch samples)

**Code sketch:**
```python
def train(encoder_model, contrast_model, data, x_cen, loader, optimizer, device):
    encoder_model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch_x_cen = x_cen[batch.n_id.cpu()].to(device)
        optimizer.zero_grad()
        _, z1, z2 = encoder_model(batch.x, batch_x_cen, batch.edge_index, batch.edge_attr)
        loss = contrast_model(z1, z2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
```

**Semantic impact**: Minimal. BarlowTwins computes feature cross-correlation, which is a good estimator even on mini-batches. The original Barlow Twins paper uses mini-batch training.

### Solution B: GRACE -- Fix NeighborLoader Explosion + Chunked InfoNCE

GRACE already has NeighborLoader but with `num_neighbors=[-1, -1]` (unlimited) or `[10, 10]` which still explodes on dense graphs.

**Option B1: Limit neighbors more aggressively**
```python
train_loader = NeighborLoader(data, num_neighbors=[5, 5], batch_size=2048, shuffle=True)
```
This caps expansion but changes the receptive field.

**Option B2: Chunked InfoNCE loss (preserves exact semantics)**

Replace `_similarity(anchor, sample)` with chunked computation:
```python
def chunked_infonce(anchor, sample, tau, chunk_size=4096):
    """Compute InfoNCE without full N x N matrix"""
    anchor = F.normalize(anchor)
    sample = F.normalize(sample)
    N = anchor.size(0)
    loss = 0.0
    for i in range(0, N, chunk_size):
        a_chunk = anchor[i:i+chunk_size]  # [C, D]
        sim = a_chunk @ sample.t() / tau  # [C, N] -- still large but C << N
        # positive is diagonal element
        pos_idx = torch.arange(i, min(i+chunk_size, N), device=anchor.device)
        pos_sim = (a_chunk * sample[pos_idx]).sum(dim=1) / tau  # [C]
        log_sum_exp = torch.logsumexp(sim, dim=1)  # [C]
        loss += (log_sum_exp - pos_sim).sum()
    return loss / N
```
Memory: O(chunk_size x N) instead of O(N x N). For chunk=4096, N=452K: 4096 x 452K x 4 = 7.4 GB per chunk -- still large.

**Option B3: Subgraph sampling (recommended)**

Sample a fixed number of nodes per batch for the contrastive loss:
```python
def train_with_subsample(loader, encoder_model, contrast_model, optimizer, x_cen, device, max_nodes=8192):
    encoder_model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        batch_x_cen = x_cen[batch.n_id.cpu()].to(device)
        optimizer.zero_grad()
        z1, z2 = encoder_model(batch.x, batch_x_cen, batch.edge_index, batch.edge_attr)
        h1, h2 = [encoder_model.project(z) for z in [z1, z2]]
        # Subsample for contrastive loss
        if h1.size(0) > max_nodes:
            idx = torch.randperm(h1.size(0))[:max_nodes]
            h1, h2 = h1[idx], h2[idx]
        loss = contrast_model(h1, h2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
```
Memory: O(max_nodes^2) = 8192^2 * 4 = 256 MB. Very manageable.

**Semantic impact**: Subsampling negatives is a well-established technique (MoCo, SimCLR). With 8K nodes per batch, you have ~8K negatives which is sufficient.

### Solution C: MVGRL -- Gradient Checkpointing + Mixed Precision

MVGRL's loss is already O(N), not O(N^2). The OOM is from 4 full-graph GNN passes.

**Option C1: Gradient checkpointing**
```python
from torch.utils.checkpoint import checkpoint

def forward(self, x_agg, x_cen, edge_index, edge_weight=None):
    aug1, aug2 = self.augmentor
    x1, edge_index1, edge_weight1 = aug1(x_agg, edge_index, edge_weight)
    x2, edge_index2, edge_weight2 = aug2(x_cen, edge_index, edge_weight)

    z1 = checkpoint(self.encoder1, x1, edge_index1, edge_weight1, use_reentrant=False)
    z2 = checkpoint(self.encoder2, x2, edge_index2, edge_weight2, use_reentrant=False)
    g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
    g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
    z1n = checkpoint(self.encoder1, *self.corruption(x1, edge_index1, edge_weight1), use_reentrant=False)
    z2n = checkpoint(self.encoder2, *self.corruption(x2, edge_index2, edge_weight2), use_reentrant=False)
    return z1, z2, g1, g2, z1n, z2n
```
This trades compute for memory: activations are recomputed during backward instead of stored.

**Option C2: Mixed precision (AMP)**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

def train(encoder_model, contrast_model, data, x_cen, optimizer, scaler):
    encoder_model.train()
    optimizer.zero_grad()
    with autocast():
        z1, z2, g1, g2, z1n, z2n = encoder_model(data.x, x_cen, data.edge_index)
        loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss.item()
```
Halves memory for activations (float16 instead of float32). Combined with checkpointing, reduces from ~20 GB to ~5-7 GB.

**Option C3: NeighborLoader mini-batching**

More invasive but most effective. Requires restructuring the G2L contrast:
- Compute graph summary `g` from full graph (once, detached)
- Mini-batch the node-level embeddings
- Loss is already 1 x N per batch, so no N^2 issue

---

## Recommendation Priority

| Model | Best Solution | Memory Reduction | Code Complexity | Semantic Change |
|-------|--------------|-----------------|-----------------|-----------------|
| **GBT** | NeighborLoader (A) | Full-batch -> mini-batch | Low (copy from GRACE pattern) | Minimal |
| **GRACE** | Subsample negatives (B3) | N^2 -> max_nodes^2 | Low (5 lines) | Minor (well-established) |
| **MVGRL** | Checkpointing + AMP (C1+C2) | ~60-70% reduction | Medium (10 lines) | None |

### Alternative: Reduce `hidden_dim`

For all models, reducing `hidden_dim` from 256 to 64 or 128 cuts activation memory by 2-4x. This is the simplest change but affects model capacity. Worth trying as a quick test.

### Alternative: `num_neighbors` tuning

For GRACE, instead of `[10, 10]`, use `[5, 3]` or `[3, 3]` to limit batch expansion. This is the simplest fix but reduces the receptive field.
