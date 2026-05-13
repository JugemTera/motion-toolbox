# MoMask Evaluation Metrics: Detailed Summary

This document summarizes how the **MoMask** repository (https://github.com/EricGuo5513/momask-codes) implements semantic evaluation metrics for text-to-motion generation. The implementation follows the **HumanML3D evaluation framework** established by Guo et al. (CVPR 2022) in "Generating Diverse and Natural 3D Human Motions From Text."

---

## 1. Repository URLs

| Resource | URL |
|----------|-----|
| Project page | https://ericguo5513.github.io/momask/ |
| GitHub repo | https://github.com/EricGuo5513/momask-codes |

---

## 2. Evaluation Metrics Implemented

MoMask evaluates text-to-motion generation with the following metrics:

### 2.1 **FID (Fréchet Inception Distance)**

**Location:** `utils/metrics.py` → `calculate_frechet_distance()`

**Formula/Logic:**
```
d² = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·sqrt(Σ₁·Σ₂))
```

- **Input:** Feature distributions from the evaluator's motion encoder
- **Process:** 
  - Compute mean (`μ`) and covariance (`Σ`) of motion embeddings for **ground truth** and **generated** motions
  - Use the standard Fréchet distance between two multivariate Gaussians
- **Lower is better** (measures distribution similarity; 0.045 reported for HumanML3D in MoMask paper)

### 2.2 **R-Precision (TOP-1, TOP-2, TOP-3)**

**Location:** `utils/metrics.py` → `calculate_R_precision()`, `euclidean_distance_matrix()`

**Logic:**
- For each text embedding, find its **K nearest motion embeddings** (K=1,2,3) in the test set
- **R@K** = fraction of cases where the **ground-truth paired motion** is among the top-K retrieved motions
- Uses Euclidean distance in the 512-dim co-embedding space
- **Higher is better** (measures text-motion alignment)

### 2.3 **Matching Score**

**Location:** `utils/eval_t2m.py` (via `euclidean_distance_matrix().trace()`)

**Logic:**
```
matching_score = (1/N) · Σᵢ dist(text_emb[i], motion_emb[i])
```
- For each text-motion pair `(i,i)`, compute Euclidean distance in the co-embedding space
- Average over all samples
- **Lower is better** (measures how well generated motions match their conditioning text)

### 2.4 **Diversity**

**Location:** `utils/metrics.py` → `calculate_diversity()`

**Logic:**
```
diversity = mean(‖activation[i] - activation[j]‖)  for random pairs (i,j)
```
- Sample `diversity_times` pairs of motion embeddings (300 if n>300, else 100)
- Compute mean L2 distance between randomly sampled pairs
- **Higher is better** (measures variety of generated motions; should be close to ground-truth diversity)

### 2.5 **MultiModality (MM)**

**Location:** `utils/metrics.py` → `calculate_multimodality()`

**Logic:**
```
mm = mean(‖activation[i,j] - activation[i,k]‖)  for same text i, different samples j,k
```
- For each text prompt `i`, generate **multiple motions** (e.g., 10 samples)
- Shape: `(num_sentences, num_samples_per_sent, embed_dim)`
- Sample pairs of different generations for the same sentence
- **Higher is better** (measures that the model can produce diverse motions for the same text)

### 2.6 **Additional Metrics (RVQ / Reconstruction)**

- **MAE / L1:** Mean absolute error between decoded and ground-truth joint positions (used in VQ evaluation)
- **MPJPE:** Mean per-joint position error (in `utils/metrics.py`); joints aligned by pelvis before computing

---

## 3. Feature Extraction Architecture

### 3.1 Pre-trained Evaluator Models

MoMask uses a **frozen text-motion co-embedding model** from the HumanML3D/T2M framework:

| Component | Module | Architecture | Output |
|-----------|--------|---------------|--------|
| **Movement Encoder** | `MovementConvEncoder` | Conv1d layers (4,2,1) × 2 → Linear | 512-dim movement features |
| **Motion Encoder** | `MotionEncoderBiGRUCo` | BiGRU over movement features | 512-dim motion embedding |
| **Text Encoder** | `TextEncoderBiGRUCo` | Word+POS emb → BiGRU | 512-dim text embedding |

**Checkpoint path:** `checkpoints/{t2m|kit}/text_mot_match/model/finest.tar`

**Input requirements:**
- **Motion:** HumanML3D format — `(batch, seq, 263)` for HumanML3D or `(batch, seq, 251)` for KIT-ML
- **Text:** GloVe word embeddings (300-dim) + POS one-hot; max 20 words

### 3.2 Motion Feature Pipeline

```
Raw motion (263/251-dim) 
  → Drop last 4 dims (motion[..., :-4]) 
  → MovementConvEncoder (conv + linear)
  → Downsample by unit_length=4 (m_lens // 4)
  → MotionEncoderBiGRUCo (BiGRU)
  → 512-dim motion embedding
```

### 3.3 Text Feature Pipeline

```
Caption → Word embeddings (GloVe 300d) + POS one-hot
  → TextEncoderBiGRUCo (BiGRU)
  → 512-dim text embedding
```

### 3.4 Co-embedding API

```python
text_emb, motion_emb = eval_wrapper.get_co_embeddings(
    word_embs, pos_ohot, cap_lens, motions, m_lens
)
```

**Note:** Outputs are **reordered** by descending motion length (for packed sequences). Order does not match input order.

---

## 4. Evaluation Pipeline Flow

### 4.1 Text-to-Motion Generation Evaluation (`eval_t2m_trans_res.py`)

```
1. Load models:
   - RVQ-VAE (for tokenization/decoding)
   - Masked Transformer (text → base tokens)
   - Residual Transformer (base → full residual tokens)

2. Load EvaluatorModelWrapper from checkpoints/.../text_mot_match/

3. For each batch from test split:
   a. Get (word_embs, pos_one_hots, caption, sent_len, motion_gt, m_length, token)
   b. Generate: trans.generate(text) → vq.decode() → pred_motions
   c. Extract co-embeddings:
      - (et, em) = get_co_embeddings(..., motion_gt, ...)   # ground truth
      - (et_pred, em_pred) = get_co_embeddings(..., pred_motions, ...)  # generated
   d. Accumulate:
      - R_precision via calculate_R_precision(et_pred, em_pred, top_k=3)
      - matching_score via euclidean_distance_matrix(et_pred, em_pred).trace()
      - motion_pred_list.append(em_pred), motion_annotation_list.append(em)

4. After all batches:
   - gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
   - mu, cov = calculate_activation_statistics(motion_pred_np)
   - fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
   - diversity = calculate_diversity(motion_pred_np, 300 or 100)
   - [If cal_mm=True] mm = calculate_multimodality(multimodal_em_pred)

5. Repeat 20 times (repeat_time=20) with different seeds; report mean ± 95% CI
```

### 4.2 RVQ Reconstruction Evaluation (`eval_t2m_vq.py`)

- Same co-embedding extraction
- Predicted motion = VQ-VAE reconstruction of input motion (no text→motion generation)
- Same metrics: FID, Diversity, R-Precision, Matching, plus MAE

---

## 5. Key File Paths in MoMask Repo

| Purpose | Path |
|---------|------|
| Main T2M evaluation script | `eval_t2m_trans_res.py` |
| RVQ evaluation script | `eval_t2m_vq.py` |
| Core evaluation logic | `utils/eval_t2m.py` |
| Metric implementations | `utils/metrics.py` |
| Evaluator wrapper | `models/t2m_eval_wrapper.py` |
| Encoder modules | `models/t2m_eval_modules.py` |
| Motion loader | `motion_loaders/dataset_motion_loader.py` |
| Download evaluator | `prepare/download_evaluator.sh` |
| Download GloVe | `prepare/download_glove.sh` |
| Dataset options (HumanML3D) | `checkpoints/t2m/Comp_v6_KLD005/opt.txt` |
| Dataset options (KIT-ML) | `checkpoints/kit/Comp_v6_KLD005/opt.txt` |

---

## 6. Pre-trained Checkpoints Required

| Checkpoint | Path | Purpose |
|------------|------|---------|
| HumanML3D evaluator | `checkpoints/t2m/humanml3d_evaluator/` (from zip) | Text/motion encoders (`text_mot_match/model/finest.tar`) |
| KIT-ML evaluator | `checkpoints/kit/kit_evaluator/` | Same, for KIT dataset |
| GloVe vectors | `./glove/` (from `download_glove.sh`) | Word embeddings for text encoder |
| RVQ model | `checkpoints/{t2m|kit}/rvq_*/model/net_best_fid.tar` | VQ tokenization |
| Masked Transformer | `checkpoints/{t2m|kit}/t2m_*/model/*.tar` | Text→motion generation |
| Residual Transformer | `checkpoints/{t2m|kit}/tres_*/model/net_best_fid.tar` | Residual refinement |

**Download commands:**
```bash
bash prepare/download_evaluator.sh   # HumanML3D + KIT evaluators
bash prepare/download_glove.sh        # GloVe word vectors
bash prepare/download_models.sh      # MoMask generation models
```

---

## 7. Data Format Requirements

### Motion Representation
- **HumanML3D:** 263-dim vector per frame (rotation-invariant + rotation features; SMPL 22 joints)
- **KIT-ML:** 251-dim vector per frame (21 joints)
- Motions padded/truncated to `max_motion_length=196` frames
- Normalized with dataset `Mean.npy` and `Std.npy`

### Text Input
- GloVe 300-dim embeddings; POS tags from `utils.word_vectorizer.POS_enumerator`
- Provided by `Text2MotionDatasetEval` via `collate_fn`

---

## 8. Dependencies

- **PyTorch**
- **CLIP** (for generation; evaluator does not use CLIP)
- **scipy** (for `linalg.sqrtm` in FID)
- **numpy**
- **GloVe** word vectors (via `download_glove.sh`)

---

## 9. References

- Guo et al., "Generating Diverse and Natural 3D Human Motions From Text," CVPR 2022 (HumanML3D evaluation framework)
- Guo et al., "MoMask: Generative Masked Modeling of 3D Human Motions," CVPR 2024
- HumanML3D dataset: https://github.com/EricGuo5513/HumanML3D
