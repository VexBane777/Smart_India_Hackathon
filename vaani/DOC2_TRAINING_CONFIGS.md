<!--
VAANI documentation suite v1.0 — generated 2026-09-03
Document: Doc 2 — Training & Inference Configuration Files · Owner: Model Lead + Systems Lead · Status: Build-ready
This file is GENERATED. Edit generate_vaani_docs.py and rerun instead.
-->
# Doc 2 — Training & Inference Configuration Files

**Owner:** Model Lead + Systems Lead · **Status:** Build-ready · **Suite:** v1.0 · **Generated:** 2026-09-03
**Depends on:** Doc 1 manifests must exist before training

---

## 2.1 Config system
Plain YAML + a single `train.py --config <file>`. No framework magic: every config is
diffable; every run logs its `config_hash`.

## 2.2 The files
```
vaani/
├── configs/  base.yaml  cnn_week1.yaml  aasist.yaml  teacher_ssl.yaml
│             teacher_ssl_lora.yaml  student_distill.yaml
│             calibrate.yaml  eval_protocols.yaml  engine.yaml
├── vaani/    models/{cnn.py, ssl_head.py, distill.py, registry.py}
│             train.py  calibrate.py  evaluate.py  export_onnx.py  bench_latency.py
├── scripts/  run_resume.sh
└── models.meta.json
```

`configs/base.yaml` — shared defaults:
```yaml
seed: 42                       # finals run seeds [42, 7, 2026] -> mean +/- std
audio:  { sr: 16000, win_s: 2.0, hop_s: 0.5 }
features: { n_mels: 64, n_fft: 512, hop: 160, fmin: 20, fmax: 8000 }
data:
  manifest: data/manifests/train.parquet
  val:       data/manifests/val.parquet
  group_split_col: source_clip        # leakage-safe (Doc 1 §1.6)
  on_the_fly_aug: [time_shift, gain_jitter, freq_mask, time_mask]
ckpt:
  dir: runs/${name}
  hub_repo: vaani/models/${name}      # auto-push to HuggingFace Hub
  every_steps: 500
  keep_last: 3
tracking: { mlflow_uri: "http://master:5000", experiment: vaani }
```

`configs/cnn_week1.yaml` — the fallback model, built Monday, never deleted:
```yaml
name: cnn_week1
model: { type: tinycnn, chs: [32, 64, 128, 256], dropout: 0.2 }
optim: { opt: adamw, lr: 3.0e-4, sched: cosine, epochs: 20, batch: 256, clip: 5.0 }
hardware: { gpu: any, vram_gb: 2 }
```

`configs/teacher_ssl.yaml` — the accuracy ceiling (5050, overnight):
```yaml
name: ssl_teacher
model: { type: ssl_head, base: facebook/wav2vec2-base, layers: 12 }
optim: { opt: adamw, lr: 1.0e-5, epochs: 10, batch: 8, accum: 2,
         bfloat16: true, grad_checkpointing: true }
freeze: [feature_extractor]
hardware: { gpu: rtx5050, vram_gb: 8, hours: overnight }
# >=300M variants (XLS-R-300M, WavLM-large): Kaggle T4 OR teacher_ssl_lora.yaml
```

`configs/student_distill.yaml` — the deployable model (4050 #2):
```yaml
name: student_v1
model: { type: tinycnn, chs: [64, 128, 256, 256] }
distill:
  teacher_logits_cache: runs/ssl_teacher/logits_cache.parquet  # pre-computed
  temperature: 2.0
  alpha: 0.5                 # loss = alpha*CE + (1-alpha)*T^2*KL
optim: { opt: adamw, lr: 3.0e-4, epochs: 30, batch: 256 }
hardware: { gpu: rtx4050_2, vram_gb: 4 }
```

`configs/eval_protocols.yaml` — held-out sets as manifest filters:
```yaml
protocols:
  in_domain:        { filter: "split == 'test'" }
  unseen_generator: { filter: "generator == 'rvc_team'" }
  unseen_codec:     { filter: "codec == 'amr_wb'" }
  unseen_noise:     { filter: "noise == 'freesound_horn'" }
  real_calls:       { filter: "origin == 'real_call'" }
  external:         { dataset: in_the_wild, adapter: itw_loader.py }
report: [ eer, fpr_at_tpr90, auc, call_level_fpr, call_level_tpr ]
```

`configs/engine.yaml` — the streaming side (what the demo runs):
```yaml
vad:    { model: silero_v5, threshold: 0.5 }
window_s: 2.0
hop_s: 0.5
ema_alpha: 0.6
alert:  { consecutive_windows: 2, score_threshold: null }  # null -> from
         # calibrate.yaml; threshold comes from the FPR budget, never guessed
cascade: { uncertain_lo: 0.35, uncertain_hi: 0.70, teacher_budget_ms: 400 }
```

`models.meta.json` — the model-swap abstraction, one line to flip:
```json
{ "streaming_default": "student_v1",
  "teacher": "ssl_teacher",
  "fallback_chain": ["student_v1", "cnn_week1", "hf_baseline"] }
```

## 2.3 Key code skeletons

```python
class TinyCNN(nn.Module):
    def __init__(self, n_mels=64, chs=(32,64,128,256), p_drop=0.2):
        super().__init__()
        layers, cin = [], 1
        for ch in chs:
            layers += [nn.Conv2d(cin, ch, 3, padding=1), nn.BatchNorm2d(ch),
                       nn.GELU(), nn.MaxPool2d(2)]
            cin = ch
        self.enc = nn.Sequential(*layers)          # 64x200 -> 4x12
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(p_drop), nn.Linear(chs[-1], 2))
    def forward(self, mel):                        # (B, 1, 64, T)
        return self.head(self.enc(mel))
```

```python
class SSLHead(nn.Module):
    """wav2vec2 (a speech-pretrained model) + small classifier. Learns
    per-layer weights over the 12 hidden layers, pools over time."""
    def __init__(self, base="facebook/wav2vec2-base"):
        self.ssl = Wav2Vec2Model.from_pretrained(base)
        for p in self.ssl.feature_extractor.parameters():
            p.requires_grad = False
        self.w = nn.Parameter(torch.zeros(12))
        self.clf = nn.Sequential(nn.Linear(768,256), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(256,2))
    def forward(self, wav):                        # (B, T) at 16 kHz
        hs = self.ssl(wav, output_hidden_states=True).hidden_states
        mix = sum(wi*hi for wi,hi in zip(torch.softmax(self.w,0), hs))
        return self.clf(mix.mean(1))
```

```python
def distill_loss(s_logits, t_logits, y, T=2.0, alpha=0.5):
    """Student learns from labels AND the teacher's soft probabilities —
    soft targets carry 'how fake-ish' a clip is, richer than hard labels."""
    ce = F.cross_entropy(s_logits, y)
    kl = F.kl_div(F.log_softmax(s_logits/T, -1),
                  F.softmax(t_logits/T, -1), reduction="batchmean") * T * T
    return alpha * ce + (1 - alpha) * kl
```

```python
def eer_and_fpr_at_tpr(y, s, tpr_target=0.90):
    """EER = where false-accepts equal false-rejects (lower better).
    FPR@TPR90 = how many innocent calls get flagged when we catch 90% of
    fakes — THE number the threshold is chosen from."""
    fpr, tpr, thr = roc_curve(y, s)                # y: 1 = fake
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    j = np.searchsorted(tpr, tpr_target)
    return fpr[i], fpr[j], thr[j]
```

Also report **call-level metrics**: concatenate each clip's windows → run EMA +
2-consecutive-window state machine → flag/no-flag per call. Window-level EER flatters;
call-level is what the product does.

Calibration: fit a single temperature T on val logits (minimize NLL) → 10-bin
reliability diagram + ECE → `runs/<name>/calibration/`. Enables threshold-from-FPR-budget.

Export + bench: `torch.onnx.export` with dynamic time axis →
`quantize_dynamic(..., weight_type=QInt8)`. **Parity gate: max score change on 100 held
clips < 0.02, else don't ship INT8.** Latency bench: 500 windows, 50 warmup,
`torch.set_num_threads(4)`, CPU forced, `perf_counter` → p50/p95/p99 + cold-start →
`runs/<name>/latency.json`.

Resume wrapper (overnight insurance, run inside tmux):
```bash
while ! python -m vaani.train --config "$1" --resume auto \
        --ckpt-hub "vaani/models/$(yq '.name' "$1")"; do
    echo "crashed $(date), retrying"; sleep 30
done
```
Checkpoint payload: `{model, optimizer, step, epoch, config_hash, RNG states}` —
reproducible resume on any GPU or Kaggle.

## 2.4 Job routing
| Config | Machine | VRAM |
|---|---|---|
| cnn_week1 | any 4050 | ~2 GB |
| aasist / rawnet2 | 4050 #1 | ~4 GB |
| ssl_teacher | **5050** | ~7–8 GB (bf16 + grad ckpt, overnight) |
| ssl_teacher ≥300M | **Kaggle T4** or LoRA on 5050 | 16 GB / LoRA |
| student_v1 | 4050 #2 | ~4 GB (teacher-logit cache) |
| finals (wks 9–10) | all 3 + one Kaggle verification run | — |

**Compute-budget rule:** CNN/student see all ~1M windows per epoch; the teacher trains
on a fixed 50k sampled windows per epoch × 20 epochs ≈ one full-corpus exposure — an
honest hackathon-scale choice, documented in the run notes.

---

*Part of the VAANI documentation suite — regenerate with `python generate_vaani_docs.py`. Placeholders marked [M] must be replaced by measured values before use in the pitch.*
