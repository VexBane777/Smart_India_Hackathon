# VoiceGuard evaluation report: split `test`

Generated 2026-09-12 05:04 by `evaluate.py`. Protocol: `voice_guard/docs/EVAL-PROTOCOL.md`. Channels: none, whatsapp, volte, cellular_3g, gsm_2g, pstn, tandem_xnet. Headline pooled over: whatsapp, volte, cellular_3g, gsm_2g, pstn, tandem_xnet.
Files: 5609 (63196 windows across channels). Operating threshold per model = phone-pooled EER threshold on the `select` split.

## Headline (core sets, phone-pooled)

| model | EER | 95% CI (file bootstrap) | threshold | FPR @thr | FNR @thr | FPR @0.60 | FNR @0.60 |
|---|---|---|---|---|---|---|---|
| v9 | 0.5087 | [0.4993, 0.5185] | 0.446 | 50.1% | 51.8% | 43.0% | 58.5% |
| v11 | 0.4853 | [0.4766, 0.4939] | 0.663 | 47.9% | 49.1% | 53.5% | 43.6% |
| v12 | 0.3223 | [0.3138, 0.3308] | 0.247 | 32.4% | 32.1% | 9.8% | 51.7% |

## EER per channel (core sets; `none` is a reference row, never a result on its own)

| channel | v9 | v11 | v12 |
|---|---|---|---|
| none | 0.2163 | 0.0788 | 0.0624 |
| whatsapp | 0.4768 | 0.4807 | 0.2036 |
| volte | 0.4657 | 0.4720 | 0.1586 |
| cellular_3g | 0.5326 | 0.4775 | 0.2770 |
| gsm_2g | 0.5398 | 0.5057 | 0.4469 |
| pstn | 0.4735 | 0.4712 | 0.2517 |
| tandem_xnet | 0.5432 | 0.4856 | 0.3946 |
| **seen_phone** | 0.4930 | 0.4865 | 0.2134 |
| **unseen_phone** | 0.5167 | 0.4899 | 0.4015 |
| **none_reference** | 0.2163 | 0.0788 | 0.0624 |
| **padded windows only** | 0.4925 | 0.4665 | 0.3477 |
| **unpadded windows only** | 0.5189 | 0.4850 | 0.3187 |
| **core + MLAAD (unseen TTS)** | 0.5023 | 0.4975 | 0.3477 |
| **accent en_native** | 0.4652 | 0.4892 | 0.4695 |
| **accent hi_native** | 0.5072 | 0.5377 | 0.5142 |

## Per eval set, phone-pooled (reals: FPR, fakes: FNR, at each model's select threshold)

| set | windows | v9 | v11 | v12 |
|---|---|---|---|---|
| accent_fake_en_native (FNR) | 2328 | 45.4% | 51.6% | 46.0% |
| accent_fake_hi_native (FNR) | 984 | 43.6% | 52.7% | 43.0% |
| accent_real_en_foreign (FPR) | 3192 | 53.1% | 50.8% | 48.0% |
| accent_real_en_native (FPR) | 2298 | 47.5% | 46.1% | 48.4% |
| accent_real_hi_native (FPR) | 2466 | 58.6% | 55.8% | 59.2% |
| itw_fake (FNR) | 7602 | 51.8% | 49.1% | 32.1% |
| itw_real (FPR) | 18546 | 50.3% | 50.9% | 31.7% |
| mlaad_fake (FNR) | 8334 | 49.2% | 53.7% | 40.4% |
| noiseaug_real_en (FPR) | 5820 | 47.3% | 40.9% | 34.1% |
| noiseaug_real_hi (FPR) | 2598 | 54.8% | 42.1% | 33.7% |

## Confound v2 (phone-pooled, core sets)

Gates: |rho| <= 0.1; a rate row fails only if worse/better half rate ratio > 1.25 AND |diff| > 1% AND the file-level bootstrap CI of the difference excludes 0 (Bonferroni alpha 0.05 over 14 rows). Rate = FPR for reals, FNR for fakes, at the select threshold.

### v9: FAIL real:pauseRatio, real:energyVariance, real:zcrVariance, real:jitter, real:shimmer, real:hnr_db, fake:energyVariance, fake:zcrVariance, fake:jitter, fake:hnr_db

| class | feature | mean p low/high | gap | ratio | rho | rate low/high | rate ratio | pass |
|---|---|---|---|---|---|---|---|---|
| real | pauseRatio | 0.522 / 0.437 | 0.085 | 1.20 | -0.109 | 54.4% / 45.5% | 1.20 | **NO** |
| real | energyVariance | 0.340 / 0.620 | 0.280 | 1.82 | 0.437 | 32.7% / 67.5% | 2.07 | **NO** |
| real | zcrVariance | 0.341 / 0.620 | 0.279 | 1.82 | 0.432 | 32.6% / 67.6% | 2.07 | **NO** |
| real | jitter | 0.604 / 0.357 | 0.247 | 1.69 | -0.392 | 65.2% / 34.9% | 1.87 | **NO** |
| real | shimmer | 0.520 / 0.441 | 0.080 | 1.18 | -0.128 | 54.8% / 45.4% | 1.21 | **NO** |
| real | hnr_db | 0.389 / 0.572 | 0.183 | 1.47 | 0.289 | 38.6% / 61.6% | 1.59 | **NO** |
| real | pad_fraction | 0.498 / 0.452 | 0.046 | 1.10 | -0.066 | 51.9% / 47.2% | 1.10 | yes |
| fake | pauseRatio | 0.477 / 0.452 | 0.024 | 1.05 | 0.004 | 50.8% / 53.0% | 1.04 | yes |
| fake | energyVariance | 0.385 / 0.545 | 0.160 | 1.41 | 0.273 | 61.6% / 42.0% | 1.47 | **NO** |
| fake | zcrVariance | 0.283 / 0.648 | 0.365 | 2.29 | 0.599 | 74.1% / 29.5% | 2.51 | **NO** |
| fake | jitter | 0.608 / 0.322 | 0.286 | 1.89 | -0.463 | 34.4% / 69.2% | 2.01 | **NO** |
| fake | shimmer | 0.481 / 0.449 | 0.032 | 1.07 | -0.053 | 49.9% / 53.7% | 1.08 | yes |
| fake | hnr_db | 0.374 / 0.556 | 0.182 | 1.49 | 0.315 | 63.0% / 40.6% | 1.55 | **NO** |
| fake | pad_fraction | 0.467 / 0.458 | 0.009 | 1.02 | -0.003 | 51.9% / 51.6% | 1.00 | yes |

### v11: FAIL real:pauseRatio, real:energyVariance, real:zcrVariance, real:jitter, real:shimmer, fake:pauseRatio, fake:energyVariance, fake:zcrVariance, fake:jitter, fake:shimmer

| class | feature | mean p low/high | gap | ratio | rho | rate low/high | rate ratio | pass |
|---|---|---|---|---|---|---|---|---|
| real | pauseRatio | 0.533 / 0.594 | 0.061 | 1.11 | 0.117 | 40.9% / 55.3% | 1.35 | **NO** |
| real | energyVariance | 0.446 / 0.679 | 0.233 | 1.52 | 0.514 | 34.8% / 61.0% | 1.75 | **NO** |
| real | zcrVariance | 0.669 / 0.456 | 0.212 | 1.47 | -0.342 | 57.5% / 38.3% | 1.50 | **NO** |
| real | jitter | 0.473 / 0.652 | 0.179 | 1.38 | 0.278 | 40.4% / 55.4% | 1.37 | **NO** |
| real | shimmer | 0.542 / 0.583 | 0.041 | 1.08 | 0.054 | 40.7% / 55.1% | 1.35 | **NO** |
| real | hnr_db | 0.583 / 0.542 | 0.042 | 1.08 | -0.025 | 46.3% / 49.5% | 1.07 | yes |
| real | pad_fraction | 0.551 / 0.581 | 0.031 | 1.06 | 0.041 | 46.0% / 50.9% | 1.11 | yes |
| fake | pauseRatio | 0.535 / 0.630 | 0.095 | 1.18 | 0.181 | 56.8% / 40.6% | 1.40 | **NO** |
| fake | energyVariance | 0.488 / 0.672 | 0.185 | 1.38 | 0.421 | 57.2% / 40.9% | 1.40 | **NO** |
| fake | zcrVariance | 0.695 / 0.465 | 0.230 | 1.50 | -0.358 | 37.9% / 60.3% | 1.59 | **NO** |
| fake | jitter | 0.494 / 0.666 | 0.173 | 1.35 | 0.252 | 55.9% / 42.3% | 1.32 | **NO** |
| fake | shimmer | 0.521 / 0.639 | 0.118 | 1.23 | 0.219 | 60.9% / 37.3% | 1.63 | **NO** |
| fake | hnr_db | 0.607 / 0.554 | 0.053 | 1.10 | -0.056 | 51.4% / 46.8% | 1.10 | yes |
| fake | pad_fraction | 0.568 / 0.629 | 0.060 | 1.11 | 0.082 | 50.9% / 41.3% | 1.23 | yes |

### v12: FAIL real:zcrVariance, real:jitter, real:hnr_db, fake:energyVariance, fake:zcrVariance, fake:jitter, fake:shimmer, fake:hnr_db, fake:pad_fraction

| class | feature | mean p low/high | gap | ratio | rho | rate low/high | rate ratio | pass |
|---|---|---|---|---|---|---|---|---|
| real | pauseRatio | 0.227 / 0.212 | 0.015 | 1.07 | -0.037 | 33.3% / 31.5% | 1.06 | yes |
| real | energyVariance | 0.208 / 0.232 | 0.024 | 1.12 | 0.055 | 29.6% / 35.2% | 1.19 | yes |
| real | zcrVariance | 0.181 / 0.259 | 0.078 | 1.43 | 0.204 | 25.3% / 39.5% | 1.56 | **NO** |
| real | jitter | 0.245 / 0.195 | 0.050 | 1.26 | -0.116 | 36.0% / 28.8% | 1.25 | **NO** |
| real | shimmer | 0.207 / 0.232 | 0.025 | 1.12 | 0.065 | 30.3% / 34.5% | 1.14 | yes |
| real | hnr_db | 0.190 / 0.249 | 0.059 | 1.31 | 0.131 | 27.9% / 36.9% | 1.33 | **NO** |
| real | pad_fraction | 0.230 / 0.204 | 0.026 | 1.13 | -0.009 | 33.7% / 30.3% | 1.11 | yes |
| fake | pauseRatio | 0.511 / 0.529 | 0.019 | 1.04 | 0.047 | 34.4% / 29.5% | 1.17 | yes |
| fake | energyVariance | 0.574 / 0.465 | 0.109 | 1.23 | -0.196 | 26.3% / 37.9% | 1.44 | **NO** |
| fake | zcrVariance | 0.349 / 0.690 | 0.341 | 1.98 | 0.544 | 53.7% / 10.5% | 5.13 | **NO** |
| fake | jitter | 0.662 / 0.377 | 0.285 | 1.76 | -0.419 | 15.2% / 48.9% | 3.21 | **NO** |
| fake | shimmer | 0.456 / 0.583 | 0.127 | 1.28 | 0.211 | 40.5% / 23.7% | 1.71 | **NO** |
| fake | hnr_db | 0.409 / 0.630 | 0.221 | 1.54 | 0.383 | 46.3% / 17.9% | 2.59 | **NO** |
| fake | pad_fraction | 0.539 / 0.438 | 0.101 | 1.23 | -0.124 | 30.6% / 38.5% | 1.26 | **NO** |

Pad diagnostics (eval data, same for every model): padded share real 38.7%, fake 19.3%, pad_fraction label AUC 0.403.

## Attack-type head

- **v11** MLAAD (all TTS, unseen systems): predicted tts 24.2%; UI sub-label shown on 42.5% of windows, tts when shown 28.4%.
- **v12** MLAAD (all TTS, unseen systems): predicted tts 74.3%; UI sub-label shown on 31.2% of windows, tts when shown 81.2%.

Training-run val fakes (`runs\voice_guard_v12`, leave-out ['A11', 'A18']). Only models trained with this run's val split held out are out-of-sample here; v9/v11 trained on (most of) these files, and v11's attack head saw A11/A18 labels.

| model | group | n | balanced acc | acc | UI coverage | UI acc when shown |
|---|---|---|---|---|---|---|
| v11 | in_distribution | 10420 | 0.712 | 0.668 | 72.4% | 72.1% |
| v11 | leave_attack_out | 858 | 0.683 | 0.676 | 69.0% | 73.6% |
| v11 | hybrids_A13_A15 | 1540 | 0.937 | 0.937 | 73.1% | 94.8% |
| v12 | in_distribution | 10420 | 0.921 | 0.927 | 84.9% | 95.9% |
| v12 | leave_attack_out | 858 | 0.763 | 0.767 | 71.3% | 81.7% |
| v12 | hybrids_A13_A15 | 1540 | 0.809 | 0.809 | 80.8% | 87.2% |

## fp16 feature-cache validation

```
{
 "n_windows": 1569,
 "channels": [
  "none",
  "whatsapp",
  "gsm_2g"
 ],
 "max_abs_feature_delta": 0.125,
 "models": {
  "runs\\voice_guard_v11_seqcnn_selected\\model.pt": {
   "arch": "seqcnn_v1",
   "max_abs_delta_p": 0.0008697509765625,
   "mean_abs_delta_p": 5.913513450650498e-05,
   "flips_at_0.5": 0,
   "flips_at_0.6": 0,
   "pass": true
  },
  "runs\\voice_guard_v9_noisefix_final\\model.pt": {
   "arch": "mlp_63d",
   "max_abs_delta_p": 0.0002142190933227539,
   "mean_abs_delta_p": 1.7240819943253882e-05,
   "flips_at_0.5": 0,
   "flips_at_0.6": 0,
   "pass": true
  }
 }
}
```

## Gates and deploy decision

- **confound gates (candidate)**: FAIL: real:zcrVariance, real:jitter, real:hnr_db, fake:energyVariance, fake:zcrVariance, fake:jitter, fake:shimmer, fake:hnr_db, fake:pad_fraction
- **beats reference on test headline EER**: True (0.3223 vs 0.4853; CIs do not overlap)
- **attack-type head gates**: PASS (leave-out balanced acc 0.763 vs >= 0.7; MLAAD tts share 74.3% vs >= 70%)
- **attack_head_ok**: True
- **deploy**: False
