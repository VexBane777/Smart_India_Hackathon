# VoiceGuard evaluation report: split `test`

Generated 2026-09-14 17:15 by `evaluate.py`. Protocol: `voice_guard/docs/EVAL-PROTOCOL.md`. Channels: none, whatsapp, volte, cellular_3g, gsm_2g, pstn, tandem_xnet, playback. Headline pooled over: whatsapp, volte, cellular_3g, gsm_2g, pstn, tandem_xnet.
Files: 5609 (72224 windows across channels). Operating threshold per model = phone-pooled EER threshold on the `select` split.

## Headline (core sets, phone-pooled)

| model | EER | 95% CI (file bootstrap) | threshold | FPR @thr | FNR @thr | FPR @0.60 | FNR @0.60 |
|---|---|---|---|---|---|---|---|
| v11 | 0.4853 | [0.4766, 0.4939] | 0.663 | 47.9% | 49.1% | 53.5% | 43.6% |
| v13 | 0.3149 | [0.3050, 0.3250] | 0.342 | 32.3% | 31.0% | 9.4% | 54.3% |

## EER per channel (core sets; `none` is a reference row, never a result on its own)

| channel | v11 | v13 |
|---|---|---|
| none | 0.0788 | 0.0694 |
| whatsapp | 0.4807 | 0.2218 |
| volte | 0.4720 | 0.1744 |
| cellular_3g | 0.4775 | 0.2904 |
| gsm_2g | 0.5057 | 0.4364 |
| pstn | 0.4712 | 0.2470 |
| tandem_xnet | 0.4856 | 0.4064 |
| playback | 0.4925 | 0.2755 |

## Acoustic headline (playback loop, core sets)

| model | EER | 95% CI (file bootstrap) | FPR @thr | FNR @thr | FPR @0.60 | FNR @0.60 |
|---|---|---|---|---|---|---|
| v11 | 0.4925 | [0.4724, 0.5106] | 63.4% | 33.4% | 70.1% | 26.4% |
| v13 | 0.2755 | [0.2582, 0.2934] | 43.7% | 15.2% | 12.3% | 44.9% |
| **seen_phone** | 0.4865 | 0.2289 |
| **unseen_phone** | 0.4899 | 0.3899 |
| **acoustic** | 0.4925 | 0.2755 |
| **none_reference** | 0.0788 | 0.0694 |
| **padded windows only** | 0.4665 | 0.3381 |
| **unpadded windows only** | 0.4850 | 0.3139 |
| **core + MLAAD (unseen TTS)** | 0.4975 | 0.3393 |
| **accent en_native** | 0.4892 | 0.4879 |
| **accent hi_native** | 0.5377 | 0.5142 |

## Per eval set, phone-pooled (reals: FPR, fakes: FNR, at each model's select threshold)

| set | windows | v11 | v13 |
|---|---|---|---|
| accent_fake_en_native (FNR) | 2328 | 51.6% | 43.2% |
| accent_fake_hi_native (FNR) | 984 | 52.7% | 43.3% |
| accent_real_en_foreign (FPR) | 3192 | 50.8% | 48.2% |
| accent_real_en_native (FPR) | 2298 | 46.1% | 54.1% |
| accent_real_hi_native (FPR) | 2466 | 55.8% | 59.4% |
| itw_fake (FNR) | 7602 | 49.1% | 31.0% |
| itw_real (FPR) | 18546 | 50.9% | 31.9% |
| mlaad_fake (FNR) | 8334 | 53.7% | 38.2% |
| noiseaug_real_en (FPR) | 5820 | 40.9% | 33.1% |
| noiseaug_real_hi (FPR) | 2598 | 42.1% | 33.0% |

## Confound v2 (phone-pooled, core sets)

Gates: |rho| <= 0.1; a rate row fails only if worse/better half rate ratio > 1.25 AND |diff| > 1% AND the file-level bootstrap CI of the difference excludes 0 (Bonferroni alpha 0.05 over 14 rows). Rate = FPR for reals, FNR for fakes, at the select threshold.

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

### v13: FAIL real:zcrVariance, real:shimmer, real:hnr_db, fake:energyVariance, fake:zcrVariance, fake:jitter, fake:shimmer, fake:hnr_db, fake:pad_fraction

| class | feature | mean p low/high | gap | ratio | rho | rate low/high | rate ratio | pass |
|---|---|---|---|---|---|---|---|---|
| real | pauseRatio | 0.278 / 0.277 | 0.001 | 1.00 | 0.012 | 32.4% / 32.1% | 1.01 | yes |
| real | energyVariance | 0.269 / 0.285 | 0.016 | 1.06 | 0.045 | 30.7% / 33.8% | 1.10 | yes |
| real | zcrVariance | 0.245 / 0.309 | 0.064 | 1.26 | 0.167 | 25.4% / 39.1% | 1.54 | **NO** |
| real | jitter | 0.296 / 0.258 | 0.039 | 1.15 | -0.086 | 35.8% / 28.7% | 1.25 | yes |
| real | shimmer | 0.254 / 0.300 | 0.046 | 1.18 | 0.138 | 28.3% / 36.2% | 1.28 | **NO** |
| real | hnr_db | 0.252 / 0.302 | 0.050 | 1.20 | 0.125 | 27.5% / 37.1% | 1.35 | **NO** |
| real | pad_fraction | 0.288 / 0.259 | 0.029 | 1.11 | -0.017 | 34.3% / 29.0% | 1.19 | yes |
| fake | pauseRatio | 0.527 / 0.530 | 0.003 | 1.00 | 0.030 | 33.4% / 28.3% | 1.18 | yes |
| fake | energyVariance | 0.581 / 0.476 | 0.106 | 1.22 | -0.233 | 24.3% / 37.7% | 1.55 | **NO** |
| fake | zcrVariance | 0.406 / 0.651 | 0.245 | 1.60 | 0.485 | 49.4% / 12.6% | 3.91 | **NO** |
| fake | jitter | 0.631 / 0.426 | 0.205 | 1.48 | -0.358 | 16.7% / 45.3% | 2.72 | **NO** |
| fake | shimmer | 0.464 / 0.593 | 0.129 | 1.28 | 0.265 | 40.2% / 21.8% | 1.84 | **NO** |
| fake | hnr_db | 0.448 / 0.608 | 0.160 | 1.36 | 0.339 | 43.1% / 18.9% | 2.27 | **NO** |
| fake | pad_fraction | 0.545 / 0.457 | 0.088 | 1.19 | -0.127 | 29.6% / 37.0% | 1.25 | **NO** |

Pad diagnostics (eval data, same for every model): padded share real 38.7%, fake 19.3%, pad_fraction label AUC 0.403.

## Attack-type head

- **v11** MLAAD (all TTS, unseen systems): predicted tts 24.2%; UI sub-label shown on 42.5% of windows, tts when shown 28.4%.
- **v13** MLAAD (all TTS, unseen systems): predicted tts 76.6%; UI sub-label shown on 29.6% of windows, tts when shown 92.3%.

Training-run val fakes (`runs\voice_guard_v13`, leave-out ['A11', 'A18']). Only models trained with this run's val split held out are out-of-sample here; v9/v11 trained on (most of) these files, and v11's attack head saw A11/A18 labels.

| model | group | n | balanced acc | acc | UI coverage | UI acc when shown |
|---|---|---|---|---|---|---|
| v11 | in_distribution | 13055 | 0.679 | 0.630 | 72.2% | 66.6% |
| v11 | leave_attack_out | 1051 | 0.653 | 0.641 | 69.4% | 68.7% |
| v11 | hybrids_A13_A15 | 1936 | 0.938 | 0.938 | 71.6% | 95.5% |
| v13 | in_distribution | 13055 | 0.874 | 0.889 | 73.0% | 95.2% |
| v13 | leave_attack_out | 1051 | 0.750 | 0.755 | 52.3% | 84.4% |
| v13 | hybrids_A13_A15 | 1936 | 0.590 | 0.590 | 56.3% | 75.1% |

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
  "runs\\voice_guard_v13_selected\\model.pt": {
   "arch": "seqtcn_v2",
   "max_abs_delta_p": 0.0008080601692199707,
   "mean_abs_delta_p": 6.69752771500498e-05,
   "flips_at_0.5": 0,
   "flips_at_0.6": 0,
   "pass": true
  }
 }
}
```

## Gates and deploy decision

- **confound gates (candidate)**: FAIL: real:zcrVariance, real:shimmer, real:hnr_db, fake:energyVariance, fake:zcrVariance, fake:jitter, fake:shimmer, fake:hnr_db, fake:pad_fraction
- **beats reference on test headline EER**: True (0.3149 vs 0.4853; CIs do not overlap)
- **acoustic gate (playback EER beats reference)**: True (0.2755 vs 0.4925; CIs do not overlap)
- **attack-type head gates**: PASS (leave-out balanced acc 0.750 vs >= 0.7; MLAAD tts share 76.6% vs >= 70%)
- **attack_head_ok**: True
- **acoustic_gate_ok**: True
- **deploy**: False
