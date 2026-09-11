# VoiceGuard evaluation report: split `select`

Generated 2026-09-11 23:53 by `evaluate.py`. Protocol: `voice_guard/docs/EVAL-PROTOCOL.md`. Channels: none, whatsapp, volte, cellular_3g, gsm_2g, pstn, tandem_xnet. Headline pooled over: whatsapp, volte, cellular_3g, gsm_2g, pstn, tandem_xnet.
Files: 3734 (39830 windows across channels). Operating threshold per model = phone-pooled EER threshold on the `select` split.

## Headline (core sets, phone-pooled)

| model | EER | 95% CI (file bootstrap) | threshold | FPR @thr | FNR @thr | FPR @0.60 | FNR @0.60 |
|---|---|---|---|---|---|---|---|
| v9 | 0.5013 | [0.4926, 0.5115] | 0.446 | 50.1% | 50.1% | 43.0% | 57.1% |
| v11 | 0.4800 | [0.4714, 0.4883] | 0.663 | 48.0% | 48.0% | 53.4% | 42.3% |

## EER per channel (core sets; `none` is a reference row, never a result on its own)

| channel | v9 | v11 |
|---|---|---|
| none | 0.2089 | 0.0708 |
| whatsapp | 0.4754 | 0.4608 |
| volte | 0.4661 | 0.4476 |
| cellular_3g | 0.5191 | 0.4669 |
| gsm_2g | 0.5177 | 0.5008 |
| pstn | 0.4722 | 0.4838 |
| tandem_xnet | 0.5417 | 0.4937 |
| **seen_phone** | 0.4866 | 0.4728 |
| **unseen_phone** | 0.5085 | 0.4903 |
| **none_reference** | 0.2089 | 0.0708 |
| **padded windows only** | 0.4928 | 0.4674 |
| **unpadded windows only** | 0.5097 | 0.4757 |

## Per eval set, phone-pooled (reals: FPR, fakes: FNR, at each model's select threshold)

| set | windows | v9 | v11 |
|---|---|---|---|
| itw_fake (FNR) | 7788 | 50.1% | 48.0% |
| itw_real (FPR) | 18162 | 50.0% | 50.6% |
| noiseaug_real_en (FPR) | 5652 | 48.0% | 41.6% |
| noiseaug_real_hi (FPR) | 2538 | 55.5% | 43.4% |

## Confound v2 (phone-pooled, core sets)

Gates: |rho| <= 0.1; worse/better half rate ratio <= 1.25 (or |diff| <= 1%). Rate = FPR for reals, FNR for fakes, at the select threshold.

### v9: FAIL real:pauseRatio, real:energyVariance, real:zcrVariance, real:jitter, real:shimmer, real:hnr_db, fake:energyVariance, fake:zcrVariance, fake:jitter, fake:hnr_db

| class | feature | mean p low/high | gap | ratio | rho | rate low/high | rate ratio | pass |
|---|---|---|---|---|---|---|---|---|
| real | pauseRatio | 0.525 / 0.434 | 0.091 | 1.21 | -0.122 | 54.9% / 45.3% | 1.21 | **NO** |
| real | energyVariance | 0.345 / 0.615 | 0.270 | 1.78 | 0.431 | 33.5% / 66.8% | 1.99 | **NO** |
| real | zcrVariance | 0.337 / 0.623 | 0.286 | 1.85 | 0.435 | 32.0% / 68.3% | 2.14 | **NO** |
| real | jitter | 0.606 / 0.354 | 0.252 | 1.71 | -0.397 | 65.7% / 34.6% | 1.90 | **NO** |
| real | shimmer | 0.523 / 0.437 | 0.085 | 1.19 | -0.132 | 55.2% / 45.1% | 1.22 | **NO** |
| real | hnr_db | 0.390 / 0.570 | 0.179 | 1.46 | 0.298 | 38.9% / 61.4% | 1.58 | **NO** |
| real | pad_fraction | 0.498 / 0.450 | 0.048 | 1.11 | -0.069 | 52.1% / 46.9% | 1.11 | yes |
| fake | pauseRatio | 0.498 / 0.455 | 0.043 | 1.09 | -0.035 | 48.2% / 52.2% | 1.08 | yes |
| fake | energyVariance | 0.399 / 0.556 | 0.157 | 1.39 | 0.270 | 59.9% / 40.4% | 1.48 | **NO** |
| fake | zcrVariance | 0.303 / 0.652 | 0.350 | 2.16 | 0.578 | 71.9% / 28.3% | 2.54 | **NO** |
| fake | jitter | 0.626 / 0.329 | 0.298 | 1.91 | -0.490 | 31.3% / 68.9% | 2.20 | **NO** |
| fake | shimmer | 0.493 / 0.462 | 0.031 | 1.07 | -0.068 | 48.4% / 51.9% | 1.07 | yes |
| fake | hnr_db | 0.382 / 0.572 | 0.190 | 1.50 | 0.318 | 62.1% / 38.2% | 1.63 | **NO** |
| fake | pad_fraction | 0.483 / 0.458 | 0.024 | 1.05 | -0.027 | 49.8% / 51.4% | 1.03 | yes |

### v11: FAIL real:pauseRatio, real:energyVariance, real:zcrVariance, real:jitter, real:shimmer, fake:pauseRatio, fake:energyVariance, fake:zcrVariance, fake:jitter, fake:shimmer

| class | feature | mean p low/high | gap | ratio | rho | rate low/high | rate ratio | pass |
|---|---|---|---|---|---|---|---|---|
| real | pauseRatio | 0.531 / 0.592 | 0.062 | 1.12 | 0.115 | 40.7% / 55.4% | 1.36 | **NO** |
| real | energyVariance | 0.442 / 0.681 | 0.238 | 1.54 | 0.515 | 34.8% / 61.2% | 1.76 | **NO** |
| real | zcrVariance | 0.668 / 0.455 | 0.213 | 1.47 | -0.338 | 57.4% / 38.6% | 1.49 | **NO** |
| real | jitter | 0.470 / 0.653 | 0.182 | 1.39 | 0.282 | 40.3% / 55.7% | 1.38 | **NO** |
| real | shimmer | 0.541 / 0.582 | 0.042 | 1.08 | 0.051 | 40.5% / 55.5% | 1.37 | **NO** |
| real | hnr_db | 0.585 / 0.538 | 0.047 | 1.09 | -0.028 | 46.8% / 49.2% | 1.05 | yes |
| real | pad_fraction | 0.549 / 0.582 | 0.033 | 1.06 | 0.047 | 46.1% / 51.1% | 1.11 | yes |
| fake | pauseRatio | 0.553 / 0.626 | 0.073 | 1.13 | 0.141 | 55.1% / 40.4% | 1.36 | **NO** |
| fake | energyVariance | 0.488 / 0.690 | 0.202 | 1.41 | 0.450 | 57.9% / 38.1% | 1.52 | **NO** |
| fake | zcrVariance | 0.701 / 0.477 | 0.224 | 1.47 | -0.337 | 37.0% / 59.0% | 1.60 | **NO** |
| fake | jitter | 0.506 / 0.672 | 0.166 | 1.33 | 0.237 | 54.4% / 41.6% | 1.31 | **NO** |
| fake | shimmer | 0.537 / 0.641 | 0.104 | 1.19 | 0.184 | 58.8% / 37.2% | 1.58 | **NO** |
| fake | hnr_db | 0.615 / 0.562 | 0.054 | 1.10 | -0.046 | 49.6% / 46.4% | 1.07 | yes |
| fake | pad_fraction | 0.580 / 0.621 | 0.042 | 1.07 | 0.059 | 49.5% / 42.7% | 1.16 | yes |

Pad diagnostics (eval data, same for every model): padded share real 38.2%, fake 21.7%, pad_fraction label AUC 0.418.

## Attack-type head

