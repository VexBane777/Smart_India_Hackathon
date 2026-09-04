<!--
VAANI documentation suite v1.0 — generated 2026-09-03
Document: Doc 5 — Real-Call Recording Protocol (The 25 Validation Calls) · Owner: Corpus Lead · Status: Build-ready
This file is GENERATED. Edit generate_vaani_docs.py and rerun instead.
-->
# Doc 5 — Real-Call Recording Protocol (The 25 Validation Calls)

**Owner:** Corpus Lead · **Status:** Build-ready · **Suite:** v1.0 · **Generated:** 2026-09-03
**Depends on:** Doc 6 Form B signed by BOTH parties before every call

---

## 5.1 Purpose
One corpus, two non-training jobs: (a) validates TeleChannel against reality (Doc 1
§1.9), (b) held-out test set #4 (master plan §5.4). Never enters any training split.

## 5.2 Call matrix (25 calls, 3–5 min each)
| Network | Calls | Why |
|---|---|---|
| Cellular VoLTE (Jio/Airtel/Vi) | 10 | default Indian path — AMR-WB wideband |
| Cellular, one party forced 2G/3G | 5 | narrowband AMR-NB path |
| WhatsApp | 7 | Opus path — the channel scams actually use |
| Landline→mobile | 3 (else redistribute to VoLTE) | G.711 PSTN |

Environments: ≥5 outdoor street, ≥5 noisy indoor, rest quiet. Participants: 4 team
members + consenting family, rotating pairs. **Both parties sign Form B before dialing.**
No third parties, no customer-service/bank/emergency numbers, ever.

## 5.3 Per-call script (the matched-passage trick)
| Segment | Duration | Content |
|---|---|---|
| Free chat | ~60 s | natural |
| **Matched passage** | ~90 s | **every call reads the same 150-word Hinglish passage** (stored in `assets/passages/matched_hi_en.txt`) — same content across networks = spectra differ only because of the channel; that comparability is the entire scientific value |
| Transaction readout | ~60 s | digits, amounts ("forty lakh, account ending 4471") |

## 5.4 Recording methods ($0) — and the subtlety everyone misses
| Network | Method | Notes |
|---|---|---|
| Cellular | Android native recorder (Google Phone app where supported) | announces "this call is being recorded" — doubles as all-party notice |
| Cellular (iPhones etc.) | Speakerphone + second device | fixed ~30 cm from the speaker |
| WhatsApp | Always speakerphone + second device | no in-app recording exists |

**Subtlety: the capture method is itself a channel stage.** Speakerphone + second device
adds room acoustics and speaker distortion on top of the network codec. Track it:
metadata records `capture: native_recorder | speakerphone_second_device`, and analysis
**stratifies by network × capture** — otherwise you misattribute recorder artifacts to
the network and "validate" the wrong thing.

## 5.5 Metadata schema
```json
{ "call_id": "rc_011",
  "network": "volte | 3g_fallback | whatsapp | pstn",
  "codec_ground_truth": "expected_from_network | confirmed",
  "operator_a": "jio", "operator_b": "airtel", "signal": "full|medium",
  "capture": "native_recorder|speakerphone_second_device",
  "devices": ["pixel_6a", "redmi_12"],
  "env": "quiet_indoor|noisy_indoor|outdoor_street",
  "duration_s": 241, "language": "hi-en",
  "consent_ids": ["c_260312_AR", "c_260312_KP"], "city_anon": "city_1" }
```
Honesty flag: VoLTE⇒AMR-WB etc. are *expectations*; mark confirmed only via radio menus.

## 5.6 Analysis
Transfer weekly → 16 kHz mono FLAC → manifest `origin: real_call,
split: validation_holdout`. Per call, over VAD-detected speech: LTAS (long-term average
spectrum), bandwidth cutoff (−20 dB point), HF ratio (energy >3.5 kHz). Stratify by
network × capture; overlay vs matching TeleChannel recipes.
**Targets: median cutoff difference ≤300 Hz per stratum; LTAS correlation ≥0.9.**
A mismatch is a finding, not a failure — it names the recipe parameter to fix; the
before/after is a genuinely good changelog slide.

## 5.7 Schedule & Done means
Weeks 2–4, parallel to TeleChannel v1; each pair records ~6–7 calls. Done: 25 calls with
≥2 min usable speech each · metadata complete · both-party consents filed · FLAC
ingested as validation_holdout · figure + numbers entered into the Doc 3 lockdown table.

---

*Part of the VAANI documentation suite — regenerate with `python generate_vaani_docs.py`. Placeholders marked [M] must be replaced by measured values before use in the pitch.*
