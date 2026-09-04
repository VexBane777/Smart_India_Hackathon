<!--
VAANI documentation suite v1.0 — generated 2026-09-03
Document: Doc 6 — Consent & Privacy Templates · Owner: Corpus Lead (templates) / Product Lead (translations) · Status: Finalize Monday morning
This file is GENERATED. Edit generate_vaani_docs.py and rerun instead.
-->
# Doc 6 — Consent & Privacy Templates

**Owner:** Corpus Lead (templates) / Product Lead (translations) · **Status:** Finalize Monday morning · **Suite:** v1.0 · **Generated:** 2026-09-03

---

> **Disclaimer:** good-faith, DPDP-aligned (India's personal-data law) internal practice
> for a student research project — not legal advice. Anything commercial or beyond the
> explicitly opted-in public release gets reviewed by actual counsel first.

## 6.1 Form A — voice donor consent (dataset, cloning, demo)
```
VAANI PROJECT — VOICE DONOR CONSENT                  Form A, v1.0
[Name] · [Team ID / relation] · [Date] · [consent_id: c_YYMMDD_initials]

I understand my voice will be recorded for a student research project on
AI voice-fraud detection, and I consent to the following (each separate):

  [ ] A1. Internal research & model training (real-speech class)
  [ ] A2. PUBLIC RELEASE in the open dataset (my anonymized clips, license
          [___]); cannot be fully recalled once published — I understand
  [ ] A3. Voice CLONING (RVC) to create synthetic attack examples that
          sound like me
  [ ] A4. Cloned voice appearing in the DEMO VIDEO and live demos,
          including PUBLIC posting of that video (independent of A3)
  [ ] A5. Live on-stage cloning demonstration (sample given minutes before)

Retention: raw recordings kept until [project end + 6 months] or withdrawal.
Withdrawal: email [corpus lead] — clips removed from future dataset versions
and training runs; published versions removed from distribution to the
maximum extent feasible.
My name is never stored with the audio — only an anonymous speaker ID
linked to this form, in a register accessible only to the corpus lead.
No payment. I am 18+. Questions first: [contact].
Signature: ____________  Date: ________  Witness: ____________
```
Design rule: A2/A4 are separate opt-ins above the A1/A3 baseline; the manifest's
license field says `public` **only** when A2 was checked.

## 6.2 Form B — call participant consent (the 25 validation calls)
```
VAANI PROJECT — RECORDED CALL CONSENT                Form B, v1.0
BOTH participants sign BEFORE the call is placed.

We will record a 3–5 minute phone call to measure how real Indian phone
networks shape audio, to validate a simulated training dataset.

  [ ] I consent to this specific call being recorded.
Storage: internal team machines only. The audio will NEVER be published or
shared — only anonymous aggregate statistics (average spectra) may appear
in reports or the public writeup.
Retention: until [project end + 6 months] or withdrawal ([contact]).
No third parties on the call. I may say "pause the recording" anytime.

Participant 1: ____________  Participant 2: ____________  Date: ________
```

## 6.3 Form C — demo-day live cloning consent
Single page, handed minutes before the "clone this voice in front of the judges"
moment; mirrors A5; volunteer keeps a copy. Saying on stage that it's consented is
itself a privacy-by-design point.

## 6.4 Consent register (paper ↔ data link)
```json
{ "consent_id": "c_260312_KP", "form": "A", "version": "1.0",
  "name": "[kept only here]", "opt_ins": ["A1","A3"], "date": "2026-03-12",
  "speaker_anon": "spk_041", "re_review_date": "2026-09-12",
  "withdrawn": false, "withdrawn_date": null }
```
Stored on master + one backup, readable only by the corpus lead. Manifest references
`speaker_anon`, never names. **Revocation workflow:** withdrawal email → flip flag →
script filters that speaker from the next dataset version and all future training
manifests → log removal in the dataset changelog.

## 6.5 DPDP privacy note (README, datasheet, app UI)
> VAANI processes voice on the device where the call is handled. No audio is stored,
> transmitted, or retained — only derived features and risk scores. Data is used solely
> for fraud-risk assessment (purpose limitation) and the minimum needed for it (data
> minimization). Consent records are maintained under the DPDP Act 2023; data principals
> may request access, correction, or erasure via [contact]. In the event of a breach
> affecting personal data, affected principals and the Data Protection Board will be
> notified as the Act requires.

## 6.6 DPDP principle map
| Practice | DPDP hook |
|---|---|
| Itemized, informed consent (A/B/C) | §6 consent standards; notice in English or an Eighth-Schedule language (Hindi/Tamil translations) |
| Audio never persisted; features only | purpose limitation + minimization |
| Retention dates on every form | storage limitation |
| Withdrawal → removal workflow | §11–13 data-principal rights |
| Named contact | grievance redressal |
| Breach-notification sentence | breach reporting duty |

## 6.7 Done means
Forms v1.0 + Hindi translation of A and B · register on master + backup · every
recording in Docs 4–5 traceable to a signed row · privacy note in README, datasheet, UI.

---

*Part of the VAANI documentation suite — regenerate with `python generate_vaani_docs.py`. Placeholders marked [M] must be replaced by measured values before use in the pitch.*
