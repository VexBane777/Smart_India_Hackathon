# Implementation Backlog: Real-Call Validation (MOD-A)

| Task ID | Subject | Technical Requirement | Definition of Done (DoD) | Deps |
|---|---|---|---|---|
| MOD-A-17 | Create Call Matrix | Setup tracking sheet for the 25 calls (Network $\times$ Env $\times$ Participant). | Matrix filled with assignments for all 25 slots. | None |
| MOD-A-18 | Prepare Matched Passage | Finalize `assets/passages/matched_hi_en.txt` for all participants. | Text file available and reviewed for readability. | None |
| MOD-A-19 | Record Validation Set | Execute the 25 calls following the Doc 5 protocol. | 25 FLAC files $\ge 2$ min speech each. | MOD-A-01 |
| MOD-A-20 | Build Analysis Tool | Create `validate_reality.py` (LTAS, Bandwidth, HF Ratio). | Script produces PSD for a given WAV file. | None |
| MOD-A-21 | Perform Stratification | Tag all recordings by `capture` method (Native vs Speakerphone). | Manifest contains `capture` metadata for every call. | MOD-A-19 |
| MOD-A-22 | Generate Validation Fig | Compare real spectra vs TeleChannel recipes; create overlay plot. | Figure showing median cutoff diff $\le 300$ Hz. | MOD-A-13, MOD-A-20 |
| MOD-A-23 | Update Lockdown Table | Enter final validation numbers into Doc 3 Slide 8. | [M] tags replaced with measured correlation/diff. | MOD-A-22 |
