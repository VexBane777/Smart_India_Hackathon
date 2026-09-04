# Implementation Backlog: Product & Demo (MOD-C)

| Task ID | Subject | Technical Requirement | Definition of Done (DoD) | Deps |
|---|---|---|---|---|
| MOD-C-01 | Record Demo Assets | Record Call A, Call N, Call B per Doc 4 §4.2. | Raw WAV takes archived; Form A signed for all. | MOD-A-01 |
| MOD-C-02 | Apply Channel to Assets | Run `assets/demo/` through TeleChannel `whatsapp` recipe. |-Assets finalized as 16kHz FLAC; no clipping. | MOD-A-13 |
| MOD-C-03 | Generate Cue JSON | Run `vaani.engine --record-cues` on assets. | `cues.json` exists with precise alert timestamps. | MOD-B-09 |
| MOD-C-04 | Build Streamlit UI | Implement Gauge, Spectrogram, and Risk Curve with `st.fragment`. | UI updates every 0.5s without page flicker. | MOD-B-09 |
| MOD-C-05 | Implement Bank Sim | Build "Transfer" $\rightarrow$ "HOLD" $\rightarrow$ "OTP Modal" workflow. | Modal triggers correctly based on engine score. | MOD-C-04 |
| MOD-C-06 | Verify Airplane Mode | Force CPU; disable Wifi; run full Demo A $\rightarrow$ Demo N. | App runs 100% offline with zero crashes. | MOD-C-05 |
| MOD-C-07 | Draft Pitch Deck | Create 12 slides following Doc 3 §3.2. | Deck structure complete; all [M] tags identified. | None |
| MOD-C-08 | Fill Numbers Lockdown | Replace [M] tags with values from `latency.json` and `leaderboard.md`. | Every number in the deck has a verified source path. | MOD-B-10 |
| MOD-C-09 | Record Backup Video | Record screen capture of full Demo A with audio. | 90s MP4 exists; timestamps match `cues.json`. | MOD-C-06 |
| MOD-C-10 | Final Q&A Drill | Rehearse "Attack $\rightarrow$ Answer" pairs from Doc 3 §3.4. | Team can answer all 10+ drill questions. | MOD-C-08 |
