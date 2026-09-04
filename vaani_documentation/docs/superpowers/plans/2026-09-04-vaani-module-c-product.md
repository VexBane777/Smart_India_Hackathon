# VAANI Module C (Product & Demo) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the end-to-end demo experience: high-fidelity audio assets, a real-time monitoring dashboard, and a "measured-only" pitch deck.

**Architecture:** A FastAPI backend serves as the audio streamer and engine wrapper, while a Streamlit frontend uses `st.fragment` to provide a low-latency risk gauge and a simulated banking workflow.

**Tech Stack:** Streamlit, FastAPI, WebSockets, RVC (for cloning), FFmpeg.

## Execution Order & Cross-Module Dependencies
**Depends on Module A and Module B.** Task 2 (Demo Asset Channeling) requires Module A
Task 8 (`telechannel/pipeline.py`) to exist. Task 4 (Streamlit UI) and Task 3 (Demo Cue
Generation) require a working, loadable model from Module B (Task 2 minimum for Week 1;
Task 9 ONNX/INT8 export before the final airplane-mode/latency-sensitive build in Task 6).
Do not start Task 2 before Module A's pipeline orchestrator lands, and do not start Task 4
before Module B has a checkpoint to load — both will block on missing files otherwise.

- **Week 1 note (master plan §8):** Tasks 1–5 (raw recording through bank-sim workflow)
  are the Week 1 critical path (Thu/Fri per the day-by-day table) and run *after* Module A
  and B's Week-1-scoped slices land, not in parallel from Monday. Tasks 6–8 (airplane-mode
  verification, pitch deck, final demo production) are Week 1 Sat/Sun and recur at the
  Week 10 freeze once real measured numbers replace Week-1 placeholders.

## Global Constraints
- **Demo Duration**: Exactly 90 seconds.
- **Connectivity**: 100% Offline (Airplane Mode). Zero external API calls.
- **Claims**: No estimated numbers. Every slide must link to a `latency.json` or `leaderboard.md` entry.
- **Consent**: All assets must be linked to a signed Form A.

---

### Task 1: Demo Asset Recording (Raw)

**Files:**
- Create: `assets/raw/call_A_raw.wav`
- Create: `assets/raw/call_N_raw.wav`
- Create: `assets/raw/call_B_raw.wav`

**Interfaces:**
- Produces: 48kHz mono RAW WAVs.

- [ ] **Step 1: Record Call A (Synthetic)**
  Record the "Vendor Payment" script. Speaker 1 (Real) $\rightarrow$ Speaker 2 (RVC Clone).
- [ ] **Step 2: Record Call N (Control)**
  Record the identical script using Speaker 2's REAL voice.
- [ ] **Step 3: Record Call B (Accent)**
  Record the script using a different accent pair (e.g., Tamil-English).
- [ ] **Step 4: Verify Audio Quality**
  Check for plosives and fan noise. Peaks at $\sim -6$ dBFS.
- [ ] **Step 5: Commit**
  (Audio not in git) $\rightarrow$ Update `assets/manifest.json` with file paths.
  `git commit -m "docs: update demo asset manifest"`

### Task 2: Demo Asset Channeling

**Files:**
- Modify: `telechannel/pipeline.py`
- Create: `assets/demo/call_A_whatsapp.wav`
- Create: `assets/demo/call_N_whatsapp.wav`

**Interfaces:**
- Consumes: `assets/raw/*.wav`
- Produces: 16kHz mono FLAC via `whatsapp` recipe.

- [ ] **Step 1: Apply WhatsApp Recipe**
  Process Call A and Call N through the `whatsapp` recipe (Opus $\rightarrow$ Noise $\rightarrow$ RIR).
- [ ] **Step 2: Verify Alert Behavior**
  Run the engine on Call A. Verify alert fires $\sim 3$s after clone entry.
- [ ] **Step 3: Verify Control Behavior**
  Run the engine on Call N. Verify no alert fires.
- [ ] **Step 4: Commit**
  `git commit -m "docs: update demo asset manifest for processed files"`

### Task 3: Demo Cue Generation

**Files:**
- Create: `assets/demo/call_A_whatsapp.cues.json`

**Interfaces:**
- Produces: JSON map of `{timestamp: score}`.

- [ ] **Step 1: Implement Cue Recorder**
  Modify `vaani.engine` to output per-window scores to a JSON file when `--record-cues` is passed.
- [ ] **Step 2: Generate Cues for Call A**
  Run the lauched model on Call A $\rightarrow$ save `cues.json`.
- [ ] **Step 3: Verify Timestamps**
  Ensure the "clone entry" timestamp matches the audio script.
- [ ] **Step 4: Commit**
  `git add assets/demo/call_A_whatsapp.cues.json`
  `git commit -m "feat: add demo cues for pitch deck timing"`

### Task 4: Streamlit UI Implementation

**Files:**
- Create: `app.py`
- Create: `app/components/gauge.py`
- Create: `app/components/spectrogram.py`

**Interfaces:**
- Consumes: WebSocket stream from FastAPI backend.
- Produces: Live risk gauge + spectrogram.

- [ ] **Step 1: Implement FastAPI Streamer**
  Build a backend that reads a WAV file and pushes 0.5s chunks to a WebSocket.
- [ ] **Step 2: Build Risk Gauge**
  In `gauge.py`, use `st.metric` or a custom HTML/CSS gauge that updates via `st.fragment`.
- [ ] **Step 3: Build Live Spectrogram**
  In `spectrogram.py`, use `st.pyplot` or `plotly` to show the current 2s window Mel-spectrogram.
- [ ] **Step 4: Implement Risk Curve**
  Add a line chart showing the smoothed EMA score over the call duration.
- [ ] **Step 5: Commit**
  `git add app.py app/components/`
  `git commit -m "feat: implement live demo dashboard"`

### Task 5: Bank Simulation & OTP Workflow

**Files:**
- Modify: `app.py`
- Create: `app/components/bank_modal.py`

**Interfaces:**
- Consumes: `risk_score > threshold`
- Produces: "HOLD" $\rightarrow$ "OTP" UI transition.

- [ ] **Step 1: Implement "Transfer" Trigger**
  Add a "Initiate Transfer" button to the UI.
- [ ] **Step 2: Implement Hold Logic**
  If the engine score is high, intercept the transfer $\rightarrow$ show "TRANSACTION HELD" warning.
- [ ] **Step 3: Build OTP Modal**
  Create a simulated OTP entry screen ("Enter 6-digit code sent to your device").
- [ ] **Step 4: Verify Workflow**
  Run Call A $\rightarrow$ Transfer $\rightarrow$ HOLD $\rightarrow$ OTP.
- [ ] **Step 5: Commit**
  `git add app/components/bank_modal.py`
  `git commit -m "feat: add mock bank simulation workflow"`

### Task 6: Airplane Mode Verification

**Files:**
- Create: `tests/demo/test_offline.py`

**Interfaces:**
- Consumes: Entire App Stack.

- [ ] **Step 1: Force CPU & Local Paths**
  Ensure `models.meta.json` points to local `.onnx` and assets are in `assets/`.
- [ ] **Step 2: Kill Network**
  Disable WiFi/Ethernet.
- [ ] **Step 3: Full Run Test**
  Launch `app.py` $\rightarrow$ Run Call A $\rightarrow$ Run Call N.
- [ ] **Step 4: Verify zero-latency**
  Ensure no "Connecting..." or "Timeout" messages appear.
- [ ] **Step 5: Commit**
  `git commit -m "test: verify airplane mode operational guarantee"`

### Task 7: Pitch Deck Draft & Number Lockdown

**Files:**
- Create: `docs/pitch/deck_structure.md`
- Create: `docs/pitch/lockdown.json`

**Interfaces:**
- Produces: 12-slide outline.

- [ ] **Step 1: Draft 12 Slides**
  Create the outline based on Doc 3 §3.2.
- [ ] **Step 2: Map [M] Tags**
  In `lockdown.json`, map every number in the deck to a source file (e.g., `S6: latency -> runs/student/latency.json`).
- [ ] **Step 3: Fill measured values**
  Replace all [M] tags with the actual numbers from the Model Engine's final runs.
- [ ] **Step 4: Commit**
  `git add docs/pitch/`
  `git commit -m "docs: finalize pitch deck numbers lockdown"`

### Task 8: Final Demo Production

**Files:**
- Create: `assets/demo/backup_video.mp4`

**Interfaces:**
- Produces: High-res screen recording.

- [ ] **Step 1: Record "Golden Run"**
  Perform the full 90s demo (Call A $\rightarrow$ Hold $\rightarrow$ OTP) and record the screen.
- [ ] **Step 2: Sync with Cues**
  Verify the video timestamps match `cues.json` exactly.
- [ ] **Step 3: Create Appendix Screenshots**
  Capture high-res images of the Risk Gauge and Spectrogram for the deck.
- [ ] **Step 4: Commit**
  `git commit -m "docs: record backup demo video"`
