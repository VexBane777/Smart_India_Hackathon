# TDD: Product & Demo Suite (MOD-C-01)
**Source Spec:** DOC3_PITCH_DECK.md + DOC4_DEMO_ASSETS.md
**Module:** C (Product)

## 1. System Overview
The Product suite transforms technical metrics into a persuasive, honest narrative. The core is the "Demo Asset" $\rightarrow$ "Live App" $\rightarrow$ "Pitch Deck" pipeline. The objective is to provide a seamless, offline-safe demonstration that proves the "measured numbers" claim.

**Input:** Recorded voice assets, Measured metrics (`latency.json`, `leaderboard.md`).
**Output:** 90s Demo Video/App, 12-Slide Pitch Deck.

## 2. Component Architecture

### 2.1 The Demo App (Streamlit + FastAPI)
A high-performance dashboard that simulates the real-time experience.

**UI Layout:**
- **Live Risk Gauge**: Real-time risk score (Student model).
- **Live Spectrogram**: Visuals of the current audio window.
- **Risk Curve**: Plot of risk over time (smoothed EMA).
- **Occlusion Panel**: Highlights segments that drove the alert.
- **Mock Bank Modal**: Triggered by "Transfer" $\rightarrow$ "HOLD" $\rightarrow$ "OTP Step-up".

**Demo Backend:**
- **Streamer**: Reads `assets/demo/callA.wav` and feeds it to the engine every 0.5s.
- **State Machine**: Tracks `Real` $\rightarrow$ `Clone` transition to trigger UI cues.

### 2.2 The Asset Production Pipeline
A rigorous process for creating "Judge-Proof" demo files.
- **Pairing**: Create Call A (Synthetic) and Call N (Real) with identical scripts.
- **Channeling**: Apply the `whatsapp` recipe to ensure a "real-world" feel.
- **Cues**: Run the engine on the asset $\rightarrow$ generate `cues.json` (timestamps of spikes).

### 2.3 The "Honesty Gate" Slide Engine
A strict mapping of Slide $\rightarrow$ Artifact.
- **Lockdown Table**: A JSON mapping of `Slide #` $\rightarrow$ `Metric` $\rightarrow$ `Source Path`.
- **Validation**: Slide is "Ready" only if the source path contains a non-placeholder value.

## 3. Implementation Deep-Dive

### 3.1 Streamlit Performance Fix
To prevent the "Full Page Refresh" freeze during live streaming:
- Use `st.fragment(run_every=0.5)` for the Risk Gauge and Spectrogram.
- Use WebSockets for the backend-to-frontend score push.

### 3.2 The "Airplane Mode" Guarantee
The app must have zero network dependencies.
- **Model**: Local `.onnx` file.
- **Assets**: Local `.wav` files.
- **Logic**: All processing on laptop CPU.
- **Verification**: Run `netstat -an` $\rightarrow$ verify no active outgoing connections during demo.

## 4. Verification Plan

| Component | Test Case | Expected Outcome |
|---|---|---|
| **Demo App** | Stream Call A | Gauge spikes $\sim 3$s after clone entry; OTP modal triggers. |
| **Negative Control** | Stream Call N | Gauge stays low; no OTP modal. |
| **Latency** | `cues.json` check | First score appears at $\sim 2.5$s; window updates every 0.5s. |
| **Offline Mode** | Wifi OFF $\rightarrow$ Run | App launches and streams without errors. |
| **Slide Audit** | [M] tag scan | Zero [M] tags in the final deck; all source paths verified. |
