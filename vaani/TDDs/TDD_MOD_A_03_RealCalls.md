# TDD: Real-Call Validation Protocol (MOD-A-03)
**Source Spec:** DOC5_REAL_CALLS.md
**Module:** A (Data Foundation)

## 1. System Overview
The Real-Call protocol provides the "ground truth" for the TeleChannel simulation. By recording real calls and comparing their spectral fingerprints to the simulated ones, we can objectively prove that our simulation is an authentic proxy for the Indian network.

**Input:** 25 consented real calls.
**Output:** `validation_results.json` $\rightarrow$ Figure for Doc 3 Slide 8.

## 2. Component Architecture

### 2.1 Recording Matrix
A strictly enforced grid of 25 calls to ensure coverage:
- **VoLTE (10)** $\rightarrow$ AMR-WB.
- **2G/3G (5)** $\rightarrow$ AMR-NB.
- **WhatsApp (7)** $\rightarrow$ Opus.
- **PSTN (3)** $\rightarrow$ G.711.

### 2.2 The "Matched Passage" Control
To eliminate the "content variable", every call must include a reading of the same 150-word passage. 
- **Logic:** $\text{Spectrum}(\text{Real Call}) - \text{Spectrum}(\text{Real Passage}) \approx \text{Channel Effect}$.

### 2.3 Capture Stratification
Because recording methods introduce their own artifacts, all analysis must be stratified:
- **Stratum 1:** Native Recorder (Cleanest).
- **Stratum 2:** Speakerphone $\rightarrow$ Second Device (Adds room/speaker distortion).
- **Requirement:** Never compare a "Native" simulation to a "Speakerphone" recording.

## 3. Implementation Deep-Dive

### 3.1 Analysis Pipeline (`validate_reality.py`)
For each recording:
1. **VAD**: Strip silence.
2. **LTAS (Long-Term Average Spectrum)**: Compute power spectral density (PSD) over the full speech duration.
3. **Bandwidth Cutoff**: Find the frequency where power drops $-20$ dB from the peak.
4. **HF Ratio**: $\frac{\int_{3.5\text{kHz}}^{\text{max}} P(f) df}{\int_{0}^{3.5\text{kHz}} P(f) df}$.

### 3.2 Comparison Logic
Compare the measured values against the corresponding TeleChannel recipe:
$\text{Error} = |\text{Measured}_{\text{Real}} - \text{Measured}_{\text{Simulated}}|$
**Targets:** Median $\text{Error} \leq 300$ Hz; Correlation $\geq 0.9$.

## 4. Verification Plan

| Test Case | Input | Expected Outcome |
|---|---|---|
| **Baseline Match** | Clean recording $\rightarrow$ 'clean' recipe | Correlation $\approx 1.0$. |
| **Codec Detection** | VoLTE call | Cutoff matches AMR-WB profile. |
| **Stratum Check** | Native vs Speakerphone | Clear spectral difference in the $2\text{kHz}-5\text{kHz}$ band. |
| **Target Hit** | 25-call aggregate | Median cutoff diff $\leq 300$ Hz. |
