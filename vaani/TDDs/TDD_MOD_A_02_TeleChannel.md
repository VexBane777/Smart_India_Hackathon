# TDD: TeleChannel Generation Pipeline (MOD-A-02)
**Source Spec:** DOC1_TELECHANNEL_SPEC.md
**Module:** A (Data Foundation)

## 1. System Overview
The TeleChannel pipeline transforms "clean" audio into "network-degraded" audio. It is a physically ordered chain of DSP (Digital Signal Processing) stages. The primary goal is to create a dataset where the degradation is an authentic simulation of Indian telephony.

**Input:** Clean WAV (Real/Fake).
**Output:** FLAC file + Metadata JSON row in `manifest.parquet`.

## 2. Component Architecture

### 2.1 The Pipeline Orchestrator (`pipeline.py`)
A sequential processor that applies stages based on a `recipe` (from `channels.yaml`).

**Data Flow:**
`Raw Audio` $\rightarrow$ `RIR` $\rightarrow$ `Noise` $\rightarrow$ `Mic` $\rightarrow$ `Codec` $\rightarrow$ `PacketLoss` $\rightarrow$ `BandLimit` $\rightarrow$ `Normalization` $\rightarrow$ `Export`.

### 2.2 Stage Implementations (`stages/`)

#### Stage 1: RIR (`rir.py`)
- **Input:** Signal $x$, RIR filter $h$.
- **Logic:** FFT convolution $\rightarrow$ align to direct path $\rightarrow$ blend wet/dry.
- **Interface:** `apply_rir(x, rir_id, wet_gain=0.7)`.

#### Stage 2: Additive Noise (`noise.py`)
- **Input:** Signal $x$, Noise clip $n$, Target SNR (dB).
- **Logic:** Calculate RMS of $x$ $\rightarrow$ scale $n$ to match SNR $\rightarrow$ add.
- **Interface:** `apply_noise(x, noise_type, snr_db)`.

#### Stage 3: Mic Stage (`mic.py`)
- **Input:** Signal $x$.
- **Logic:** Random gain ($-6$ to $+12$ dB) $\rightarrow$ Hard clip at $\pm 0.5$ (probabilistic).
- **Interface:** `apply_mic(x, clip_prob=0.2)`.

#### Stage 4: Codec Round-trip (`codec.py`)
- **Input:** Signal $x$, Codec name (e.g., `amr_nb`), Bitrate.
- **Logic:** 
    1. Write to temp WAV.
    2. `ffmpeg -i temp.wav -acodec <codec> -ab <bitrate> out.<ext>`.
    3. `ffmpeg -i out.<ext> -ar 16000 out_final.wav`.
    4. Read `out_final.wav` $\rightarrow$ pad/trim to original length.
- **Interface:** `codec_roundtrip(x, codec, bitrate)`.

#### Stage 5: Packet Loss (`packetloss.py`)
- **Input:** Signal $x$.
- **Logic:** 
    1. Generate Gilbert-Elliott mask (binary array of 20ms frames).
    2. Zero out dropped frames.
    3. `conceal_loss()`: Repeat last good frame with 5ms crossfade.
- **Interface:** `apply_packet_loss(x, p_gb, p_bg)`.

#### Stage 6: Band-Limiting (`bandlimit.py`)
- **Input:** Signal $x$.
- **Logic:** 5th order Butterworth bandpass filter (300–3400 Hz).
- **Interface:** `apply_bandlimit(x, sr=16000)`.

### 2.3 Manifest & Metadata (`manifest.py`)
Every clip is registered in a Parquet file.
- **Schema:** See Doc 1 §1.6.
- **Splits:** `train`, `val`, `test`, `demo`.
- **Hold-outs:** Implemented as filters: `df[df['generator'] == 'rvc_team']`.

## 3. Implementation Deep-Dive

### 3.1 The "Tandem" Transcoding
To simulate a call crossing networks (e.g., VoLTE $\rightarrow$ Landline), the pipeline supports calling `codec_roundtrip` twice with different configs.

### 3.2 Performance Optimization
- **Parallelism:** Use `multiprocessing.Pool` to process clips.
- **FFmpeg Overhead:** Use a single long-running FFmpeg process or batch files to avoid process-start latency.

## 4. Verification Plan

| Component | Test Case | Expected Outcome |
|---|---|---|
| **RIR** | Impulse response input | Output is exactly the RIR filter. |
| **Codec** | G.711 $\rightarrow$ PCM | 8kHz sampling rate; quantization noise present. |
| **PacketLoss** | 10% loss rate | Histogram of drop lengths shows "bursts" (not random). |
| **BandLimit** | Sine wave at 5kHz | Output energy at 5kHz is $<-40$ dB. |
| **Manifest** | Split check | No `source_clip` appears in both `train` and `test`. |
