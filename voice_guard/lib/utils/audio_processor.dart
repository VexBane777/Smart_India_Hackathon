import 'dart:math' as math;

/// Lightweight LFCC + prosody feature extraction (Dart side).
/// Mirrors the logic described in the SIH master prompt:
/// 16kHz, 3s chunks, 1024-pt FFT, 256 hop, 513 linear banks, DCT→60 LFCC.
/// Prosody: pitch-variance proxy (zero-crossing / energy variance) + pause ratio.
/// Raw PCM is processed in-memory and discarded — never written to disk.

class AudioProcessor {
  static const int sampleRate = 16000;
  static const int chunkSamples = 48000; // 3s
  static const int fftSize = 1024;
  static const int hopLength = 256;
  static const int nLfcc = 60;
  static const int nFilterBanks = 513;

  /// 3s PCM16 buffer → 60 LFCC coefficients (mean-pooled over frames).
  static List<double> extractLfcc(List<double> pcm) {
    if (pcm.length < fftSize) return List.filled(nLfcc, 0);
    final frames = _frameSignal(pcm);
    final filterbankEnergies = <List<double>>[];
    for (final frame in frames) {
      final windowed = _hamming(frame);
      final spectrum = _magnitudeSpectrum(windowed);
      final energies = _linearFilterbank(spectrum);
      filterbankEnergies.add(energies);
    }
    // Log + DCT per frame, then mean-pool.
    final lfccFrames = filterbankEnergies.map((e) {
      final logE = e.map((v) => math.log(v + 1e-10)).toList();
      return _dct(logE).sublist(0, nLfcc);
    }).toList();

    final mean = List<double>.filled(nLfcc, 0);
    for (final f in lfccFrames) {
      for (int i = 0; i < nLfcc; i++) { mean[i] += f[i]; }
    }
    for (int i = 0; i < nLfcc; i++) { mean[i] /= lfccFrames.length; }
    // Zero-mean / unit-ish normalization.
    final m = mean.reduce((a, b) => a + b) / mean.length;
    final variance = mean.map((v) => (v - m) * (v - m)).reduce((a, b) => a + b) / mean.length;
    final std = math.sqrt(variance + 1e-8);
    return mean.map((v) => (v - m) / std).toList();
  }

  /// Prosody features: [pauseRatio, energyVariance, zcrVariance]
  static List<double> extractProsody(List<double> pcm) {
    if (pcm.isEmpty) return [0, 0, 0];
    const frameLen = 512;
    final energies = <double>[];
    final zcrs = <double>[];
    for (int i = 0; i + frameLen <= pcm.length; i += frameLen) {
      final frame = pcm.sublist(i, i + frameLen);
      final energy = frame.map((v) => v * v).reduce((a, b) => a + b) / frameLen;
      energies.add(energy);
      int zc = 0;
      for (int j = 1; j < frame.length; j++) {
        if ((frame[j] >= 0) != (frame[j - 1] >= 0)) zc++;
      }
      zcrs.add(zc / frameLen);
    }
    if (energies.isEmpty) return [0, 0, 0];
    final maxE = energies.reduce(math.max);
    final thresh = maxE * 0.02;
    final pauseRatio = energies.where((e) => e < thresh).length / energies.length;
    double variance(List<double> xs) {
      final mean = xs.reduce((a, b) => a + b) / xs.length;
      return xs.map((v) => (v - mean) * (v - mean)).reduce((a, b) => a + b) / xs.length;
    }

    return [pauseRatio, variance(energies), variance(zcrs)];
  }

  // ---- helpers ----

  static List<List<double>> _frameSignal(List<double> pcm) {
    final frames = <List<double>>[];
    for (int i = 0; i + fftSize <= pcm.length; i += hopLength) {
      frames.add(pcm.sublist(i, i + fftSize));
    }
    return frames;
  }

  static List<double> _hamming(List<double> frame) {
    return List.generate(frame.length, (n) {
      final w = 0.54 - 0.46 * math.cos(2 * math.pi * n / (frame.length - 1));
      return frame[n] * w;
    });
  }

  static List<double> _magnitudeSpectrum(List<double> frame) {
    final n = frame.length;
    final re = List<double>.from(frame);
    final im = List<double>.filled(n, 0);
    _fft(re, im);
    final mag = List<double>.filled(n ~/ 2 + 1, 0);
    for (int k = 0; k <= n ~/ 2; k++) {
      mag[k] = math.sqrt(re[k] * re[k] + im[k] * im[k]) / n;
    }
    return mag;
  }

  static void _fft(List<double> re, List<double> im) {
    final n = re.length;
    // Bit-reversal permutation
    int j = 0;
    for (int i = 1; i < n; i++) {
      int bit = n >> 1;
      while ((j & bit) != 0) {
        j ^= bit;
        bit >>= 1;
      }
      j ^= bit;
      if (i < j) {
        final tr = re[i]; re[i] = re[j]; re[j] = tr;
        final ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }
    // Cooley-Tukey
    for (int len = 2; len <= n; len <<= 1) {
      final ang = -2 * math.pi / len;
      final wlenRe = math.cos(ang), wlenIm = math.sin(ang);
      for (int i = 0; i < n; i += len) {
        double wRe = 1, wIm = 0;
        for (int k = 0; k < len ~/ 2; k++) {
          final uRe = re[i + k], uIm = im[i + k];
          final vRe = re[i + k + len ~/ 2] * wRe - im[i + k + len ~/ 2] * wIm;
          final vIm = re[i + k + len ~/ 2] * wIm + im[i + k + len ~/ 2] * wRe;
          re[i + k] = uRe + vRe; im[i + k] = uIm + vIm;
          re[i + k + len ~/ 2] = uRe - vRe; im[i + k + len ~/ 2] = uIm - vIm;
          final nwRe = wRe * wlenRe - wIm * wlenIm;
          final nwIm = wRe * wlenIm + wIm * wlenRe;
          wRe = nwRe; wIm = nwIm;
        }
      }
    }
  }

  static List<double> _linearFilterbank(List<double> mag) {
    // Uniform linear bins: simple downsample/average of magnitude spectrum
    // into nFilterBanks energies. For a 513-pt spectrum and 513 banks this
    // is identity; kept as averaging to satisfy the "linear filterbank" framing.
    if (mag.length == nFilterBanks) {
      return mag.map((v) => v * v + 1e-10).toList();
    }
    final out = List<double>.filled(nFilterBanks, 0);
    final ratio = mag.length / nFilterBanks;
    for (int i = 0; i < nFilterBanks; i++) {
      final start = (i * ratio).floor().clamp(0, mag.length - 1);
      final end = ((i + 1) * ratio).floor().clamp(0, mag.length);
      double s = 0;
      for (int k = start; k < end; k++) { s += mag[k] * mag[k]; }
      out[i] = s / (end - start).clamp(1, 9999) + 1e-10;
    }
    return out;
  }

  static List<double> _dct(List<double> x) {
    final n = x.length;
    final out = List<double>.filled(n, 0);
    for (int k = 0; k < n; k++) {
      double s = 0;
      for (int ni = 0; ni < n; ni++) {
        s += x[ni] * math.cos(math.pi * k * (2 * ni + 1) / (2 * n));
      }
      out[k] = s * math.sqrt(2 / n);
      if (k == 0) out[k] *= 1 / math.sqrt(2);
    }
    return out;
  }

  // --- Physiological voice-quality features (remediation track 2) ---
  static const int _pitchFrameLen = 480; // 30ms @ 16kHz
  static const int _pitchHop = 160; // 10ms @ 16kHz
  static const double _minF0 = 75.0;
  static const double _maxF0 = 500.0;
  static final int _minLag = (sampleRate / _maxF0).round(); // 32
  static final int _maxLag = (sampleRate / _minF0).round(); // 213
  static const double _voicingThreshold = 0.30;

  /// [periods(s), amplitudes, autocorrPeaks, voicedMask] — one entry per
  /// frame, mirrors features.py::_pitch_track exactly.
  static (List<double>, List<double>, List<double>, List<bool>) _pitchTrack(List<double> pcm) {
    final n = pcm.length;
    if (n < _pitchFrameLen) return (const [], const [], const [], const []);
    final nFrames = 1 + (n - _pitchFrameLen) ~/ _pitchHop;
    final periods = List<double>.filled(nFrames, 0);
    final amps = List<double>.filled(nFrames, 0);
    final peaks = List<double>.filled(nFrames, 0);
    final voiced = List<bool>.filled(nFrames, false);
    for (int i = 0; i < nFrames; i++) {
      final start = i * _pitchHop;
      final frame = pcm.sublist(start, start + _pitchFrameLen);
      double sumSq = 0;
      for (final v in frame) { sumSq += v * v; }
      amps[i] = math.sqrt(sumSq / _pitchFrameLen);
      if (sumSq <= 1e-12) continue;
      int bestLag = 0;
      double bestR = 0.0;
      final maxLagForFrame = math.min(_maxLag, _pitchFrameLen - 1);
      for (int lag = _minLag; lag <= maxLagForFrame; lag++) {
        double dotAB = 0, dotAA = 0, dotBB = 0;
        for (int k = 0; k < _pitchFrameLen - lag; k++) {
          final a = frame[k], b = frame[k + lag];
          dotAB += a * b; dotAA += a * a; dotBB += b * b;
        }
        final denom = math.sqrt(dotAA * dotBB);
        if (denom <= 1e-12) continue;
        final r = dotAB / denom;
        if (r > bestR) { bestR = r; bestLag = lag; }
      }
      peaks[i] = bestR;
      if (bestR >= _voicingThreshold && bestLag > 0) {
        voiced[i] = true;
        periods[i] = bestLag / sampleRate;
      }
    }
    return (periods, amps, peaks, voiced);
  }

  /// 3s PCM buffer -> [jitterLocal, shimmerLocal, hnrDb]. Mirrors
  /// features.py::extract_physio exactly — see that function's docstring
  /// for the algorithm and its deliberate simplifications.
  static List<double> extractPhysio(List<double> pcm) {
    final (periods, amps, peaks, voiced) = _pitchTrack(pcm);
    final voicedCount = voiced.where((v) => v).length;
    if (voicedCount < 2) return [0, 0, 0];

    final pairIdx = <int>[];
    for (int i = 0; i < voiced.length - 1; i++) {
      if (voiced[i] && voiced[i + 1]) pairIdx.add(i);
    }
    if (pairIdx.isEmpty) return [0, 0, 0];

    double sumAbsPeriodDiff = 0, sumMeanPeriod = 0;
    double sumAbsAmpDiff = 0, sumMeanAmp = 0;
    for (final i in pairIdx) {
      final p0 = periods[i], p1 = periods[i + 1];
      sumAbsPeriodDiff += (p1 - p0).abs();
      sumMeanPeriod += (p0 + p1) / 2;
      final a0 = amps[i], a1 = amps[i + 1];
      sumAbsAmpDiff += (a1 - a0).abs();
      sumMeanAmp += (a0 + a1) / 2;
    }
    final meanPeriod = sumMeanPeriod / pairIdx.length;
    final meanAmp = sumMeanAmp / pairIdx.length;
    final jitterLocal = meanPeriod > 1e-12 ? (sumAbsPeriodDiff / pairIdx.length) / meanPeriod : 0.0;
    final shimmerLocal = meanAmp > 1e-12 ? (sumAbsAmpDiff / pairIdx.length) / meanAmp : 0.0;

    double sumHnr = 0;
    for (int i = 0; i < voiced.length; i++) {
      if (!voiced[i]) continue;
      final r = peaks[i].clamp(0.0, 0.999999);
      sumHnr += 10 * math.log(r / (1 - r) + 1e-12) / math.ln10;
    }
    final hnrDb = sumHnr / voicedCount;

    return [jitterLocal, shimmerLocal, hnrDb];
  }

  /// Convert PCM16 bytes (little-endian) → normalized double [-1, 1].
  static List<double> pcm16ToDouble(List<int> bytes) {
    final out = <double>[];
    for (int i = 0; i + 1 < bytes.length; i += 2) {
      int v = bytes[i] | (bytes[i + 1] << 8);
      if (v >= 32768) v -= 65536;
      out.add(v / 32768.0);
    }
    return out;
  }
}