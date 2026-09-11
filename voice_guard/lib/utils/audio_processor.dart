import 'dart:math' as math;
import 'dart:typed_data';

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

  /// Per-frame LFCC matrix (unpooled) — one inner list per frame, each
  /// length 60. Mirrors features.py::extract_lfcc_sequence exactly.
  static List<List<double>> extractLfccSequence(List<double> pcm) {
    if (pcm.length < fftSize) return [];
    final frames = _frameSignal(pcm);
    return frames.map((frame) {
      final windowed = _hamming(frame);
      final spectrum = _magnitudeSpectrum(windowed);
      final energies = _linearFilterbank(spectrum);
      final logE = energies.map((v) => math.log(v + 1e-10)).toList();
      return _dct(logE).sublist(0, nLfcc);
    }).toList();
  }

  /// [pauseRatio, energyVariance, zcrVariance, jitterLocal, shimmerLocal,
  /// hnrDb] — prosody then physio. Mirrors features.py::extract_scalars.
  static List<double> extractScalars(List<double> pcm) {
    return [...extractProsody(pcm), ...extractPhysio(pcm)];
  }

  /// 3s PCM16 buffer → 60 LFCC coefficients (mean-pooled over frames).
  static List<double> extractLfcc(List<double> pcm) {
    if (pcm.length < fftSize) return List.filled(nLfcc, 0);
    final lfccFrames = extractLfccSequence(pcm);

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

  /// Decodes a 16-bit PCM .wav (mono or stereo, arbitrary sample rate) into
  /// float samples in [-1, 1] plus the file's sample rate. Returns null for
  /// non-WAV / non-16-bit-PCM input instead of throwing, so the file-import
  /// scan path can show a friendly error. Handles the canonical 'fmt ' chunk
  /// (audioFormat == 1) and the WAVE_FORMAT_EXTENSIBLE variant (0xFFFE)
  /// whose subformat GUID is PCM.
  static ({List<double> samples, int sampleRate})? decodePcm16Wav(Uint8List bytes) {
    if (bytes.length < 44) return null;
    // 'RIFF' / 'WAVE'
    if (bytes[0] != 0x52 || bytes[1] != 0x49 || bytes[2] != 0x46 || bytes[3] != 0x46) return null;
    if (bytes[8] != 0x57 || bytes[9] != 0x41 || bytes[10] != 0x56 || bytes[11] != 0x45) return null;

    int? channels;
    int? sampleRate;
    int? bitsPerSample;
    int? dataOffset;
    int? dataLen;

    int pos = 12;
    while (pos + 8 <= bytes.length) {
      final id = _ascii(bytes, pos, 4);
      final size = _le32(bytes, pos + 4);
      if (id == 'fmt ') {
        if (size < 16) return null;
        int fmt = _le16(bytes, pos + 8);
        if (fmt == 0xFFFE) {
          // WAVE_FORMAT_EXTENSIBLE: the real format code is the first two
          // bytes of the subformat GUID at payload offset 24.
          if (pos + 8 + 24 + 2 > bytes.length) return null;
          fmt = _le16(bytes, pos + 8 + 24);
        }
        if (fmt != 1) return null; // uncompressed 16-bit PCM only
        channels = _le16(bytes, pos + 10);
        sampleRate = _le32(bytes, pos + 12);
        bitsPerSample = _le16(bytes, pos + 22);
        if (channels <= 0 || sampleRate <= 0 || bitsPerSample != 16) {
          return null;
        }
      } else if (id == 'data') {
        dataOffset = pos + 8;
        dataLen = size;
        break;
      }
      pos += 8 + size + (size & 1); // chunks are word-aligned
    }
    if (channels == null || sampleRate == null || dataOffset == null || dataLen == null) return null;

    final count = dataLen ~/ 2; // int16 samples, interleaved across channels
    final out = <double>[];
    if (channels == 1) {
      for (int i = 0; i < count; i++) {
        final o = dataOffset + i * 2;
        if (o + 1 >= bytes.length) break;
        out.add(_asInt16(bytes, o) / 32768.0);
      }
    } else {
      for (int i = 0; i + channels <= count; i += channels) {
        int acc = 0;
        for (int c = 0; c < channels; c++) {
          acc += _asInt16(bytes, dataOffset + (i + c) * 2);
        }
        out.add(acc / (channels * 32768.0));
      }
    }
    return (samples: out, sampleRate: sampleRate);
  }

  /// Linear-interpolation resample to `toRate`, mirroring
  /// vaani/app/server.py's `_load_mono_float` (np.interp), so arbitrary
  /// sample-rate WAVs (e.g. 44.1/48k phone recordings) land on the model's
  /// native 16 kHz.
  static List<double> resampleLinear(List<double> x, int fromRate, int toRate) {
    if (fromRate == toRate) return x;
    final n = x.length;
    if (n < 2) return x;
    final nOut = (n * toRate / fromRate).round();
    if (nOut <= 1) return [x[0]];
    final out = <double>[];
    final last = n - 1;
    for (int i = 0; i < nOut; i++) {
      final t = last * i / (nOut - 1);
      final i0 = t.floor();
      final frac = t - i0;
      final i1 = i0 < last ? i0 + 1 : last;
      out.add(x[i0] + (x[i1] - x[i0]) * frac);
    }
    return out;
  }

  /// Bundles both extraction passes into one `compute()`-friendly entry
  /// point. Must be a static method (not a closure) so it can be shipped to
  /// a background isolate — extractPhysio's pitch-tracking autocorrelation
  /// is O(frames * lagRange * frameLen) (tens of millions of ops per 3s
  /// chunk) and was blocking the UI isolate for 1-2.5s every scoring tick
  /// (seen as "Skipped N frames" / multi-second MotionEvent processing in
  /// logcat), freezing the app during calibration/Live Mic Test/calls.
  static (List<List<double>>, List<double>) extractFeaturesIsolate(List<double> pcm) {
    return (extractLfccSequence(pcm), extractScalars(pcm));
  }

  // ---- little-endian byte helpers (WAV decoding) ----

  static int _le16(Uint8List b, int o) => b[o] | (b[o + 1] << 8);

  static int _le32(Uint8List b, int o) =>
      b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24);

  static int _asInt16(Uint8List b, int o) {
    var v = _le16(b, o);
    if (v >= 32768) v -= 65536;
    return v;
  }

  static String _ascii(Uint8List b, int o, int n) =>
      String.fromCharCodes(b.sublist(o, o + n));
}