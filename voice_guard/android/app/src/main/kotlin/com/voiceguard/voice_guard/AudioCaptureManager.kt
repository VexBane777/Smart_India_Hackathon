package com.voiceguard.voice_guard

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * AudioCaptureManager — Captures 16kHz 16-bit mono PCM for on-device scoring + WAV recording.
 *
 * Source cascade (see module docstring in call_service.dart for the platform rationale):
 * a regular app cannot read live cellular call audio once the telephony HAL owns the mic
 * for an active call — VOICE_COMMUNICATION's hardware AEC actively cancels the acoustic
 * speaker-to-mic loop the app depends on for hearing the far end, and on most stock builds
 * AudioRecord reads silently zero-fill instead of erroring once the real call audio path
 * takes over. So capture is attempted in this order and the first source that actually
 * initializes is used:
 *   1. VOICE_CALL     — only a handful of OEM builds honor this for non-privileged apps,
 *                       but costs nothing to try first since it needs no acoustic coupling.
 *   2. VOICE_DOWNLINK / VOICE_UPLINK — legacy 2G/3G-era raw call-leg taps (far end / near
 *                       end respectively). Deprecated and CAPTURE_AUDIO_OUTPUT-gated on most
 *                       modern builds, so construction almost always fails outright rather
 *                       than silently zero-filling — cheap to try, and on the rare OEM build
 *                       that still honors them for a non-privileged app, they're a real tap
 *                       into the call audio rather than relying on acoustic speaker-to-mic
 *                       coupling at all.
 *   3. VOICE_RECOGNITION — AGC/noise-suppression are specified OFF for this source, which
 *                       matters for feeding a classifier: MIC's default DSP chain would
 *                       otherwise flatten exactly the spectral/prosodic cues LFCC extraction
 *                       depends on. Relies on speakerphone being on for the far end to reach it.
 *   4. MIC            — universal fallback if VOICE_RECOGNITION tuning isn't available.
 *
 * Important caveat proven out in testing (2026-09-08): "initializes" is necessary but not
 * sufficient — VOICE_CALL/VOICE_DOWNLINK/VOICE_UPLINK have been observed to report
 * STATE_INITIALIZED while every subsequent read() is silently zero-filled for the entire
 * call, which openRecorder() cannot detect at construction time. hasRecentSignal (silence
 * tracking below) and the Dart-side per-window RMS gate are what actually catch this at
 * runtime; a source higher in this cascade is not guaranteed to be better than one lower.
 *
 * Test-device note (2026-09-08, an "Oplus"/CPH2613-family build, Android 16 / API 36):
 * VOICE_CALL, VOICE_DOWNLINK, and VOICE_UPLINK all failed AudioRecord construction outright
 * (falls through to VOICE_RECOGNITION every time — check logcat for
 * "AudioCaptureManager: Using audio source: ..." to see which one won on a given device).
 * If you're bringing this up on a new OEM/Android version and want to know whether any of
 * the three privileged sources are usable there: that log line is step one, and even if one
 * of them wins, don't stop there — per the caveat above, confirm real (non-zero) RMS is
 * actually arriving during a live call too, since "won the cascade" and "delivers real call
 * audio" are two different, independently-failing things.
 * Real Call Recording: Writes PCM chunks to a standard 44-byte WAV file in the app's
 * external files directory so calls are permanently recorded and playable.
 * Real-Time Stream: Pushes PCM chunks to Flutter via EventChannel for on-device TFLite inference.
 */
object AudioCaptureManager {
    private const val TAG = "AudioCaptureManager"
    private const val SAMPLE_RATE = 16000
    private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
    private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT

    // Ordered by preference; see cascade rationale above.
    @Suppress("DEPRECATION")
    private val SOURCE_CASCADE = listOf(
        "VOICE_CALL" to MediaRecorder.AudioSource.VOICE_CALL,
        "VOICE_DOWNLINK" to MediaRecorder.AudioSource.VOICE_DOWNLINK,
        "VOICE_UPLINK" to MediaRecorder.AudioSource.VOICE_UPLINK,
        "VOICE_RECOGNITION" to MediaRecorder.AudioSource.VOICE_RECOGNITION,
        "MIC" to MediaRecorder.AudioSource.MIC,
    )

    private var recorder: AudioRecord? = null
    private var thread: Thread? = null
    @Volatile private var running = false
    private var sink: ((ByteArray) -> Unit)? = null
    // Flutter's EventChannel.EventSink must only be invoked from the platform
    // (main) thread; the capture loop runs on a background Thread, so every
    // sink call is hopped back to the main looper before it fires.
    private val mainHandler = Handler(Looper.getMainLooper())

    var lastRecordingPath: String? = null
        private set
    var activeSourceName: String? = null
        private set

    // Rolling capture-quality signal: true once any recent chunk had real signal.
    // Lets the UI distinguish "genuinely quiet call" from "OS silently zero-filled
    // our reads" instead of presenting both as the same flat low-risk score.
    @Volatile var hasRecentSignal: Boolean = false
        private set
    private var silentReadStreak = 0
    private const val SILENCE_RMS_THRESHOLD = 50.0
    private const val SILENT_STREAK_TO_FLAG = 20 // ~2s at 100ms/read

    private var currentRecordingFile: File? = null
    private var recordingStream: FileOutputStream? = null
    private var totalBytesRecorded = 0L

    fun setSink(s: ((ByteArray) -> Unit)?) { sink = s }

    private fun openRecorder(bufSize: Int): Boolean {
        for ((name, source) in SOURCE_CASCADE) {
            val candidate = try {
                AudioRecord(source, SAMPLE_RATE, CHANNEL, ENCODING, bufSize)
            } catch (e: Exception) {
                Log.w(TAG, "Source $name threw on construction: ${e.message}")
                null
            }
            if (candidate?.state == AudioRecord.STATE_INITIALIZED) {
                Log.i(TAG, "Using audio source: $name")
                recorder = candidate
                activeSourceName = name
                return true
            }
            candidate?.release()
        }
        return false
    }

    fun start(context: Context, onBytes: ((ByteArray) -> Unit)? = null) {
        if (onBytes != null) {
            sink = onBytes
        }
        if (running) return
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        if (minBuf <= 0) { Log.e(TAG, "Invalid min buffer size: $minBuf"); return }
        val bufSize = minBuf * 4
        hasRecentSignal = false
        silentReadStreak = 0
        if (!openRecorder(bufSize)) {
            Log.e(TAG, "No audio source could be initialized"); return
        }

        // Initialize Call Recording WAV File
        try {
            val recDir = File(context.getExternalFilesDir(null), "Recordings").apply { mkdirs() }
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val recFile = File(recDir, "call_${timeStamp}.wav")
            currentRecordingFile = recFile
            totalBytesRecorded = 0L
            val fos = FileOutputStream(recFile)
            fos.write(ByteArray(44)) // placeholder for WAV header
            recordingStream = fos
            Log.i(TAG, "Call recording started: ${recFile.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize call recording file", e)
        }

        recorder?.startRecording()
        running = true
        thread = Thread {
            val buf = ByteArray(3200) // 100ms @16kHz mono 16-bit
            while (running) {
                val n = recorder?.read(buf, 0, buf.size) ?: 0
                if (n > 0) {
                    val copy = buf.copyOf(n)
                    // Track whether the OS is actually handing us signal or silently
                    // zero-filling reads (see class docstring) so the UI can tell
                    // "quiet call" apart from "capture source lost the call audio".
                    var sumSq = 0.0
                    var i = 0
                    while (i + 1 < n) {
                        val sample = ((copy[i + 1].toInt() shl 8) or (copy[i].toInt() and 0xFF)).toShort().toDouble()
                        sumSq += sample * sample
                        i += 2
                    }
                    val sampleCount = n / 2
                    val rms = if (sampleCount > 0) Math.sqrt(sumSq / sampleCount) else 0.0
                    if (rms < SILENCE_RMS_THRESHOLD) {
                        silentReadStreak++
                        if (silentReadStreak >= SILENT_STREAK_TO_FLAG) hasRecentSignal = false
                    } else {
                        silentReadStreak = 0
                        hasRecentSignal = true
                    }
                    // 1. Write audio chunk to WAV file on disk
                    try {
                        recordingStream?.write(buf, 0, n)
                        totalBytesRecorded += n
                    } catch (e: Exception) {
                        Log.w(TAG, "Error writing audio chunk to recording", e)
                    }
                    // 2. Stream chunk to Flutter for real-time model scoring & waveform
                    val cb = sink
                    if (cb != null) {
                        mainHandler.post {
                            try { cb.invoke(copy) } catch (_: Exception) {}
                        }
                    }
                }
            }
        }.also { it.isDaemon = true; it.start() }
        Log.i(TAG, "Audio capture and call recording started (16kHz mono)")
    }

    fun stop() {
        // Telecom fires onStateChanged(DISCONNECTING), onStateChanged(DISCONNECTED)
        // and onCallRemoved for a single hangup, each of which calls this — make
        // repeat calls a no-op instead of re-finalizing/re-saving the same WAV.
        if (recorder == null && recordingStream == null) return
        running = false
        thread?.interrupt()
        thread = null
        try { recorder?.stop() } catch (_: Exception) {}
        try { recorder?.release() } catch (_: Exception) {}
        recorder = null
        activeSourceName = null
        hasRecentSignal = false

        // Finalize WAV Header on disk
        try {
            recordingStream?.flush()
            recordingStream?.close()
            recordingStream = null

            val recFile = currentRecordingFile
            if (recFile != null && recFile.exists() && totalBytesRecorded > 0) {
                RandomAccessFile(recFile, "rw").use { raf ->
                    writeWavHeader(
                        raf,
                        totalBytesRecorded,
                        totalBytesRecorded + 36,
                        SAMPLE_RATE.toLong(),
                        1,
                        (SAMPLE_RATE * 2).toLong()
                    )
                }
                lastRecordingPath = recFile.absolutePath
                Log.i(TAG, "Saved call recording to: ${recFile.absolutePath} ($totalBytesRecorded bytes)")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error finalizing WAV file", e)
        }

        Log.i(TAG, "Audio capture stopped")
    }

    private fun writeWavHeader(
        out: RandomAccessFile,
        totalAudioLen: Long,
        totalDataLen: Long,
        sampleRate: Long,
        channels: Int,
        byteRate: Long
    ) {
        val header = ByteArray(44)
        header[0] = 'R'.code.toByte() // RIFF/WAVE header
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte() // 'fmt ' chunk
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1 // format = 1 (PCM)
        header[21] = 0
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (sampleRate and 0xff).toByte()
        header[25] = ((sampleRate shr 8) and 0xff).toByte()
        header[26] = ((sampleRate shr 16) and 0xff).toByte()
        header[27] = ((sampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = (channels * 2).toByte() // block align
        header[33] = 0
        header[34] = 16 // bits per sample
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
        header[43] = ((totalAudioLen shr 24) and 0xff).toByte()
        out.seek(0)
        out.write(header, 0, 44)
    }
}
