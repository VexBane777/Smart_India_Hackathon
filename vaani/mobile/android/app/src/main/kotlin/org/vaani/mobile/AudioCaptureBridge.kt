package org.vaani.mobile

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import io.flutter.plugin.common.EventChannel

/** Streams the phone's own mic as 0.5s (8000-sample) mono 16kHz PCM chunks
 *  over an EventChannel. Captures only the device's own microphone — this
 *  is legal and unrestricted, but for a VoIP call it captures whatever
 *  reaches the room mic, not the app's internal audio stream (spec §1). */
class AudioCaptureBridge : EventChannel.StreamHandler {
    companion object {
        const val SAMPLE_RATE = 16000
        const val CHUNK_SAMPLES = SAMPLE_RATE / 2 // 0.5s
    }

    private var recorder: AudioRecord? = null
    @Volatile private var running = false
    private var thread: Thread? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onListen(arguments: Any?, events: EventChannel.EventSink) {
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT
        )
        val bufSize = maxOf(minBuf, CHUNK_SAMPLES * 4)
        recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT, bufSize
        )
        recorder!!.startRecording()
        running = true
        thread = Thread {
            val buf = FloatArray(CHUNK_SAMPLES)
            while (running) {
                val read = recorder!!.read(buf, 0, CHUNK_SAMPLES, AudioRecord.READ_BLOCKING)
                if (read > 0) {
                    val chunk = if (read == CHUNK_SAMPLES) buf.toList() else buf.copyOf(read).toList()
                    // EventSink.success is @UiThread-only; this loop runs on a
                    // background Thread, so the call must be posted to the
                    // main Looper — calling it directly here crashed with
                    // "Methods marked with @UiThread must be executed on the
                    // main thread" on a real device (caught only by actually
                    // running this on hardware, not by unit tests or a build).
                    mainHandler.post { events.success(chunk) }
                }
            }
        }
        thread!!.start()
    }

    override fun onCancel(arguments: Any?) {
        running = false
        thread?.join(500)
        recorder?.stop()
        recorder?.release()
        recorder = null
    }
}
