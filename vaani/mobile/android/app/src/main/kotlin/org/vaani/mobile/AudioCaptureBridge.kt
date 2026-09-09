package org.vaani.mobile

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
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
                    events.success(chunk)
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
