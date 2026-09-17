package com.voiceguard.voice_guard

import com.cloudwebrtc.webrtc.FlutterWebRTCPlugin
import org.webrtc.AudioTrack
import org.webrtc.AudioTrackSink

/**
 * Taps raw PCM off a WebRTC audio track for VoiceGuard's "Protected Call"
 * scoring pipeline (docs/superpowers/plans/2026-09-08-voice-guard-self-owned-voip-call.md,
 * Task 4).
 *
 * flutter_webrtc's own Dart/Java API has no equivalent of the plan's assumed
 * `RTCAudioSink` — confirmed absent across the plugin's entire published
 * history. But the native org.webrtc.AudioTrack it wraps (from
 * io.github.webrtc-sdk:android, bundled by the plugin) exposes a real,
 * public, JNI-backed addSink(AudioTrackSink)/removeSink pair that works for
 * any AudioTrack regardless of direction — flutter_webrtc's own
 * LocalAudioTrack.java uses this identical native mechanism for local
 * capture. This reaches the same native capability for the *remote* track
 * via the plugin's public getRemoteTrack(trackId) accessor — no reflection
 * into plugin internals, unlike flutter_webrtc's own package-private
 * WebRtcAudioTrackUtils hack used for its MediaRecorder feature.
 *
 * The raw buffers this sink receives are whatever rate/channel-count the
 * remote track happens to be running at (typically 48kHz, possibly
 * stereo) — not the 16kHz mono PCM16 the scoring pipeline expects. Each
 * tap now routes through [AudioResampler16k] before [onData] fires, so
 * callers always get 100ms/3200-byte 16kHz mono chunks regardless of the
 * source format.
 */
object RemoteAudioTap {
    private var sink: AudioTrackSink? = null
    private var track: AudioTrack? = null

    /** Returns true if a remote audio track with [trackId] was found and tapped. */
    fun attach(plugin: FlutterWebRTCPlugin, trackId: String, onData: (ByteArray) -> Unit): Boolean {
        val remoteTrack = plugin.getRemoteTrack(trackId) as? AudioTrack ?: return false
        detach()
        val resampler = AudioResampler16k(onChunkReady = onData)
        val newSink = AudioTrackSink { buffer, _, sampleRate, numberOfChannels, numberOfFrames, _ ->
            resampler.processAudio(buffer, sampleRate, numberOfChannels, numberOfFrames)
        }
        remoteTrack.addSink(newSink)
        sink = newSink
        track = remoteTrack
        return true
    }

    fun detach() {
        val currentSink = sink
        val currentTrack = track
        if (currentSink != null && currentTrack != null) {
            currentTrack.removeSink(currentSink)
        }
        sink = null
        track = null
    }
}
