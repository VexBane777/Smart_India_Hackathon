package org.vaani.mobile

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // Mel spectrogram computation
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "vaani/mel")
            .setMethodCallHandler { call, result ->
                if (call.method == "computeMelDb") {
                    @Suppress("UNCHECKED_CAST")
                    val pcm = (call.argument<List<Double>>("pcm"))!!.map { it.toFloat() }.toFloatArray()
                    val sr = call.argument<Int>("sr")!!
                    val mel = MelBridge.computeMelDb(pcm, sr)
                    result.success(mel.map { it.toList() })
                } else {
                    result.notImplemented()
                }
            }

        // Audio file decoding
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "vaani/decode")
            .setMethodCallHandler { call, result ->
                if (call.method == "decodeFile") {
                    val path = call.argument<String>("path")!!
                    try {
                        val pcm = AudioDecodeBridge.decodeToPcm16kMono(path)
                        result.success(pcm.map { it.toDouble() })
                    } catch (e: Exception) {
                        result.error("decode_failed", e.message, null)
                    }
                } else {
                    result.notImplemented()
                }
            }

        // Audio playback with mute control (for file import + real-time testing)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "vaani/playback")
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "start" -> {
                        val success = AudioPlaybackBridge.start()
                        result.success(success)
                    }
                    "playChunk" -> {
                        @Suppress("UNCHECKED_CAST")
                        val samples = call.argument<List<Int>>("samples")!!
                        AudioPlaybackBridge.playChunk(samples)
                        result.success(null)
                    }
                    "setMuted" -> {
                        val mute = call.argument<Boolean>("muted")!!
                        AudioPlaybackBridge.setMuted(mute)
                        result.success(null)
                    }
                    "stop" -> {
                        AudioPlaybackBridge.stop()
                        result.success(null)
                    }
                    "isPlaying" -> {
                        result.success(AudioPlaybackBridge.isPlaying())
                    }
                    "isMuted" -> {
                        result.success(AudioPlaybackBridge.isMuted())
                    }
                    else -> result.notImplemented()
                }
            }

        // Mic capture
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, "vaani/mic")
            .setStreamHandler(AudioCaptureBridge())
    }
}
