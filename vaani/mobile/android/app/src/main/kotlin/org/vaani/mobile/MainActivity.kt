package org.vaani.mobile

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "vaani/mel")
            .setMethodCallHandler { call, result ->
                if (call.method == "computeMelDb") {
                    @Suppress("UNCHECKED_CAST")
                    val pcm = (call.argument<List<Double>>("pcm"))!!
                        .map { it.toFloat() }.toFloatArray()
                    val sr = call.argument<Int>("sr")!!
                    val mel = MelBridge.computeMelDb(pcm, sr)
                    result.success(mel.map { it.toList() })
                } else {
                    result.notImplemented()
                }
            }
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
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, "vaani/mic")
            .setStreamHandler(AudioCaptureBridge())
    }
}
