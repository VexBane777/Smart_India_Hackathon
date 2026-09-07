package com.voiceguard.voice_guard

import android.app.role.RoleManager
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.provider.Settings
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    private val methodChannel = "com.voiceguard/calls"
    private val eventChannel = "com.voiceguard/audio_stream"
    private var eventSink: EventChannel.EventSink? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, methodChannel).setMethodCallHandler { call, result ->
            when (call.method) {
                "isDefaultDialer" -> {
                    result.success(isDefaultDialer())
                }
                "setAsDefaultDialer" -> {
                    requestDefaultDialer()
                    result.success(null)
                }
                "hasOverlayPermission" -> {
                    result.success(Settings.canDrawOverlays(this))
                }
                "requestOverlayPermission" -> {
                    val intent = Intent(
                        Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:$packageName")
                    )
                    startActivity(intent)
                    result.success(null)
                }
                "startCallDetection" -> {
                    AudioCaptureManager.start(this) { bytes ->
                        eventSink?.success(bytes)
                    }
                    result.success(null)
                }
                "stopCallDetection" -> {
                    AudioCaptureManager.stop()
                    result.success(null)
                }
                "showOverlay" -> {
                    val score = (call.argument<Double>("riskScore") ?: 0.0)
                    val verdict = call.argument<String>("verdict") ?: "SUSPICIOUS"
                    OverlayService.show(this, score, verdict)
                    result.success(null)
                }
                "hideOverlay" -> {
                    OverlayService.hide(this)
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        }

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, eventChannel).setStreamHandler(
            object : EventChannel.StreamHandler {
                override fun onListen(args: Any?, sink: EventChannel.EventSink) {
                    eventSink = sink
                    // Forward events from AudioCaptureManager if already running
                    AudioCaptureManager.setSink { bytes -> sink.success(bytes) }
                }
                override fun onCancel(args: Any?) {
                    eventSink = null
                    AudioCaptureManager.setSink(null)
                }
            }
        )
    }

    private fun isDefaultDialer(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val rm = getSystemService(RoleManager::class.java)
            rm.isRoleHeld(RoleManager.ROLE_DIALER)
        } else {
            val tm = getSystemService(android.telecom.TelecomManager::class.java)
            packageName == tm.defaultDialerPackage
        }
    }

    private fun requestDefaultDialer() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val rm = getSystemService(RoleManager::class.java)
            val intent = rm.createRequestRoleIntent(RoleManager.ROLE_DIALER)
            startActivityForResult(intent, 1001)
        } else {
            val intent = Intent(android.telecom.TelecomManager.ACTION_CHANGE_DEFAULT_DIALER)
            intent.putExtra(android.telecom.TelecomManager.EXTRA_CHANGE_DEFAULT_DIALER_PACKAGE_NAME, packageName)
            startActivity(intent)
        }
    }
}
