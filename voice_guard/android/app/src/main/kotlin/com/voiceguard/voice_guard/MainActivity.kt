package com.voiceguard.voice_guard

import android.app.Activity
import android.app.role.RoleManager
import android.content.Intent
import android.media.projection.MediaProjectionManager
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
    private val webrtcAudioTapChannel = "com.voiceguard/webrtc_audio_tap"
    private val webrtcAudioTapEventChannel = "com.voiceguard/webrtc_audio_tap_stream"
    private var webrtcAudioTapSink: EventChannel.EventSink? = null
    private var pendingProjectionResult: MethodChannel.Result? = null
    private var pendingMediaProjection: android.media.projection.MediaProjection? = null
    private val PLAYBACK_CAPTURE_REQUEST_CODE = 2001

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        instance = this
        val mc = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, methodChannel)
        channel = mc
        mc.setMethodCallHandler { call, result ->
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
                    if (InCallServiceImpl.hasActiveCall()) {
                        // Real telecom call: route via InCallServiceImpl so the speaker
                        // gets forced (needed for the mic to overhear the far end) instead
                        // of poking AudioManager/AudioCaptureManager directly.
                        InCallServiceImpl.startDetection(this) { bytes ->
                            eventSink?.success(bytes)
                        }
                    } else {
                        // No real call (e.g. the Live Mic self-test) — no routing to touch.
                        AudioCaptureManager.start(this) { bytes ->
                            eventSink?.success(bytes)
                        }
                    }
                    result.success(null)
                }
                "stopCallDetection" -> {
                    if (InCallServiceImpl.hasActiveCall()) {
                        InCallServiceImpl.stopDetection()
                    } else {
                        AudioCaptureManager.stop()
                    }
                    result.success(null)
                }
                "getLastRecordingPath" -> {
                    result.success(AudioCaptureManager.lastRecordingPath)
                }
                "getCaptureStatus" -> {
                    result.success(mapOf(
                        "source" to AudioCaptureManager.activeSourceName,
                        "hasSignal" to AudioCaptureManager.hasRecentSignal
                    ))
                }
                "placeCall" -> {
                    val number = call.argument<String>("number") ?: ""
                    try {
                        val hasPerm = checkSelfPermission(android.Manifest.permission.CALL_PHONE) == android.content.pm.PackageManager.PERMISSION_GRANTED
                        val intent = if (hasPerm) {
                            Intent(Intent.ACTION_CALL, Uri.parse("tel:${Uri.encode(number)}")).apply {
                                flags = Intent.FLAG_ACTIVITY_NEW_TASK
                            }
                        } else {
                            Intent(Intent.ACTION_DIAL, Uri.parse("tel:${Uri.encode(number)}")).apply {
                                flags = Intent.FLAG_ACTIVITY_NEW_TASK
                            }
                        }
                        startActivity(intent)
                        result.success(true)
                    } catch (e: Exception) {
                        result.error("CALL_FAILED", e.message, null)
                    }
                }
                "endCall" -> {
                    InCallServiceImpl.endCurrentCall()
                    result.success(null)
                }
                "toggleSpeakerphone" -> {
                    val enable = call.argument<Boolean>("enable") ?: true
                    if (InCallServiceImpl.hasActiveCall()) {
                        // Real telecom call: must route via InCallService.setAudioRoute —
                        // AudioManager.isSpeakerphoneOn gets silently overridden by
                        // Telecom's own CallAudioRouteStateMachine otherwise.
                        InCallServiceImpl.setSpeakerphone(enable)
                        result.success(enable)
                    } else {
                        // No real call (e.g. the Live Mic self-test) — plain AudioManager works.
                        val am = getSystemService(android.media.AudioManager::class.java)
                        am.isSpeakerphoneOn = enable
                        result.success(am.isSpeakerphoneOn)
                    }
                }
                "toggleMicMute" -> {
                    val muted = call.argument<Boolean>("muted") ?: true
                    if (InCallServiceImpl.hasActiveCall()) {
                        InCallServiceImpl.setMuted(muted)
                        result.success(muted)
                    } else {
                        val am = getSystemService(android.media.AudioManager::class.java)
                        am.isMicrophoneMute = muted
                        result.success(am.isMicrophoneMute)
                    }
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
                "requestPlaybackCaptureConsent" -> {
                    if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
                        result.error("UNSUPPORTED", "Requires Android 10+", null)
                    } else {
                        val mpm = getSystemService(MediaProjectionManager::class.java)
                        pendingProjectionResult = result
                        startActivityForResult(mpm.createScreenCaptureIntent(), PLAYBACK_CAPTURE_REQUEST_CODE)
                    }
                }
                "startPlaybackCapture" -> {
                    val projection = pendingMediaProjection
                    if (projection == null) {
                        result.success(false)
                    } else {
                        val started = PlaybackCaptureManager.start(this, projection) { bytes ->
                            eventSink?.success(bytes)
                        }
                        result.success(started)
                    }
                }
                "stopPlaybackCapture" -> {
                    PlaybackCaptureManager.stop()
                    pendingMediaProjection?.stop()
                    pendingMediaProjection = null
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

        // Taps raw PCM off the remote party's WebRTC audio track for the
        // Protected Call scoring pipeline — see RemoteAudioTap for why this
        // exists (flutter_webrtc has no Dart-level RTCAudioSink equivalent).
        val webrtcPlugin = flutterEngine.plugins.get(
            com.cloudwebrtc.webrtc.FlutterWebRTCPlugin::class.java
        ) as? com.cloudwebrtc.webrtc.FlutterWebRTCPlugin

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, webrtcAudioTapChannel).setMethodCallHandler { call, result ->
            when (call.method) {
                "attach" -> {
                    val trackId = call.argument<String>("trackId")
                    if (trackId == null || webrtcPlugin == null) {
                        result.success(false)
                    } else {
                        val attached = RemoteAudioTap.attach(webrtcPlugin, trackId) { bytes ->
                            runOnUiThread { webrtcAudioTapSink?.success(bytes) }
                        }
                        result.success(attached)
                    }
                }
                "detach" -> {
                    RemoteAudioTap.detach()
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        }

        EventChannel(flutterEngine.dartExecutor.binaryMessenger, webrtcAudioTapEventChannel).setStreamHandler(
            object : EventChannel.StreamHandler {
                override fun onListen(args: Any?, sink: EventChannel.EventSink) {
                    webrtcAudioTapSink = sink
                }
                override fun onCancel(args: Any?) {
                    webrtcAudioTapSink = null
                }
            }
        )
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != PLAYBACK_CAPTURE_REQUEST_CODE) return
        val granted = resultCode == Activity.RESULT_OK && data != null
        if (granted) {
            val mpm = getSystemService(MediaProjectionManager::class.java)
            pendingMediaProjection = mpm.getMediaProjection(resultCode, data!!)
            startForegroundService(Intent(this, PlaybackCaptureForegroundService::class.java))
        }
        pendingProjectionResult?.success(granted)
        pendingProjectionResult = null
        channel?.invokeMethod("onPlaybackCaptureConsent", mapOf("granted" to granted))
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

    companion object {
        var instance: MainActivity? = null
        var channel: MethodChannel? = null

        fun notifyCallState(status: String, number: String?) {
            instance?.runOnUiThread {
                channel?.invokeMethod("onCallStateChanged", mapOf("status" to status, "number" to number))
            }
        }
    }

    override fun onDestroy() {
        PlaybackCaptureManager.stop()
        pendingMediaProjection?.stop()
        super.onDestroy()
        if (instance == this) {
            instance = null
            channel = null
        }
    }
}

