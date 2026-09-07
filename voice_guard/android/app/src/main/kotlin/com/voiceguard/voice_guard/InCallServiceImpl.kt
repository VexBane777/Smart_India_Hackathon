package com.voiceguard.voice_guard

import android.telecom.Call
import android.telecom.InCallService
import android.util.Log

/**
 * Intercepts telecom calls. On STATE_ACTIVE starts AudioCaptureManager;
 * on STATE_DISCONNECTED / onCallRemoved stops it.
 * Real call audio via VOICE_CALL is blocked on Android 10+ for non-system apps,
 * so this POC captures via MIC (caller must use speakerphone for demo).
 */
class InCallServiceImpl : InCallService() {
    companion object { private const val TAG = "InCallServiceImpl" }

    private val callback = object : Call.Callback() {
        override fun onStateChanged(call: Call, state: Int) {
            when (state) {
                Call.STATE_ACTIVE -> {
                    Log.i(TAG, "Call active — starting capture")
                    AudioCaptureManager.start(this@InCallServiceImpl) { _ -> }
                }
                Call.STATE_DISCONNECTED, Call.STATE_DISCONNECTING -> {
                    Log.i(TAG, "Call ended — stopping capture")
                    AudioCaptureManager.stop()
                }
                else -> {}
            }
        }
    }

    override fun onCallAdded(call: Call) {
        super.onCallAdded(call)
        Log.i(TAG, "onCallAdded: ${call.details.handle}")
        call.registerCallback(callback)
        if (call.state == Call.STATE_ACTIVE) {
            AudioCaptureManager.start(this) { _ -> }
        }
    }

    override fun onCallRemoved(call: Call) {
        super.onCallRemoved(call)
        Log.i(TAG, "onCallRemoved")
        call.unregisterCallback(callback)
        AudioCaptureManager.stop()
    }
}
