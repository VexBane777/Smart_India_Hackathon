package com.voiceguard.voice_guard

import android.telecom.Call
import android.telecom.InCallService
import android.util.Log

class InCallServiceImpl : InCallService() {
    companion object {
        private const val TAG = "InCallServiceImpl"
        private var activeCall: Call? = null

        fun endCurrentCall() {
            try {
                activeCall?.disconnect()
                activeCall = null
            } catch (e: Exception) {
                Log.e(TAG, "Failed to disconnect call", e)
            }
        }
    }

    private fun mapState(state: Int): String {
        return when (state) {
            Call.STATE_DIALING, Call.STATE_CONNECTING -> "dialing"
            Call.STATE_RINGING -> "incoming"
            Call.STATE_ACTIVE -> "active"
            Call.STATE_HOLDING -> "holding"
            Call.STATE_DISCONNECTED, Call.STATE_DISCONNECTING -> "disconnected"
            else -> "idle"
        }
    }

    private fun handleState(call: Call, state: Int) {
        val number = call.details.handle?.schemeSpecificPart ?: "Unknown"
        val status = mapState(state)
        Log.i(TAG, "Call state changed: $status ($state) for $number")
        MainActivity.notifyCallState(status, number)

        when (state) {
            Call.STATE_ACTIVE -> {
                Log.i(TAG, "Call active — starting audio capture")
                AudioCaptureManager.start(this)
            }
            Call.STATE_DISCONNECTED, Call.STATE_DISCONNECTING -> {
                Log.i(TAG, "Call disconnected — stopping audio capture")
                AudioCaptureManager.stop()
                if (activeCall == call) {
                    activeCall = null
                }
            }
            else -> {}
        }
    }

    private val callback = object : Call.Callback() {
        override fun onStateChanged(call: Call, state: Int) {
            handleState(call, state)
        }
    }

    override fun onCallAdded(call: Call) {
        super.onCallAdded(call)
        activeCall = call
        val number = call.details.handle?.schemeSpecificPart ?: "Unknown"
        Log.i(TAG, "onCallAdded: $number, initial state: ${call.state}")
        call.registerCallback(callback)
        handleState(call, call.state)
    }

    override fun onCallRemoved(call: Call) {
        super.onCallRemoved(call)
        Log.i(TAG, "onCallRemoved")
        call.unregisterCallback(callback)
        handleState(call, Call.STATE_DISCONNECTED)
        AudioCaptureManager.stop()
        if (activeCall == call) {
            activeCall = null
        }
    }
}

