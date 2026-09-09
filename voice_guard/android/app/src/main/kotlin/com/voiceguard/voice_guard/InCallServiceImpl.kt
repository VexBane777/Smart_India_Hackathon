package com.voiceguard.voice_guard

import android.media.AudioManager
import android.telecom.Call
import android.telecom.CallAudioState
import android.telecom.InCallService
import android.telephony.TelephonyManager
import android.util.Log

class InCallServiceImpl : InCallService() {
    companion object {
        private const val TAG = "InCallServiceImpl"
        private var activeCall: Call? = null

        // The bound InCallService instance, when a real telecom call is in progress.
        // Telecom-managed calls own their own CallAudioRouteStateMachine — poking
        // AudioManager.isSpeakerphoneOn directly gets silently overridden by it, so
        // speaker/mute must go through InCallService's own setAudioRoute/setMuted.
        private var instance: InCallServiceImpl? = null

        fun endCurrentCall() {
            try {
                activeCall?.disconnect()
                activeCall = null
            } catch (e: Exception) {
                Log.e(TAG, "Failed to disconnect call", e)
            }
        }

        /** Returns true if a real telecom call is active and the route call was made. */
        fun setSpeakerphone(enable: Boolean): Boolean {
            val svc = instance ?: return false
            return try {
                svc.setAudioRoute(if (enable) CallAudioState.ROUTE_SPEAKER else CallAudioState.ROUTE_EARPIECE)
                true
            } catch (e: Exception) {
                Log.e(TAG, "setAudioRoute failed", e)
                false
            }
        }

        /** Returns true if a real telecom call is active and the mute call was made. */
        fun setMuted(muted: Boolean): Boolean {
            val svc = instance ?: return false
            return try {
                svc.setMuted(muted)
                true
            } catch (e: Exception) {
                Log.e(TAG, "setMuted failed", e)
                false
            }
        }

        fun hasActiveCall(): Boolean = activeCall != null

        // state.md "Ideas not yet tried" -> "VoLTE vs. legacy circuit-switched
        // call": human-readable network-type name so a logcat grep can tell
        // which call type root cause #1 (silent zero-fill) was reproduced on,
        // without needing a second device/tool to decode the raw int.
        private fun networkTypeName(type: Int?): String = when (type) {
            null -> "unknown(no-permission-or-error)"
            TelephonyManager.NETWORK_TYPE_LTE -> "LTE(13, likely VoLTE if HD)"
            TelephonyManager.NETWORK_TYPE_NR -> "NR(20, 5G, likely VoNR if HD)"
            TelephonyManager.NETWORK_TYPE_UMTS -> "UMTS(3, 3G circuit-switched)"
            TelephonyManager.NETWORK_TYPE_HSPA -> "HSPA(15, 3G circuit-switched)"
            TelephonyManager.NETWORK_TYPE_HSPAP -> "HSPA+(17, 3G circuit-switched)"
            TelephonyManager.NETWORK_TYPE_GSM -> "GSM(16, 2G circuit-switched)"
            TelephonyManager.NETWORK_TYPE_EDGE -> "EDGE(2, 2G circuit-switched)"
            TelephonyManager.NETWORK_TYPE_UNKNOWN -> "UNKNOWN(0)"
            else -> "type=$type (see TelephonyManager.NETWORK_TYPE_* for meaning)"
        }
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    override fun onDestroy() {
        if (instance == this) instance = null
        super.onDestroy()
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
                logCallAudioDiagnostics(call)
                // MODE_IN_COMMUNICATION: some OEMs require this for any non-VOICE_CALL
                // source to keep receiving real samples once the carrier call audio path
                // takes over. See AudioCaptureManager's docstring.
                val am = getSystemService(AudioManager::class.java)
                am?.mode = AudioManager.MODE_IN_COMMUNICATION
                // Route via Telecom, not AudioManager directly (see setSpeakerphone doc) —
                // without speaker, the far end's voice never physically reaches the mic.
                setSpeakerphone(true)
                AudioCaptureManager.start(this)
            }
            Call.STATE_DISCONNECTED, Call.STATE_DISCONNECTING -> {
                Log.i(TAG, "Call disconnected — stopping audio capture")
                AudioCaptureManager.stop()
                val am = getSystemService(AudioManager::class.java)
                am?.mode = AudioManager.MODE_NORMAL
                if (activeCall == call) {
                    activeCall = null
                }
            }
            else -> {}
        }
    }

    // Best-effort diagnostic: logs which call-audio path this call is riding
    // (VoLTE/VoNR/CS-fallback and HD/WiFi flags) so a captured-silent-call
    // logcat can be correlated against network type without extra tooling.
    // See state.md "Ideas not yet tried" -> "VoLTE vs. legacy circuit-switched
    // call". Read-only, never affects call handling — any failure here must
    // not disrupt STATE_ACTIVE's actual job of starting audio capture.
    private fun logCallAudioDiagnostics(call: Call) {
        try {
            val tm = getSystemService(TelephonyManager::class.java)
            @Suppress("DEPRECATION")
            val networkType = try { tm?.networkType } catch (e: SecurityException) { null }
            val props = call.details.callProperties
            val highDef = call.details.hasProperty(Call.Details.PROPERTY_HIGH_DEF_AUDIO)
            val wifi = call.details.hasProperty(Call.Details.PROPERTY_WIFI)
            Log.i(
                TAG,
                "Call audio diagnostics: networkType=${networkTypeName(networkType)} " +
                    "highDefAudio=$highDef wifiCall=$wifi rawProperties=$props"
            )
        } catch (e: Exception) {
            Log.w(TAG, "logCallAudioDiagnostics failed (non-fatal)", e)
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
        if (activeCall == call) {
            activeCall = null
        }
    }
}

