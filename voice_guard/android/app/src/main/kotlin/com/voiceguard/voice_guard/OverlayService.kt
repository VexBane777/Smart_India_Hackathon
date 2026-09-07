package com.voiceguard.voice_guard

import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.os.Build
import android.provider.Settings
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import android.app.Service
import android.os.IBinder

/**
 * System overlay warning (TYPE_APPLICATION_OVERLAY, FLAG_NOT_FOCUSABLE, TOP gravity).
 * Shows risk score + verdict + recommended action. Triggered from Flutter when
 * risk exceeds threshold; auto-dismisses after 5s.
 */
class OverlayService : Service() {
    companion object {
        private var overlayView: View? = null
        private var windowManager: WindowManager? = null

        fun show(context: Context, riskScore: Double, verdict: String) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(context)) return
            hide(context)
            val wm = context.getSystemService(Context.WINDOW_SERVICE) as WindowManager
            windowManager = wm
            val view = LayoutInflater.from(context).inflate(R.layout.overlay_warning, null)
            view.findViewById<TextView>(R.id.overlayScore).text = "${(riskScore * 100).toInt()}% RISK"
            view.findViewById<TextView>(R.id.overlayVerdict).text = verdict
            view.findViewById<View>(R.id.overlayDismiss).setOnClickListener { hide(context) }
            // auto-dismiss after 5s
            view.postDelayed({ hide(context) }, 5000)
            val params = WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.WRAP_CONTENT,
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY else WindowManager.LayoutParams.TYPE_PHONE,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
                PixelFormat.TRANSLUCENT
            ).apply { gravity = Gravity.TOP; y = 0 }
            try { wm.addView(view, params); overlayView = view } catch (_: Exception) {}
        }

        fun hide(context: Context) {
            overlayView?.let {
                try { (context.getSystemService(Context.WINDOW_SERVICE) as WindowManager).removeView(it) } catch (_: Exception) {}
                try { windowManager?.removeView(it) } catch (_: Exception) {}
            }
            overlayView = null
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val score = intent?.getDoubleExtra("riskScore", 0.0) ?: 0.0
        val verdict = intent?.getStringExtra("verdict") ?: "SUSPICIOUS"
        show(this, score, verdict)
        return START_NOT_STICKY
    }
}
