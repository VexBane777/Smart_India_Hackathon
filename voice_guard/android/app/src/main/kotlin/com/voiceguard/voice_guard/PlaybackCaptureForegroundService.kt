package com.voiceguard.voice_guard

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder

/**
 * Exists only to satisfy Android 10+'s requirement that AudioPlaybackCapture
 * run inside a foreground service. All real capture logic lives in
 * PlaybackCaptureManager; this class just keeps the process alive and shows
 * the mandatory "VoiceGuard is monitoring a call" notification.
 */
class PlaybackCaptureForegroundService : Service() {
    companion object {
        private const val CHANNEL_ID = "playback_capture_channel"
        private const val NOTIFICATION_ID = 4201
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "VoIP Call Protection", NotificationManager.IMPORTANCE_LOW)
            )
        }
        val notification: Notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("VoiceGuard is monitoring this call")
            .setContentText("Analyzing the caller's voice for AI cloning risk")
            .setSmallIcon(android.R.drawable.stat_notify_call_mute)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    override fun onDestroy() {
        PlaybackCaptureManager.stop()
        super.onDestroy()
    }
}
