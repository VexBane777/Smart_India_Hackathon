import 'package:flutter/material.dart';

/// Controls for audio playback during analysis: mute/unmute toggle and
/// visual status indicator.
///
/// Shows:
/// - Whether audio is currently playing/processing
/// - Mute state (with toggle button)
/// - Visual indicator for playback status
class PlaybackControl extends StatelessWidget {
  const PlaybackControl({
    super.key,
    required this.isPlaying,
    required this.isMuted,
    required this.isProcessing,
    required this.onMuteToggle,
  });

  final bool isPlaying;
  final bool isMuted;
  final bool isProcessing;
  final VoidCallback onMuteToggle;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF1E293B),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isMuted ? Colors.orange : Color(0xFF334155),
          width: 2,
        ),
      ),
      child: Row(
        children: [
          // Status indicator
          Icon(
            isProcessing
                ? (isMuted ? Icons.volume_off : Icons.volume_up)
                : Icons.check_circle_outline,
            color: isProcessing
                ? (isMuted ? Colors.orange : Colors.green)
                : Colors.grey,
            size: 28,
          ),
          const SizedBox(width: 12),

          // Status text
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  isProcessing ? 'Playing audio...' : 'Ready',
                  style: TextStyle(
                    color: isProcessing ? Colors.white : Colors.grey,
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                if (isPlaying) ...[
                  const SizedBox(height: 2),
                  Text(
                    isMuted
                        ? 'Muted — scoring continues'
                        : 'Playing through speaker',
                    style: TextStyle(
                      color: isMuted ? Colors.orange : Colors.green,
                      fontSize: 12,
                    ),
                  ),
                ],
              ],
            ),
          ),

          // Mute toggle button
          if (isPlaying)
            IconButton(
              onPressed: onMuteToggle,
              icon: Icon(
                isMuted ? Icons.volume_up : Icons.volume_off,
                color: isMuted ? Colors.orange : Colors.white,
                size: 32,
              ),
              tooltip: isMuted ? 'Unmute' : 'Mute',
            ),
        ],
      ),
    );
  }
}
