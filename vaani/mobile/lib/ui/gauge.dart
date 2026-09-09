import 'package:flutter/material.dart';
import '../decision/decision_engine.dart';

const _stateColors = {
  AlertState.normal: Color(0xFF22C55E),
  AlertState.warn: Color(0xFFF59E0B),
  AlertState.alert: Color(0xFFEF4444),
};
const _stateLabels = {
  AlertState.normal: 'NORMAL',
  AlertState.warn: 'WATCH',
  AlertState.alert: 'ALERT — 2+ consecutive windows above threshold',
};

class RiskGauge extends StatelessWidget {
  const RiskGauge({
    super.key,
    required this.ema,
    required this.state,
    this.raw,
    required this.backendLabel,
  });

  final double? ema;
  final AlertState state;
  final double? raw;
  final String backendLabel;

  @override
  Widget build(BuildContext context) {
    final score = ema ?? 0.0;
    final color = _stateColors[state]!;
    final pct = (score * 100).round();
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      decoration: BoxDecoration(
        border: Border.all(color: color, width: 2),
        borderRadius: BorderRadius.circular(16),
        color: const Color(0xFF0F172A),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Flexible(
                child: Text('Synthetic-voice risk',
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(color: Color(0xFFE2E8F0), fontWeight: FontWeight.w600)),
              ),
              const SizedBox(width: 8),
              Flexible(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
                  decoration: BoxDecoration(
                    border: Border.all(color: const Color(0xFF475569)),
                    borderRadius: BorderRadius.circular(999),
                  ),
                  child: Text('backend: $backendLabel',
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 12)),
                ),
              ),
            ],
          ),
          Text('$pct%',
              style: TextStyle(color: color, fontSize: 48, fontWeight: FontWeight.w800)),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: pct / 100.0,
              minHeight: 14,
              backgroundColor: const Color(0xFF1E293B),
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
          const SizedBox(height: 8),
          Text(_stateLabels[state]!, style: TextStyle(color: color, fontWeight: FontWeight.w600)),
          if (raw != null)
            Text('raw window: ${raw!.toStringAsFixed(2)}',
                style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 13)),
        ],
      ),
    );
  }
}
