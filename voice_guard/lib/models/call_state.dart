enum CallStatus { idle, ringing, active, disconnected }

class CallState {
  final CallStatus status;
  final String? number;
  final DateTime? startedAt;
  final Duration elapsed;

  const CallState({
    this.status = CallStatus.idle,
    this.number,
    this.startedAt,
    this.elapsed = Duration.zero,
  });

  CallState copyWith({
    CallStatus? status,
    String? number,
    DateTime? startedAt,
    Duration? elapsed,
  }) =>
      CallState(
        status: status ?? this.status,
        number: number ?? this.number,
        startedAt: startedAt ?? this.startedAt,
        elapsed: elapsed ?? this.elapsed,
      );

  bool get isActive => status == CallStatus.active;
  bool get isIdle => status == CallStatus.idle;
}
