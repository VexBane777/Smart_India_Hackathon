import 'call_service.dart';

class OverlayService {
  final CallService _calls;
  OverlayService(this._calls);

  Future<void> show(double score, String verdict) =>
      _calls.showOverlay(riskScore: score, verdict: verdict);

  Future<void> hide() => _calls.hideOverlay();
}
