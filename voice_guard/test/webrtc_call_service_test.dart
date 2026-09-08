import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/services/webrtc_call_service.dart';

void main() {
  test('caller role sends an offer-type signal first, callee does not', () {
    expect(WebRtcCallService.initialSignalType(isCaller: true), 'offer');
    expect(WebRtcCallService.initialSignalType(isCaller: false), null);
  });
}
