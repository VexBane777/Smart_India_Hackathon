import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/decision/decision_engine.dart';

void main() {
  test('first low score stays normal', () {
    final engine = DecisionEngine();
    expect(engine.update(0.1), AlertState.normal);
    expect(engine.ema, closeTo(0.1, 1e-9));
  });

  test('two consecutive high-EMA windows trigger alert', () {
    final engine = DecisionEngine();
    expect(engine.update(0.9), AlertState.warn); // ema = 0.9 >= 0.6, count=1
    expect(engine.update(0.9), AlertState.alert); // count=2
  });

  test('a single high window then a low one never alerts', () {
    final engine = DecisionEngine();
    expect(engine.update(0.9), AlertState.warn);
    expect(engine.update(0.0), AlertState.normal); // ema drops below 0.6
  });

  test('ema formula matches alpha * raw + (1 - alpha) * prev', () {
    final engine = DecisionEngine(); // alpha = 0.7
    engine.update(1.0);
    expect(engine.ema, closeTo(1.0, 1e-9));
    engine.update(0.0);
    // 0.7*0.0 + 0.3*1.0 = 0.3
    expect(engine.ema, closeTo(0.3, 1e-9));
  });
}
