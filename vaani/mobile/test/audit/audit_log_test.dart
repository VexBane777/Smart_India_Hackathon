import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/audit/audit_log.dart';

void main() {
  test('append chains prevHash to the previous entry hash', () {
    final log = AuditLog();
    final a = log.append('transfer_held', payload: {'amount_inr': 4000000});
    final b = log.append('otp_verified');
    expect(a.prevHash, '0' * 64);
    expect(b.prevHash, a.entryHash);
    expect(log.length, 2);
  });

  test('verifyChain reports intact chain as (true, null)', () {
    final log = AuditLog();
    log.append('a');
    log.append('b');
    final (ok, brokenSeq) = log.verifyChain();
    expect(ok, isTrue);
    expect(brokenSeq, isNull);
  });

  test('verifyChain detects a tampered payload', () {
    final log = AuditLog();
    log.append('a');
    final entry = log.append('b', payload: {'x': 1});
    entry.payload['x'] = 999; // mutate after the hash was computed
    final (ok, brokenSeq) = log.verifyChain();
    expect(ok, isFalse);
    expect(brokenSeq, entry.seq);
  });

  test('entries() returns hash alongside the same fields as Python\'s to_dict', () {
    final log = AuditLog();
    final e = log.append('transfer_approved', payload: {'amount_inr': 1});
    final dict = log.entries().single;
    expect(dict['seq'], e.seq);
    expect(dict['event'], 'transfer_approved');
    expect(dict['hash'], e.entryHash);
    expect(dict['prev_hash'], e.prevHash);
  });
}
