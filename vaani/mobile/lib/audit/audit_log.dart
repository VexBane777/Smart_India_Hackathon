import 'dart:convert';
import 'package:crypto/crypto.dart';

/// Ported from app/components/audit_log.py: each entry embeds the SHA-256
/// fingerprint of the previous entry, so editing any past entry breaks
/// every hash after it. Single-writer, in-memory — a decision log, not a
/// distributed ledger (master plan §1 item 5: "our honest answer to the
/// Blockchain theme").
///
/// Built with List.filled rather than a string literal so the length (64,
/// matching a SHA-256 hex digest) is exact by construction, not by manual
/// counting.
final String kGenesisHash = List.filled(64, '0').join();

class AuditEntry {
  AuditEntry({
    required this.seq,
    required this.ts,
    required this.event,
    required this.payload,
    required this.prevHash,
  }) : entryHash = _computeHash(seq, ts, event, payload, prevHash);

  final int seq;
  final double ts;
  final String event;
  final Map<String, dynamic> payload;
  final String prevHash;
  final String entryHash;

  static String _computeHash(int seq, double ts, String event,
      Map<String, dynamic> payload, String prevHash) {
    final body = jsonEncode(_sortedMap({
      'seq': seq,
      'ts': ts,
      'event': event,
      'payload': _sortedMap(payload),
      'prev_hash': prevHash,
    }));
    return sha256.convert(utf8.encode(body)).toString();
  }

  /// json.dumps(..., sort_keys=True) equivalent: sort map keys recursively.
  static Map<String, dynamic> _sortedMap(Map<String, dynamic> m) {
    final sortedKeys = m.keys.toList()..sort();
    return {for (final k in sortedKeys) k: m[k]};
  }

  String recomputeHash() => _computeHash(seq, ts, event, payload, prevHash);

  Map<String, dynamic> toDict() => {
        'seq': seq,
        'ts': ts,
        'event': event,
        'payload': payload,
        'prev_hash': prevHash,
        'hash': entryHash,
      };
}

class AuditLog {
  final List<AuditEntry> _entries = [];

  int get length => _entries.length;

  AuditEntry append(String event,
      {Map<String, dynamic> payload = const {}, double? ts}) {
    final prevHash = _entries.isEmpty ? kGenesisHash : _entries.last.entryHash;
    final entry = AuditEntry(
      seq: _entries.length,
      ts: ts ?? DateTime.now().millisecondsSinceEpoch / 1000.0,
      event: event,
      payload: Map<String, dynamic>.from(payload),
      prevHash: prevHash,
    );
    _entries.add(entry);
    return entry;
  }

  List<Map<String, dynamic>> entries() =>
      _entries.map((e) => e.toDict()).toList();

  (bool, int?) verifyChain() {
    var prevHash = kGenesisHash;
    for (final entry in _entries) {
      if (entry.prevHash != prevHash) return (false, entry.seq);
      if (entry.recomputeHash() != entry.entryHash) return (false, entry.seq);
      prevHash = entry.entryHash;
    }
    return (true, null);
  }
}
