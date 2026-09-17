# VoIP Signaling Tailscale Backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let two VoiceGuard "Protected Call" devices connect across real,
different networks (not just the same LAN) by pointing the existing
signaling relay at a Tailscale IP instead of the hardcoded `10.0.2.2`
emulator default — with no new NAT-traversal infrastructure (no coturn, no
VPS) because Tailscale's mesh handles that.

**Architecture:** No protocol or backend logic changes. `backend/signaling.py`
already implements the right relay behavior; it just needs to be reachable
at a Tailscale IP. The only real code changes are on the Flutter client:
persist a configurable signaling host/port in `SettingsProvider`, expose it
in Settings UI, and make `ProtectedCallScreen` use it instead of
`SignalingService.connect`'s baked-in default.

**Tech Stack:** Flutter/Dart (`shared_preferences`, `provider`), FastAPI
WebSocket relay (unchanged), Tailscale (operational infra, no SDK
integration).

**Spec:** `voice_guard/docs/superpowers/specs/2026-09-17-voip-signaling-tailscale-backbone-design.md`

## Global Constraints

- No TURN/coturn, no cloud VPS — Tailscale replaces that layer entirely (spec §1).
- Default `signalingHost`/`signalingPort` must fall back to the current
  `10.0.2.2:8001` values so emulator/dev workflows are unaffected (spec §3).
- `backend/signaling.py` gets no code changes (spec §3) — this plan's
  backend tasks are operational (run it reachably), not code changes.

---

### Task 1: Persist configurable signaling host/port in `SettingsProvider`

**Files:**
- Modify: `voice_guard/lib/providers/settings_provider.dart`
- Test: `voice_guard/test/settings_provider_test.dart` (new)

**Interfaces:**
- Produces: `SettingsProvider.signalingHost` (`String`, default `'10.0.2.2'`),
  `SettingsProvider.signalingPort` (`int`, default `8001`),
  `Future<void> setSignalingHost(String v)`, `Future<void> setSignalingPort(int v)`.
  `load()` reads both from `SharedPreferences` with those same defaults.

- [ ] **Step 1: Write the failing test**

```dart
// voice_guard/test/settings_provider_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:voice_guard/providers/settings_provider.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  test('signalingHost/signalingPort default to the emulator loopback values', () async {
    final settings = SettingsProvider();
    await settings.load();
    expect(settings.signalingHost, '10.0.2.2');
    expect(settings.signalingPort, 8001);
  });

  test('setSignalingHost persists and updates the getter', () async {
    final settings = SettingsProvider();
    await settings.load();
    await settings.setSignalingHost('100.101.102.5');
    expect(settings.signalingHost, '100.101.102.5');

    final reloaded = SettingsProvider();
    await reloaded.load();
    expect(reloaded.signalingHost, '100.101.102.5');
  });

  test('setSignalingPort persists and updates the getter', () async {
    final settings = SettingsProvider();
    await settings.load();
    await settings.setSignalingPort(9000);
    expect(settings.signalingPort, 9000);

    final reloaded = SettingsProvider();
    await reloaded.load();
    expect(reloaded.signalingPort, 9000);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd voice_guard && flutter test test/settings_provider_test.dart`
Expected: FAIL — `signalingHost`/`signalingPort`/`setSignalingHost`/`setSignalingPort` are not defined on `SettingsProvider`.

- [ ] **Step 3: Add the fields to `SettingsProvider`**

In `voice_guard/lib/providers/settings_provider.dart`, add alongside the
existing fields (mirror the existing `_sensitivity`/`setSensitivity` pattern
exactly):

```dart
  String _signalingHost = '10.0.2.2';
  int _signalingPort = 8001;

  String get signalingHost => _signalingHost;
  int get signalingPort => _signalingPort;
```

In `load()`, add:

```dart
    _signalingHost = p.getString('signalingHost') ?? '10.0.2.2';
    _signalingPort = p.getInt('signalingPort') ?? 8001;
```

Add setters (same pattern as `setSensitivity`):

```dart
  Future<void> setSignalingHost(String v) async {
    _signalingHost = v;
    (await SharedPreferences.getInstance()).setString('signalingHost', v);
    notifyListeners();
  }

  Future<void> setSignalingPort(int v) async {
    _signalingPort = v;
    (await SharedPreferences.getInstance()).setInt('signalingPort', v);
    notifyListeners();
  }
```

Leave `resetToDefaults()` untouched — signaling host/port is deployment
config, not a "reset my alert preferences" concern.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd voice_guard && flutter test test/settings_provider_test.dart`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add voice_guard/lib/providers/settings_provider.dart voice_guard/test/settings_provider_test.dart
git commit -m "feat(voice_guard): persist configurable signaling host/port in SettingsProvider"
```

---

### Task 2: Expose signaling host/port in Settings UI

**Files:**
- Modify: `voice_guard/lib/screens/settings_screen.dart`

**Interfaces:**
- Consumes: `SettingsProvider.signalingHost`, `.signalingPort`,
  `.setSignalingHost(String)`, `.setSignalingPort(int)` from Task 1.

This task is a UI addition with no new business logic to unit-test in
isolation (persistence is already covered by Task 1); verification is
manual (Step 3) per this codebase's existing settings-screen pattern (other
settings toggles in this file have no dedicated widget tests either).

- [ ] **Step 1: Locate the insertion point**

Read `voice_guard/lib/screens/settings_screen.dart` and find the existing
sensitivity slider section (search for `sensitivity`) — add the new fields
immediately after it, inside the same settings list/card structure the file
already uses for other fields.

- [ ] **Step 2: Add host/port text fields**

Add a small section with two `TextFormField`s (or this file's existing
input widget, matching whatever `protected_call_screen.dart` uses —
`ShadInput`) seeded from `context.watch<SettingsProvider>()` and committing
via `context.read<SettingsProvider>().setSignalingHost(...)` /
`.setSignalingPort(...)` on submit:

```dart
ShadInput(
  initialValue: context.watch<SettingsProvider>().signalingHost,
  placeholder: 'Signaling host (e.g. Tailscale IP)',
  onSubmitted: (v) => context.read<SettingsProvider>().setSignalingHost(v.trim()),
),
const SizedBox(height: 8),
ShadInput(
  initialValue: context.watch<SettingsProvider>().signalingPort.toString(),
  placeholder: 'Signaling port',
  keyboardType: TextInputType.number,
  onSubmitted: (v) {
    final port = int.tryParse(v.trim());
    if (port != null) context.read<SettingsProvider>().setSignalingPort(port);
  },
),
```

(If `ShadInput` doesn't support `initialValue`/`onSubmitted` — check its
definition in `voice_guard/lib/widgets/shad_input.dart` first — use a
`TextEditingController` seeded in `initState` instead, following whatever
pattern `protected_call_screen.dart`'s `_roomController` already
establishes.)

- [ ] **Step 3: Manual verification**

Run: `cd voice_guard && flutter run`
Navigate to Settings, confirm the two new fields appear, show the current
persisted defaults (`10.0.2.2` / `8001`), and that editing + leaving the
field persists (verify by hot-restarting and confirming the edited value is
still shown).

- [ ] **Step 4: Commit**

```bash
git add voice_guard/lib/screens/settings_screen.dart
git commit -m "feat(voice_guard): add signaling host/port fields to Settings screen"
```

---

### Task 3: Wire `ProtectedCallScreen` to use the configured signaling host/port

**Files:**
- Modify: `voice_guard/lib/screens/protected_call_screen.dart:36`

**Interfaces:**
- Consumes: `SettingsProvider.signalingHost`, `.signalingPort` (Task 1);
  `SignalingService.connect(String roomId, {String host, int port})`
  (`voice_guard/lib/services/signaling_service.dart:21` — already accepts
  these parameters, unchanged by this plan).

- [ ] **Step 1: Change the connect call**

In `_connect()` (`protected_call_screen.dart:27-43`), replace:

```dart
    final signaling = SignalingService.connect(roomId);
```

with:

```dart
    final settings = context.read<SettingsProvider>();
    final signaling = SignalingService.connect(
      roomId,
      host: settings.signalingHost,
      port: settings.signalingPort,
    );
```

Add the import: `import '../providers/settings_provider.dart';`

- [ ] **Step 2: Manual verification (single device, sanity check only)**

Run: `cd voice_guard && flutter run`
Open Settings, confirm host/port default to `10.0.2.2`/`8001`. Open
Protected Call, initiate a call — confirm it behaves exactly as before this
change (no regression) when host/port are left at defaults. Full
cross-device verification happens in Task 6.

- [ ] **Step 3: Commit**

```bash
git add voice_guard/lib/screens/protected_call_screen.dart
git commit -m "feat(voice_guard): use configured signaling host/port in Protected Call"
```

---

### Task 4: Stand up Tailscale on both devices and confirm mesh connectivity

This task is operational, not code — it's the load-bearing verification
step the rest of the plan depends on (spec §5's flagged risk: whether the
demo's NAT-traversal assumption actually holds).

**Interfaces:** none (infra step).

- [ ] **Step 1: Install Tailscale**

On both phones: install the official "Tailscale" app from the Play Store.
On the machine that will run `signaling.py` (recommended: a laptop, per
spec §4): install the Tailscale client for that OS from
https://tailscale.com/download.

- [ ] **Step 2: Log in to the same tailnet**

Open the Tailscale app on all three devices (2 phones + the signaling
host), sign in with the same account (Google/GitHub/Microsoft/Apple — no
credit card required for the free "Personal" plan). Confirm all three
appear in the tailnet's device list (visible in the Tailscale app or at
https://login.tailscale.com/admin/machines).

- [ ] **Step 3: Confirm cross-network reachability**

Put the two phones on genuinely different networks (e.g. one on Wi-Fi, one
on cellular data — this is the actual scenario being validated, not a
same-LAN shortcut). From one phone's Tailscale app, note the other
device's `100.x.x.x` address. Use a terminal/network-utility app (or the
laptop, if easier) to `ping` that address and confirm replies.

If this step fails: do not proceed to Task 5/6 — this is spec §5's flagged
risk materializing. Fall back to investigating Tailscale's own
diagnostics (`tailscale status`, `tailscale ping <address>`) before
touching any app code.

- [ ] **Step 4: No commit** (infra-only step, nothing to check in)

---

### Task 5: Run `signaling.py` reachably on the tailnet

**Files:** none modified — `backend/signaling.py` and `backend/main.py`
already mount it correctly (`signal.py`'s docstring and Task-1-era design
are unchanged by this plan, per spec §3's "no code change").

**Interfaces:**
- Consumes: `signaling_router` mounted in `backend/main.py` (verify this
  mount exists before assuming it — read `backend/main.py` for
  `include_router(signaling_router` or equivalent if Step 1 below doesn't
  already confirm it).

- [ ] **Step 1: Confirm the router is mounted**

Run: `grep -n "signaling_router" voice_guard/backend/main.py`
Expected: a line showing `app.include_router(signaling_router)` or
equivalent. If missing, this is a pre-existing gap outside this plan's
scope — stop and flag it rather than silently patching `main.py`, since
that's not a change this plan's spec anticipated.

- [ ] **Step 2: Start the server on the tailnet host**

On the chosen host (laptop, per Task 4):

```bash
cd voice_guard/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
```

`--host 0.0.0.0` is required — binding to `127.0.0.1` would make it
unreachable from the phones even over Tailscale.

- [ ] **Step 3: Confirm it's reachable from both phones' Tailscale IP**

From a phone (or via `adb` if easier), confirm a WebSocket handshake
succeeds against `ws://<laptop-tailscale-ip>:8001/v1/signal/test-room` —
easiest check is opening Protected Call, setting the signaling host in
Settings (Task 2) to the laptop's Tailscale IP, and initiating a call;
watch the `uvicorn` server logs for the incoming WebSocket connection.

- [ ] **Step 4: No commit** (operational step)

---

### Task 6: Regression-check the signaling test suite, then verify end-to-end across networks

**Files:**
- Test: `voice_guard/backend/test_signaling.py` (unchanged — regression check only)

**Interfaces:** none new.

- [ ] **Step 1: Run the existing backend signaling tests**

Run: `cd voice_guard/backend && pytest test_signaling.py -v`
Expected: PASS (2 tests — `test_two_clients_relay_messages_to_each_other`,
`test_third_client_to_occupied_room_is_rejected`). Per spec §3, this
sub-project makes no protocol changes, so these must pass unmodified — a
failure here means something in Tasks 1-3 broke assumptions the relay
depends on, not that the test needs updating.

- [ ] **Step 2: Manual end-to-end verification (the actual acceptance criterion)**

With both phones on different networks (Task 4's setup) and
`signaling.py` running reachably (Task 5):
1. On Phone A: Settings → set signaling host to the laptop's Tailscale IP, port `8001`.
2. On Phone B: same.
3. Phone A: Protected Call → enter a room code → "Initiate Call".
4. Phone B: Protected Call → same room code → "Answer Room".
5. Confirm the call connects (`Peer Status` in the UI moves to `Connected`)
   and audio is audible in both directions.

This is the sub-project's real acceptance test per spec §6 step 4. If it
fails despite Task 4's ping succeeding, the most likely cause is spec §5's
flagged risk (ICE not picking up the Tailscale interface as a candidate) —
capture the WebRTC connection state/ICE gathering logs before debugging
further, since that's the one assumption this design didn't have certainty
on going in.

- [ ] **Step 3: No commit** (verification step; if it reveals a bug, that
  becomes a new task/commit, not part of this checklist)
