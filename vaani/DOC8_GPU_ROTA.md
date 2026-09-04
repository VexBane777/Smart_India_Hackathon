<!--
VAANI documentation suite v1.0 — generated 2026-09-03
Document: Doc 8 — GPU Rota Doc & Overnight-Run Checklist · Owner: Systems Lead · Status: Live document
This file is GENERATED. Edit generate_vaani_docs.py and rerun instead.
-->
# Doc 8 — GPU Rota Doc & Overnight-Run Checklist

**Owner:** Systems Lead · **Status:** Live document · **Suite:** v1.0 · **Generated:** 2026-09-03

---

Lives at `ops/rota.md`, versioned and visible to everyone. Principle: these are
teammates' daily laptops — the cluster is *borrowed politely and crashed cheaply.*

## 8.1 Machines
| Label | Hardware | Owner | Standing constraint |
|---|---|---|---|
| gpu-master | RTX 5050, 8 GB | [member] | MLflow + canonical dataset live here; ≥50 GB free; teacher gets night priority |
| gpu-w1 | RTX 4050, 6 GB | [member] | [classes / travel / needs it mornings] |
| gpu-w2 | RTX 4050, 6 GB | [member] | [idem] |

## 8.2 Rota template (updated in the Sunday ritual)
```
Week of 2026-04-06
| Slot            | gpu-master      | gpu-w1         | gpu-w2         |
| Mon–Fri day     | interactive dev | interactive    | TTS generation |
| Mon 22:00–08:00 | teacher ep.4-6  | aasist sweep A | student distill|
| Tue 22:00–08:00 | teacher ep.6-8  | (rest / owner) | student distill|
| Sat marathon    | teacher cont.   | 3-seed final 1 | 3-seed final 2 |
Withdrawal: "I need my laptop Tue 06:00" -> run migrates via one-line resume
on the next free GPU — never a negotiation, always a command.
```
Booking rules: teacher (5050) reserves nights by default · interactive dev beats batch
during the day · no run without a rota row · a borrowed machine's owner gets a text
before 21:00.

## 8.3 One-time machine setup (per laptop)
- [ ] PyTorch ≥2.7 **cu128** verified via `torch.cuda.get_device_capability()` —
      catches the 5050's sm_120 silent-CPU-fallback before it eats a night
- [ ] Power plan: never sleep on AC; **Windows Update paused** (03:00 reboots are the
      #1 overnight killer)
- [ ] WSL2 + SSH + tmux; HF token stored; MLflow reachable at `http://master:5000`
- [ ] Syncthing: corpus folder fully synced (initial sync overnight)
- [ ] `scripts/run_resume.sh` (Doc 2) deployed and tested
- [ ] Thermals: hard flat surface, lid open, vents clear — no beds/sofas; expect 60–80%
      of nominal sustained; USB fan if anyone owns one
- [ ] ≥50 GB free; `runs/` excluded from sync

## 8.4 Pre-run checklist (every overnight run — 5 minutes)
- [ ] Config valid; `config_hash` logged
- [ ] **Checkpoint round-trip tested once per model** — 20-step throwaway run, push to
      HF Hub, restore on a *different* machine. Test the parachute before you need it;
      this converts every future crash from a lost night into 30 lost seconds
- [ ] tmux session named `vaani_<model>`; runbook entry (model, machine, ETA, owner)
- [ ] Owner texted; rota row updated; power cable physically checked (half of "mystery
      crashes" are unplugged chargers)
- [ ] Optional $0 crash alert: `curl -d "student_v1 crashed, retrying"
      ntfy.sh/vaani-<random-suffix>` wired into the resume loop's failure branch —
      **message content carries run name and status only, never data** (the topic is
      public-but-unlisted)

## 8.5 Morning-after ritual (10 min, whoever's up first)
1. MLflow: loss curve smooth? heartbeat recent?
2. tmux alive? auto-resume loop showing retries?
3. `nvidia-smi` temps; steps/sec vs. first hour — **>25% drop = thermal throttling:
   reassign, add cooling, or shorten runs**
4. Disk space; runbook updated with progress + ETA

## 8.6 Crash protocol
1. Script auto-retries once (its job).
2. Second failure: capture last 100 log lines, exit code, `nvidia-smi`,
   `dmesg | tail -20` (kernel messages — reveals driver/OOM causes) into
   `runs/<name>/crash/<date>.log`.
3. Migrate: one-line resume on the next free GPU.
4. Escalate to Systems Lead only after migration; repeat crashes on one machine →
   it drops off the rota until diagnosed.

## 8.7 The Sunday ritual (15 min, whole team)
Next week's runs → GPU mapping → rota updated → travel/exam conflicts → consent/etiquette
check ("is anyone's laptop silently becoming a server they resent?"). The rota is a social
document as much as a compute one; keep it honest or it stops being followed.

---

*Part of the VAANI documentation suite — regenerate with `python generate_vaani_docs.py`. Placeholders marked [M] must be replaced by measured values before use in the pitch.*
