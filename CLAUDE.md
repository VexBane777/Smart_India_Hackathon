# SIH — project context (accumulated by Claude Code sessions)

This file is auto-loaded by Claude Code whenever a session starts with this
directory as the working directory. It exists so a fresh session resumes
with the context built up in prior sessions, without needing the old
conversation transcript.

> Note: at the time this file was written, `.claude`'s global memory store
> (`~/.claude/projects/.../memory/`) contained nothing about SIH, VAANI,
> telechannel, or nyaya — only unrelated projects (photo metadata recovery,
> GN-PNS/NMA research systems). So there was nothing to import from there;
> everything below comes from working sessions directly in this repo.

## Repo shape

- Remote: `origin` → `https://github.com/VexBane777/Smart_India_Hackathon.git`
- **Worktree-per-branch layout**: the SIH root itself is the main worktree,
  checked out on branch **`vaani`**. Its content lives directly under
  `vaani/` (renamed from `vaani_documentation/` on 2026-09-04).
  A second branch, **`nyaya`**, is checked out as a linked worktree at
  `./nyaya/` (previously at `./DVR/NVR/`, which is why `.gitignore` still
  has a stale `/DVR/` line — `/nyaya/` is the current, correct exclusion).
  `nyaya/` and `vaani/` are **separate git histories** — never `git add`
  across that boundary from the main worktree.
- If a worktree ever shows `prunable` in `git worktree list` after a manual
  directory move, fix it with `git worktree repair <new-path>` (must pass
  the actual current path — bare `git worktree repair` with no args did
  nothing in practice here).

## `vaani/` — VAANI Documentation Suite

A telephony-channel audio-simulation pipeline project, organized into
Modules A–D (see `vaani/00_MASTER_PLAN.md`, `vaani/state.md` for live
status, and `vaani/docs/superpowers/plans/2026-09-04-vaani-module-{a,b,c,d}-*.md`
for the per-module implementation plans). Module A ("TeleChannel") is the
audio DSP pipeline: `vaani/telechannel/` (stages: rir → noise → mic → codec
→ packetloss → bandlimit, orchestrated by `pipeline.py`, indexed via
`manifest.py`, gated by `qa.py`), tests under `vaani/tests/telechannel/`
and `vaani/tests/legal/` (DPDP consent/revocation).

### FFmpeg (2026-09-04)

Module A's codec stage (`vaani/telechannel/stages/codec.py`) shells out to
`ffmpeg` for real codec round-trips (G.711, GSM-FR, AMR-NB/WB, Opus). It was
previously blocked — no ffmpeg on PATH, and the docstrings said so. Resolved:

- Installed via `winget install --id Gyan.FFmpeg -e` — the **full** GPL
  build (has libopencore-amrnb, libvo-amrwbenc, libgsm, libopus). Per-user,
  **no admin rights needed**. Lives under
  `AppData\Local\Microsoft\WinGet\Links`, already on PATH permanently.
- Full suite (`pytest tests/` from `vaani/`) → **148 passed**, 0 skipped/failed.
- **Bug found by having a real encoder**: `CODEC_SPECS["libvo_amrwbenc"]["ext"]`
  was `"awb"`; ffmpeg has no muxer for that extension (only `.amr`, which
  auto-detects NB vs. WB from the stream). Fixed to `"amr"`. This class of
  bug is invisible until a real ffmpeg binary is actually present — worth
  remembering if similar "verified only with mocks" caveats show up
  elsewhere in this codebase.
- Committed as `4459f31` (rename + fix + stale-caveat cleanup) and `915e198`
  (old-path removal) on `vaani`, 2026-09-04.

## Open / unresolved — Module A "final scoped re-review"

A user request came in to: adjudicate a "Module A final scoped re-review"
(park residual findings with rulings, no more fix rounds), push to
`origin/vaani`, delete "the SDD workspace", and report a completion summary.

**This has not been done.** Investigation turned up **no review artifacts
anywhere** — not in the repo, not in global Claude memory, not in any
findings/report file. Two questions are parked, unanswered, for whoever
picks this back up:

1. **Where do the Module A review findings actually live?** Candidates
   raised (not yet resolved): (a) no review has been run yet — run
   `/code-review` scoped to Module A now; (b) it happened in a different,
   `/clear`'d session — need the transcript or a pasted findings list;
   (c) it's GitHub PR review comments (no `gh` CLI available in-session,
   would need repo/PR info to fetch via API).
2. **What is "the SDD workspace" to delete?** Best guess:
   `.superpowers/sdd/` — it's git-ignored specifically as "Superpowers SDD
   scratch workspace (git-ignored per skill)" per `.gitignore`. The other
   candidate considered (the `nyaya` worktree) is very unlikely — it's a
   separate, live, unrelated project, not an SDD scratch space.

Do not silently invent findings or rulings to satisfy this request — ask.

## Working notes / preferences observed this session

- The user runs many parallel Claude Code sessions (local + cloud) that can
  touch this same filesystem concurrently. If repo state looks different
  than expected, don't assume corruption — ask whether another session is
  mid-task before doing anything destructive.
- Prefers real verification over narrated confidence: install the actual
  tool and run the actual tests rather than reasoning about whether it
  would probably work.
