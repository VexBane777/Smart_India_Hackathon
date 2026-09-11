# Test assets

Small, force-committed audio clips (bypassing the repo's normal `*.wav`
gitignore rule — see the exception this directory was originally added
under, commit `3a2fc32`) for cross-device manual testing via VoiceGuard's
Live Mic Test, played through an external speaker into the phone's mic
(never the phone's own speaker — see `voice_guard/state.md`'s documented
acoustic anti-loopback clamp).

These are all public research-dataset clips, not real user call recordings
— the DPDP/no-audio-retained privacy principle in `vaani/00_MASTER_PLAN.md`
is about the app not recording real calls in production, not about test
fixtures like these.

- `ai_clone_test_clip.wav` — pre-existing AI voice-clone test clip.
- `tts_elevenlabs_sample.wav` — pure TTS (no voice conversion), ElevenLabs
  v2 Multilingual, from MLAAD (`data/mlaad_en500`).
- `tts_chattts_sample.wav` — pure TTS, ChatTTS architecture, from MLAAD.
- `voice_conversion_asvspoof_a17.wav` — genuine **voice conversion** (not
  TTS) attack, ASVspoof2021 LA attack type A17
  (`LA_E_1428848`, `data/fake2021`) — per the ASVspoof2019 taxonomy A17-A19
  are actual VC systems (waveform/spectral filtering), distinct from the
  TTS attack types (A01-A16). Use this to reproduce the "voice-transformed
  clip triggers AI detection but plain TTS doesn't" observation — the two
  TTS samples above and this VC sample are deliberately different attack
  *families*, not just different generators, for exactly that comparison.
