#!/bin/bash
# Chains English XTTS cloning -> Hindi XTTS cloning, sequentially (single GPU).
cd "/c/Users/Tuviksh Murudkar/SIH/voice_guard/model_training/data"
source ../.venv_tts/Scripts/activate
export COQUI_TOS_AGREED=1

# Wait for the English run (already in progress) to finish.
while ! grep -q "^Done\." ../runs/gen_fake_en.log 2>/dev/null; do
  sleep 15
done
echo "English cloning done, starting Hindi" >> ../runs/chain_xtts.log

python gen_accents_fake.py --lang hi --limit 400 --local-model-dir xtts_v2_local > ../runs/gen_fake_hi.log 2>&1

echo "Hindi cloning done" >> ../runs/chain_xtts.log
echo "XTTS_CHAIN_COMPLETE" >> ../runs/chain_xtts.log
