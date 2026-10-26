#!/bin/zsh
# v2 controls, cross-model pairings and repeated samples. Writes new files only.
export $(grep '^OPENROUTER_API_KEY=' ~/Desktop/Scientific-Eval/.env | tr -d '"')
export SPEND_CAP_USD=40
cd "$(dirname $0)/src"
M=(openai/gpt-4.1-mini google/gemini-2.5-flash meta-llama/llama-3.3-70b-instruct openai/gpt-4o-mini)
for m in $M; do ../.venv/bin/python run_v2.py controls $m 100 > ../results/logv2_${m:t}.txt 2>&1 & done
for m in openai/gpt-4.1-mini google/gemini-2.5-flash; do ../.venv/bin/python run_v2.py repeat $m 100 > ../results/logrep_${m:t}.txt 2>&1 & done
for a in $M; do for b in $M; do
  [[ $a == $b ]] && continue
  ../.venv/bin/python run_v2.py cross $a $b 100 > ../results/logx_${a:t}__${b:t}.txt 2>&1 &
done; done
wait
