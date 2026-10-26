#!/bin/zsh
# Loads the OpenRouter key from the Scientific-Eval project without copying it here.
export $(grep '^OPENROUTER_API_KEY=' ~/Desktop/Scientific-Eval/.env | tr -d '"')
cd "$(dirname $0)/src"
for spec in openai/gpt-4.1-mini:150 meta-llama/llama-3.3-70b-instruct:100 openai/gpt-4o-mini:100 google/gemini-2.5-flash:100 anthropic/claude-haiku-4.5:40; do
  m=${spec%:*}; n=${N:-${spec##*:}}
  ../.venv/bin/python run.py $m $n > ../results/log_${m:t}.txt 2>&1 &
done
wait
