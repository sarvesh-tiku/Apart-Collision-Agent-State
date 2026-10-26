#!/bin/zsh
# Resume one model: ./resume.sh MODEL N [--codebook]
export $(grep '^OPENROUTER_API_KEY=' ~/Desktop/Scientific-Eval/.env | tr -d '"')
cd "$(dirname $0)/src"
../.venv/bin/python run.py "$@"
