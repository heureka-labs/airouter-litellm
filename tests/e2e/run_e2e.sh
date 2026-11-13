#!/usr/bin/env bash
# End-to-end check against a real LiteLLM proxy with mock deployments and a fake AI Router.
# Usage: run_e2e.sh <venv-python-bin-dir>
set -u
BIN=${1:-/tmp/llm-venv/bin}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
LOG=$(mktemp -d)
"$BIN/python" "$HERE/fake_airouter.py" 4012 gpt-5.2 low >"$LOG/airouter.log" 2>&1 &
FAKE=$!
# Install the way the README describes it: the callback file next to config.yaml, no PYTHONPATH.
cp "$HERE/config.yaml" "$ROOT/airouter_litellm.py" "$LOG/"
(cd "$LOG" && AIROUTER_API_KEY=test AIROUTER_BASE_URL=http://127.0.0.1:4012 PYTHONUNBUFFERED=1 \
  AIROUTER_SHADOW_MODELS=azure-mini-eu,shadow-broken \
  "$BIN/litellm" --config "$LOG/config.yaml" --port 4011 >"$LOG/proxy.log" 2>&1) &
PROXY=$!
trap 'kill $FAKE 2>/dev/null; pkill -P $PROXY 2>/dev/null; kill $PROXY 2>/dev/null; fuser -k 4011/tcp >/dev/null 2>&1' EXIT
for _ in $(seq 1 60); do curl -sf localhost:4011/health/liveliness >/dev/null && break; sleep 2; done
H=(-H "Authorization: Bearer sk-test-1234" -H "Content-Type: application/json")
fail=0
check() { if echo "$2" | grep -q "$3"; then echo "PASS $1"; else echo "FAIL $1: $(echo "$2" | head -c 300)"; fail=1; fi; }

r=$(curl -s localhost:4011/v1/chat/completions "${H[@]}" -d '{"model":"airouter-auto","messages":[{"role":"user","content":"hi"}]}')
check "chat completions routed" "$r" '"BIG"'
r=$(curl -sN localhost:4011/v1/chat/completions "${H[@]}" -d '{"model":"airouter-auto","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}')
check "chat streaming routed" "$(echo "$r" | grep -o '"content":"[^"]*"' | tr -d '\n')" 'B.*I.*G'
r=$(curl -s localhost:4011/v1/messages "${H[@]}" -d '{"model":"airouter-auto","max_tokens":100,"system":"be brief","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}')
check "anthropic messages routed" "$r" 'BIG'
r=$(curl -s localhost:4011/v1/responses "${H[@]}" -d '{"model":"airouter-auto","instructions":"be brief","input":"hi"}')
check "responses api routed" "$r" 'BIG'
r=$(curl -s localhost:4011/v1/chat/completions "${H[@]}" -d '{"model":"azure-mini-eu","messages":[{"role":"user","content":"hi"}]}')
check "direct model untouched" "$r" '"MINI"'
r=$(curl -s localhost:4011/v1/chat/completions "${H[@]}" -d '{"model":"azure-mini-eu","reasoning_effort":"high","messages":[{"role":"user","content":"observed with effort"}]}')
check "observed request with effort untouched" "$r" '"MINI"'
r=$(curl -sN localhost:4011/v1/chat/completions "${H[@]}" -d '{"model":"azure-mini-eu","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"observed stream"}]}')
check "observed streaming untouched" "$(echo "$r" | grep -o '"content":"[^"]*"' | tr -d '\n')" 'M.*I.*N.*I'
r=$(curl -s localhost:4011/v1/messages "${H[@]}" -d '{"model":"azure-mini-eu","max_tokens":100,"output_config":{"effort":"xhigh"},"messages":[{"role":"user","content":"observed messages"}]}')
check "observed anthropic messages untouched" "$r" 'MINI'
r=$(curl -s localhost:4011/v1/chat/completions "${H[@]}" -d '{"model":"shadow-broken","messages":[{"role":"user","content":"observed fallback"}]}')
check "observed request falls back as configured" "$r" '"MINI"'
r=$(curl -s localhost:4011/v1/chat/completions "${H[@]}" -d '{"model":"auto-quality","messages":[{"role":"user","content":"quality"}]}')
check "second virtual model routed" "$r" '"BIG"'
r=$(curl -s localhost:4011/v1/chat/completions "${H[@]}" -d '{"model":"project-cheap","messages":[{"role":"user","content":"cheap"}]}')
check "project virtual model limited to its candidates" "$r" '"MINI"'
sleep 8
check "select got mapped catalog names" "$(grep '^SELECT' "$LOG/airouter.log" | head -1)" '"models": \["gpt-5-mini", "gpt-5.2", "claude-4-5-sonnet"\]'
check "virtual model sent its weighting" "$(grep '"quality"' "$LOG/airouter.log" | grep '^SELECT')" '"weighting": {"quality": 3.0, "costs": 1.0, "latency": 0.5}'
check "project virtual model sent only its candidate" "$(grep '"cheap"' "$LOG/airouter.log" | grep '^SELECT')" '"models": \["gpt-5-mini"\]'
check "select got anthropic system text" "$(grep '^SELECT' "$LOG/airouter.log")" '"content": "be brief"'
reports=$(grep '^REPORTS' "$LOG/airouter.log" | sed 's/^REPORTS //' | "$BIN/python" -c '
import json, sys
for line in sys.stdin:
    for report in json.loads(line)["reports"]:
        kind = "shadow" if report["decision_id"].startswith("shadow-") else "routed"
        print(kind, report["model"], report["effort"], report["input_tokens"] > 0)')
check "usage reports delivered (6 routed calls)" "$(echo "$reports" | grep -c '^routed')" '^6$'
check "routed reports carry the selected effort on every route" "$(echo "$reports" | grep -c '^routed [^ ]* low True')" '^6$'
check "shadow reports delivered (5 observed calls)" "$(echo "$reports" | grep -c '^shadow')" '^5$'
check "shadow selections marked, naming the requested model" "$(grep -c '^SHADOW-SELECT.*"shadow": true.*"model": "gpt-5-mini"' "$LOG/airouter.log")" '^4$'
check "shadow selection of the fallback group names its own model" "$(grep '^SHADOW-SELECT' "$LOG/airouter.log")" '"model": "claude-4-5-sonnet"'
check "shadow reports name the model that ran (incl. the fallback)" "$(echo "$reports" | grep '^shadow' | grep -c 'gpt-5-mini')" '^5$'
check "shadow report carries the chat effort" "$reports" '^shadow gpt-5-mini high True'
check "shadow report carries the messages effort" "$reports" '^shadow gpt-5-mini xhigh True'
check "shadow report without effort" "$reports" '^shadow gpt-5-mini None True'
check "unmapped models logged" "$(cat "$LOG/proxy.log")" "'vertex-sonnet' has no AI Router catalog match"

kill $FAKE; sleep 1
r=$(curl -s localhost:4011/v1/chat/completions "${H[@]}" -d '{"model":"airouter-auto","messages":[{"role":"user","content":"hi"}]}')
check "ai router down falls back to default" "$r" '"DEFAULT"'
echo "logs: $LOG"
exit $fail
