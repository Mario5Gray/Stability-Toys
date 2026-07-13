#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

fixture="testdata/history.jsonl"

human="$(jq -Rrs 'include "st-history"; st_history_human(20)' -L . "$fixture")"
grep -F "ID      EXIT  FAMILY" <<<"$human" >/dev/null
grep -F "4       0     gen       2026-07-13 12:03" <<<"$human" >/dev/null
grep -F "st gen --prompt owl --cfg 4.5" <<<"$human" >/dev/null

json="$(jq -Rrs 'include "st-history"; st_history_json(2)' -L . "$fixture")"
[[ "$(jq 'length' <<<"$json")" == "2" ]]
[[ "$(jq -r '.[0].id' <<<"$json")" == "4" ]]
[[ "$(jq -r '.[0].command' <<<"$json")" == "st gen --prompt owl --cfg 4.5" ]]
[[ "$(jq -r '.[1].derived_from_history_id' <<<"$json")" == "1" ]]

entries="$(jq -Rrs 'include "st-history"; st_history_entries | length' -L . "$fixture")"
[[ "$entries" == "4" ]]
