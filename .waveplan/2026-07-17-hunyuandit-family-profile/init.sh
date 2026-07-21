source /Users/darkbit1001/workspace/Stability-Toys/.waveplan/.envrc
source /Users/darkbit1001/workspace/Stability-Toys/.waveplan/2026-07-17-hunyuandit-family-profile/env.hunyuandit

jq -n --arg sched "$WAVEPLAN_SCHED" '{
  schema_version: 1,
  schedule_path: $sched,
  cursor: 0,
  events: []
}' > "$WAVEPLAN_JOURNAL"

waveplan-cli swim validate --kind journal --in "$WAVEPLAN_JOURNAL"
waveplan-cli swim run \
  --schedule "$WAVEPLAN_SCHED" \
  --journal "$WAVEPLAN_JOURNAL" \
  --state "$WAVEPLAN_STATE" \
  --review-schedule "$WAVEPLAN_SCHED_REVIEW" \
  --until T1 \
  --dry-run \
  --max-steps 1