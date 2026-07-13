# jq helpers for inspecting st history.jsonl.
#
# Usage:
#   jq -Rrs 'include "st-history"; st_history_json(20)' "$XDG_STATE_HOME/st/history.jsonl"
#   jq -Rrs 'include "st-history"; st_history_human(20)' "$XDG_STATE_HOME/st/history.jsonl"

def st_history_entries:
  if type == "string" then
    split("\n") | map(select(test("\\S")) | fromjson?)
  elif type == "array" then
    .
  else
    [.]
  end
  | map(select(. != null));

def st_history_command:
  .effective.display // .raw.display // ((.raw.argv // []) | join(" "));

def st_history_summary:
  {
    id,
    exit_code,
    family,
    started_at,
    finished_at,
    command: st_history_command,
    raw: .raw,
    effective: .effective,
    derived_from_history_id,
    replayed_from_history_id,
    conflate_policy,
    error
  };

def st_history_summaries:
  st_history_entries
  | sort_by(.id)
  | reverse
  | map(st_history_summary);

def st_history_json($limit):
  st_history_summaries | .[:$limit];

def st_history_pad($width):
  tostring as $s
  | if ($s | length) >= $width then
      $s
    else
      $s + (" " * ($width - ($s | length)))
    end;

def st_history_when:
  (.started_at // "")
  | sub("\\.[0-9]+Z$"; "Z")
  | sub("T"; " ")
  | sub("Z$"; "");

def st_history_human_line:
  [
    (.id | st_history_pad(6)),
    (.exit_code | st_history_pad(4)),
    ((.family // "") | st_history_pad(8)),
    (st_history_when | st_history_pad(19)),
    st_history_command
  ]
  | join("  ");

def st_history_human($limit):
  (
    "ID      EXIT  FAMILY    STARTED              COMMAND",
    "------  ----  --------  -------------------  -------",
    (st_history_entries | sort_by(.id) | reverse | .[:$limit][] | st_history_human_line)
  );
