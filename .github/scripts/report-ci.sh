set -euo pipefail

failed=$(gh run view "$GITHUB_RUN_ID" --json jobs -q '.jobs[] | select(.conclusion == "failure") | .name' | sort)
issue=$(gh issue list --state open --label "$ALERT_LABEL" --json number -q '.[0].number')
if [ -z "$failed" ]; then
  [ "$(jq 'all(.[]; .result == "success")' <<<"$NEEDS")" = true ] || exit 0
  if [ -n "$issue" ]; then gh issue close "$issue" -c "Passing again at $GITHUB_SHA: $RUN_URL"; fi
  exit 0
fi
labels=$(printf '%s\n' "$failed" | python3 .github/scripts/nightly.py labels --alert-label "$ALERT_LABEL")
if [ -n "$issue" ]; then
  gh issue edit "$issue" --add-label "$labels"
  if [[ ",$labels," != *,area:benchmarks,* ]]; then gh issue edit "$issue" --remove-label area:benchmarks; fi
fi
list() { if [ -n "$1" ]; then printf -- '- `%s`\n' $1; else echo "- none"; fi; }
if [ -z "$issue" ]; then
  gh issue create --title "$TITLE" --label "$labels" \
    --body "$(printf 'The %s run at %s failed: %s\n\nFailing jobs:\n%s\n\n<!-- failing\n%s\n-->\n' "$WORKFLOW_NAME" "$GITHUB_SHA" "$RUN_URL" "$(IFS=$'\n'; list "$failed")" "$failed")"
  exit 0
fi
previous=$(gh issue view "$issue" --json body,comments -q '[.body] + [.comments[].body] | last' | sed -n '/<!-- failing/,/-->/p' | sed '1d;$d' | sort)
IFS=$'\n'
new=$(comm -13 <(echo "$previous") <(echo "$failed") | grep . || true)
fixed=$(comm -23 <(echo "$previous") <(echo "$failed") | grep . || true)
still=$(comm -12 <(echo "$previous") <(echo "$failed") | grep . || true)
gh issue comment "$issue" --body "$(printf 'Still failing at %s: %s\n\nNewly failing:\n%s\n\nStill failing:\n%s\n\nFixed since the last report:\n%s\n\n<!-- failing\n%s\n-->\n' \
  "$GITHUB_SHA" "$RUN_URL" "$(list "$new")" "$(list "$still")" "$(list "$fixed")" "$failed")"
