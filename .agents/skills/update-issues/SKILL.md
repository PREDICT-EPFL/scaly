---
name: update-issues
description: Create or update Scaly GitHub issues, their classification, dependencies, status, and progress or handoff records within an authorized task. Not for choosing tasks or resolving ownership conflicts. Use only when the user or another skill asks for it by name, never on your own initiative.
---

# Update Scaly issues

Read [the tracking conventions](../../../docs/dev/issue_tracking.md) and inspect the issue's live
metadata, discussion, dependencies, and related PRs before changing it. GitHub Issues is the
actionable list. Resolve legacy references by searching issue bodies for their original todo
IDs; do not create duplicate tasks or interpret old todo IDs as GitHub issue numbers.

Apply only updates authorized by the current task. Using this skill does not authorize publishing
the backlog, choosing work, taking over an issue assigned to someone else, changing an issue's
examples, or committing and pushing code. Report missing access or conflicting ownership without
guessing a replacement.

Use an explicit repository, normally `PREDICT-EPFL/scaly`. Preserve unrelated labels, assignments
and body content. Prefer additive or subtractive edits for the specific change. Use a temporary
body file for multiline comments or descriptions.

```bash
gh issue edit ISSUE -R PREDICT-EPFL/scaly --add-label LABEL --remove-label OLD_LABEL
gh issue edit ISSUE -R PREDICT-EPFL/scaly --add-blocked-by PREREQUISITE
gh issue edit ISSUE -R PREDICT-EPFL/scaly --milestone MILESTONE
gh issue edit ISSUE -R PREDICT-EPFL/scaly --add-assignee LOGIN
gh issue comment ISSUE -R PREDICT-EPFL/scaly --body-file COMMENT_FILE
```

When writing an issue, follow *What an issue states* in the tracking conventions, and also:

- describe one coherent result, not a step toward one, and do not plan sub-issues
- write each example so that a reviewer can check it: target code with its Pin and Leave free
  lists, a command and its result, or an observable behaviour
- link the roadmap section or design note the issue implements, by its heading anchor, so that a
  revised decision can find every issue depending on it

Use kind labels and leave native issue types unset. Record prerequisites with dependency fields.
Origin describes where the report came from, not its current priority or who implements it. An
issue an agent files on its own carries `source:agent` and no status.

The status is the issue's `workflow status` field. Set it with the option's name, through `POST`,
which leaves the issue's other fields alone. `PUT` on the same endpoint replaces them all.

```bash
echo '{"issue_field_values":[{"field_id":FIELD_ID,"value":"In progress"}]}' > BODY_FILE
gh api -X POST repos/PREDICT-EPFL/scaly/issues/ISSUE/issue-field-values --input BODY_FILE
```

Find `FIELD_ID` once with `gh api orgs/PREDICT-EPFL/issue-fields -q '.[] | select(.name ==
"workflow status") | .id'`. Set only the requested field. Nothing changes the status when a pull
request opens, becomes ready or merges, so the thread doing the work sets In progress and In
review itself. A closed issue is done and its status is left as it was.

When recording work, identify the active session through its branch, its draft pull request, or
a brief comment. Progress and handoff notes state what is done, what remains, relevant validation,
and blockers. Do not replace the contributor's conflict-management decisions with a claiming
protocol.

Use `Closes #N` in the PR description when the PR completes the issue, and `Part of #N` when it
lands a part early. Keep the issue open during implementation and review. Deferred work stays
open, and declined or duplicate work uses the appropriate closure reason with an explanation. An
issue folded into a larger one is closed as a duplicate, with a comment naming the issue that
carries its content.

Re-read the affected issue and its status after updating. Before retrying a failed creation or
comment, check whether the first request succeeded to avoid duplicates. Report what changed and
any unfinished update without claiming success from the command alone.
