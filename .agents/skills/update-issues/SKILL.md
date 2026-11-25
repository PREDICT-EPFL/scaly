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
issue an agent files on its own carries `source:agent` and stays in Backlog.

The status is GitHub's built-in issue status: Backlog, Todo, In progress or In review while the
issue is open. GitHub sets it by itself when someone is assigned (Todo), when a draft pull request
that says `Closes #N` is opened (In progress), and when that pull request is marked ready (In
review). Read the status first, and set it only when it is not what it should be, for example for
a pull request that says `Part of #N`, which does not link the issue.

```bash
gh api graphql -f query='{ repository(owner: "PREDICT-EPFL", name: "scaly") { planningStatuses(first: 20) { nodes { id name } } } }'
gh api graphql -f query='mutation($issue: ID!, $status: ID!) { setIssueStatus(input: {issueId: $issue, statusId: $status}) { clientMutationId } }' \
  -f issue="$(gh api repos/PREDICT-EPFL/scaly/issues/ISSUE -q .node_id)" -f status=STATUS_ID
```

The first command lists the statuses with their ids. Closing an issue gives it its closed status,
so never set Done, Not planned or Duplicate by hand.

When recording work, identify the active session through its branch, its draft pull request, or
a brief comment. Progress and handoff notes state what is done, what remains, relevant validation,
and blockers. Do not replace the contributor's conflict-management decisions with a claiming
protocol.

Use `Closes #N` in the PR description when the PR completes the issue, and `Part of #N` when it
lands a part early. Keep the issue open during implementation and review. Postponed work stays
open in Backlog, and declined or duplicate work uses the appropriate closure reason with an explanation. An
issue folded into a larger one is closed as a duplicate, with a comment naming the issue that
carries its content.

Re-read the affected issue and its status after updating. Before retrying a failed creation or
comment, check whether the first request succeeded to avoid duplicates. Report what changed and
any unfinished update without claiming success from the command alone.
