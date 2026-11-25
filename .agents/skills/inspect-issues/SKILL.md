---
name: inspect-issues
description: Inspect Scaly GitHub issues, their status, dependencies, ownership, and overlapping work before planning or starting a task. Makes no tracker updates. Use only when the user or another skill asks for it by name, never on your own initiative.
---

# Inspect Scaly issues

Read [the tracking conventions](../../../docs/dev/issue_tracking.md) for the form of an issue and
the meaning of labels, statuses, assignment and milestones. GitHub Issues is the actionable list.
Resolve a legacy reference by searching issue bodies for its original todo ID; old todo IDs are
not GitHub issue numbers.

Use `gh` with an explicit repository, normally `PREDICT-EPFL/scaly`. Respect another repository
or issue URL supplied by the user. Read the issue and its recent discussion before judging readiness.

```bash
gh issue view ISSUE -R PREDICT-EPFL/scaly --comments
gh issue view ISSUE -R PREDICT-EPFL/scaly --json number,title,body,state,stateReason,labels,assignees,milestone,blockedBy,blocking,closedByPullRequestsReferences,url
gh pr list -R PREDICT-EPFL/scaly --state open --json number,title,isDraft,headRefName,url
gh pr diff PR -R PREDICT-EPFL/scaly --name-only
```

The status is GitHub's built-in issue status, which `gh issue view` does not return. Read it
separately. Backlog also holds what nobody has assessed yet. A closed issue is Done, Not planned
or Duplicate.

```bash
gh api repos/PREDICT-EPFL/scaly/issues/ISSUE -q .status.name
gh api graphql --paginate -f query='query($endCursor: String) { repository(owner: "PREDICT-EPFL", name: "scaly") {
  issues(first: 100, states: OPEN, after: $endCursor) { pageInfo { hasNextPage endCursor }
    nodes { number title status { name } } } } }' \
  -q '.data.repository.issues.nodes[] | "\(.status.name)\t\(.number)\t\(.title)"'
```

The search qualifier takes the field's name with a hyphen. Raise `--limit` above the default 30
when completeness matters. Many threads share one GitHub account and its hourly API allowance, so
read once and keep the result instead of repeating a query.

Read prerequisite issues to establish their state and outcome. A prerequisite closed as not
planned does not prove the required behavior exists, and an issue depends on what is on main, not
on a branch. Mentions are not prerequisite relationships.

Read assignment together with status: unassigned is free, assigned with Todo or Backlog means its
assignee wants it soon, assigned with In progress means it is being worked on. Several agent
sessions may share one GitHub account, so the branch and the draft pull request identify the
active work, not the assignee.

Before recommending parallel work, read the relevant design note and the
[compiler roadmap's order of work](../../../internal/notes/core_compiler_roadmap.md#order-of-work).
Check the open pull requests, drafts included, for the files they touch, and check snapshot
changes and shared contracts.

Return a concise briefing with the goal and examples, current planning metadata, unresolved
prerequisites, active work, overlap risks, and the next useful action. Distinguish verified facts
from missing information. Do not claim work or change issues or authorization scopes while
inspecting.
