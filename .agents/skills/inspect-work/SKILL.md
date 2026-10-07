---
name: inspect-work
description: Inspect Scaly GitHub issues, Project status, dependencies, ownership, and overlapping work before planning or starting a task. Makes no tracker updates. Use only when the user or another skill asks for it by name, never on your own initiative.
---

# Inspect Scaly work

Read [the tracking conventions](../../../docs/dev/issue_tracking.md) for the meaning of labels,
statuses, and milestones. GitHub Issues is the actionable list. Resolve a legacy reference by
searching issue bodies for its original todo ID; old todo IDs are not GitHub issue numbers.

Use `gh` with an explicit repository, normally `PREDICT-EPFL/scaly`. Respect another repository
or issue URL supplied by the user. Read the issue and its recent discussion before judging readiness.

```bash
gh issue view ISSUE -R PREDICT-EPFL/scaly --comments
gh issue view ISSUE -R PREDICT-EPFL/scaly --json number,title,body,state,stateReason,labels,issueType,assignees,milestone,blockedBy,blocking,parent,subIssues,projectItems,closedByPullRequestsReferences,url
gh pr list -R PREDICT-EPFL/scaly --state open --json number,title,body,headRefName,url
```

When an issue has a parent, read the parent body for the shared contract and combined completion
criteria. Keep readiness and ownership at the child level; parent membership does not establish
prerequisite order. The published Project is [scaly roadmap #2](https://github.com/orgs/PREDICT-EPFL/projects/2).

Read prerequisite issues to establish their state and outcome. A prerequisite closed as not
planned does not prove the required behavior exists. Inspect related PRs and design notes when
they affect the task. Mentions and sub-issues are not prerequisite relationships.

Discover the relevant Project from the issue's `projectItems` or the owner's Projects. Inspect
its field schema and the item's values rather than inferring status from labels or issue state.

```bash
gh project list --owner PREDICT-EPFL --format json
gh project field-list PROJECT_NUMBER --owner PREDICT-EPFL --format json
gh project item-list PROJECT_NUMBER --owner PREDICT-EPFL --limit 1000 --format json
```

Match the item by issue URL or repository and number. Project numbers, Project node IDs, item IDs,
field IDs, and option IDs are distinct. Check listing limits and pagination when completeness
matters. A missing value is unknown, not Backlog or Ready. Report missing access without treating
an inaccessible Project as absent.

Before recommending parallel work, read the relevant design note and the
[compiler roadmap's order of work](../../../internal/notes/core_compiler_roadmap.md#order-of-work).
Check active ownership, likely shared files, snapshot changes, and shared contracts. Assignment
is not a lock, and several agent sessions may share one GitHub account.

Return a concise briefing with scope and completion criteria, current planning metadata, unresolved
prerequisites, active work, overlap risks, and the next useful action. Distinguish verified facts
from missing information. Do not claim work or change issues, Projects, or authorization scopes
while inspecting.
