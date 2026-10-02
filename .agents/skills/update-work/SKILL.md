---
name: update-work
description: Update Scaly GitHub issue classification, dependencies, Project status, and progress or handoff records within an authorized task. Use when maintaining tracking metadata or linking completed work, not to choose tasks or resolve ownership conflicts.
---

# Update Scaly work

Read [the tracking conventions](../../../docs/dev/issue_tracking.md) and inspect the issue's live
metadata, discussion, dependencies, and related PRs before changing it. GitHub Issues is the
actionable list. Resolve legacy references by searching issue bodies for their original todo
IDs; do not create duplicate tasks or interpret old todo IDs as GitHub issue numbers.

Apply only updates authorized by the current task. Using this skill does not authorize publishing
the backlog, creating a Project, choosing work, taking over another session's issue, or committing
and pushing code. Report missing access or conflicting ownership without guessing a replacement.

Use an explicit repository, normally `PREDICT-EPFL/scaly`. Preserve unrelated labels, assignments,
body content, and Project fields. Prefer additive or subtractive edits for the specific change.
Use a temporary body file for multiline comments or descriptions.

```bash
gh issue edit ISSUE -R PREDICT-EPFL/scaly --add-label LABEL --remove-label OLD_LABEL
gh issue edit ISSUE -R PREDICT-EPFL/scaly --add-blocked-by PREREQUISITE
gh issue edit ISSUE -R PREDICT-EPFL/scaly --milestone MILESTONE
gh issue comment ISSUE -R PREDICT-EPFL/scaly --body-file COMMENT_FILE
```

Choose native issue types or kind labels according to the configured tracker, not both. Record
prerequisites with dependency fields and decomposition with sub-issues. Origin describes where
the report came from, not its current priority or who implements it.

For Project updates, discover the Project, field options, and issue item through `gh project list`,
`field-list`, and `item-list`. Do not hardcode undiscovered IDs or overwrite the Status options.
Edit the existing item using its actual identifiers.

```bash
gh project item-edit --id ITEM_ID --project-id PROJECT_ID --field-id STATUS_FIELD_ID --single-select-option-id OPTION_ID
```

For a permitted item addition, use `gh project item-add PROJECT_NUMBER --owner PREDICT-EPFL --url ISSUE_URL`.
Set only the requested fields. Do not assume a linked PR's state automatically changes its issue's
Project status. Check the configured workflows and the issue item itself.

When recording work, identify the active session through its branch, PR, or a brief comment.
Progress and handoff notes state what is done, what remains, relevant validation, and blockers.
Do not replace the contributor's conflict-management decisions with a claiming protocol.

Use a closing reference in the PR description when it completes the issue. Keep the issue open
during implementation and review. A partial PR references the issue without a closing keyword.
For a parent task, verify its completion criteria before closing it; closed children alone are
insufficient evidence. Deferred work stays open, and declined or duplicate work uses the appropriate
closure reason with an explanation.

Re-read the affected issue and Project item after updating. Before retrying a failed creation or
comment, check whether the first request succeeded to avoid duplicates. Report what changed and
any unfinished update without claiming success from the command alone.
