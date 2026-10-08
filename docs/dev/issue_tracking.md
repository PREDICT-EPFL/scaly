# Issue tracking and parallel work

[GitHub Issues](https://github.com/PREDICT-EPFL/scaly/issues) is Scaly's single actionable list, and the public
[scaly roadmap Project](https://github.com/orgs/PREDICT-EPFL/projects/2) tracks planning and progress.
These conventions cover classification, dependencies, and risks when contributors or agents work
in parallel.

## One actionable list

Every task, bug report, research question, and deferred idea has an issue in `scaly`. Design notes explain decisions and
approaches and link to issues instead of maintaining separate task checklists.

An open issue is not a promise to implement a feature. Triage distinguishes incoming proposals
from accepted work, and milestones show release scope or a broad capability goal.

Each actionable issue states the problem, scope, completion criteria, and relevant design notes.
A research issue states the question and the evidence or decision that completes the investigation.
Preserve old todo identifiers in a small source permalink during migration so existing references
remain searchable. Keep titles imperative and free of legacy IDs. Use compact prose; add headings
only when they help. Keep labels, milestone, status, and prerequisite lists in tracker fields, not
in the body. Include dependency explanations only when they clarify a task-specific constraint.
Contributor instructions belong in this guide rather than repeated in every issue.

## Labels

Origin and planning are independent. A community report can become release-critical, and a
maintainer's proposal can remain deferred. Apply one origin label and keep it after triage.

| Group | Labels | Meaning |
| --- | --- | --- |
| Origin | `source:community`, `source:maintainer` | Where the report or proposal originated, regardless of who filed it. |
| Kind | `kind:bug`, `kind:feature`, `kind:maintenance`, `kind:research`, `kind:question` | One primary kind. Research can finish with a decision rather than code. |
| Area | `area:api`, `area:compiler`, `area:solvers`, `area:c-api`, `area:benchmarks`, `area:docs`, `area:tooling` | Usually one area, several for work that crosses boundaries. Tooling includes packaging, continuous integration, and platform support. |
| Assistance | `needs:reproduction`, `needs:feedback`, `help wanted`, `good first issue` | Missing information or opportunities to contribute. Remove `needs:*` labels when resolved. |

Scaly uses kind labels and leaves native issue types unset to avoid duplicate classification.
Add area labels only when they help readers find work.

Use Project status for progress and milestones for release or capability scope. Avoid labels that
duplicate those fields. Issue forms can apply default labels, but issues filed without a form still
need triage.

## Project status and views

Use the public scaly roadmap Project for all Scaly issues, with this Status field.

| Status | Meaning |
| --- | --- |
| Triage | The report or proposal has not been assessed. |
| Backlog | Accepted work with no immediate start planned. |
| Ready | Scope and completion criteria are clear, and prerequisites permit starting. |
| In progress | Someone owns the work and is implementing or investigating it. |
| In review | The implementation or investigation result awaits review. |
| Deferred | Deliberately postponed, with a reason or condition for revisiting it. |
| Done | Closed. Check the closure reason to distinguish completed work from declined or duplicate work. |

New submissions normally enter Triage. Accepted work enters Backlog or Deferred until it is ready
or active. Close duplicates and declined proposals with an explanatory comment and
the appropriate closure reason. Deferred means postponed, not rejected.

Keep blocked active work In progress and record its prerequisites as issue dependencies. Add a
comment for a blocker that has no corresponding issue, such as missing access to a Windows machine.

| View | Purpose |
| --- | --- |
| Current release | Default board filtered to the current release milestone, grouped by Status. |
| Triage | Unassessed issues, oldest first. |
| Ready to pick up | Unassigned Ready issues, showing area and milestone. |
| Roadmap | Accepted work grouped by milestone. |
| Community | Community-origin issues across statuses. |
| All work | Complete table, including deferred work. |

## Milestones

Use milestones for releases such as `scaly 0.1.0`, or broad capabilities such as GPU support,
control flow support, or a fully sparse intermediate representation (IR). Keep changing
implementation details in the issues and design notes.

Each milestone describes its scope and what completion means. A capability milestone need not
have a version or due date. Give a release milestone a date only when it is an intended target.
Use package-qualified release names when solver plugins release independently.

An issue has one milestone. When capability work is scheduled into a release, assign the release
milestone. If simultaneous capability grouping becomes necessary, use a Project field instead of
creating duplicate issues. Do not assign future versions to the entire backlog.

Use the milestone for current release scope. The
[compiler roadmap](https://github.com/PREDICT-EPFL/scaly/blob/main/internal/notes/core_compiler_roadmap.md#order-of-work)
records design and implementation order. Its phases do not require matching parent issues or
permanent milestone names.

## References, sub-issues, and dependencies

GitHub provides separate relationships for context, decomposition, and prerequisite order.

| Relationship | Meaning | GitHub behavior |
| --- | --- | --- |
| Mention, such as `Related to #42` | Relevant context. | Links the issues and normally adds a reference to the other issue's timeline. Does not create a dependency. |
| Sub-issue | Part of a larger task. | Creates a parent-child relationship and exposes sub-issue completion progress and parent grouping in Projects. Does not define execution order. |
| Blocked by or blocking | A prerequisite must be resolved. | Creates a queryable dependency and displays a blocked indicator on issues and Project boards. |
| Closing reference in a pull request (PR) | The PR completes the issue. | `Closes #42` closes the issue when the PR merges into the default branch, with auto-closing enabled. |

Use sub-issues when a concrete task needs several independently owned pieces. Broad capability
milestones do not need matching parent issues. Record dependencies separately, including between
siblings when one must precede another. Explain a prerequisite in the body only when the reason
is useful, such as a condition that applies to part of the task. Keep the relationship itself in
GitHub's dependency fields.

To record a prerequisite, open the dependent issue and select **Relationships**, then
**Mark as blocked by**, then the prerequisite issue. Agents can use the same relationships through the CLI.
The installed GitHub CLI supports these flags and JSON fields.

```bash
gh issue edit ISSUE -R PREDICT-EPFL/scaly --add-blocked-by PREREQUISITE
gh issue view ISSUE -R PREDICT-EPFL/scaly --json state,assignees,blockedBy,blocking
```

Replace `ISSUE` and `PREREQUISITE` with GitHub issue numbers.

Relationships provide tracking information. Do not treat them as enforcement that prevents
starting work, closing an issue, or merging a PR. Review a parent before closing it rather than
assuming that child completion closes it automatically. Closing a prerequisite also does not
mean its dependent task is ready: review scope, other prerequisites, and the merged behavior.

Start with Project auto-add and issue-closure status updates. Keep admission to Ready and parent
completion deliberate. Additional scheduling or completion automation needs its own explicit rules.

## Parallel work

Before starting, inspect the issue's dependencies, owner, recent comments, and related open PRs.
Assignment records responsibility, but is not an exclusive lock. Several agent sessions can also
share one GitHub account, so assignment alone does not identify the active session.

Make active work identifiable through its branch, PR, or a short issue comment. Leave enough
context to resume an interrupted task. Do not assume that old activity means ownership is released.

Use separate worktrees for independent implementations. Separate checkouts prevent accidental
file overwrites, but do not prevent incompatible designs. Compare likely files and shared
contracts before running tasks together. Area labels alone do not establish independence.

The [compiler roadmap's parallel-work rules](https://github.com/PREDICT-EPFL/scaly/blob/main/internal/notes/core_compiler_roadmap.md#order-of-work) keep
one shared constraint: PRs that regenerate generated-C snapshots must not be in flight
simultaneously. Two such PRs always conflict in their snapshots, and the second must regenerate them
after the first merges.

File moves, operation definitions, derivative rules, generated symbols, and plugin contracts can
affect work outside the edited files. Land prerequisite changes before dependent implementation,
and coordinate an expansion of scope that overlaps active work. Record real prerequisite
dependencies, not artificial dependency chains for changes that merely need merge coordination.

Passing tests on independent branches do not prove that their combination is correct. Main merges
through a merge queue, which runs the required checks on each PR on top of main and the PRs queued
ahead of it, and removes a PR whose combination fails. A branch therefore need not be updated after
every merge, only when it conflicts with main or its PR leaves the queue. Follow
[the contribution checks](contributing.md#run-the-checks), including the full suite for IR,
differentiation, or code generation changes.

Task claiming, conflict resolution, dispatch, and abandoned-session recovery procedures remain
outside these conventions. The tracking conventions must expose enough information for contributors
and agents to make those decisions.

## Sources

- [GitHub issue forms](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms)
- [Organization issue types](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/managing-issue-types-in-an-organization)
- [Sub-issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/adding-sub-issues)
- [Issue dependencies](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-issue-dependencies)
- [References](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls)
- [Closing references](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue)
- [Project auto-add](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/adding-items-automatically)
- [Project workflows](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/using-the-built-in-automations)
