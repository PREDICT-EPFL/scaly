# Issue tracking and parallel work

[GitHub Issues](https://github.com/PREDICT-EPFL/scaly/issues) is Scaly's single actionable list.
These conventions cover what an issue states, how it is classified, who holds it, and the risks
when contributors or agents work in parallel.

## One actionable list

Every task, bug report, research question, and deferred idea has an issue in `scaly`. Design notes
explain decisions and approaches and link to issues instead of maintaining separate task checklists.

An open issue is not a promise to implement a feature. Its status distinguishes incoming proposals
from accepted work, and its milestone says which capability it belongs to.

## What an issue states

An issue describes one coherent result: something a user or the compiler can do once it is closed.
It is not one step toward a result. Steps belong to whoever does the work, on their branch.
An issue is normally built on one branch and lands as one pull request (PR).

An actionable issue states:

- **The goal:** what becomes possible, and why it matters.
- **The examples:** what must be true at the end, written as code where it can be.
- **What is out of scope**, when a reader could expect it to be included.
- **The design section or note** it implements, linked by its heading.

An example pairs what goes in with what should come out: the Python a user writes and the C the
compiler should generate for it, an error message, or an interface. Two short lists follow it.
**Pin** names what a reviewer checks. **Leave free** names what the implementer decides.

````markdown
```python
L, d, status = linalg.ldl(A)
sc.gradient(L.sum(), A)
# TypeError: ldl: output 'L' is not differentiable; differentiate linalg.solve(..., assume="sym") instead
```

Pin: the message names the Function and the output and says what to do instead.
Leave free: the exception's exact wording beyond that.
````

A result that touches several parts of the library gets one example per part. Not every issue
needs target code: a measurement states its numbers, a bug report carries its reproduction, and a
research issue states the question and the evidence or decision that completes it.

The examples are what the PR is reviewed against, so they are agreed before the work starts and
changed only by a maintainer.

Keep titles imperative. Use compact prose and add headings only when they help. Keep labels,
milestone, status, and prerequisites in tracker fields, not in the body. Contributor instructions
belong in this guide, not in every issue.

## Labels

Source and planning are independent. A community report can become release-critical, and a
maintainer's proposal can remain deferred. Apply one source label and keep it after triage.

| Group | Labels | Meaning |
| --- | --- | --- |
| Source | `source:community`, `source:maintainer`, `source:agent` | Where the report or proposal originated, regardless of who filed it. `source:agent` marks an issue an agent created during unattended work, which no maintainer has assessed yet. |
| Kind | `kind:bug`, `kind:feature`, `kind:maintenance`, `kind:research`, `kind:question` | One primary kind. Research can finish with a decision rather than code. |
| Area | `area:api`, `area:compiler`, `area:solvers`, `area:c-api`, `area:benchmarks`, `area:docs`, `area:tooling` | Usually one area, several for work that crosses boundaries. Tooling includes packaging, continuous integration, and platform support. |
| Needs | `needs:examples`, `needs:decision` | What a maintainer must add before the work can start: an agreed example such as target code or numbers, or a decision. The issue names the missing piece under a *Needs* heading. Remove the label when it is settled. |
| Alert | `ci:nightly-failing`, `ci:platforms-failing` | Separate failure issues for Nightly and Platforms. Only their workflows apply these labels. |

Scaly uses kind labels and leaves native issue types unset to avoid duplicate classification.
Add area labels only when they help readers find work.

## Status

Each issue carries one of GitHub's built-in statuses, shown beside its title and as the columns of
the repository's issue board.

| Status | Meaning |
| --- | --- |
| Backlog | Not started, and nobody plans to start it now. New reports arrive here. Postponed work stays here too, with the reason in a comment. |
| Todo | The goal and examples are clear and nothing it needs is missing from main, or someone has taken it. |
| In progress | Someone is implementing or investigating it. Its draft PR shows the work. |
| In review | The result awaits a maintainer's review. |
| Done, Not planned, Duplicate | The three ways an issue closes. |

GitHub moves the status by itself in three cases, so set it by hand only in the others:

- assigning someone moves the issue to Todo
- a draft PR that links the issue, with `Closes #N` in its description, moves it to In progress
- marking that PR ready for review, or linking a PR that is already open, moves it to In review

An issue with a `needs:` label is not started, whatever its status. An issue nobody has assessed
is in Backlog with the `source:agent` label, or with no kind label yet. Close duplicates and
declined proposals with an explanatory comment, as Duplicate or Not planned.

Keep blocked active work In progress and record what it waits for as an issue dependency. Add a
comment for a blocker that has no corresponding issue, such as missing access to a Windows machine.

To read one issue's status, and to list the open issues with theirs:

```bash
gh api repos/PREDICT-EPFL/scaly/issues/ISSUE -q .status.name
gh api graphql --paginate -f query='query($endCursor: String) { repository(owner: "PREDICT-EPFL", name: "scaly") {
  issues(first: 100, states: OPEN, after: $endCursor) { pageInfo { hasNextPage endCursor }
    nodes { number title status { name } } } } }' \
  -q '.data.repository.issues.nodes[] | select(.status.name == "Todo") | "\(.number)\t\(.title)"'
```

The statuses are a GitHub public preview, released on 2026-11-23, so these commands may change.

## Assignment

Assignment says who holds an issue, and reads together with its status.

| On the issue | Meaning | To take it |
| --- | --- | --- |
| Unassigned | Free | Take it: comment on the issue, and assign yourself if you can. |
| Assigned, Todo or Backlog | The assignee wants to do it soon. | Ask them. |
| Assigned, In progress or In review | The assignee is working on it now. | Ask them, and expect a no. |

Ask wherever is convenient, and record the outcome on the issue, since that is the only place
everyone, agents included, can see it. Several agent sessions can share one GitHub account, so
assignment alone does not identify the active session. The branch and the draft PR do.

## Milestones

A milestone is a capability, such as a fully sparse intermediate representation (IR), the loop
compiler, control flow, or generated solvers. Its description says what it covers and what
completing it means. A capability milestone needs no version or due date.

An issue has one milestone. A release is a list of milestones, kept in the
[compiler roadmap's release scope](https://github.com/PREDICT-EPFL/scaly/blob/main/internal/notes/core_compiler_roadmap.md#release-scope),
which also records design and implementation order. Issues carry no release field: the project
works on one release at a time.

## References and dependencies

| Relationship | Meaning | GitHub behavior |
| --- | --- | --- |
| Mention, such as `Related to #42` | Relevant context. | Links the issues and normally adds a reference to the other issue's timeline. Does not create a dependency. |
| Blocked by or blocking | A prerequisite must be on main first. | Creates a queryable dependency and displays a blocked indicator on issues. |
| `Closes #42` in a PR | The PR completes the issue. | Closes the issue when the PR merges into the default branch. |
| `Part of #42` in a PR | The PR lands a part of the issue early. | Links them. The issue stays open. |

An issue depends on what is on main, never on someone's branch. To record a prerequisite, open the
dependent issue and select **Relationships**, then **Mark as blocked by**, or use the CLI:

```bash
gh issue edit ISSUE -R PREDICT-EPFL/scaly --add-blocked-by PREREQUISITE
gh issue view ISSUE -R PREDICT-EPFL/scaly --json state,assignees,blockedBy,blocking
```

Work is not planned as sub-issues. Cutting a result into child issues before the work starts
guesses which parts someone else will wait for, and the draft PR already shows which examples
hold. Split an issue only when a need appears: someone has to wait for a part of it, or wants to
take a part over. That part then becomes its own issue, with its own examples, and the reason is
visible to everyone. Fold small issues that one result covers into a single issue, and close them
as duplicates of it.

Relationships provide tracking information. Do not treat them as enforcement that prevents
starting work, closing an issue, or merging a PR. Closing a prerequisite does not mean its
dependent is ready: check the merged behavior and the other prerequisites.

## Parallel work

Before starting, inspect the issue's dependencies, assignee, recent comments, and the open PRs,
drafts included. Open a draft PR with your first commit and keep the branch pushed. That is how
others see which files your work touches, and how interrupted work is resumed.

Use separate worktrees for independent implementations. Separate checkouts prevent accidental
file overwrites, but do not prevent incompatible designs. Two issues collide when they edit the
same code. Compare the files and the shared contracts before running them together: file moves,
operation definitions, derivative rules, generated symbols, and plugin contracts can affect work
outside the edited files. Area labels alone do not establish independence. A collision depends on
the code as it is today, so it is worked out when planning and never recorded as a dependency.

PRs that regenerate the generated-C snapshots land one after another. Two such PRs always conflict
in their snapshots. Both can be worked on at once, but the second is marked ready only after the
first has merged, once it has rebased, regenerated its snapshots and rerun its checks.

Passing tests on independent branches do not prove that their combination is correct. Main merges
through a merge queue, which runs the required checks on each PR on top of main and the PRs queued
ahead of it, and removes a PR whose combination fails. A branch therefore need not be updated after
every merge, only when it conflicts with main or its PR leaves the queue. Follow
[the contribution checks](contributing.md#run-the-checks), including the full suite for IR,
differentiation, or code generation changes.

## Sources

- [GitHub issue forms](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms)
- [Issue dependencies](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/creating-issue-dependencies)
- [References](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls)
- [Closing references](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue)
