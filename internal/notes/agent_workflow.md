> Maintained note on how Scaly's maintainers plan, dispatch and review work done by coding agents.
> **Status: a proposal of 2026-11-24.** One maintainer has worked it out in discussion with an
> agent. The other has not agreed to it yet, and no trial has run. The skills, `AGENTS.md` and
> `docs/dev/issue_tracking.md` on this branch are drafts of it. Nothing in the tracker has
> changed. Once it is agreed, this header is replaced by the date of the agreement and the note
> describes the workflow in force. It replaces the design note of October 2026, which was kept
> outside the repository.

# Agent workflow

Issues become bigger. One issue describes one coherent result, one agent thread builds all of it
on one branch with the help of subagents, and it lands as one pull request that a maintainer
reviews from examples written when the feature was designed. That is the whole change. The rest
of this note says why, and how each step works.

[Tracking work](../../docs/dev/issue_tracking.md) stays the reference for the tracker itself:
labels, status, milestones and what an issue must state. It applies to every contributor, with or
without agents. This note covers only the part that involves agents, and the procedures themselves
are the skills in `.agents/skills/`.

## Contents

1. [The problems this answers](#the-problems-this-answers)
2. [The workflow in order](#the-workflow-in-order)
3. [What each thing is](#what-each-thing-is)
4. [Writing an issue](#writing-an-issue)
5. [Running an issue](#running-an-issue)
6. [Reviewing the pull request](#reviewing-the-pull-request)
7. [Choosing what starts next](#choosing-what-starts-next)
8. [Skills and where things live](#skills-and-where-things-live)
9. [Decisions, and what they rejected](#decisions-and-what-they-rejected)
10. [What adopting this changes](#what-adopting-this-changes)
11. [Open questions](#open-questions)

## The problems this answers

Until now an issue was sized to fit one agent thread and one small pull request, so that each
part could be reviewed thoroughly and poor code would be caught early. Measured on 2026-11-24, the
result was this:

- **Weak tests still merge.** The last 60 merged pull requests added 16,300 lines of tests for
  9,300 lines of source. The suite has needed two pruning passes (#260 and #460). The cause is a
  rule applied to small issues: the implementer writes one test per completion criterion, and the
  criterion of a small issue is often "this exists", so the test shows only that it exists.
- **Decisions are made without context.** An implementer reads one small issue. Anything that
  depends on the larger goal becomes either a question to the maintainer or a silent departure
  from the plan. #75 alone raised two such questions.
- **Follow-up issues pile up.** An implementer must not widen its issue, so whatever it finds
  becomes a new one. Between 2026-11-10 and 2026-11-24, 118 issues were opened and 106 closed.
  Of the 113 open issues, 43 were filed by agents. #75 produced three (#512, #513, #514).
- **Review is spread thin.** 120 pull requests merged in 13 days, with a median of 397 changed
  lines and 4 hours open. Each one is reviewed alone, so nobody checks that the parts fit.
- **Work that regenerates the C snapshots waits in line.** Such pull requests land one at a time,
  and a feature cut into six issues waits six times.

The opposite extreme failed too. The compiler run of 2026-11-10 started agents on the whole
roadmap at once. In 46 hours it merged about 95 pull requests, kept about 15 open at any time on
the same lowering code and snapshots, had 78 of 154 CI runs cancelled by a newer push in one day,
and stopped when the coordinating threads ran out of quota. Its mistakes were having no limit on
parallel work and letting pull requests collide, and small pull requests made the collisions
worse.

The workflow before both of these, described in the October note, also recorded problems that
still apply: decisions lost between threads, "done" meaning less than the maintainer's standard,
and repeated full reviews that find nothing after the first. Those answers stay: decisions are
written where the next planner looks, every change has checkable finish conditions, and a change
gets one review from another model and a short check on the fixes.

## The workflow in order

1. **A design discussion** between a maintainer and an agent ends with `record-decisions`. It
   presents one checklist: the decisions, and one issue per coherent result, each with its goal
   and the examples that must hold at the end. The maintainer approves the checklist, and only
   then are the design document and the issues written.
2. **`plan-next` proposes what to start.** It reads what is running, what is assigned and what
   would collide, and proposes an order. A maintainer approves.
3. **A maintainer starts one thread on the issue** with `implement`. The thread opens a draft pull
   request with its first commit, delegates parts to subagents when the issue is large, and
   integrates their work on its branch.
4. **The thread finishes** by pruning its tests, regenerating the snapshots once, getting one
   review from another model and writing the description the maintainer reviews from.
5. **A maintainer reviews** by comparing each example in the issue with what the description
   shows, reading the decisions the thread took, and merges.

A bug or a small task skips step 1. It is one issue, one thread and one pull request, as before.

## What each thing is

- **Milestone.** A capability, such as the loop compiler, with its checks in the
  [compiler roadmap](core_compiler_roadmap.md#milestone-checks). A release is a list of
  milestones, kept in the roadmap's *Release scope*. An issue has one milestone.
- **Issue.** One coherent result. By default it is sized for one to three days of one thread. It
  is what a maintainer assigns, starts a thread on and reviews as a whole.
- **Pull request.** A change that stands on its own on main and passes every check. Normally one
  per issue. It is what gets reviewed and merged, and the only place the snapshots are regenerated.
- **`PLAN.md` on the branch.** The thread's steps and their state. It lets a new thread resume the
  work, and it is deleted in the last pull request. Steps never go into the issue.
- **Delegated task.** One part of an issue given to a subagent in its own worktree. It leaves no
  public record: no pull request, no issue, no tracker update.

There is no level between milestone and issue. An issue is split only when a need appears:
someone has to wait for a part of it, someone wants to take a part over, or the work has grown
well beyond its planned size and a maintainer decides to cut it. That part then becomes its own
issue, and everyone can see why it exists.

## Writing an issue

An issue states:

- **The goal:** what a user or the compiler can do afterwards, and why it matters.
- **The examples:** what must be true at the end, each written as code.
- **What is out of scope**, when a reader could expect it to be included.
- **The design section** it implements.

An example pairs what goes in with what should come out: the Python a user writes and the C the
compiler should generate for it, or the error message, or the interface. Under it are two short
lists. **Pin** names what the review checks. **Leave free** names what the implementer decides.
This form was first used on 2026-11-19 for the linear-algebra and generated-solver issues, where
the maintainer agreed each target before any code was written. This example is from #496. Its
names are illustrative, and the two lists under it are from its draft:

```python
# lower triangle of an arrow matrix: (0,0) (1,1) (2,0) (2,1) (2,2)
K_type = sc.TensorType((3, 3), pattern=arrow)

@sc.function(sc.arg("K", K_type, layout="csc"), sc.arg("b", 3),
             outputs=sc.group(sc.arg("x", 3), sc.arg("status", status_type)))
def kkt_solve(K, b):
    return linalg.solve(K, b, assume="sym", ordering="natural")
```

```c
double d0  = K[0];
double l20 = K[1] / d0;
double d1  = K[2];
double l21 = K[3] / d1;
double d2  = K[4] - l20 * l20 * d0 - l21 * l21 * d1;
int32_t s  = unusable(d0, K[0]) ? 1 : unusable(d1, K[2]) ? 2 : unusable(d2, K[4]) ? 3 : 0;
double y2  = b[2] - l20 * b[0] - l21 * b[1];
x[2] = y2 / d2;
x[1] = b[1] / d1 - l21 * x[2];
x[0] = b[0] / d0 - l20 * x[2];
*status = s;
```

> **Pin:** no loop and no index table for a matrix under the budget. Each stored entry of `L` is
> computed once. The factor lives in C locals. The status never causes an early return.
>
> **Leave free:** the order of independent statements, the names, whether a division becomes a
> multiplication by a stored reciprocal.

A feature that touches several parts of the library gets one example per part. A change to the
generated entry point, for instance, would show the default entry, the CasADi export and a solver
bridge separately.

Not every issue needs target code. A measurement states its numbers as the gate, and a bug carries
its reproduction. Those are its examples.

`record-decisions` writes issues in this form, and checks the code and the existing issues when it
does, so that an issue is ready to start when it is created. Small existing issues that the new
one covers are closed as merged into it, with their content carried in its examples.

## Running an issue

`implement` is the only entry point, for a large issue and a small one alike.

- **One thread owns the issue.** It owns the branch, the pull request and its CI, and every
  decision inside the agreed design. A question reaches the maintainer only when it would change
  that design.
- **A draft pull request is open from the first commit**, and the branch is always pushed. Other
  threads and the other maintainer can then see which files the work touches, read `PLAN.md`, and
  see in the description which examples already hold.
- **Each example becomes a test first**, of the kind the area prescribes, and fails on main for
  the reason the issue gives.
- **Large issues are delegated in parts.** The thread gives each subagent a part, a worktree and
  the files it owns, then merges the result into its branch and runs the suite. Whether to delegate
  is the thread's judgement. A small issue is written by the thread itself.
- **The thread may edit code.** It resolves conflicts, fixes CI failures and applies review
  remarks itself. When a fix made after the review changes logic in the intermediate
  representation (IR), differentiation or code generation, the reviewer checks that fix alone.
- **Work that serves the issue's goal is done in the issue.** It is added to `PLAN.md`, not filed
  as a new issue. The thread writes an example for it and marks it as added in the description,
  where the maintainer may refuse it.
- **Leaving something out needs a strong reason.** There are two: doing it would widen the issue
  beyond its goal, or it waits on work outside the issue. Each thing left out is named in the
  description with its reason, and the maintainer can refuse it. The thread files what remains in
  one batch once the pull request has merged.
- **If the design cannot work as written**, the thread stops that part and posts the evidence on
  the issue. The maintainer revises the design through `record-decisions`.

A part can land early when someone outside needs it. The thread finishes that part first, runs the
finishing steps on what exists, and opens a pull request that says `Part of #N`. After the merge
it rebases and continues, and the issue stays open. The part must stand on its own on main. Each
early landing costs a full review, so an issue that needs several was drawn wrong.

The finishing steps, in order:

1. Rebase onto main.
2. `prune-tests` on the tests the branch added or changed.
3. Regenerate the C snapshots, once. If another pull request that regenerates them is ready for
   review or queued, the thread waits for it to merge first.
4. The checks a merge needs, and the measurements the issue asks for, with the raw data committed
   under `internal/notes/<topic>_<date>/`.
5. `cross-review`: one review from another model, then a short check on the fixes.
6. The description, as the next section says.

## Reviewing the pull request

The description is what the maintainer reads, so it carries the review. It holds:

1. **Each example from the issue, with the real before and after.** The maintainer compares what
   was promised with what was delivered.
2. **The decisions the thread took** where the design was silent, each with the alternative.
3. **What was left out**, with the reason for each.
4. **The other model's verdict**, the points it left open, and anything it could not check.
5. **The measurements**, where the issue asked for them.

The maintainer reads the examples, the decisions and the test names, and relies on the other
model's review for the rest of the lines. Reading less per line is the accepted cost of this
workflow. It is acceptable for three reasons. Reviewing every small pull request did not stop
weak tests. The examples were agreed before the code existed. And correctness in the IR,
differentiation and code generation does not rest on a maintainer reading lines: each example is
a differential test against NumPy or an unrolled reference, and every such change gets a second
review that only hunts bugs.

The maintainer who agreed the examples reviews the pull request. When the other one reviews, they
read and agree the examples before the work starts. Either may read the draft pull request and
`PLAN.md` at any time and object early.

Main takes squash merges, and GitHub's default commit message for them is the pull request's
title and description. With the review written in the description, that setting should become
"Pull request title". The commit then carries the title and the pull request number, which leads
to the description. What is lost is the summary in `git log` for someone reading offline.

## Choosing what starts next

`plan-next` answers one question: what do we start now? It does not reshape issues that a design
discussion wrote. It changes an issue's content only when the code has changed since the issue was
written. The one thing it creates is an issue that groups loose bug reports and agent follow-ups
in the same code, since those never had a design discussion.

It reads the draft pull requests to see what running work touches, and proposes issues that do not
collide with it or with each other. Two issues collide when they edit the same code. A claim that
they do names the file and the function.

**Assignment** tells the maintainers apart, and reads together with the status:

| On the issue | Meaning | To take it |
|---|---|---|
| Unassigned | Free | Take it |
| Assigned, Ready or Backlog | Someone wants to do it soon | Ask them |
| Assigned, In progress | Being worked on now | Ask them, and expect a no |

Asking can happen anywhere, including in a chat between the maintainers. The outcome goes on the
issue, because agents only see the tracker. An agent never takes an assigned issue on its own. It
brings the question to whoever started it.

**How much runs at once** has no fixed number per person. Three things limit it:

- Issues running at the same time must not collide.
- Pull requests that regenerate the snapshots land one after another. Two such issues can be
  worked on together. The second rebases, regenerates and reruns its checks after the first merges.
- Every issue ends in a review by a maintainer. This is the limit to watch, since several large
  reviews arriving on the same day are worse than the same work arriving in sequence.

Each maintainer also keeps their own rule on usage limits, since every running issue is a long
thread.

**An issue that depends on another** depends on what is on main, never on a branch. If it needs
part of an issue that is still running, that part either moves into the issue that needs it, or
lands early as described above, or the second issue waits.

## Skills and where things live

| Where | What it holds |
|---|---|
| This note | Why the workflow is what it is, the sequence, and each decision with what it rejected |
| `docs/dev/issue_tracking.md` | The tracker rules and the form of an issue, for every contributor |
| `.agents/skills/` | The procedures |
| Each maintainer's private instructions | Which model fills each role, their machines, their usage rule |

The skills that form this workflow move into the repository so that both maintainers run the same
ones: `implement`, `plan-next`, `record-decisions`, `cross-review` and `prune-tests`.
`coordinate` is removed, since `implement` covers what it did. They are written for Scaly: they
name its checks and its tracker directly, and they name T3 Code's tools directly, since both
maintainers use it. They never name models, because the maintainers use different ones.

They name five roles, and each maintainer's private instructions say which model fills each and
at which effort:

| Role | What it does |
|---|---|
| The implementer | Writes the delegated parts of an issue |
| The reviewer for taste | Reviews design, public interfaces, user-facing behaviour and the shape of generated C |
| The second reviewer for taste | Reviews what the first one wrote |
| The reviewer for prose | Reviews documentation |
| The reviewer for bugs | Hunts bugs and edge cases, and adds a review to every high-risk change |

Each skill still says to use it only when a maintainer or another skill asks for it by name, so
that an outside contributor's agent never runs into them.

A maintainer who has a personal skill with the same name must remove or rename it. Claude Code
gives a personal skill priority over a project skill of the same name, and Codex lists both
without choosing.

## Decisions, and what they rejected

1. **An issue is one coherent result, not one small step.** Rejected: issues sized for one small
   pull request, for the problems listed at the top.
2. **Two levels only, milestone and issue.** Rejected: parent issues with planned sub-issues.
   Sub-issues written at design time were a guess about which parts someone else would wait for.
   They looked arbitrary to whoever did the work and could not be judged by anyone outside it, and
   the draft pull request already shows which parts are done. Also rejected: a third level for the
   sections of the compiler run. Milestones do that job and need no closing by hand.
3. **A design discussion writes issues that are ready to start.** Rejected: a second pass in
   `plan-next` that reshapes every issue. Design and planning are apart in time, not in kind, and
   a mandatory second pass could not be explained to a new collaborator.
4. **One thread per issue, which may edit code, with subagents for the parts.** Rejected: a
   coordinator that never edits code and starts a separate top-level thread per part. That needed
   messages relayed between threads, watches handed over, and a subagent for every CI failure.
5. **One entry point, `implement`.** Rejected: separate `coordinate` and `implement` skills. A
   small issue is a large one with a single part.
6. **No stacked pull requests.** A stack meant "review these one by one, in order", which this
   workflow gives up. Rejected along with it: a stack that another thread watches in order to
   build on an unmerged part.
7. **No design skill.** Design discussions stay free in form. Their ending is scripted by
   `record-decisions`, and the form of an issue is defined once, in `issue_tracking.md`.
8. **Milestones are capabilities, and a release is a list of them.** Rejected: one milestone per
   release, which held 95 issues for 0.2.0 and said nothing about progress on any capability.
   Rejected too: a release field on each issue, since the project works on one release at a time.
9. **No new word for a group of work.** "Lane" was tried and dropped. In Scaly it already means a
   vector lane, and the roadmap uses it for its order of work.
10. **Status is a field on the issue, and the GitHub Project is retired.** The organization's
    `workflow status` field takes Backlog, Ready, In progress, In review and Deferred. No value
    means the issue has not been assessed, and a closed issue is done. Rejected: keeping the
    Project, whose status had to be kept in step with the issues and whose views neither
    maintainer's tools need.

## What adopting this changes

**Drafted on this branch**, to be read and corrected before anything else moves:

- the five skills in `.agents/skills/`, with `inspect-issues` and `update-issues` changed to read
  and set the status field
- `docs/dev/issue_tracking.md`, with the form of an issue, the status field, the meaning of
  assignment, and without sub-issues or the Project
- `AGENTS.md`, which points to the workflow and lists the skills

**To do once both maintainers agree**, in this order:

1. **The status field.** It currently has two options that this workflow does not use, Done and
   Triage. Remove them. Setting the field from the command line has not been tried yet, so try the
   command in `update-issues` on one issue first.
2. **The tracker.** Regroup the existing issues into issues of the new form, closing the small
   ones they replace. Close the section issues of the compiler run (#292 to #298). Repopulate the
   capability milestones (sparse IR, loop compiler, control flow, generated solvers) and retire
   `scaly 0.2.0` as a milestone. Clear assignments on issues nobody plans to start soon. The work
   on #70 continues on its branch through all of this.
3. **Status on every issue**, as one mechanical pass after the regrouping, then retire the Project.
4. **Each maintainer's own setup.** Remove or rename personal skills with the same names, and map
   the five roles to models in private instructions.
5. **Repository settings.** The default squash commit message becomes "Pull request title".
6. **The October note** is marked as replaced by this one.

## Open questions

1. **What is the trial?** One issue of the new form, run from design to merge, compared with the
   numbers at the top of this note: issues filed, questions asked, test lines per source line. Both
   maintainers agree on which issue before it starts.
2. **Is one to three days the right size?** It is a default with no evidence behind it. The trial
   should give a first number.
3. **Can an outside contributor see the status field?** Its visibility is set to organization
   members only. If it is hidden from others on a public repository, `issue_tracking.md` should
   say so.
4. **What should a view of the tracker show for this workflow?** At least: which issues are
   running, with their draft pull request; who holds what and whether they are working on it now;
   and progress per milestone. Changes to a particular tool belong in that tool's own tracker.
