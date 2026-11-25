---
name: implement
description: Build one Scaly issue end to end in its own worktree, from its examples and failing tests through delegated parts, the finishing steps, one cross-review and the pull request a maintainer reviews, then address that review. Use only when a maintainer or another skill asks for it by name, never on your own initiative.
---

# Implement an issue

The input is a Scaly issue, large or small. One thread owns it from the first commit to the merge:
the branch, the pull request, its CI and every decision inside the agreed design. Without an issue,
the input is a small change the maintainer describes, and [Without an issue](#without-an-issue)
says what changes.

[Tracking work](../../../docs/dev/issue_tracking.md) defines the form of an issue, its status and
what assignment means. [Agent workflow](../../../internal/notes/agent_workflow.md) gives the
reasons behind this procedure. Read the tracker through `inspect-issues` and change it through
`update-issues`.

This skill names roles, not models: **the implementer** (writes delegated parts), **the reviewer
for taste**, **the second reviewer for taste**, **the reviewer for prose** and **the reviewer for
bugs**. Your maintainer's private instructions say which model fills each, on which machine work
runs, and when to stop starting agents because of usage limits. If they name no model for a role
you need, ask once.

Invoking this skill authorizes committing, pushing the issue's branch, opening and updating its
pull request, delegating parts, and updating the issue. It does not authorize merging or queueing
a pull request, pushing to main, creating tags or releases, running the deployment workflows,
closing issues by hand, changing an issue's examples, or taking an issue assigned to someone
other than the maintainer who started you.

## 1. Start

1. Read the issue with `inspect-issues`: its goal, examples, discussion, blockers, assignee, status
   and linked pull requests. Link it to this thread with `link_issue`. Title the thread with a
   short name for the result, without the issue number.
2. Check the checkout. `git rev-parse --git-dir` and `git rev-parse --git-common-dir` differ in a
   linked worktree. In the main checkout, stop and ask for a worktree.
3. If the issue already has a branch and a draft pull request, an earlier thread started it.
   Confirm with the maintainer who started you that the earlier thread has stopped, then resume:
   check out its branch and read `PLAN.md` and the last comment on the issue. `PLAN.md` says
   which step the work is at and lists the subagents that were running. Read each one's thread
   and branch, and merge or discard its work before giving its files to anyone else. Without a
   `PLAN.md`, the pull request tells you where the work is: one that is ready for review is at
   step 8, and a draft belongs to a small issue that never had a plan, whose commits and
   description show what is done. Do not open a second pull request.
4. Otherwise, stop and report if any of these holds:
   - a blocker is open. An issue depends on what is on main, never on another branch.
   - the issue is assigned to someone else
   - another open pull request, drafts included, edits code this issue will edit (`gh pr list`, then
     `gh pr diff <n> --name-only`). Name the file and the function where they meet.
   - the issue carries a `needs:` label. Its *Needs* section says what a maintainer must settle
     first.
   - the issue has no examples and is more than a bug with a reproduction or a measurement with
     its numbers. Recommend `record-decisions` to write them.
5. Assign the issue to the maintainer who started you. GitHub moves it to In progress when the
   draft pull request of step 3 opens, and to In review when you mark it ready.

## 2. Read

- The design section the issue links, and the roadmap's *Rules for every item*.
- For compiler work, `internal/notes/compiler_maintenance_context.md` and *Where to add things*
  in `docs/dev/codebase.md`.
- The tests and examples on the frozen experimental branch that the issue names, through
  `port-from-devrush`.

The decided design and the alternatives it rejected are fixed. Two cases come up:

- **The design is silent.** Decide it yourself, prefer the choice that is easy to undo, and record
  it under *Decisions* in `PLAN.md` with the alternative. Ask the maintainer only when the choice
  changes a public interface, changes behaviour the documentation states, or contradicts the
  design.
- **The design cannot work as written.** Stop that part, post the evidence on the issue, and
  continue with the parts that do not depend on it. The maintainer revises the design through
  `record-decisions`.

## 3. Plan, and open the draft

Write `PLAN.md` at the root of the worktree. It lists the parts of the work, and for each part the
examples it serves, the files it owns and its state. For a delegated part it also records the
subagent's thread, its branch and when it started. It holds *Decisions* and *Left out* too.
Steps live here and never in the issue or the pull request.

A small issue that you will write alone in a few commits needs no `PLAN.md`. Its decisions and
left-out items go straight into the description, and its first commit is its first change,
normally the failing test from step 4.

Commit, push the branch and open a draft pull request at once with `pull_request_create`. Its
description starts as the issue's examples in a checklist and `Closes #N`. Link it with
`link_pull_request` if creation did not. From here on:

- push after every change you integrate, so that the draft always shows which files the work
  touches
- tick an example in the description when its test passes on the branch

## 4. Turn the examples into tests

Before any implementation, write one test per example, of the kind the area prescribes: a
differential test against NumPy or an unrolled reference for the IR, differentiation and code
generation, and a check on the lowered program for a statement about loops, buffers or counts.

- Each item under **Pin** becomes an assertion. Nothing under **Leave free** is asserted.
- Run each test and see it fail on main for the reason the issue gives. Once the implementation
  passes it, perturb the implementation and see the test fail again.
- An example that states behaviour main already has, which the change must keep, needs no new
  test when an existing one asserts it. Name that test. An example on a documentation page is
  checked by running it.
- Add no other test, with two exceptions: a regression test for a bug found on the way, and a
  test a reviewer asks for with a reason. A test that only shows that new code exists is not
  written.

Commit the failing tests before delegating, because a delegated worktree starts from your last
commit and does not see uncommitted edits.

## 5. Build

Write a small issue yourself. Delegate when the issue has two or more parts that own different
files. Parts that edit the same function stay with one writer.

To delegate, call `orchestrator_capabilities` once, then start each part with `delegate_task`,
`isolation: "worktree"`, `role: "implementation"` and the implementer's model. Start with two at
a time. The brief is all the subagent knows, so it states:

- the issue's goal in two sentences, and the part
- the examples this part serves, with their Pin and Leave free lists, and the failing tests
  already on the branch
- the files it owns, and the files it must not touch
- the design section, and what was decided in `PLAN.md` that affects it
- the checks to run: `uv run ruff format`, `uv run ruff check`, `uv run ty check` and the tests of
  its part
- where measurements and scripts are committed: `internal/notes/<topic>_<date>/`, never `/tmp`
- what it must not do: open a pull request, push, touch the tracker, file issues, weaken an
  assertion, regenerate the C snapshots, or widen its part
- the report: its branch and head commit, what it did, what it decided, the test results, and
  anything it found outside its part

When one part needs another, merge the first before starting the second, so that the second's
worktree starts from a commit that has it.

Follow the task's `next` and wait for its completion. Do not poll. Answer a subagent's question
yourself, with `t3_pending_request`, when the issue or `PLAN.md` settles it. Pass a permission
request on to the maintainer. Then, for each part:

1. Read the diff before merging it. If it weakened an assertion, restore the assertion first.
2. Merge its branch into yours and resolve conflicts yourself.
3. Run the tests of the parts merged so far, and push.

Parts that own different files can still disagree, for instance on a layout, a signature or a
generated symbol. State such a shared contract in both briefs, and after merging both parts run an
example that crosses it. When a design assumption fails, stop the subagents whose parts depend on
it with `task_cancel`.

Fix small defects in a returned part yourself. If a part misses its examples, keep what is usable
and delegate the remainder as a new part with a new brief.

You may edit code throughout: conflicts, CI failures and review remarks are yours. A fix that
changes logic in the IR, differentiation or code generation after the review gets a delta check,
the short check on a fix alone that `cross-review` defines, from the main reviewer.

What you or a subagent finds on the way goes one of two ways:

- **It serves the issue's goal:** add it to `PLAN.md` and do it. The goal bounds the issue, and
  the examples illustrate it. Write an example for the new work in the issue's form, test it as
  in step 4, and mark it "added by the thread" in the description. The maintainer may refuse it.
- **It does not:** add it to *Left out* with the reason. There are two good reasons: doing it
  would widen the issue beyond its goal, or it waits on work outside the issue, which you name.
  "It was not listed", "it is hard" and "it can come later" are not reasons.

If the issue turns out much larger than its one to three days, say so on the issue with what is
done and what remains. The maintainer decides whether a part lands early or the issue is split.

Run the whole suite once all parts are merged. The snapshot tests may fail until step 6
regenerates them. No other failure is acceptable there.

## 6. Finish

1. **Catch up with main.** Run `git fetch`, rebase onto `origin/main` and push with
   `--force-with-lease`, as you also do earlier whenever the draft shows a conflict. If the branch
   changes generated C, keep your pull request in draft while another one that regenerates the
   snapshots is ready for review or queued, or is the one your start message says lands first.
   Once that one has merged, fetch and rebase again.
2. **Prune.** Run `prune-tests` on the tests the branch added or changed, with the issue as
   context.
3. **Regenerate the C snapshots, once**, and regenerate the node-ID baseline with the recipe in
   `docs/dev/contributing.md`. Read the snapshot diff line by line.
4. **Prove it.** Run `wt hook pre-merge --yes`, then add what the change type needs:

   | Change | Also needed |
   |---|---|
   | IR, differentiation or code generation | Snapshots byte-identical, unless the issue says to regenerate them. Then the reviewed diff and measurements of source size, generation time and run time. |
   | User-visible feature | Its guide page and API entries, written with `write-docs`, and the README and `examples/` still running |
   | Bug fix | The new test reproduces the reported failure before the fix |
   | Performance claim | A measurement with `benchmark-study`. A quick timing is never evidence. |
   | Tooling or CI | The changed command run locally where possible, and the pull request's own CI run |

   Commit raw measurements, scripts and the snapshot audit under `internal/notes/<topic>_<date>/`.
5. **Review.** Commit and push everything first: a reviewer reads the committed diff and sees
   nothing else. Then run `cross-review`, giving it the issue, the diff, and `PLAN.md` or, without
   one, your decisions and left-out items.
6. **Hand over.** Check once more that no other pull request that regenerates the snapshots has
   become ready in the meantime. Delete `PLAN.md` in the last commit, write the description below,
   and mark the pull request ready. Check that the issue's status is now In review.

## 7. The description

A maintainer reviews from the description, so write it for that. The visible part holds, in this
order:

1. **What changed and why**, in a few sentences a reader can follow without opening the issue.
2. **Examples.** Each example from the issue with the same input, what main does and what the
   branch does, using real output, and the name of its test. A pull request with an unmet Pin
   item stays in draft: meet it, or ask the maintainer to change the example. For an internal
   change, show what happens inside the compiler even when the output stays the same.
3. **Decisions.** Each choice you made where the design was silent, with the alternative.
   "None" when there were none.
4. **Left out.** Each item with its reason. The maintainer may refuse any of them. "None" when
   nothing was.
5. **Review.** The line from `cross-review`, the points it left open, and anything a reviewer
   could not check.
6. **Measurements**, when the issue asked for them.
7. `Closes #N`, or `Part of #N` for a part that lands early.

Below it, a `<details>` block labelled with your model and effort (`<summary>Opus 5.5
high</summary>`, then a blank line) holds the validation run, which model built which part, and
the commands behind each number.

The title states the result in plain words. It becomes the commit on main.

## 8. Until it merges

Watch the pull request with `watch_pull_request` and `until: "merged"`, and keep the watch between
turns.

- **CI fails:** fix and push. A failure the project lists as a known flake gets one rerun with
  `ci_rerun`.
- **The code changes after the review**, through a fix or a rebase: rerun the checks it affects.
  If generated C could change, regenerate the snapshots and compare the diff with the reviewed
  one. Any hunk that differs gets a delta check.
- **The maintainer reviews:** read the summary and the unresolved threads with
  `pull_request_read`. Treat each comment as a rule: a remark on one example applies to every
  example and every page the pull request touched. Fix, run the checks for what changed, push, and
  reply to each comment with `pull_request_respond`. Resolve the comments applied as written and
  leave open the ones you interpreted or dispute. A refused *Left out* item is work to do now,
  and the description and its evidence are updated when it is done.
- **A decision is needed:** ask with your question tool, with a recommendation. If you are blocked
  without one, stop your watches and end with **Needs you** and the exact action.
- **It merges:** file each remaining *Left out* item as an issue in one batch, with `update-issues`
  and the label `source:agent`, each with its goal and whatever example or reproduction you have.
  Then unwatch.

Never merge or queue. The maintainer does both.

## Landing a part early

When the maintainer says that other work needs a part of this issue now:

1. Finish that part first, on your branch as usual.
2. Make a second branch from `origin/main` that holds only that part's commits and tests, and
   open a pull request for it that says `Part of #N`. The part must stand on its own on main. The
   issue's own draft stays open with `PLAN.md` in it.
3. Run steps 6 and 7 on the second branch, for the examples of that part only. The issue's
   status stays In progress, and a comment on the issue says which part is in review.
4. After it merges, fetch, rebase the issue's branch onto main and continue. *Left out* items are
   filed only when the issue's last pull request has merged.

## Without an issue

The pull request is the record. Before writing code:

1. Write the change's examples in a few lines in the conversation, unless the request already
   states them. They go into the description.
2. Recommend an issue, with a draft in the form `docs/dev/issue_tracking.md` gives, and wait for
   the maintainer when the change needs a design decision or reopens one, regenerates the C
   snapshots, is unlikely to fit one thread, or reaches beyond what was described. Check again
   whenever the work grows.

The steps above then apply, except that nothing reads or updates an issue, the description names
the pull request this change follows up, if any, and a handoff comment goes on the draft pull
request.

## Running out of time or budget

Update `PLAN.md`, or the draft's description when there is no plan, with what is done, what
remains and the state of every running subagent, commit
and push, and post a short comment on the issue: the branch, what was run with its result, and
what blocks. Leave the draft open. The next thread starts from the comment, the branch and
`PLAN.md`.

## Report

End with the pull request link, each example with its evidence, the decisions, the left-out items,
the review findings and how each was handled, and anything still open.
