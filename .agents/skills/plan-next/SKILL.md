---
name: plan-next
description: Survey Scaly's open issues and running work, propose which issues to start now and in what order they land, and after the maintainer approves, write the message that starts each one. Never edits code or launches threads.
disable-model-invocation: true
---

# Plan the next work

This skill answers one question: what do we start now? The input is optional: a milestone, an
area or a list of issues. Without one, survey every open issue. The output is a proposal. Nothing
changes on the tracker and nothing starts before the maintainer approves.

Issues written by `record-decisions` are ready to start. Do not reshape them. Change an issue's
content only when the code has moved since it was written, and say what moved.

[Tracking work](../../../docs/dev/issue_tracking.md) defines status, assignment and the form of an
issue. Read the tracker through `inspect-issues` and change it through `update-issues`.

## 1. Survey

Collect:

- every open issue in scope, with its status, milestone, blockers, assignee and recent comments
- the open pull requests, drafts included, and the files each touches (`gh pr diff <n>
  --name-only`). A draft is how running work shows what it edits.
- the running threads (`t3_thread_list`), so that nothing starts twice
- the *Order of work* and *Release scope* of `internal/notes/core_compiler_roadmap.md`

## 2. Check each candidate against the code

The tracker can be wrong. For each issue that looks ready:

- **Is it already done?** Search main for the change. Before proposing to close an issue, run the
  check the issue names and report what it showed.
- **Is it still true?** Run an example's input on main. If "what happens today" has changed, say
  so and propose the edit.
- **Is the failure real?** For a CI failure, read the run history. Waiting for the next nightly
  is a valid proposal.
- **Can an agent finish it?** Name anything it needs that an agent lacks: a kind of machine, a
  measurement on the reference machine, a platform only CI has. The start message then says who
  provides it.

## 3. Read who holds what

| On the issue | Meaning | What to propose |
|---|---|---|
| Unassigned | Free | Either maintainer may take it |
| Assigned, Ready or Backlog | Its assignee wants it soon | Only for its assignee, or ask them |
| Assigned, In progress | Being worked on | Leave it |

Never propose taking an issue from its assignee. Put the question to the maintainer instead.

## 4. Work out what can run together

Two relations matter, and only the first belongs on the tracker:

- **B needs A** when B uses something A builds. This is recorded as *blocked by*. B waits for A
  to be on main. If B needs only a part of a running A, propose one of: move that part into B,
  have A's thread land the part early, or let B wait.
- **A and B collide** when they edit the same code. This depends on the code today and on what is
  running, so work it out each time and never record it. Every claim names the file and the
  function where the two meet. A claim without that evidence is dropped.

Then:

| Situation | Handling |
|---|---|
| B collides with running work or with A | One after the other, or propose merging them into one issue when they are the same result |
| Both regenerate the C snapshots | They may run together. The second marks its pull request ready only after the first has merged. Say which goes first. |
| Same file, different functions | Together. The merge queue tests each on top of the ones ahead of it. |
| Changes that `AGENTS.md` groups (the plugins' `hatch_build.py`, renames of generated C symbols), performance changes measured against one baseline, renames of public names | One issue, or one right after the other |
| No overlap | Together |

Each issue ends in a review by a maintainer, and several large reviews on one day are worse than
the same work in sequence. Say who reviews what, and spread the endings.

## 5. Group the loose issues

Bug reports and agent-filed follow-ups never had a design discussion, and issues without a status
have not been assessed. For those only:

- propose a status for each unassessed issue, or its closure with a reason
- where several loose issues ask for changes to the same code toward one result, propose one
  issue in the form `issue_tracking.md` gives, carrying their content as its examples, and the
  closure of the ones it folds

## 6. Propose

Send one message:

1. the tracker changes: statuses, issues to close with the evidence, grouped issues to create,
   edits to issues that the code has overtaken
2. the issues to start now, those in the current release's milestones first, each with its
   evidence that nothing collides, the machine it needs and who reviews it
3. the order in which they land, with the snapshot order stated
4. every open question, each with a recommended answer

Then stop and wait for the maintainer.

## 7. Hand over

Apply the approved tracker changes through `update-issues`. Then write the message that starts
each issue: `implement` and the issue, plus only what the issue does not say, such as "keep the
pull request in draft until #N merges". The maintainer chooses the model and the machine and
starts each thread.

Forbidden: editing code, merging, launching threads, changing an issue's examples beyond what
step 2 showed, and closing an issue without the check from step 2.
