---
name: record-decisions
description: Close a Scaly design discussion by listing every decision for the maintainer to confirm, writing each where the next planner looks and writing the issues that are ready to start, or revise a recorded decision and update what depends on it. Use only when a maintainer or another skill asks for it by name, never on your own initiative.
---

# Record decisions

A design discussion is free in form. This skill is its ending: it turns what was decided into the
design document and into issues that a thread can start on without further conversation.

[Tracking work](../../../docs/dev/issue_tracking.md) defines the form of an issue.
[Agent workflow](../../../internal/notes/agent_workflow.md) gives the reasons. Read the tracker
through `inspect-issues` and change it through `update-issues`.

Invoking this skill authorizes editing the design documents and notes, committing them on the
thread's branch and opening or updating its pull request, and creating, editing and closing the
issues the maintainer approved in the checklist. In the main checkout, ask before committing.

## Close a discussion

1. **Gather.** Read the whole discussion. Each quoted passage with a comment in the maintainer's
   messages is a separate instruction. List every one, including the small ones.
2. **Check the code and the tracker.** For each result the discussion wants:
   - read the code it will change, and run each example's input on main, so that "what happens
     today" in the issue is observed and not guessed
   - search the open issues for ones it covers, overlaps or depends on
   - list the open pull requests, drafts included, that touch the same code
3. **Draft the examples.** Where the Python a user writes or the C the compiler generates is a
   matter of taste, write the target code and, under it, **Pin** (what a reviewer checks) and
   **Leave free** (what the implementer decides). A result that touches several parts of the
   library gets one example per part. A measurement states its numbers. A bug carries its
   reproduction.
4. **Present one checklist** and wait for the maintainer's answer. This is the only stop.
   - each decision, with the alternatives it rejected and where it will be written
   - each issue to create: title, goal, examples, milestone and blockers
   - each existing issue to fold into a new one, and each to edit
   - the questions still open, each with a recommended answer and the issue that will settle it
5. **Write each decision where the next planner looks.**
   - Compiler and API design goes in the decisions of `internal/notes/core_compiler_roadmap.md`.
   - Other designs go in a dated note in `internal/notes/`.
   - A choice that matters to one issue only goes in that issue.

   State the rejected alternatives and the reason in a sentence or two, so that an implementer
   does not reopen them.
6. **Write the issues.** Each one:
   - describes one coherent result, sized by default for one to three days of one thread. A
     result is coherent when it stands on its own on main and can be reviewed against its examples
     as a whole. Do not cut a result into steps, and do not plan sub-issues.
   - follows the form in `issue_tracking.md`: goal, examples, out of scope, design section
   - carries a milestone, and a blocked-by relation for each prerequisite that is not on main
   - gets the status Ready when nothing blocks it, otherwise Backlog. An issue whose examples or
     decisions the maintainer left open gets Backlog, a *Needs* section naming what is missing,
     and the `needs:examples` or `needs:decision` label.

   Fold each small issue the new one covers: carry its content into the examples, then close it as
   a duplicate with a comment naming the new issue.
7. **Report** the commit or pull request, the sections written, the issues created, folded and
   edited, and the open questions.

An issue written here is ready to start. `plan-next` chooses when, and changes its content only if
the code has moved since.

## Revise a decision

Implemented decisions change through a new issue. A decision that is still only planned already
has issues, pull requests and threads depending on it, so revising it takes three steps:

1. Edit the decision in its design document or note. Keep one line saying what it replaced, when
   and why, so that the next planner does not raise the old version again.
2. Find every issue that links the decision's section, by searching the tracker for the section's
   anchor. Update the goals and examples that depend on it.
3. List the open pull requests and the running threads that implement those issues
   (`t3_thread_list`). Tell each one what changed, with `t3_thread_send` or a pull request comment,
   or stop it if its work no longer applies.

Report what changed and every issue, pull request and thread touched.
