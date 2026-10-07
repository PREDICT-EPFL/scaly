---
name: record-decisions
description: Close a Scaly design discussion by listing every decision for the maintainer to confirm, writing each where the next planner looks and slicing the remaining work into issues, or revise a recorded decision and update what depends on it. Use only when the user or another skill asks for it by name, never on your own initiative.
---

# Record decisions

Invoking this skill authorizes editing the roadmap and design notes, committing them on the
thread's branch and opening or updating its pull request, and creating or editing issues through
`update-work`. In the main checkout, ask before committing.

## Close a discussion

1. **Gather.** Read the whole discussion. Each quoted passage with a comment in the maintainer's
   messages is a separate instruction. List every one, including the small ones.
2. **Present one checklist** and wait for the maintainer's answer. This is the only stop.
   - each decision, with the alternatives it rejected and where it will be written
   - the issues to create or edit, each with its title, completion criteria and prerequisites
   - the questions still open, each with a recommended answer and the issue that will settle it
3. **Write each decision where the next planner looks.**
   - Compiler and API design goes in the decisions of `internal/notes/core_compiler_roadmap.md`.
   - Other designs go in a dated note in `internal/notes/`.
   - Choices that matter to one task only go in that issue's body.

   State the rejected alternatives and the reason in a sentence or two, so an implementer does not
   reopen them.
4. **Slice the work** into issues through `update-work`. Each issue fits one fresh thread and one
   reviewable pull request. Each includes its test, its change and its documentation, has
   checkable completion criteria, and links the roadmap section it implements.
5. **Report** the commit or pull request, the sections written, the issues created and the open
   questions.

## Revise a decision

Implemented decisions change through a new issue. A decision that is still only planned already
has issues, pull requests and threads depending on it, so revising it takes three steps:

1. Edit the decision in the roadmap or note. Keep one line saying what it replaced, when and why,
   so the next planner does not raise the old version again.
2. Find every issue that links the decision's section (`gh search issues --repo
   PREDICT-EPFL/scaly "<section anchor>"`, and the issues of the roadmap phase). Update the bodies
   and completion criteria that depend on it.
3. List the open pull requests and running T3 threads that implement those issues. Tell each one
   what changed, through `t3_thread_send` or a pull request comment, or stop it if its work no
   longer applies.

Report what changed and every issue, pull request and thread touched.
