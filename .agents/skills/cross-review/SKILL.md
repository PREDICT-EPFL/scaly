---
name: cross-review
description: Get one independent review of a Scaly change from the model whose role fits it, then confirm the fixes with a short check on the fix alone, or collect a design-panel opinion on a decision that is hard to undo. Use only when a maintainer or another skill asks for it by name, never on your own initiative.
---

# Cross-review

A change gets one review and one check on its fixes. Repeating a full review finds little after
the first and costs the most.

This skill names roles: **the reviewer for taste**, **the second reviewer for taste**, **the
reviewer for prose** and **the reviewer for bugs**. Your maintainer's private instructions say
which model fills each and at which effort. High-risk changes are those to the IR,
differentiation and code generation.

## Review a change

1. **Run the checks first.** `wt hook pre-merge --yes` passes before a reviewer is asked. A review
   is never spent on lint, formatting or types. Commit and push everything, and note the head
   commit: the reviewer reads the committed diff only.
2. **Choose the main reviewer by role.** It is never the model that wrote most of the change:
   - anything that needs taste (design, public interface, user-facing behaviour, the shape of
     generated C): the reviewer for taste, or the second one when the first wrote the change
   - prose: the reviewer for prose
   - purely technical changes where only correctness matters: the reviewer for bugs

   A high-risk change adds a second review by the reviewer for bugs, in a fresh task, that only
   hunts bugs and edge cases. It runs even on code that model wrote, unless it is already the main
   reviewer. Set the main reviewer's effort by risk: its default for code, one level higher for
   high-risk changes, where the reviewer reproduces each finding with a command or a test, and low
   for documentation, tooling, tracker and CI changes. A one-line change of that kind needs only
   CI.
3. **Write the brief:**
   - the issue: its goal and its examples with their **Pin** and **Leave free** lists, and where
     the decided design lives
   - the diff to read, `git diff origin/main...HEAD`. For a large change, the parts from `PLAN.md`
     and the files of each, so that it can be read part by part.
   - the questions to answer:
     1. Does the code meet every Pin item, and does a test assert it? Nothing under Leave free is
        a finding.
     2. Is each choice under *Decisions* sound, and does any contradict the design?
     3. Should any item under *Left out* have been done, because it serves the issue's goal?

     Both lists are in `PLAN.md`. Put them in the brief when the change has no `PLAN.md`.
     4. Does every new test check an example or a real bug? Name tests that only show that code
        exists.
     5. When the snapshots were regenerated, is every changed line explained by the change?
   - the standards: `AGENTS.md`, `docs/dev/conventions.md` and `docs/dev/codebase.md`, and
     `.agents/skills/write-docs/brief.md` for prose
   - for a mechanical migration, whether the change serves each caller and does not merely
     type-check
   - "Do not edit files. Do not comment on the pull request or the issue. Do not report lint,
     formatting or type issues."
   - the output: each finding marked "act on" or "note", with a reproduction or a file and line
     and a reason, then one verdict line, "approve" or "changes needed"
4. **Fix** every "act on" finding, and the notes that are cheap.
5. **Confirm with a delta check, not a new review.** A new task for the same reviewer model, at
   low effort, receives the findings, the responses and `git diff <reviewed commit>..HEAD` only. It
   answers whether each finding is resolved and whether the fix broke anything next to it. It does
   not look for new problems elsewhere.
6. **Stop there.** If the delta check still finds something to act on, fix it and list it as an
   open point in the pull request description. Do not start a third round. A change that needs
   one rests on an unclear decision, and the maintainer should see that.

## Collect a design panel

For a decision that is hard to undo, send the same brief to the reviewer for taste and the
reviewer for bugs, and to the second reviewer for taste for the hardest decisions. Ask each for a
recommendation and its main risks. Sort the findings into act on, consider, noted and dismissed,
and say where the models agree. Keep the brief and the memos in `internal/notes/<topic>_<date>/`,
never only in `/tmp`.

## Launch

A reviewer is a fresh agent that has none of your history. Call `orchestrator_capabilities` once
per thread to find the provider and model for the role, and start the reviewer with
`delegate_task`, `role: "review"`, in async mode. Follow the task's `next`
and do not poll it. Use a new `delegate_task` and a new `clientRequestId` for the delta check,
carrying the brief, the findings and the responses. Do not send it to the earlier child thread.

If the model for a role is unavailable, say so and ask. Do not switch to another model unasked.

## Record

The pull request shows the result, not the rounds. Post no review comments. Put one line in the
description, naming both reviewers when there were two:

```text
Reviewed by <model> (<effort>) at <commit>; fixes confirmed at <commit>. Open: <point> | none.
```

The findings and how each was handled go in the final report to whoever started the work. A
reviewer that failed, timed out, was cut short or hit a usage limit is reported as missing on that
line and in the report, never as a pass. Retry a missing review once. If it is still missing, the
pull request may be marked ready only with that stated first under *Review* in its description,
and the maintainer decides whether to merge without it.
