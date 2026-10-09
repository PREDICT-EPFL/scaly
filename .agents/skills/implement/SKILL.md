---
name: implement
description: Implement one Scaly GitHub issue, or a small change the maintainer describes, end to end in its own worktree, from the brief and a failing test through the checks, one cross-review and the pull request, then address the maintainer's review. Use only when the user or another skill asks for it by name, never on your own initiative.
---

# Implement a Scaly change

The input is an issue number, and optionally a brief from `coordinate` that names the base
branch, the position in a stack and any limits on scope. The brief and the issue together replace
a conversation with the maintainer. Without an issue, the input is a small change the maintainer
describes in the conversation, and [Without an issue](#without-an-issue) says what changes.

Invoking this skill authorizes committing, pushing the change's branch, opening and updating its
pull request, and updating the issue and its Project item when there is one. It does not authorize merging a pull
request, pushing to main, creating tags or releases, running the deployment workflows, closing
issues by hand or changing an issue's completion criteria. Those stay with the maintainer.

Implementation steps stay out of public records. When the work has several steps, keep them in a
`PLAN.md` at the root of the worktree, committed on the branch so another thread can resume it, and
delete it in the last commit before the pull request is marked ready. Issue comments and the pull
request state scope, status, results and blockers.

## Without an issue

The pull request takes the issue's place as the record. Before writing code:

1. Write the completion criteria down in a few lines in the conversation, unless the request
   already states them. They go into the pull request description in step 6.
2. Decide whether the change should have an issue after all. Recommend one, with a draft title and
   body, and wait for the maintainer when the change:
   - needs a design decision, or reopens a decided one
   - regenerates C snapshots
   - is unlikely to finish in one thread
   - reaches beyond what the maintainer described

   Check again whenever the work grows, and stop to recommend the issue when it crosses one of
   these lines. Once the maintainer agrees, file it through `update-work` and continue as for an
   issue.

The steps below then apply with these differences:

- In step 1, only check the checkout.
- In step 2, read what the change touches, and the roadmap section it falls under if there is one.
- In step 6, the description states the completion criteria itself and names the pull request
  the change follows up, if any. There is no `Closes` line and no Project item.
- In step 8, the handoff comment goes on the draft pull request.

## 1. Brief

1. Run `inspect-work` on the issue: the body, comments, prerequisites, parent, linked pull requests.
2. Check the checkout. `git rev-parse --git-dir` and `git rev-parse --git-common-dir` differ in a
   linked worktree. In the main checkout, stop and ask for a worktree.
3. Stop and report on the issue if any of these holds:
   - a prerequisite is open, unless the brief stacks this branch on top of it
   - this issue will regenerate C snapshots and another open pull request does too
     (`gh pr list` and `gh pr diff <n> --name-only`), unless the brief stacks this branch on top
     of that pull request
4. Comment on the issue with the branch, and move its Project item to In progress through
   `update-work`.

## 2. Read

- The roadmap section the issue links and the roadmap's *Rules for every item*.
- The design notes the issue links, and `internal/notes/compiler_maintenance_context.md` for
  compiler work.
- *Where to add things* in `docs/dev/codebase.md`.
- The tests and examples devrush has for the feature, ported with `port-from-devrush`.

The decided design and the alternatives it rejected are fixed. If the code shows that the design
cannot work as written, stop and post the evidence on the issue rather than choosing another
design. When the maintainer changes the design in a comment, apply the change and record it with
`record-decisions`.

## 3. Test first

Write the test that shows each completion criterion. For IR, differentiation and code generation,
that is a differential test against NumPy or an unrolled reference. Run it and see it fail for the
reason the issue describes. Once it passes, perturb the implementation and see it fail again.

## 4. Change

Make the smallest change that follows *Where to add things*, the import layers and the
public-name rules. Something needed but outside the issue's scope becomes a follow-up issue
through `update-work`, named in the pull request. Commit in small steps with plain titles,
without conventional prefixes and without citing hashes from this branch.

## 5. Prove it

Run `wt hook pre-merge --yes` for every change; without `--yes` it waits for an approval that
nobody gives in an unattended thread. It runs `ty`, both Ruff checks, the full suite, the
benchmark smoke, the documentation build and the private-name check. Then add the proof the change
type needs:

| Change | Also needed |
|---|---|
| IR, differentiation, code generation | C snapshots byte-identical, or regenerated only when the issue says so, with the reviewed diff and measurements of source size, generation time and run time |
| User-visible feature | Its guide page and API entries, written with `write-docs`, and the README and `examples/` still running |
| Bug fix | The new test reproduces the reported failure before the fix |
| Performance claim | A measurement with `benchmark-study`. A Python-level timing is never evidence. |
| Tooling or CI | The changed command run locally where possible, and the pull request's own CI run |

## 6. Pull request and review

1. Push and open a draft pull request. Base it on the branch below in a stack, otherwise on main.
   Link it in T3 with `link_pull_request`.
2. Run the `cross-review` skill: one review and one delta check. It lives in the maintainer's
   shared skills. Without it, ask a model of another family for one review of the diff
   against the issue, fix what it finds, and ask once more about the fixes only. Post the verdict
   on the pull request.
3. Write the description: what changed and why in a few sentences of prose, each completion
   criterion quoted with its evidence (a test, a command and its result), the validation run, any
   point the review left open, and the follow-up issues. End with `Closes #N`, or `Part of #N`
   for a partial change.
4. Mark the pull request ready, move the Project item to In review, and call
   `watch_pull_request`. CI results, review comments and conflicts wake the thread.

When CI fails, fix and push. A failure listed among the known flakes in
`docs/dev/contributing.md` gets one rerun.

## 7. When the maintainer reviews

1. Read every unresolved review thread through the GraphQL `reviewThreads` field, and the review
   summary.
2. Treat each comment as a rule, not a single line. A comment on one example applies to every
   example in the pull request, and to every page it touched.
3. Fix, run the checks for what changed, push, and reply to each comment with what changed.
   Resolve the comments applied as written. Leave open the ones interpreted, questioned or
   disputed.
4. Ask for a delta check only when a fix changed IR, differentiation or code generation logic.
   The maintainer's comments were the review.

## 8. Running out of time or budget

Update `PLAN.md` with what is done and what remains, commit and push the branch, and post a short
handoff comment on the issue: the branch, what was run with its result, and what blocks. The next
thread starts from the comment, the branch and its `PLAN.md`.

## Report

End with the pull request link, the evidence for each completion criterion, the review outcome,
the open points and the follow-up issues.
