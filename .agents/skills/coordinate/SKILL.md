---
name: coordinate
description: Coordinate a batch of related, partly dependent Scaly issues that belong in one pull request or one stack, launching one implement thread per issue in its own worktree. Use when the maintainer hands over a batch of issues. The coordinator never edits code.
disable-model-invocation: true
---

# Coordinate issues

The input is a list of issues, or a parent issue whose children form the batch. The coordinator
plans, launches, watches and relays. It never edits code, and it inherits the forbidden actions of
`implement`.

## Plan

1. Read each issue with `inspect-work`: prerequisites, the roadmap sections linked, the files
   likely touched.
2. Order the issues with the table in `plan-next`'s *Order and group the Ready issues*. When the
   invocation already states the order, follow it.
3. Choose the shape by asking whether the pieces will land together anyway:
   - **One lane branch** when they will, or when they cannot pass CI separately. Each child works
     in its own worktree and merges back with `wt merge <lane-branch>`. Never run `wt merge`
     without a target, because the default target is main. One pull request carries the lane.
   - **A stack** when each piece should be reviewed and landed on its own. One branch and one pull
     request per issue, each based on the branch below. Manage it with GitHub's stacked pull
     requests through the `gh stack` extension (`github/gh-stack`), which tracks the stack across
     worktrees and rebases the remaining branches after a squash merge. Read `gh stack --help`
     before the first use in a session.
4. Post the plan on the parent issue, or on the first issue: the order, the shape, the branches,
   and which issues can start at once.

## Launch

Launch one thread per issue with `t3_thread_launch`, a title naming the issue, and:

```json
{"workspaceStrategy": {"type": "worktree", "baseRef": "<branch below, or main>",
  "branch": "t3code/issue-<n>-<slug>", "startFromOrigin": false},
 "message": "<brief>"}
```

Pick the provider with `modelSelection` (`instanceId` `codex` or `claudeAgent`). Keep each
returned thread id. A launch has no retry key, so after an error check `t3_thread_list` before
launching again.

Independent issues start at once. A dependent issue starts when the branch below has its review
verdict, not when it merges. Start with two threads at a time across the batch, and do not launch
past about 70% of a provider's usage window. Alternate providers between implementation and review
so neither subscription carries both.

The brief, pasted into each launch:

```text
Use the implement skill on #<n>.
Goal: <one sentence>.
Scope: <files and behaviour in scope>. Out of scope: <what neighbours own>.
Design: <roadmap section>. It is decided.
Base: <branch>. Stack position: <k of m>. Pull request base: <branch>.
Completion: the issue's criteria. Also: <anything the batch adds>.
Forbidden: merging, pushing to main, tags, and editing files another issue in this batch owns.
Report: the pull request link, evidence per criterion, open points, follow-up issues.
```

## Watch

Launched threads are top-level, so nothing notifies the coordinator. While threads run, loop:
`t3_thread_wait` on one thread with a timeout of about ten minutes, then `t3_thread_list` across
the batch for any thread that finished or is `waiting`. A timeout does not stop the thread. Read new
output with `t3_thread_read` from the last position. Answer a child's question with
`t3_pending_request_respond`, or relay it to the maintainer in one batched message per round, with
a recommended answer for each. Permission requests need the maintainer. Send instructions with
`t3_thread_send`, `mode: queue` unless the running turn must change. Stop a thread that has become
pointless with `t3_thread_interrupt`. Once every running thread waits on the maintainer's review,
end the turn and let the maintainer wake the coordinator.

The maintainer lands a stack through the merge queue, with `gh stack merge`. A layer ejected from
the queue takes every layer above it with it: tell the thread that owns the ejected layer, and once
it is fixed, tell the maintainer the stack can be queued again. When the bottom of a stack is
squash-merged on its own, let `gh stack` rebase the rest and tell each thread to rerun its checks. Without the extension, rebase the next branch with
`git rebase --onto origin/main <old base tip>` in its own worktree, push with
`--force-with-lease`, and retarget its pull request with `gh pr edit --base main`. When an
unrelated pull request merges into main, tell in-flight threads whose files it touched to rebase
and rerun.

## Finish

The batch is done when every pull request is open with a verdict and green checks, or merged.
Report each issue's pull request, verdict and open points, and the issues still blocked.
