---
name: plan-work
description: Survey the Scaly backlog, propose which issues to make Ready or move into the current release, group the Ready ones by what they share, and after the maintainer approves, write the coordinate-issues and implement-issue messages that start them. Never edits code or launches threads.
disable-model-invocation: true
---

# Plan the next work

The input is optional: a milestone, an area, or a list of issues to consider. Without one, survey
the current release milestone and the open issues outside it. The output is a proposal for the
maintainer. Nothing changes on the tracker and nothing starts before the maintainer approves.

## 1. Survey

Use `inspect-work` for the reading. Collect, for every open issue in scope: its Project status,
milestone, blockers, assignee and recent comments. Also collect the open pull requests and the
files they touch (`gh pr diff <n> --name-only`), and the running threads (`t3_thread_list`), so
that nothing already in progress gets started twice. Read the roadmap's *Order of work* and
*Release scope*. An open issue missing from the Project is a finding in itself.

## 2. Check each candidate against the code

The tracker can be wrong. For each issue that looks ready, check:

- **Is it already done?** Search main for the change the issue asks for. A fix can land as a side
  effect of other work. Before proposing to close an issue, run the check the issue itself names
  and report what it showed.
- **Is the failure real?** For a CI failure, look at the run history. A failure seen once may be a
  one-off, and waiting for the next nightly run is a valid proposal.
- **Can an agent finish it?** Name anything it needs that an agent lacks: a Windows machine, a
  measurement on la015, a platform only CI has. Such an issue can still be Ready, but its brief
  says who provides the missing part.
- **Does it fit the milestone?** The roadmap's *Release scope* says what a release takes. Propose
  adding an issue only on those grounds, and say which one applies.

## 3. Order and group the Ready issues

Two relations between issues decide the order, and only the first belongs on the tracker:

- **B needs A** when B uses something A builds. This is a prerequisite, recorded as *blocked by*.
- **A and B collide** when they edit the same code or the same generated files. This depends on
  the current code and on what else is in flight, so work it out here each time and never record
  it as a dependency.

Every ordering claim names the file and the function where the two changes meet. A claim without
that evidence is dropped. Then handle each pair:

| Situation | Handling |
|---|---|
| B needs A, or B edits the lines A edits | A stack, or B starts once A has its review verdict |
| Both regenerate the C snapshots | One after the other: B starts once A has merged |
| Same file, different places | In parallel. The merge queue tests each on top of the ones ahead of it |
| No overlap | In parallel |

Three more reasons put issues in one group even without a collision: two performance changes
measured against the same baseline on la015, renames of generated C symbols or public API that
users should meet in one release, and changes to `plugins/*/hatch_build.py`, which each cost a cold
solver rebuild.

Keep stacks short. In the merge queue, a layer that is ejected takes every layer above it with it,
so a stack ties together work that does not otherwise depend on each other.

## 4. Choose how each group starts

- A group of more than one issue goes to `coordinate-issues`, with its shape stated in the invocation:
  which issues start at once, which wait and for what.
- A lone issue goes to its own `implement-issue` thread, with the brief from `coordinate-issues`.
- Follow the budget in the global `AGENTS.md`: at most two implementing threads per provider at
  once, no new launch past about 70% of a provider's window, and implementation and review on
  different providers. Put Linux-bound and measurement work on la015.

## 5. Propose

Send one message:

1. the tracker changes: milestone additions, moves to Ready, issues to close with the evidence,
   issues missing from the Project
2. the groups, each with its evidence and how it starts
3. the expected merge order, as chains of issues plus the issues that can merge whenever
4. every open question, each with a recommended answer

Then stop and wait for the maintainer.

## 6. Hand over

Apply the approved tracker changes through `update-work`. Then write out, for each group and each
lone issue, the message that starts it: the `coordinate-issues` invocation with the shape, or the
`implement-issue` brief. Recommend a provider for each, and a machine where the work needs one. The
maintainer chooses where each one runs and starts it.

Forbidden: editing code, merging, launching threads, and closing an issue without the check from
step 2.
