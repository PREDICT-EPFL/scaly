# Compiler run, started 2026-11-10: plan

One unattended run that takes the core compiler roadmap from the sparse stack to the generated
solvers, on `main`, without the maintainer reviewing along the way. It is an experiment: the
maintainer reviews afterwards, section by section, and what the review changes lands on top as
ordinary work. Nothing is tagged or released during the run.

This file is the brief of the thread that drives the run, called the super coordinator below, and
of every thread it starts. Where it contradicts the `coordinate`, `implement` or `plan-next`
skills, this file wins. Everything else in those skills, in `AGENTS.md` and in the roadmap's
*Rules for every item* still applies.

## Context

- Design: `internal/notes/core_compiler_roadmap.md`. Its decisions are fixed, its *Order of work*
  gives the order, its *Milestone checks* table says what each milestone must show, and its
  *Open questions* each name the issue that settles them.
- Tracking: `docs/dev/issue_tracking.md`, read and changed with `inspect-issues` and
  `update-issues`.
- The first section is already built: GitHub stack #278 (pull requests #274, #277, #286, #287,
  #289, for #39 to #43). Its review packet is untracked in the worktree of the thread
  "Coordinate Sparse Compiler Tasks" on la015, under
  `internal/notes/sparse_stack_review_2026-11-10/`.
- Everything runs on la015. All timing measurements also run there, so they need the machine
  quiet (see *Measurements*).

## What changes from the usual workflow

1. **Agents merge.** A thread queues its own pull request with `gh pr merge --squash` as soon as it
   has its review verdict and green checks. Nothing waits for the maintainer. Tags, releases,
   deployment workflows and direct pushes to main stay forbidden.
2. **No limit on parallel work.** Start every child whose blockers are closed. There is no cap on
   sections or threads and no usage threshold: T3 resumes a thread when its quota resets. If a
   thread is found stopped and not resuming, send it "continue".
3. **Snapshots are not serialized.** Two pull requests may regenerate the C snapshots at once.
   The second rebases, regenerates, and its reviewer reads the regenerated diff.
4. **Sections.** The compiler work is grouped under section issues (table below). A section is
   one coordinator, one idea and one review packet. Sections never carry blocked-by edges;
   ordering lives only on the children.
5. **Coordinators may re-cut.** A section coordinator may split, merge or close its children when
   the work shows the cut was wrong. It moves the blocked-by edges onto the new pieces, labels
   issues it creates `source:agent`, and closes a replaced issue as not planned with a comment
   naming what replaced it.
6. **Roadmap departures are applied, not proposed.** When a section has to depart from a roadmap
   decision, take the choice that is easiest to undo, edit the roadmap in the same pull request,
   keep one line saying what was replaced and why, and list it in the packet and on the status
   issue. Never change a decision silently.

## Sections

`->` is a blocked-by chain. "Later" children run only when nothing else in the section can start.
Confirm each attachment against the issue before relying on it.

| # | Section | Parent | Children | Later |
|---|---|---|---|---|
| 1 | Sparse values into generated C | #37 | #39 to #43 (stack #278); follow-ups #288, #290, #103 | #44 |
| 2 | Sparse kernels | new | #45 -> #46 -> #47 | #91 |
| 3 | Sparse derivatives | new | #221 -> #225; #214, #167 | #48 |
| 4 | Dtypes | #24 | #30 -> #55 -> #56 -> #207; #168, #76 | |
| 5 | Calls and carried loops | new | #28 -> #29 | #83 |
| 6 | JIT cache | new | #64, #163, #193, #269 | |
| 7 | Tensor operations | new | #66 -> #67 -> #68 | #65 |
| 8 | Fusion and lanes | new | #69 -> #71 -> #73 -> #90; #75; #183, #223, #98, #93 | |
| 9 | Control flow | #23 | #35, #31, #36, #34, #50, #53, #32 -> #33 | #78 |
| 10 | Dense linear algebra | #49 | #51 -> #52 | #54 |
| 11 | Generated solvers | new | #74, #72, #205, #206, #208 | #70 |

Out of scope for this run: solver plugins, benchmark problems, research issues not listed above,
and API work outside the roadmap. Open issues labelled `kind:bug` in `area:compiler` are in scope
as lone `implement` threads. An issue labelled `ci:nightly-failing` is dispatched at once, ahead
of everything else, and does not pause the run.

## Phases

### Phase 0: set up (super coordinator, before anything else)

- [ ] Before anything merges, push a branch `compiler-run/start` at the current tip of main. It
      marks the state to return to (see *Rolling back*).
- [ ] Commit this file through a pull request so every thread can read it on main.
- [ ] Add the `source:agent` label, and its row in the Labels table of
      `docs/dev/issue_tracking.md`.
- [ ] Create the seven new section issues and attach the children as in the table. Move #28 and
      #29 out of #23 and #225 out of #220. Set #183's milestone to `scaly 0.3.0`. Close nothing
      and change no other milestone.
- [ ] Create one issue, "Compiler run: status", and keep its body current for the whole run: one
      row per section (state, coordinator thread, packet link), then every hard-to-undo decision
      and roadmap departure in one line each, then the parked children. The maintainer reads only
      this issue.

### Phase 1: land the sparse stack

- [ ] Ask the thread "Coordinate Sparse Compiler Tasks" to add the cover page (see *Packet*) to
      its packet and to move the packet to `internal/notes/compiler_run_2026_11_10/sparse_values/`
      in a pull request.
- [ ] Queue stack #278 with `gh stack merge`. Fix and requeue any ejected layer.
- [ ] Dispatch the follow-ups #288, #290 and #103.

### Phase 2: run the sections (repeats until nothing can start)

The super coordinator reads only the tracker and thread states. It never reads or edits code.

1. Start a section coordinator, with `t3_thread_launch` in its own worktree, for every section
   that has a child whose blockers are closed. Its message: this file's path, the section's row,
   and "Use the coordinate skill on these children under the compiler run plan".
2. Loop: wait on the section threads, and after each wake look for children that a merge has
   unblocked. Wake a section coordinator that reported itself idle when its blocker closes.
3. After any context summary, re-read this file and the status issue before acting.

Each section coordinator:

1. Writes or rewrites its section issue body: what the section is for, the roadmap part that
   decides it, one demo a user could run at the end (taken from the milestone checks where one
   fits), limits, what is out of scope and which later section owns it, and which sections it
   interacts with. Comments with its thread link.
2. Runs its children with `coordinate`. A linear chain is a stack whose next layer starts on the
   verdict below it; every layer is queued as soon as it has a verdict and green checks. Lists
   open pull requests touching the same files before each launch and tells the child to expect a
   rebase.
3. Parks a child that still fails after its review and one delta check: sets it back to Ready with
   the reason, reports it, and continues with the rest.
4. When it has nothing it can start, ends its turn with one line naming the issue it waits on.

### Phase 3: close each section

When a section's scheduled children are merged, its coordinator:

- [ ] Starts one simplification pass on `claude-fable-5-2` at high effort over the whole section's
      diff: reuse, dead code, needless generality, defensive checks inside the compiler. It lands
      as ordinary pull requests with the C snapshots byte-identical, each with one correctness
      review by `gpt-6.1-sol`.
- [ ] Runs the section's demo and the milestone checks that the section completes.
- [ ] Writes the packet (below) and lands it in a pull request.
- [ ] Reports to the super coordinator, which updates the status issue.

### Phase 4: milestone checks and the end

- [ ] After the sections of each milestone, the super coordinator has one thread run that
      milestone's row of the roadmap's *Milestone checks* table, timing checks included. A failing
      check becomes a `kind:bug` issue, is dispatched, and does not stop work that does not
      depend on it.
- [ ] When a milestone's checks all pass, push a branch at that commit of main:
      `compiler-run/after-0.2.0`, `compiler-run/after-0.3.0`, `compiler-run/after-0.4.0`. Record
      it on the status issue. These are the verified states to return to.
- [ ] The run ends when the Milestone 3 checks pass, or when nothing more can start. The final
      message gives: the order in which to read the packets, the checks that fail, the parked
      children, and what needs hardware or a person (#208 needs the STM32F405 board).

## Packet

One folder per section, `internal/notes/compiler_run_2026_11_10/<section>/`, with a `README.md`
that opens with a cover page of at most 60 lines:

1. **Where this section sits.** What a user can do after it, and what each later section adds.
   Five lines.
2. **Needs the maintainer's decision.** Only where the section departs from the roadmap or
   commits a public interface. One line each, with a recommendation.
3. **Expected gaps.** Every limitation an open issue already owns, one line each with its number.
   No discussion.
4. **Unowned.** Limitations no open issue covers. File each as an issue and give its number.

Below a "Reference" heading: the full decision list, hardest to undo first; the demo script and
its real output; a reading list of the 200 to 300 source lines that carry the design, as file and
line ranges in reading order; source lines added under `src/` before and after the simplification
pass; and what the coordinator is least sure of. No per-pull-request summaries and no tables of
completion evidence.

## Measurements

Timings taken while other threads run on la015 are not evidence. During the run:

- A pull request proves a performance claim with operation counts, generated source size and
  compile time of the generated C. A snapshot regeneration shows the reviewed diff and those
  three numbers.
- Run-time measurements happen only in Phase 4. The super coordinator stops launching, waits
  for running children to finish, takes `~/.scaly-measuring`, runs the study with the
  `benchmark-study` skill, and releases the lock.

## Rolling back

Only the maintainer decides to roll back. Main is never force-pushed. To return main to a marker
branch, land one pull request whose single commit makes every file match the marker, while
keeping this folder:

```bash
git fetch origin
git switch -c restore-main origin/main
git restore --source=origin/compiler-run/start --staged --worktree -- .
git restore --source=HEAD --staged --worktree -- internal/notes/compiler_run_2026_11_10
git commit -m "Restore main to its state before the compiler run"
```

The first `git restore` rewrites, adds and deletes files until the checkout equals the marker.
The second puts this folder back as it is on main, so the packets survive. The run's commits
stay in history. Afterwards, reopen the issues the run closed, close the ones labelled
`source:agent`, and remove the run's leftover branches, worktrees and pull requests.

A single section cannot be rolled back this way, because later sections build on it. Problems
inside the run are fixed on top.

## Shortcuts this run takes

Each is deliberate and is to be revisited in the maintainer's review.

- No maintainer review before merge, including for changes to the IR (intermediate
  representation), differentiation and code generation.
- Run-time measurements are deferred from each pull request to the milestone boundary.
- Section demos are chosen by the coordinators, not written by the maintainer.
- Roadmap edits are applied by agents and reviewed afterwards.
- Unfinished features still stay out of `docs/` and are filtered from the API page, as
  `AGENTS.md` requires; the published site must stay true while main moves.
