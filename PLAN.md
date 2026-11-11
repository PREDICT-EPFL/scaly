# Integrate mapped loops onto main

Complete the paused rebase of #425 onto the supplied main base. Preserve COND, sparse leaves,
operand indexing, mask activity and forwarding, in-place carries, and lowering fusion. Keep mapped
loops on the common is_map predicate, and preserve result numbering and residual discovery.
Read the project guidance, compiler maintenance context, original patch, and suite rules.

- [x] Inspect all 15 conflicted files and both commits being rebased.
- [ ] Resolve conflicts and migrate references introduced on main, preserving the pruning fix.
- [ ] Execute documentation examples, regenerate actual listings and the collection baseline.
- [ ] Verify focused tests and all nine C/header snapshots against the supplied base.
- [ ] Finish the rebase and pass the detached full pre-merge hook through ~/.scaly-suite/run.
- [ ] Push once with the specified lease, link the PR, preserve verdicts in its updated body,
      write the ready marker, and report the final SHA. No delegation, queueing or watching.

The maintainer explicitly authorizes integration edits, commits, one force push and the ready marker.
The final body must list each integration edit beyond conflict markers on its own line.
This temporary plan is deleted before final validation.
