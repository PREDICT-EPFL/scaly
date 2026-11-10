# Keep program nodes with distinct dtype descriptors apart

Issue #345 follows #318. Its branch starts on pull request #327 until that merges, then targets main. The compiler-run plan and `~/.scaly-suite/RULES.md` apply. Full checks run detached through the shared runner. Finish local validation and review before the first push, then enable automatic merge after the verdict. No tags, releases, deployments, or direct pushes to main.

- [x] Read the issue and relevant implementation. Pull requests #331 and #320 edit program definitions, and #331 also edits the program tests. Expect a rebase if conflicts arise.
- [x] Add a regression for every dtype field, both construction orders, and reuse of equal copies. The old key fails twelve cases.
- [x] Key program nodes on the complete dtype descriptor, using #327's structural equality.
- [x] Refresh the collection baseline and pass focused checks. Seventy-three program, type, and C snapshot tests pass, as do Ruff and strict ty. Restoring the name-only key reproduces the same twelve failures.
- [ ] Run the full pre-merge check detached through `~/.scaly-suite/run`. C snapshots must stay byte-identical.
- [ ] Obtain the independent reviews and address findings together. Remove this temporary plan before pushing.
- [ ] Open the stacked pull request, make it ready, and enable auto-merge after its verdict. Retarget to main once #327 merges, then handle any conflicts or merge-queue ejection.

The regression precedes the fix. Local checks precede review. Review and all fixes precede the first push. Preserve #327's worktree for its ongoing merge events.

Validation logs are under `/tmp/scaly-345.RkPWGG/`.
