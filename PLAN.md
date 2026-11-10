# Keep program nodes with distinct dtype descriptors apart

Issue #345 follows #318. Its branch starts on pull request #327 until that merges, then targets main. The compiler-run plan and `~/.scaly-suite/RULES.md` apply. Full checks run detached through the shared runner. Finish local validation and review before the first push, then enable automatic merge after the verdict. No tags, releases, deployments, or direct pushes to main.

- [x] Read the issue and relevant implementation. Pull requests #331 and #320 edit program definitions, and #331 also edits the program tests. Expect a rebase if conflicts arise.
- [ ] Add a regression for every dtype field, both construction orders, and reuse of equal copies. Show it fails on the old key.
- [ ] Key program nodes on the complete dtype descriptor, using #327's structural equality.
- [ ] Refresh the collection baseline, pass focused checks, and prove the regression fails under a name-only key perturbation.
- [ ] Run the full pre-merge check detached through `~/.scaly-suite/run`. C snapshots must stay byte-identical.
- [ ] Obtain the independent reviews and address findings together. Remove this temporary plan before pushing.
- [ ] Open the stacked pull request, make it ready, and enable auto-merge after its verdict. Retarget to main once #327 merges, then handle any conflicts or merge-queue ejection.

The regression precedes the fix. Local checks precede review. Review and all fixes precede the first push. Preserve #327's worktree for its ongoing merge events.
