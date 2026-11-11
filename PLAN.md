# Finish issue #339 after the dtype merges

The section 4 coordinator changed merge order: #358 and #361 may proceed together
after #360. Do not push until the coordinator says #360 has merged. The compiler run plan
on main and `~/.scaly-suite/RULES.md` still govern tests, GitHub and pushes.

- [x] Rebase the task branch onto main after #320's squash merge, dropping its old commits.
- [x] Trial-merge the fetched `pr360` and `pr361` heads on the local scratch branch
      `t3code/339-trial-360-361-20261111`.
- [x] Run the focused trial checks with two workers. 690 tests passed, plus both added
      nested-block sanitizer cases. All 96 no-select C/header artifacts match main.
- [x] Have Opus check the logic-changing conflict resolutions alone. Approved with
      no act-on findings. The reviewer also passed 309 focused tests and eight
      integer-sum sanitizer probes across both widths and lowering modes.
- [x] Redo the trial against #360's newest head, checking the shared AD rules and
      expression files. The local branch `t3code/339-trial360-latest-20261111`
      passes 418 focused tests, Ruff and strict ty. All 96 no-select C/header
      artifacts match the fetched #360 head. Keep the earlier trial resolutions
      for reference.
- [ ] Wait for the coordinator's confirmation that #360 merged.
- [ ] Fetch and rebase this task branch onto main, resolve any remaining overlap,
      remove this plan and run the full pre-merge check through the shared runner.
- [ ] Check the PR targets main, push once, retarget if needed and queue with
      `gh pr merge 358 --squash --auto`. Watch through merge and report an ejection immediately.

The scratch branch preserves both upstream heads and their trial resolutions. Its old #30
history is deliberately absent from the task branch. Do not copy the scratch branch's
#55 or #340 implementation into this task. Get those changes only through main.

The newest #360 trial has one conflict in the SCATTER transpose. Keep its sparse
dispatch and the reviewed dense-source reshape. The shared `ad/rules.py` and
`ir/expr.py` changes merge cleanly. No new AD logic was needed. The newest trial's
resolution and comparisons are `trial360-latest-resolution.diff`, `before360-latest`
and `after360-latest` under the artifact directory below.

Trial resolutions retain the newer carried-loop and array-based fusion implementations,
make newer ProgramNode callers supply dtype explicitly, retain both verifier checks,
and carry #55's typed zeros into the shared reduction and stored-map emitters.
The #361 renderer keeps carried exit variables and float32 BIT_SELECT handling,
uses index rendering for bit positions, and passes integer-data context into blocks
and loops. Removing block context makes both new sanitizer cases trap.

`ad/reverse.py` and the functions guide needed no trial logic change. `ad/rules.py`
only gains #55's integer ABS spelling. The task branch also converts its new numerical
test results with `np.asarray`, as main's numerical-return typing now requires.

Review artifacts and comparisons live under `/tmp/scaly-339-g5fjmqv1/`:
`trial360-resolutions.diff`, `trial361-resolutions.diff`, and the trial comparison
directories. The original #339 reviews and their fix checks approved the change.
Follow-up #364 remains documented and outside this task.
