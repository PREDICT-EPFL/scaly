# Shared wrapper parameter names

The compiler-run coordinator assigned #353 and authorized combining it with #344 when one general wrapper naming fix covers both. PR #352 will close both. The branch now builds on reviewed #351 so it does not duplicate that PR's C++ workspace change. Follow the compiler-run plan and `~/.scaly-suite/RULES.md`, including rare pushes and settling used review delegates.

- [x] Read #353, its discussion and prerequisites, the current naming code and solver-options compile tests. Settle all seven completed prior reviews.
- [x] Record the combined scope on #344 and #353 and return #352 to draft. Restack locally onto #351 without pushing work in progress.
- [x] Add six solver-marked C/C++ compile-and-call regressions for input and output buffer names matching `scaly_solver_option`, including a suffix collision. All six fail before the fix.
- [x] Move parameter allocation into one helper in `codegen/abi.py` used by both renderers. Buffer typedefs and pointer entries stay unchanged, as do all C snapshots. Removing the option-type reservation in memory makes all six tests fail again. The focused batch passes 79 tests, the full collection has 3,002 IDs, and format, lint and strict types pass.
- [ ] Prove the regressions detect the original fault, regenerate test IDs, run focused checks and the detached full pre-merge runner. Commit and remove this plan before a single complete push.
- [ ] Rewrite #352's title and description for the combined scope and new base. Obtain one review of the new change and its correctness check, batch any fixes, and settle their delegate threads once used.
- [ ] Queue #351 and #352 onto main after their parent merges, as their checks permit. Keep #325's existing queue request and watches intact.

Evidence for this extension is under the directory named by `/tmp/scaly-353-active`. Macro handling remains #326.
