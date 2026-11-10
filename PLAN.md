# #64 JIT index: working plan

Brief: implement skill under internal/notes/compiler_run_2026_11_10/plan.md, child of section 6 (#295).
Deleted in the last commit before the pull request is marked ready.

## Run-wide rules

Read and follow `~/.scaly-suite/RULES.md`: full suites only through `~/.scaly-suite/run`, detached,
with a `run_in_background` until-loop on the exit file and `t3_background_wait`; focused tests at
most `-n 2`; GitHub through REST; queue once with `gh pr merge 321 --squash --auto` after #317
(#163) merges. #318 belongs to the run coordinator's thread. Do not edit `src/` or `plugins/` while a
test run is going: an edit under a live process turns the index off in it.

## Done

- `function/digest.py`: `graph_digest`, `source_digest`. `codegen/jit.py`: index key, index read and atomic write,
  cache-relative paths, rebuild behind an entry, `invalidate_cache` through the index, `SCALY_JIT_KEY=verify`.
- Tests in `tests/codegen/test_jit_index.py`, demo `tests/codegen/jit_warm_start.py`, CI job `verify`.
- How it works page updated.

## Remaining

- sol's and Fable's findings fixed; delta checks (Fable, sol) next, then full suite through ~/.scaly-suite/run, rebase.
- Rebase on #193 (#316): take its `invalidate_cache` as is (one epoch increment under the lock, no render, no index
  lookup), drop the index lookup there and `test_invalidation_finds_the_library_through_the_index`, keep `_index_key`
  returning None while the epoch is set so an epoch-bearing build never reads or writes the shared entry, and take
  #316's text for the `recompile` docs paragraph, #269 (solver fingerprint into the index key),
  #163 (deterministic C, needed for verify across xdist workers), stack #278.
- Docs on rebase: #310 edits the cache-key list (native solver inputs) and the new-process reuse paragraph
  (initial directory reads); #316 edits the last `recompile` paragraph. Keep their wording and fold the index into it.
- Push rarely (RULES.md GitHub item 5): no push until #317, #316 and #310 have all merged. Then rebase once (fold in
  #269's fingerprint and #193's one-line `invalidate_cache`), delete PLAN.md, run the full suite once through
  ~/.scaly-suite/run, push once, mark ready and queue with `gh pr merge 321 --squash --auto`.
- Full pre-merge already passed on 34f05d9c (2649 passed, 52 skipped).
- Queue only after #163 merged.
- Follow-ups filed: #318 (dtype interning), #319 (index off while bytecode is first written). Add both to Project 2
  once GraphQL answers again (it was rate limited).
