---
name: prune-tests
description: Spawn Test Sicko to delete useless, redundant and outdated Scaly tests, check its kills, and remove the dead code they kept alive. Use only when a maintainer or another skill asks for it by name, never on your own initiative.
---

# Prune tests

Spawn Test Sicko. Check its kills. Act on the accepted findings.

Whoever wrote the tests will defend them. Trust the fresh reviewer's read.

## Scope

Use the caller's files, directories or diff. Otherwise, use the tests touched by the current diff
against `origin/main`, including the working tree, plus the tests of the code that diff touches.
Prune the whole suite only when the caller asks for it.

When `implement` calls this at the end of an issue, the scope is the tests the branch added or
changed, and the issue comes along as context: the last test that asserts a **Pin** item of one of
its examples must be kept.

## Steps

1. **Baseline.** Run the scoped tests with `uv run pytest` and record the test count and any
   failures that already exist.
2. **Spawn** a fresh subagent with the contents of `test-sicko.md` (next to this file) as its
   instructions, the scope, the issue when there is one, and the baseline failures, and wait for
   its report. Do not restate its rules.
3. **Inspect** its report and diff. Reject edits to production code, deletions outside the scope,
   kills without a reason line, twins whose named survivor does not exist or does not catch the
   same breakage, and deletions of tests that its own rules keep or that the issue's Pin items need. For a doubtful twin kill, break the behavior
   on purpose, run the survivors, and restore the code after. Delete fixtures, helpers and
   snapshots it left orphaned. If the report is rejected, revert it and rerun once with the failure
   named. If the second report is also rejected, report it as open and stop.
4. **Verify.** Run the scoped tests again. Apart from the deleted tests, the results must match the
   baseline, with no new import or collection errors. Regenerate the node-ID baseline with the
   recipe in `docs/dev/contributing.md`.
5. **Act on `MUST KILL`.** Check that the flagged code has no callers outside tests, then delete it
   and its tests. If it is out of scope or still has callers, report it as open. Report `REWRITE`
   and `SUGGEST` flags as open work. Write new tests only when the caller asked for it.
6. **Report** the tests deleted, grouped by reason, the net lines removed, restored tests, reruns,
   the production code deleted, open `MUST KILL`, `REWRITE` and `SUGGEST` flags, the test count
   before and after, and the final test result.
