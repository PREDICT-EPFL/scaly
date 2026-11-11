# Section 6: JIT cache

Packet of section 6 of the [compiler run](../plan.md), parent #295. Built by #317 (#163), #316
(#193), #310 (#269) and #321 (#64), then simplified by #399. #395 (#398) and #400 (#402) joined
the section during its close. JIT is the just-in-time compile path in `codegen/jit.py`. A digest
is a SHA-256 hash taken over an encoding of a value.

## Where this section sits

- A second process finds a Function's compiled library from a digest of its graph without lowering or rendering C, whatever its `PYTHONHASHSEED`. This is the cross-process cache test of Milestone 1 (0.2.0), and `demo.py` shows it.
- The generated C no longer depends on the hash seed, and recompiling a Function in one process no longer breaks another process compiling or loading the same library.
- A cached solver module is reused only against the solver headers and libraries it was built with.
- On Windows too, a source file rewritten after scaly loaded turns the index off for that process (#398), and taking the digest walks the graph once whatever the number of outputs (#402).
- Later sections only extend the digest. Each object they add to a Function's graph must be encodable by `function/digest.py` or the index stays off for it. #355 (section 5, #83) did that for external C: its library build keys enter the index key.
- Run-time numbers for the warm start (devrush measured 1.3 s to 0.17 s) wait for the Phase 4 measurements (#309).

## Needs the maintainer's decision

1. Roadmap departure, applied in this packet's pull request: `invalidate_cache` neither renders nor reads the index. It only moves this process's next build to a fresh path (#316). The roadmap said it would use the index. Recommend keeping.
2. Documented behaviour (#316): invalidation and `recompile()` delete nothing. A fresh process after a `recompile()` elsewhere still loads the original library, and the cache grows until deleted by hand (`docs/guide/codegen.md`). Recommend keeping, with #394 for pruning.
3. Debugging switch (#321): `SCALY_JIT_KEY=verify` renders on every index hit and raises when the C, workspace size or link flags differ from the entry's. CI runs the suite under it (`verify / linux-x86_64`). It is kept out of the guide, as `AGENTS.md` asks. Recommend keeping it internal.
4. Cost (#310): the first lookup of a solver Function in a process hashes the solver's native include and library folders, about 63 MB for PIQP, IPOPT and Fatrop together. Recommend deciding after #309 measures it.

## Expected gaps

- Warm-start and fingerprint timings: #309.
- No pruning of old libraries and index entries: #394.
- An index entry that is valid JSON with a wrongly typed value is accepted and the call fails later with a `TypeError`. It needs a hand edit or a corruption that keeps the JSON well formed: #397.

## Unowned, filed by this section

- #394 prune the JIT cache without breaking running processes.
- #397, filed by the simplification pass's review: a malformed index entry fails late.
- Filed and fixed during the close: #395, in-place source rewrites were not detected on Windows (#398), and #400, the digest walked the graph once per output (#402).
- Filed by #321 and since closed: #318 (dtypes with one name and different C types were interned together) and #319 (the index switched off while Python wrote bytecode).

# Reference

Line numbers are on main at `6b36854f`, after #399, #398, #402 and #355, and paths are under `src/scaly/`.
Beside this file: `demo.py` and `demo_output.txt`.

## Decisions

Hardest to undo first.

### Documented behaviour

1. **Invalidation deletes nothing** (#316; `codegen/jit.py`, `invalidate_cache`). It increments
   `_cache_epochs[fun.name]` under `_artifact_lock`. The epoch and the pid then enter this
   process's next rendered key, so the next build goes to a new directory, and loaded handles,
   other processes and the shared index entry are untouched. On Windows a loaded DLL cannot be
   deleted anyway (#154). The guide and `docs/how_it_works/generated_interface.md` say a fresh
   process keeps the original library and the cache is cleaned by hand.
2. **`SCALY_JIT_KEY`** (#321; `codegen/jit.py`, `_build_artifact`). `index`, the default, uses
   the index. `verify` renders on each hit and compares. Any other value raises. CI runs the suite under
   `verify` on linux-x86_64.

### The index key

3. **What the index key holds** (#321, #310; `_index_key`). The graph digest of the Function and
   its callees; the source digest, over the installed scaly and solver-plugin files and the Python
   and NumPy versions; the scaly version; the host build recipe, which carries
   `SCALY_VECTOR_LIBM`; the compiler fingerprint (#63), with the build keys of external
   libraries since #355 and the pid and epoch after an invalidation; the compile command, which
   carries `SCALY_CC_OPT` and the link flags; the solver-inputs fingerprint (#310); and the glibc
   version when vector math uses it. Files over a megabyte in the plugin sources, the built
   libraries, enter by size and times; their content is the solver fingerprint's. The entry maps
   to the artifact directory named by the rendered key, as a path relative to the cache root, and
   is written through a temporary file and `os.replace`.
4. **When there is no index key** (#321). The digest meets a value it cannot encode (it raises
   `Unkeyed`); a file under the scaly or plugin sources changed after scaly loaded, or Python
   would load bytecode older than its source (two-second clock slack); the pass pipeline was
   changed; a render observer watches this Function; or this process invalidated it. The Function
   then renders as before #321. On Windows, where `os.stat` gives the creation time in place of
   the change time, `_source_stat` reads the NTFS change time instead, and a zero change time
   gives no key (#398).
5. **The digest is an explicit encoder** (#321; `function/digest.py`, `_Walk.value`). Plain values,
   arrays of plain kinds by bytes, floats by bits, enums by class and value, frozen dataclasses by
   class and compared, initialized fields, callees by their order in the walk, nodes numbered in
   walk order, classes only by a module-level name that leads back to them. Anything else raises.
   One traversal covers all of a Function's outputs (#402).
6. **Field lists with guard tests** (#321, after the sparse stack merged). `_FIELDS` and
   `_NOT_READ` classify every field of a `ConcreteFunction`, `_TREE_READ` and `_TREE_NOT_READ`
   every attribute of its input and output trees. A test fails when either class gains an
   unclassified attribute. The tree guard caught `_layout_decls` from section 1 the day it
   merged. It was added because the first CI run of #321 showed CSR and CSC leaves sharing a key.
7. **The solver-inputs fingerprint** (#310; `solver_inputs_fingerprint`). Hashes the content of the
   native include and library folders that a solver module's link flags name, memoized per folder
   per process. A file it may not read enters by its name and a marker.

### Generated C

8. **Statement order no longer follows set iteration** (#317). `fuse_ranges` emits its final
   stores in declared output order, and `pack_workspace` takes each statement's dependency
   closure before updating the written buffers. No C snapshot changed.

### Left as it is by the simplification pass

Judgement calls that #399 did not take.

- `_compute_cache_key` and `_index_key` each build the compile command and the solver fingerprint in two lines. Sharing them needs a new signature for `_compute_cache_key`, which the JIT tests call with keywords and patch.
- `_index_write` writes its own temporary file, because `_write_cached_text` skips a file that exists and an index entry must be replaced.
- `_solver_directory_fingerprint` and `_scan` both walk a folder, under different rules: the first reads every file once and assumes it fixed, the second detects change. They stay apart.
- The field lists, the per-walk type memo, the per-Function tables and the per-file content and bytecode memos stay: they are what makes a warm lookup open no file.

## Demo

`uv run internal/notes/compiler_run_2026_11_10/jit_cache/demo.py` from a checkout. It renders
#163's mapped stage under five hash seeds, then runs `tests/codegen/jit_warm_start.py` three times
against one fresh cache: seed 1, seed 2, and seed 3 under `SCALY_JIT_KEY=verify`.
`demo_output.txt` is its output on la015:

```text
== #163: the mapped stage of the issue, rendered under five hash seeds ==
PYTHONHASHSEED=0: sha256 of the C ded46add3b39a1eb
PYTHONHASHSEED=1: sha256 of the C ded46add3b39a1eb
PYTHONHASHSEED=4: sha256 of the C ded46add3b39a1eb
PYTHONHASHSEED=5: sha256 of the C ded46add3b39a1eb
PYTHONHASHSEED=7: sha256 of the C ded46add3b39a1eb
one C for every seed: True (on main before #317, seeds 4 and 5 gave another)

== #64: the cross-process cache test of Milestone 1 ==
first process, PYTHONHASHSEED=1: lower_function calls 1, result 8.15097141822, library <cache>/173a72363ea5…/libwarm_start_outer.so
second process, PYTHONHASHSEED=2: lower_function calls 0, result 8.15097141822, library <cache>/173a72363ea5…/libwarm_start_outer.so
third process, PYTHONHASHSEED=3, SCALY_JIT_KEY=verify: lower_function calls 1, result 8.15097141822, library <cache>/173a72363ea5…/libwarm_start_outer.so
index entries: 1
same result from the same library in all three: True
second process lowered nothing: True
```

## Reading list

About 320 lines, in reading order, on main at `6b36854f`.

1. `codegen/jit.py:633-648`. `invalidate_cache`: one epoch increment, and why nothing is deleted.
2. `codegen/jit.py:255-265`. `_toolchain`, where the external library keys and the epoch enter both keys.
3. `codegen/jit.py:287-312`. `_index_key`: when there is no key, and what the key holds.
4. `codegen/jit.py:395-425`. The index hit in `_build_artifact`, and verify mode comparing the entry with what rendering gives.
5. `codegen/jit.py:217-233`. The solver-inputs fingerprint, read once per folder per process.
6. `function/digest.py:26-44`. The field lists that the guard tests hold complete.
7. `function/digest.py:73-128`. `_Walk.value`, the explicit encoder.
8. `function/digest.py:139-208`. The graph walk, a Function's own digest, and `graph_digest`, which numbers callees in the order the walk meets them.
9. `function/digest.py:283-330`. `source_digest` and the rules in `_scan` for a changed file or folder.
10. `passes/program/fuse_ranges.py:524-528` and `passes/program/pack_workspace.py:59-81`. The two orderings that made the C depend on the hash seed.

## Source lines

Lines under `src/`, from `git show --numstat`.

- #317 added 8 and removed 5, #316 added 12 and removed 20, #310 added 45 and removed 6, #321 added 468 and removed 14.
- Before the simplification pass the section had added 533 lines, a net 488, of which `function/digest.py` is 315.
- #399, the simplification pass, added 37 and removed 76, in three files.
- #398 added 61 and removed 6, and #402 added 16 and removed 4, all in `function/digest.py`.
- After the close the section's net addition is 516 lines. `function/digest.py` is 383 lines and `codegen/jit.py` 646, with #355's external libraries in it.

## What the coordinator was least sure of

**That the digest covers everything rendering reads.** The evidence is three checks. The CI job
`verify / linux-x86_64` renders every index hit in the suite again and compares it with the
entry's C, workspace size and link flags. The guard tests fail when a `ConcreteFunction` or a tree
gains an attribute nobody classified, and the tree guard caught section 1's `_layout_decls` the day
it merged. And a review asked this exact question after the sparse-layout miss and found nothing
else. Verify mode sees only the C and the flags, though. An input that changes the built library
without changing the C is invisible to it. Two such inputs are known and keyed: solver headers and
libraries (#310) and external libraries (#355), each with its own cross-process test. A third would
pass verify mode and every test I know of.

**That the source digest's time rules hold on every file system.** They compare modification and
change times with the moment scaly loaded, less a two-second slack, and accept a folder whose only
change is Python creating `__pycache__`. They are tested on ext4 here and on the four CI platforms,
including NTFS since #398. A network or overlay file system with coarse or rewritten times may turn
the index off, which is safe, or miss a change, which is not. I did not test one.

**That #317 removed every hash-seed dependence.** Its test renders 11 Functions under six seeds.
The verify job is a broader check if the xdist workers draw different seeds and share one cache,
which I did not confirm. A seed-dependent pass that the corpus does not reach would show up as a
verify failure in one CI run and not the next.

**What I did not expect.** The first CI run of #321 failed on 19 sparse-boundary tests, because
the sparse stack had merged between the review and the run. The fix and its guard test came from
that failure, not from the design. Each section that adds fields to Functions or their trees will
meet the guard test in the same way. The guard makes that a decision, not a silent stale hit.
