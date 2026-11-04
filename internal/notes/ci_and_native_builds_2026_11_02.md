# CI, nightly and native builds

Decided with the maintainer on 2026-11-02. This replaces the native-bundle plan of #228, #229 and
#230 (PRs #231, #233 and #234), which were closed. The work is listed under [Work](#work).

## The problem

Measured on 2026-11-02:

- **Native rebuilds come from a cache key that is too broad.** `wheels.yml` keys the wheel cache on
  the whole plugin directory, so a version bump, a README edit or a wrapper change rebuilds Ipopt:
  17 minutes on Linux x86-64, and about 61 minutes on Windows with #218, 53 of them for OpenBLAS.
  Of the 13 commits since August that touched `plugins/scaly-ipopt`, 6 changed the build recipe.
- **New machines compile from source.** The solver cache under `~/.cache/scaly/solvers` is shared
  by the worktrees of one machine, but a new machine, agent sandbox or Windows box compiles once and
  needs gfortran, or MSYS2 on Windows. Google Benchmark builds once per checkout.
- **CI is slow and hard to read.** Queue run 40911243562 had 25 jobs, 93 runner minutes and 21
  minutes of wall time. Nine of its jobs were cached wheel builds of about 10 seconds. The slowest
  Windows test job spent 8 of its 21 minutes in pytest; the rest went to install checks (5
  minutes), the Google Benchmark build (3.4) and the benchmark smoke run (3.1), all repeated for
  three Python versions. Every reviewed change ran CI twice on the same commit, in the queue and
  again on push. Only `lint + type` and the five install checks were required. Job names were cut
  short in the sidebar, and up to ten cibuildwheel summaries buried the run page.
- **The test suite is not the problem.** Its 2,747 tests run in 44 seconds with a cold compile
  cache and 26 seconds warm on the maintainer's Mac, and in 1.6 to 2.8 minutes on Linux and macOS
  CI.
- Nightly run 41408147359 had 27 jobs, 223 runner minutes and 46 minutes of wall time, at a single
  Python version.

## Decisions

### Native code

1. **Keep the plugin architecture and fix the speed with caching.** Rejected:
   - the pinned native bundles of #231, #233 and #234: about 7,000 lines, no speed-up until the
     last layer, manual publication, and a recipe that can drift from the shipped files;
   - a custom artifact store keyed by the recipe, or a separate build-farm repository: both rebuild
     a package manager, and a separate repository needs a lock file and two PRs per native change;
   - native-only wheels as extra PyPI packages: a second package per solver, and PyPI's limits on
     creating projects;
   - conda-forge, vcpkg, Conan, Spack and sccache: none produces wheels PyPI accepts, or every
     machine still needs the compilers. conda-forge binaries depend on conda's own runtime, so
     developers would test different binaries from the ones users install.
2. **Native builds are keyed on their inputs only**: `hatch_build.py`, `build_config.json` and
   `licenses/`, plus the platform. Python sources, READMEs and the version stay out, so wrapper
   edits and version bumps reuse the build, in CI and locally. If the key proves too coarse because
   `hatch_build.py` mixes recipe and packaging code, split the file rather than hashing part of it
   as #231 did.
3. **The hook reuses released native libraries when the recipe is unchanged.** Each built wheel
   records the hash of its native inputs. The hook looks in the local solver cache, then at the
   newest release on PyPI: if its wheel for this platform records the same hash, the hook takes
   `lib/`, `include/` and the licenses from it, verified against PyPI's SHA-256. Otherwise it builds
   from source, and a `SCALY_BUILD_SOLVERS` value forces that. Libraries taken from PyPI are cached
   under the hash alone, while local builds keep the compiler identity in their key, so a key always
   names one set of files. Releases take the same path, so the native libraries stay byte-identical
   across plugin versions until the recipe changes. Rejected: installing the released wheel
   directly, which would hide a recipe change in the checkout.
4. **Native plugins are not released automatically** when their version is bumped. Until a recipe
   change is released, CI builds it once and keeps it in its cache, and each machine compiles it
   once.
5. **Google Benchmark builds once per machine** under Scaly's cache directory, and CI caches it.
6. **Moving the solver interfaces into core is parked**, without an issue. It would spare the
   plugins a release on every protocol bump and fold `scaly-sqp` away, but the speed work does not
   need it. It would cost the one-directory-per-solver layout, an unambiguous place for solver
   tests, and Scaly's own use of the plugin contract.

### CI

7. **Pull requests and the merge queue run the same checks**, so tests run on every push to a PR.
   Rejected: running the tests only in the queue, which reports failures too late.
8. **One Python version, from `.python-version`.** Ruff's banned-API rule (TID251) forbids
   `sys.version_info`, and ty runs at its default (the 3.12 floor from `requires-python`), at 3.14
   and for Windows. Rejected: several Python versions per PR, and install checks per Python
   version. Raising the floor under SPEC 0 is parked for a later discussion, without an issue.
9. **All five platforms build their wheels once, then run tests and an install check, as
   separate jobs**: `build / <platform>`, `test / <platform>` and `install / <platform>` for
   linux-x86_64, linux-arm64, macos-arm64, macos-x86_64 and windows-x86_64. The build job restores
   the native libraries from the cache, makes the platform's wheels and hands them on as artifacts,
   so the install check tests the files a release publishes. Separate test and install jobs keep
   the fresh Windows install off the slowest path, 2.6 to 3.1 minutes per run. The whole test suite
   runs; the benchmark and closed-loop smoke run only on linux-x86_64, the reference platform.
   Revised on 2026-11-02 during #248: at first each test and install job built its own wheels,
   which doubled the build's runner minutes, about 42 extra with a cold cache and 4.5 warm in runs
   41456863572 and 41460383908. Rejected: the install check as steps of the test job, which puts
   it on the slowest path. The build job costs a job start and an artifact round trip per
   platform, and every test job waits for the slowest build. Cutting cibuildwheel's own overhead
   of 0.5 to 1.6 minutes per warm build is left for a later discussion.
   Revised on 2026-11-04: the benchmark and closed-loop smoke run in a separate
   `smoke / linux-x86_64` job after `build`, in parallel with the tests and install checks.
   The Linux test job previously added 1.3 to 1.7 minutes of benchmark smoke and 4 to 7 minutes
   of closed-loop smoke after pytest. The separate job repeats environment setup but removes
   that serial work from the test job. It runs the same checks and feeds the required `ci`
   check and its platform summary table.
10. **One required check, `ci`**, which depends on every other job and fails unless all of them
    succeeded. The ruleset requires only it, and the merge queue and the maintainer's bypass stay.
    Rejected: required checks named after matrix entries, which never report, or report "skipped",
    when an entry changes.
11. **Pushes to main run CI unless they only touch `internal/**`**, and the run ends after its first
    job when the merge queue already tested the same commit. Runs on main do not cancel each other.
    A ruleset cannot make a bypass depend on file paths: bypass is granted to roles, teams and apps,
    and path rules can only block.
12. **Runs are readable.** Jobs are named after platforms rather than runner labels, with no ` / `
    inside a job's own name. The `ci` job writes one summary table, with a row per platform and
    columns for native code, install and tests. The docs build and the PR title check move into
    `ci.yml`, and `docs.yml` keeps the deployment.

### Nightly and release

13. **Each nightly entry differs from CI in one thing and is named after it**:
    `toolchain / <image> <compiler>` for today's compilers and images, `python / <version>` running
    only the unit tests on linux-x86_64 for every version in the package classifiers, and
    `benchmarks / <platform>` running the full benchmarks and every closed-loop pair once per
    platform. Caches start cold, and the nightly saves the compile cache that CI restores. On
    Mondays it rebuilds the native libraries from source.
14. **The failure issue's labels follow what failed**: `source:maintainer`, `kind:bug` and
    `area:tooling`, plus `area:benchmarks` when a benchmark entry fails. Putting the issue on the
    Project as Ready needs a credential the default Actions token lacks, and is dropped for now.
15. **A release reuses completed validation and the exact files it tested for the release commit.**
    It first looks for completed TestPyPI verification on main, then a completed full nightly on
    main, and runs a fresh build and full suite when neither has usable artifacts. Reuse requires
    a validation record, the same commit and all referenced artifacts still available. An unchanged
    nightly that skipped its tests does not qualify. A PyPI release proceeds automatically through
    every remaining stage, without an approval pause. Revised on 2026-11-03 to avoid repeating the
    full suite for TestPyPI-to-PyPI publication or a release after that commit's overnight nightly.
    This replaces the original decision to rebuild and validate within every release dispatch, and
    the earlier requirement for a push run of `ci.yml`.

## Work

1. #248 Restructure CI around one required check: decisions 2, 5 and 7 to 12. It comes first,
   since it rewrites the workflows the others touch.
2. After #248, in parallel:
   - #249 Reorganize the nightly by axis and release from a full run: decisions 13 to 15.
   - #196 Support IPOPT on Windows: PR #218 rebased onto #248 without its version bumps, adding two
     fixes from #231, the serialised `ar` step that kept OpenBLAS's static library from being
     corrupted on many-core machines, and the `pefile` scan of Windows imports.
3. #250 Reuse released native builds when the recipe is unchanged: decision 3, after #196 since
   both rewrite the Ipopt hook. It pays off from the next plugin release, which also ships Windows
   Ipopt. `scaly-fatrop` (#237) follows decisions 2 and 3.
