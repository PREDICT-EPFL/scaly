# Native Windows Ipopt archive evidence

This records the workstation evidence for [#228](https://github.com/PREDICT-EPFL/scaly/issues/228)
and [PR #231](https://github.com/PREDICT-EPFL/scaly/pull/231). The source foundation is #218,
with the toolchain and numerical configuration recorded in the
[five-target controls](ipopt_native_controls.md). The archive/install interfaces are in the
[native archive contract](ipopt_native_bundle_contract.md).

## OpenBLAS archive corruption

The first producer attempts failed while updating OpenBLAS's shared static archive. GNU ar
reported truncated members and unrecognized file formats. Lowering Python's build-job count
alone did not serialize recursive makes, since OpenBLAS's generated `Makefile.conf` adds its
detected parallelism back to `MAKEFLAGS`. The attempted manual diagnosis and aggregate rebuild
under `build/native-bundle/` is not acceptance evidence for the producer command.

A separate control used the real pinned GNU archiver and 48 retained OpenBLAS object files,
with 12 concurrent writers to one archive. Without locking, ten writers failed and only four
of the 48 members survived. With the hook's archive wrapper, every writer succeeded and all
48 members survived. [The control JSON](ipopt_native_archive_race.json) records the archiver
identity, every object hash and both results. This demonstrates an archive-write defect, rather
than a numerical or thread-policy change.

The shared source recipe wraps `ar` and `ranlib` in one cross-process lock per OpenBLAS tree.
It keeps ordinary compilation parallelism and the existing `NO_SHARED=1`, `USE_OPENMP=0`,
`DYNAMIC_ARCH=1`, `TARGET=PRESCOTT` settings. It sets no runtime thread count. A regression
launches six subprocess writers and fails if their exclusive write sections overlap.

The acceptance invocation uses a fresh `build/native-locked/` work directory, seeded with a
copy of #218's retained OpenBLAS source tree, compiled objects and complete original archive.
That is an incremental input, not a fresh compilation of every OpenBLAS object. All other native
components start from their pinned sources. Nothing in that work directory receives a manual
native repair. The original #218 work directory is read only throughout this task.

The retained-checkout check also exposed MSYS argument expansion of `HEAD^{commit}` into
`HEAD^commit`. The equivalent `HEAD^0` peels both lightweight and annotated tags without braces.
The corrected check passed against the retained OpenBLAS HEAD and v0.3.28 tag before building.

## Successful producer and local installation

The documented producer command completed successfully on 2026-10-27:

```powershell
$env:PATH = "$toolroot/ucrt64/bin;$toolroot/usr/bin;$env:PATH"
$env:MSYSTEM = 'UCRT64'
$env:CHERE_INVOKING = '1'
$env:MSYS2_PATH_TYPE = 'inherit'
$env:CC = 'gcc'
$env:CXX = 'g++'
$env:FC = 'gfortran'
uv run plugins/scaly-ipopt/hatch_build.py bundle --work-dir build/native-locked --output build/scaly-ipopt-native-windows-x86_64.tar.gz
```

Here `$toolroot` is the existing MSYS2 installation. The command used ordinary `make -j20`,
without `MAKE_NB_JOBS` or `PYTHON_CPU_COUNT` overrides. The effective OpenBLAS configuration
still has `CORE=PRESCOTT`, `NUM_CORES=20`, pthreads and dynamic CPU dispatch. The host-dependent
compiled maximum differs from the CI controls' four, as it did before this extraction. This
does not measure active runtime threads or introduce a new thread policy.

The archive SHA-256 is
`2304c2ecfad7078813d4d4a68d4c90c4f94f18799e8ada654d98a5d778d0893d`.
Its recipe SHA-256 is
`80d56564653b6a7859688edeefe4fdd1d623effac7efa2e734e5714f15697789`.
These identify a native recipe and archive, not a Git revision. The
[machine-readable evidence](ipopt_native_windows_demo.json) records the target, source commits,
patch identities, toolchain, effective OpenBLAS files, complete payload hashes, dependency graph,
probe source and results. Full native work, commands, logs, manifest and archive remain under
the ignored `build/` directory. The evidence records their hashes and replaces workstation paths
with `$WORKTREE` and `$MSYS2` where possible.

The explicit local path completed with these commands:

```powershell
$env:SCALY_IPOPT_BUNDLE = (Resolve-Path build/scaly-ipopt-native-windows-x86_64.tar.gz).Path
$env:SCALY_SOLVER_CACHE = "$PWD/build/bundle-cache"
$env:SCALY_REQUIRE_SOLVERS = 'ipopt,piqp'
uv sync --reinstall-package scaly-ipopt
uv run --no-sync pytest plugins/scaly-ipopt/tests -q
uv build plugins/scaly-ipopt --wheel --out-dir build/demo-wheels
uv run --no-sync .github/scripts/windows_install.py build/demo-wheels 3.14
```

All 60 Ipopt plugin tests passed with the solver required. The first run exposed six mocked
source-toolchain cases inheriting the archive override. Clearing that variable in those cases
restores their intended source-path coverage. Every payload file in the editable installation
and built wheel matched its manifest hash. The wheel is tagged `py3-none-win_amd64`.

The installed-wheel probe used Python 3.14.8, NumPy 2.5.4, SciPy 1.18.1 and packaged Zig 0.17.0.
It installed the four local Scaly wheels in a fresh environment, then passed direct and nested
PIQP/Ipopt and SQP solves. The child process used only that environment's Scripts directory and
Windows system directories on `PATH`. It checked every loaded solver DLL's installed location.

## Standalone Zig and import-library consumer

The consumer source in the evidence calls Ipopt's C interface with double precision and 32-bit
indices. It minimizes `x0*x0 + x1*x1` subject to `x0+x1=10`, using MUMPS. Packaged Zig compiled it
as a DLL with the archive's `include/coin-or` headers and `-L.../lib -lipopt`, resolving the
archive's `ipopt.lib`. The wheel's extracted native bytes were the only payload used.

A fresh isolated Python process loaded this consumer with only `Windows/System32` and `Windows`
on `PATH`. It returned status 0, solution `[5, 5]`, objective 50 and seven iterations. Every
loaded native DLL came from the extracted wheel and matched its recorded hash. CasADi was not
imported. Hiding the private Fortran runtime made another fresh load fail, and restoring it
restored the complete payload. The control therefore exercises the bundled runtime dependency
chain rather than relying on a toolchain directory on `PATH`.

## Repository checks

With this archive installed and both native solvers required, `wt hook pre-merge --yes`
passed type, format and lint checks. Its full suite reported 2746 passed, 104 skipped and
one failure in `test_native_recipe_render_policies_separate_cache_entries`. That test probes a
hard-coded `cc` executable which is absent on this workstation, and the maintainer confirmed
the same failure on main. It is left unchanged. Four pytest workers avoided the Zig thread
exhaustion seen with 20 workers earlier in the task.

The benchmark smoke passed against the installed archive, including native consumers and solver
calls. A clean Zensical documentation build exited 0 with no issues. The clean build cleared
stale incremental link warnings from the workstation's documentation cache.
