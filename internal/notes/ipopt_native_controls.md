# Ipopt native payload controls

This inventory records the five source-built wheel controls from [PR #218](https://github.com/PREDICT-EPFL/scaly/pull/218).
It supplies the observed payload boundary for #228 and the comparison inputs for #229 and #230.
The [fixed design](https://github.com/PREDICT-EPFL/scaly/issues/228#issuecomment-6440493339)
keeps the supplier, numerical versions, threading policy and platform floors unchanged.

## Inputs and inspection

Downloaded every `dist-scaly-ipopt-*` artifact from
[run 39988283717](https://github.com/PREDICT-EPFL/scaly/actions/runs/39988283717) on 2026-10-27.
The refreshed [run 40588152775](https://github.com/PREDICT-EPFL/scaly/actions/runs/40588152775)
reports the same artifact ZIP SHA-256 for all five Ipopt artifacts. Its rerun reused these native bytes.
This does not claim that rebuilding the recipe reproduces the bytes.

```powershell
gh run download 39988283717 --repo PREDICT-EPFL/scaly --pattern 'dist-scaly-ipopt-*' --dir build/control-inventory
gh run view 39988283717 --repo PREDICT-EPFL/scaly --job JOB_ID --log
```

The five JSON files under [ipopt_native_controls/](ipopt_native_controls/) record wheel names,
wheel SHA-256 values, every native payload file's path, size and SHA-256, declared imports for
every shared library, loader paths and compatibility metadata. They include the matching six C
headers and copied license texts. They exclude Python wrappers, wheel metadata and the Python
distribution's own license. Inspect the JSON rather than matching a shortened hash from prose.

Inspection used CPython 3.13, LIEF `1.0.0-d05b3499b` for Portable Executable, Mach-O and ELF
headers, and pyelftools `0.33` for ELF symbol-version requirements. SHA-256 values come from the
uncompressed wheel member bytes. Binary inspection records declared imports; it does not replace
clean-machine execution or a compatibility-container audit. Build-log excerpts carry their job
and line identities in each JSON. Downloaded wheels, extracted members, full job logs and temporary
inspection scripts remain in the ignored `build/control-inventory/` directory.

| Control | Job | Native files | Observed compatibility |
|---|---:|---:|---|
| Windows x86-64 | 120533454608 | 27 | `win_amd64`, Universal C Runtime imports |
| Linux x86-64 | 120533454383 | 24 | Both `manylinux_2_27_x86_64` and `manylinux_2_28_x86_64`; highest required GLIBC symbol version is 2.27 |
| Linux aarch64 | 120533454267 | 22 | Both `manylinux_2_27_aarch64` and `manylinux_2_28_aarch64`; highest required GLIBC symbol version is 2.27 |
| macOS x86-64 | 120533454506 | 21 | `macosx_15_0_x86_64`; maximum recorded Mach-O minimum is 15.0 |
| macOS arm64 | 120533454354 | 21 | `macosx_15_0_arm64`; maximum recorded Mach-O minimum is 15.0 |

The Windows optional-header OS/subsystem versions are recorded verbatim. They do not establish
the actual minimum supported Windows version. In particular, those old linker header numbers
do not supersede the Universal C Runtime requirement or the project's supported Windows floor.
The GLIBC symbol maximum alone does not establish a CPU baseline or every manylinux constraint.

## Actual native layouts

Windows contains `scaly_ipopt/lib/libscaly_ipopt-3.dll`, `ipopt.lib`, and these five runtime DLLs:

- `scaly_ipopt_dc059d083519_libgcc_s_seh-1.dll`
- `scaly_ipopt_dc059d083519_libgfortran-5.dll`
- `scaly_ipopt_dc059d083519_libquadmath-0.dll`
- `scaly_ipopt_dc059d083519_libstdc++-6.dll`
- `scaly_ipopt_dc059d083519_libwinpthread-1.dll`

The runtime DLLs import the same private names where required. Their remaining imports are
Windows system DLLs and `api-ms-win-crt-*` contracts. None has a delay import. The ordinary Ipopt
import library must keep targeting `libscaly_ipopt-3.dll`; copying a toolchain's differently named
import library would break this boundary. The exact per-file hashes distinguish these CI bytes
from another native build with a different private-name digest.

Both macOS wheels contain `libipopt.dylib`, `libgfortran.5.dylib`, `libquadmath.0.dylib` and
`libgcc_s.1.1.dylib` under `scaly_ipopt/lib/`. Non-system imports use `@loader_path` siblings.
The remaining imports are `/usr/lib/libSystem.B.dylib`, `/usr/lib/libc++.1.dylib` and the system
Accelerate framework. Ipopt, gfortran and quadmath record minimum macOS 15.0. The gcc runtime
records minimum 11.0. Lowering the wheel floor to 11.0 would therefore misdescribe the payload.

Both Linux wheels contain `scaly_ipopt/lib/libipopt.so`, `libipopt.so.3` and `libgfortran.so.5`.
The two Ipopt filenames contain identical bytes within each wheel and retain `$ORIGIN` RUNPATH.
The aarch64 wheel has no quadmath import or quadmath payload. Its Fortran runtime additionally
imports the system `libz.so.1`, `libm.so.6`, `libgcc_s.so.1`, `libc.so.6` and
`ld-linux-aarch64.so.1`.

The Linux x86-64 wheel additionally contains both:

- `scaly_ipopt/lib/libquadmath.so.0`
- `scaly_ipopt.libs/libquadmath-2284e583.so.0.0.0`

Ipopt imports `libquadmath.so.0`. The repaired `libgfortran.so.5` imports
`libquadmath-2284e583.so.0.0.0` and has RPATH `$ORIGIN/../../scaly_ipopt.libs`.
That sibling directory is part of the complete payload. A producer that archives only the source
hook's `lib/include/licenses` output would lose it after this repair. Preserve the directory and
its relative path when consuming a completed repaired bundle. The other Linux imports are system
libraries, including libstdc++, libgcc, zlib, libm, libc, the dynamic loader, libpthread and libdl.
The per-object lists and required GLIBC, GLIBCXX, CXXABI and GCC symbol versions are in the JSON.

## Numerical components and effective BLAS build

The shipped notices identify Ipopt 3.14.19, MUMPS 5.8.2 through ThirdParty-Mumps 3.0.12,
METIS 5.2.1 and the pinned GKlib revision. MUMPS, METIS and GKlib are static components of
Ipopt. Linux and Windows also link OpenBLAS 0.3.28 statically. macOS imports Accelerate.
The copied `IpoptConfig.h` leaves `IPOPT_SINGLE` and `IPOPT_INT64` undefined, so the matching
public headers use double precision and the 32-bit integer interface.

The three OpenBLAS job summaries explicitly say
`libopenblasp-r0.3.28.a (Multi-threading; Max num-threads is 4)`.
Compilation uses `DYNAMIC_ARCH`, `SMP_SERVER`, `MAX_CPU_NUMBER=4`, `MAX_PARALLEL_NUMBER=1`
and `NO_AFFINITY`. The source recipe passes `USE_OPENMP=0`; the effective thread implementation
is pthreads. These are compiled settings. They do not prove the number of threads used by a
particular solver call. The `OPENBLAS_NUM_THREADS=1` and `=2` commands in these logs run
OpenBLAS's own tests and must not be reported as the installed solver's default.

The x86-64 Linux and Windows summaries report `PRESCOTT` as the common-code minimum while
retaining runtime dispatch. The Linux aarch64 summary reports `NEOVERSEN2`. Its ordinary BLAS
interface objects compile with `-march=armv8.5-a+sve+sve2+bf16 -mtune=neoverse-n2`, alongside
separately compiled dispatch kernels for other CPUs. Therefore these controls do not establish
a generic Armv8 CPU baseline merely because `DYNAMIC_ARCH=1` was set. #229 must carry this
observed baseline into the audit and surface any proposed change separately from delivery work.

The Linux controls use GCC and GFortran 14.2.1 from the manylinux toolset. The Windows control
uses MSYS2 GCC and GFortran 16.2.0. The source hook pins the Windows runtime package transaction;
the wheel notices record GCC runtime 16.2.0-6 and winpthreads 14.0.0.r495.g63ebe6cdb-2.
The control JSON records hashes of those notices and actual runtime files.

The retained local Windows build's `openblas/Makefile.conf` records
`CORE=PRESCOTT` and `NUM_CORES=20`.
This is a different host build from the four-thread CI controls. Do not assign the CI's compiled
thread capacity to a new workstation build without inspecting that build's generated settings.

## Remaining evidence

#229 owns execution of the five-target producer gates and complete compatibility audits. This
inventory does not claim runtime validation of Linux or macOS binaries on this Windows machine.
The installed thread count and selected dispatch kernel depend on the actual process environment
and CPU; those need runtime observations if used in comparisons. Accelerate's effective threading
is not established by the recorded imports. A local manifest must state what was observed and
must not replace missing evidence with an assumed single-thread or portable-baseline claim.

#230 must install the complete repaired layout, preserve its compatibility tag, and compare the
actual loaded Ipopt hash. The source controls already show why a tag based only on the newer
consumer host or a copy limited to `scaly_ipopt/lib/` cannot satisfy that contract.
