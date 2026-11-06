# Add Windows to the Fatrop package in PR #267

The maintainer expanded issue #237 to Windows in this conversation. Preserve Fatrop 1.2.0,
BLASFEO 0.1.4.3, Haswell on x86-64, private static BLASFEO and the existing C-interface tests.
This work stays on `ted/issue-237` and updates the existing pull request.

References: the existing Fatrop recipe and tests, PIQP's Windows Zig/Ninja toolchain and cache
locking, `internal/notes/vendored_solvers.md`, and the native-build reuse decisions. The separate
PIQP update in PR #268 does not change its build hooks. Avoid its code and snapshot changes.

1. [x] Extend regression tests to Windows and reproduce the current rejection.
2. [x] Add Windows toolchain, DLL/import-library installation, locking and export isolation.
3. [x] Enable Windows wheel build, installed-wheel tests and clean installation checks. Update
   the issue's corrected platform scope and the package's build notes and README.
4. [ ] Run focused and full merge checks, prove the Windows C solve and PIQP coexistence, then
   push to PR #267 and address CI failures.
5. [ ] Review the Windows delta once, fix findings, check only those fixes, remove this plan,
   update the PR and tracker, and watch final CI.

Local development is on la015. The Windows machine thebeauty is linked and reachable. Prefer
the existing CI for final wheel evidence. Source downloads and temporary build probes live
under the ignored `third_party/` directory. IPOPT's pre-existing editable runtime issue #266
still requires the cached `wheel-libs/` directory on `LD_LIBRARY_PATH` for strict local checks.

Local proof: the initial Windows branch raised its existing unsupported-platform error. The
expanded focused suite passes 27 checks, and all merge checks pass with 2,462 passed tests and
50 platform/environment skips. A Zig 0.16 cross-build produces the real Windows DLL and import
library, with exactly the 12 C exports and only system DLL imports. Both the C executable and
shared probe link. Zig rejected `--exclude-all-symbols`; removed it and verified that explicit
`__declspec(dllexport)` annotations restrict the DLL exports. CI uses the existing Zig 0.17 pin
and must establish Windows execution and coexistence.

The independent Windows review approved with two notes. Added public C-consumer macro and DLL
search-path instructions, and restricted Windows imports to Kernel32 and the Universal C Runtime.
Validate the import gate against the real cross-built DLL and an injected C++ runtime dependency,
then run the focused delta check. The native Windows CI run is still in progress.
