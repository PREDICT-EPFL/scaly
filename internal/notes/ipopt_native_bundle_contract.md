# Ipopt native archive contract

The archive boundary and explicit local input implement [#228](https://github.com/PREDICT-EPFL/scaly/issues/228)
and the [fixed design](https://github.com/PREDICT-EPFL/scaly/issues/228#issuecomment-6440493339).
The source recipe remains in `plugins/scaly-ipopt/hatch_build.py`. The command in
`docs/dev/contributing.md` calls that recipe. This contract is shared with producer issue #229
and ordinary-consumer issue #230. No producer workflow, release pin or automatic download is
introduced here.

## Layout and identity

Version 1 is a gzip-compressed POSIX tar archive with these paths relative to its root:

```text
manifest.json
scaly_ipopt/lib/...
scaly_ipopt/include/coin-or/...
scaly_ipopt/licenses/...
scaly_ipopt.libs/...                 optional repaired Linux sibling
```

`lib/` contains the final shared library and complete non-system dependency closure. Windows
also contains `ipopt.lib` targeting `libscaly_ipopt-3.dll`. Headers and license texts come from
the same source build. Repair, private DLL renaming, import-library creation and macOS signing
precede hashing and archiving. Python wrappers, wheel metadata and build directories are outside
the archive. The producer writes a copy of the manifest and a SHA-256 checksum beside the archive.

`recipe_sha256` hashes the syntax trees of the native recipe and repair functions, the six
native configuration entries in `build_config.json` and checked-in license texts. It excludes
Python wrapper code, hook consumer logic, archive metadata and consumer bundle pins. Changes in
compiler inputs also appear in the manifest's toolchain and environment records. Identical recipe
hashes do not imply identical native bytes. Every archive has its own full SHA-256, computed after
compression. Tar timestamps may change that identity without changing the native payload.

## Manifest version 1

The top-level fields are:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1`; unknown versions fail |
| `target` | `system`, normalized `arch`, one `platform_tag`, and explicit compatibility `floor` |
| `recipe_sha256` | Native recipe identity, independent of the output archive's checksum |
| `inputs.sources` | Resolved Git commits, source URLs, tracked source patches and patch hashes; individual downloaded MUMPS source hashes |
| `inputs.build_config` | Exact native source configuration, including static linkage and Windows package versions |
| `inputs.toolchain` | Compiler/build/repair identities and Windows `pacman -Q` inventory |
| `inputs.environment` | Compiler flags and relevant BLAS/thread environment values, recorded without changing policy |
| `inputs.diagnostics` | Generated configure logs, configure status, CMake caches and OpenBLAS settings, with hashes and text |
| `inputs.commands` | Producer command log with native arguments and working directories |
| `configuration` | Double precision, 32-bit integers, BLAS implementation/effective settings, CPU baseline/dispatch and effective threading settings |
| `components` | Names indexed by the copied license directories; static provenance is in native configuration, source and diagnostics records |
| `dependencies` | Every shared object's direct `bundled` payload paths and `system` imports, including transitive objects |
| `files` | Exact native file inventory keyed by archive path, with `sha256`, byte `size` and POSIX `mode`; symlinks also have `link` |

The OpenBLAS generated settings record compiled thread capacity and flags. They do not claim a
measured runtime thread count. Accelerate remains OS-provided, with runtime threading unmeasured.
The [control inventory](ipopt_native_controls.md) records the actual five-target baselines.
In particular, the existing ARM Linux common code targets NEOVERSEN2. Delivery work must not
silently substitute a generic ARM baseline or a different thread policy.

Targets are Windows/x86_64, Linux/x86_64, Linux/aarch64, Darwin/x86_64 and Darwin/arm64.
Windows uses `win_amd64` and the Windows 10 Universal C Runtime floor. Local Linux uses
`linux_<arch>` and its host glibc version. An explicit `manylinux_2_N_<arch>` invokes auditwheel
repair and must succeed in an appropriate compatibility environment. The floor and tag must
agree and cannot be older than glibc 2.27. Darwin tags are `macosx_MAJOR_MINOR_<arch>` and reflect
the maximum native deployment minimum, at least macOS 15. Validation rejects a payload requiring
a newer OS than the consumer host. None of these metadata checks replaces #229's binary and
compatibility-container audits.

The five fixtures in `plugins/scaly-ipopt/tests/fixtures/bundles/` model the control layouts,
including Linux x86-64's required hashed quadmath sibling. They use text in place of binaries,
so tests can exercise the boundary on every host without running another target's native code.

## Input boundary and installation

`_read_bundle(bundle: bytes, out: Path | None = None)` validates the entire immutable byte string
before writing anything. It checks target, floor, schema, required solver/header/import-library
and notice entries, required language runtimes, declared dependency closure and all file hashes.
Only declared native paths and their parent directories are allowed. Absolute, traversing,
backslash, drive-qualified, Windows-reserved, duplicate and case-colliding paths fail.
Devices, FIFOs, hardlinks and directory symlinks fail. A relative file symlink may resolve to one
regular payload file in the archive, never another link. Its recorded size/hash describe the
UTF-8 link target text. Ordinary entries describe the file bytes. Archive mode and manifest mode
must agree, without setuid/setgid/sticky bits.

`BuildHook.initialize` returns immediately for sdists. Skip mode preserves its existing precedence.
Otherwise an explicit `SCALY_IPOPT_BUNDLE` is strict, including editable automatic mode: missing,
corrupt or incompatible archives fail without a source fallback. With no override, the existing
source path and editable automatic/required behaviour remain in effect.

`_cache_key(root, bundle=None)` retains source-cache identity for the default hook. Passing archive
bytes returns `bundle-` plus their full SHA-256 and requires no compiler. The hook reads the
archive once and uses those same bytes for verification, identity and extraction. A caller in #230
can pass bytes already checked against a published pin. It must not key on the download path.

`_install_build(..., manifest=None)` retains the source cache's `lib/include/licenses` layout.
Passing the verified manifest selects the complete archive-root layout. It uses the same exclusive
cache lock, fresh scratch output and rename-on-success publication. It verifies cached and
installed native bytes, copies repaired siblings, removes stale native outputs from a previous
selection, and installs `manifest.json` as `scaly_ipopt/native_manifest.json`. The checkout's
`.build_key` stores the content identity. A wheel includes the installed manifest and sibling
directory, and its tag comes directly from the manifest. It performs no second native repair.

Changing an archive at the same path requires the documented `uv sync --reinstall-package
scaly-ipopt`, since uv's environment-variable cache key tracks the value, not that file's bytes.
The native hook's cache key distinguishes the replacement. Unsetting the variable and
reinstalling resets to the ordinary source selection. No official release pin changes.

## Producer and downstream boundaries

`_build_ipopt_stack` is the only numerical source recipe. It accepts the hook logger, retained
source directory, output lib/include/licenses paths and checked-in license directory. Both
source and producer paths call it. Retained Git checkouts must match pinned revisions. Tracked
local patches are recorded rather than erased. `_produce_bundle` checks native tools, retains
sources/build trees/logs on success and failure, reconstructs the payload, finishes repair and
freezes its manifest/archive/checksum. `recipe.json` prevents reusing the work directory after a
recipe, target or compiler-environment change.

#229 should invoke the same command on native runners, pass the explicit manylinux tag in the
compatibility containers, retain manifests/checksums and run its five-target gates. It owns
workflow trigger selection, candidate validation, container-image provenance and authorized
publication. Producer command success alone is not permission to publish or evidence for every
CPU in a platform tag.

#230 should preserve `_read_bundle` and `_install_build` as the verified input/install boundary,
pass pinned downloaded bytes to the content key, and keep finished native bytes and sibling
paths unchanged. It owns the default download/cache policy, checked-in release pins, build-job
tool removal, installed provenance and public-asset acceptance. Consumer repair must not silently
rewrite this frozen payload. Its publication gate remains the maintainer's action after #229.
