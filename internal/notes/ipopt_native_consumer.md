# Ipopt bundle consumer, publication handoff

This is part 1 of [#230](https://github.com/PREDICT-EPFL/scaly/issues/230), stacked on
[PR #233](https://github.com/PREDICT-EPFL/scaly/pull/233). The fixed design is
[#228's consumer and acceptance contract](https://github.com/PREDICT-EPFL/scaly/issues/228#issuecomment-6440493339).
The producer's publication procedure and pin document are in
[`ipopt_native_producer.md`](ipopt_native_producer.md). The archive boundary stays in
[`ipopt_native_bundle_contract.md`](ipopt_native_bundle_contract.md).

## Current behavior

No official native release exists at this stage. Checked-in configuration contains no
`native_bundle` entry, and ordinary jobs still use the existing source default. Nothing here
publishes a release, treats a candidate as official, or changes the numerical recipe.

The consumer accepts the producer's complete `consumer-pins.json` object under
`native_bundle` in `plugins/scaly-ipopt/src/scaly_ipopt/build_config.json`. It requires the
first-party repository, release identity and five target entries, each with its archive name,
full SHA-256 and platform tag. There is no mirror environment variable. Tests replace the URL
constant in a temporary copy of the hook, and their pins never enter the checked-in config.

The hook selects by operating system, normalized architecture and glibc availability. It
checks the pinned platform floor before downloading, checks the full archive hash, and passes
the same immutable bytes to #228's manifest validator and locked installer. Download entries
use `ipopt-download-SHA256/bundle.tar.gz` in the existing solver cache. Their exclusive locks
also protect them from the existing cache pruner. Extraction uses the existing
`ipopt-bundle-SHA256` entry. The hook rechecks the compressed hash and full native inventory
on offline reuse. Corruption fails with the affected cache path and never triggers compilation.

Required mode and wheel builds fail when an official input is unavailable. Editable auto mode
permits core-only work when the host has no compatible target or the network is unavailable.
It removes stale native outputs before skipping. Invalid pins, checksum failures, malformed
manifests and corrupt extracted caches remain fatal. Skip mode retains its existing precedence.
`SCALY_IPOPT_BUNDLE` remains a strict explicit local archive input, including in editable auto
mode, and takes precedence over official pins. Installed identity distinguishes the override
from an official selection, even when they contain identical bytes.

`native_bundle.json` records selection identity beside the unchanged `native_manifest.json`.
Both wheel and editable installation preserve the payload and tag without repair. Sdists
include the checked-in config and hook but exclude generated native files and identity.
The benchmark's existing `native_solvers.ipopt` record includes `sha256` for its resolved
library and `bundle` for installed identity, or null for source/system inputs. The solver
discovery and application binary interface in `src/scaly/solvers/paths.py` are unchanged.

## Prepared activation

`ipopt_native_consumer_activation.patch` is deliberately unapplied. It removes the ordinary
hook's source branch and compiler/repair build requirements, packages Ipopt directly with
`uv build` on all five native hosts, and retains PIQP's existing cibuildwheel path. Ipopt
packaging performs no second auditwheel/delocate/delvewheel transformation. Installed test,
clean-install and package-release jobs check the pinned identity and every manifest file with
`.github/scripts/ipopt_bundle_verify.py`. The existing solver and Python-version matrices stay
in place, including Windows 3.12 through 3.14. Nightly rebuilds the wrapper wheel with an empty
native cache. Producer health remains separate.

The patch also contains install/developer documentation for the activated behavior. Those
claims stay out of the published site until official inputs pass acceptance. A source producer
still uses the existing explicit `bundle` command and its script dependencies. Its native
recipe, archive validator and producer/publication workflows are unchanged.

Apply the patch only after receiving all official inputs, then regenerate `uv.lock` for the
removed ordinary build requirements. Recheck the patch after every code change or lower-layer
rebase. `git apply --check internal/notes/ipopt_native_consumer_activation.patch` checks its
context without activating it. The patch is a preparation artifact, not finished acceptance.

## Inputs and work required for part 2

The coordinator must supply the maintainer-published native release identity and complete
`consumer-pins.json`, including five actual archive SHA-256 values, asset names and platform
tags. Also retain the trusted producer run URL/attempt, reviewed main commit, five acceptance
reports and source controls. The producer must land on main and the maintainer must publish
before these are used for final consumer acceptance. A candidate artifact does not satisfy this.

1. Download the published pin document and all five assets. Check their hashes and provenance
   against the supplied acceptance record. Insert the exact object as `native_bundle` in
   `build_config.json`. Do not copy candidate or fixture pins.
2. Apply and review the activation patch. Remove any remaining ordinary source-cache branch
   that has become unused, while retaining the explicit producer and its dependencies. Refresh
   the lockfile and check both checkout and sdist-to-wheel packaging. Retarget the old
   source-hook toolchain-mode tests to the explicit producer's missing-tool check, since the
   ordinary hook will no longer inspect native tools. Regenerate the node-ID baseline.
3. Run #229's packaging, binary/floor, C/C++, sparse MUMPS, direct/nested statistics,
   coexistence and missing-runtime gates against the final installed consumer wheel on all
   five targets. Compare status, iterations and solutions with the retained source controls.
   Preserve installed headers, Windows import library, notices and repaired sibling hashes.
4. Run the existing solver-required suite and release verification across their established
   Python coverage. Keep the Linux clean-container and macOS install jobs.
5. Run three independent cold-download-cache CI builds per target with wheel caching disabled,
   retain run URLs and job times excluding runner queues, and check logs contain no producer
   command. The acceptance target is under five minutes for each Ipopt wheel job. Time the
   editable native installation separately from Python environment creation on an ordinary
   network, targeting under two minutes. Local loopback timings are not this evidence.
6. Publish the activated docs, record misses and unresolved evidence, update the draft PR,
   complete the independent review, and remove temporary plan/activation artifacts when done.

Rollback selects a previous retained and tested release's complete `consumer-pins.json` and
repeats consumer checks. It changes neither release bytes nor the source recipe and never
enables source fallback. If no previous official revision exists, no native-release rollback
target exists. Unsetting the local override and reinstalling restores the configured official
selection after activation.

## Part 1 evidence

The focused consumer tests exercise all five layouts, concurrent single-download behavior,
offline cache reuse, corrupt archives/caches, unsupported libc/architecture/floor, mode behavior,
override precedence, installed-byte checking and real Hatch checkout/sdist wheels. The first
new test failed before the consumer interface existed. The native recipe hash equals the
stack base's recipe hash when both source files are decoded as UTF-8.

On native Windows, the retained real #228 producer archive has SHA-256
`312e69b5c3020491f114fcaea6d7a56bc61a8753a2c64f3bd106a9fe94a35fc6`. A temporary
checkout used it through a loopback HTTP mirror and test-only pins. The checkout wheel made
one request. With the mirror shut down, sdist-to-wheel and editable installation reused the
validated cache. Both wheels retained all 27 native file hashes, headers, import library and
notices. The installed-wheel byte/identity check, `solve_smoke.py` and
`windows_install.py --installed` pass in a fresh Python 3.14 environment. The latter verifies
direct/nested PIQP and Ipopt and SQP with all solver DLLs from the installed wheels.

The local cold-cache wheel command took 3.42 seconds. A warm-cache editable install took
2.60 seconds, measured after creating its Python environment. The archive was served on
loopback. These are development diagnostics, not public-network or five-target timing evidence.
No producer command ran during these consumer checks.

The prepared activation hook and reduced build requirements were also exercised in a temporary
copy. With pins absent it fails rather than compiling. With the real archive supplied by the
test-only mirror, its isolated wheel build and subsequent offline editable install pass with
only Hatchling as a declared build requirement. The five prepared workflows pass actionlint,
and the activation patch passes `git apply --check`.

The producer's existing audit/probe functions pass against the final installed Windows consumer
wheel. C and C++ both solve the 512-variable sparse MUMPS problem with status 0 in seven
iterations. Direct and nested Python solves, real generated-module binding checks and both
coexistence load orders match the retained source wheel's status, iterations and solutions.
The source wheel SHA-256 is
`06b05a83a26d914e52fe5f31e84ebc819e1d05b860cdd87317fc192141a6f200`.
The isolated runtime-removal control passes before hiding the private Fortran DLL and fails
afterwards. All native hashes remain unchanged after packaging.

`wt hook pre-merge --yes` passes type, Ruff format and lint. The full solver-required suite
reports 2772 passed, 107 skipped and one existing workstation failure in
`test_native_recipe_render_policies_separate_cache_entries`, which invokes missing bare `cc`.
Both prerequisite PRs record the same failure. No compiler or code-generation file changed.
The remaining hook commands were run separately after that failure stopped the hook.
Benchmark smoke passes and a documentation build exits 0 with no issues. The final focused
hook/provenance run reports 93 passed. A mutation removing the archive checksum comparison
makes the checksum test fail while the archive remains structurally valid.

Project updates are unavailable because the GitHub token lacks `read:project`. Publication,
final all-target consumer gates and timing acceptance remain open in #230. No new follow-up
issue has been identified yet.
