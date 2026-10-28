# Ipopt native candidates and publication

The producer uses the shared command and archive boundary described in
[the archive contract](ipopt_native_bundle_contract.md). The
[source controls](ipopt_native_controls.md) retain the numerical baseline.
Ordinary package CI and the hook's default remain unchanged by this producer.

## Candidate generation

`ipopt-native.yml` builds Windows x86-64, Linux x86-64 and aarch64, and macOS x86-64 and arm64.
Linux uses the digest-pinned manylinux 2.28 images from cibuildwheel 4.2.1 and repairs to the
existing manylinux 2.27 floor. Native inputs trigger PR and main candidates. The selector compares
the builder's recipe hash, so a wrapper or consumer-pin edit alone does not compile the stack.
Changes to the validation scripts or producer workflows rerun acceptance too.

A manual candidate uses the branch input and selects one target or all five. It has no publication
switch or release permission. Dispatch uses the workflow on main once it has landed:

```bash
gh workflow run ipopt-native.yml -R PREDICT-EPFL/scaly --ref main -f branch=main -f target=all
gh workflow run ipopt-native.yml -R PREDICT-EPFL/scaly --ref main -f branch=BRANCH -f target=windows-x86_64
```

The independent monthly schedule runs all five targets from main. It records health evidence
without attestations for publication and cannot supply an official revision. Neither candidate
generation nor the monthly run delays ordinary package tests.

Each target artifact contains `scaly-ipopt-native-TARGET.tar.gz`, the builder's `.manifest.json`
and `.sha256` siblings, and `.provenance.json`. Provenance records the actual source commit,
run and attempt, runner image, Linux image reference/digest and Docker identity. The manifest
records source revisions and patches, flags, compiler and runtime identities, effective OpenBLAS
settings and the dependency closure. Compare generated `NUM_CORES` and `MAX_CPU_NUMBER` with
the controls' compiled maximum of four. A different host maximum is reported without setting
a different thread policy. Accelerate's active thread count remains unmeasured.

Validation runs on a separate runner, without the producer work directory. It audits binary
imports and floors, including delay imports on Windows, and installs through the explicit local
archive override. It verifies installed native hashes, compiles C and C++ consumers against the
matching public C headers and ordinary `-lipopt` linkage, and checks the libraries those consumers
actually load. Windows uses the packaged Zig compiler. The sparse consumer requests MUMPS for a
512-variable problem with a tridiagonal Hessian and 256 equality constraints.

Fresh Python processes exercise direct and nested solves, statistics and both load orders with
CasADi, NumPy and SciPy. The same probes run against the recorded source-built wheel, whose hash
must match the control inventory. Status, iteration and solution differences fail acceptance for
investigation. Equal source versions alone never establish native equivalence. The report also
states whether native bytes match and records configuration differences. These checks make no
hosted-CI performance claim.

Before the first official revision, the controls come from the recorded Actions run. Publication
attaches an `ipopt-source-control-TARGET.zip` containing the original Ipopt and PIQP wheels for
each target. Later health runs read those retained controls from `scaly-ipopt-native-v1`, with
the Ipopt wheel still required to match the checked-in control hash. This preserves the source
baseline after Actions artifacts expire. It does not select a native bundle for ordinary consumers.

The existing solver-required tests, `solve_smoke.py`, `windows_install.py` on Windows, and the
chain and compiled CasADi race-cars closed-loop smokes supply the remaining acceptance checks. Linux also runs a standalone
consumer in glibc 2.27 with only allowed system libraries. Hiding the bundled Fortran runtime must
break a fresh load. Its positive control must pass first. No build or host search path is mounted
in that container. Each target produces a validation report and detailed audit/probe evidence.
The final `ipopt-accepted` artifact binds the full set to one commit, run and attempt. Trial
summaries are marked `trial` and cannot publish.

## Maintainer publication

Publication is separate from candidate generation. Do not run the commands in this section until
the maintainer authorizes publication after the producer lands. No native revision is published
by this implementation task.

1. Review and land the producer, then generate a full candidate from current main. Retain the
   run URL, attempt, reviewed main commit, five archive hashes and all acceptance evidence.
2. Check every target report, including effective threading and CPU flags against the controls.
   Investigate changed numerical results or native configuration before admitting the run.
3. The `ipopt-native-publication` environment must require an independent reviewer with
   self-review disabled. Its setup is a maintainer repository-setting decision. This work neither
   creates that setting nor enables repository-wide release immutability. The existing Python
   `release.yml` attaches assets to its releases and must be examined before immutability changes.
4. Dispatch a read-only preflight with the full reviewed commit and exact candidate run:

```bash
gh workflow run ipopt-native-publish.yml -R PREDICT-EPFL/scaly --ref main \
  -f candidate_run=RUN_ID -f reviewed_sha=FULL_MAIN_SHA \
  -f revision=scaly-ipopt-native-v1 -F publish=false
```

The preflight requires a repository maintainer, a successful full main producer run, an approved
merged main PR, current main identity, five complete reports from the same attempt, exact asset
hashes and GitHub attestations signed by the producer on that main commit. PR, trial and monthly
output is rejected. It downloads the tested bytes and emits a reviewable publication plan and
`consumer-pins.json`. A successful trusted full main run attests the archive bytes after all
targets pass. Publication also attaches the verified attestation bundles for offline checks.
PR jobs have no attestation or release-write permissions.

After reviewing the plan, the maintainer authorizes the explicit publication dispatch:

```bash
gh workflow run ipopt-native-publish.yml -R PREDICT-EPFL/scaly --ref main \
  -f candidate_run=RUN_ID -f reviewed_sha=FULL_MAIN_SHA \
  -f revision=scaly-ipopt-native-v1 -F publish=true
```

The environment approval is the final gate. Publication downloads the preflight's exact assets
without rebuilding, creates a draft with `--latest=false`, attaches every asset, downloads them
again and compares every byte hash before publishing. An existing tag or release is rejected.
An interrupted upload leaves an unpublished draft for maintainer inspection. Do not overwrite
an earlier revision or use a package-version tag.

The expected pin document has this shape, with all five target keys and full hashes:

```json
{
  "release": "scaly-ipopt-native-v1",
  "repository": "PREDICT-EPFL/scaly",
  "targets": {
    "windows-x86_64": {
      "asset": "scaly-ipopt-native-windows-x86_64.tar.gz",
      "sha256": "FULL_ARCHIVE_SHA256",
      "platform_tag": "win_amd64"
    }
  }
}
```

The other keys are `linux-x86_64`, `linux-aarch64`, `macos-x86_64`, and `macos-arm64`. This is
the producer's handoff format. The consumer follow-up owns its checked-in pin representation
and the final public-download acceptance checks.

## Rollback

Keep every previous release revision and its exact assets. Rollback selects an earlier reviewed
pin document in the consumer change, then runs that change's normal acceptance checks. It does
not edit, delete or republish native release bytes:

```bash
gh release download scaly-ipopt-native-vPREVIOUS -R PREDICT-EPFL/scaly \
  --pattern consumer-pins.json --dir build/native-rollback
gh release view scaly-ipopt-native-vPREVIOUS -R PREDICT-EPFL/scaly
```

Use the downloaded release identity, target assets and hashes in the consumer's pin file and
submit a reviewed rollback PR. The consumer issue owns that file and its exact update command.
If no previous official revision exists, there is no native-release rollback target. Keep the
ordinary source baseline available until initial consumer adoption passes.
