# Consume pinned Ipopt bundles, issue #230

Stack base: `t3code/issue-229-native-bundle-producer`, PR #233. This branch owns only
the consumer. Do not edit the builder/archive contract or producer/publication scripts.
The fixed design is issue #228's consumer and stack contract comment. Read PRs #231
and #233, `internal/notes/ipopt_native_bundle_contract.md`,
`internal/notes/ipopt_native_producer.md`, the build hook, packaging tests, workflows,
and existing benchmark native-solver provenance before implementation.

## Part 1, before publication

- [x] Inspect prerequisites, stack, checkout and interfaces. Project access is blocked
  by missing `read:project` token scope. No tracker field has been changed.
- [x] Add failing consumer tests for five-target selection, verified downloads,
  locked/offline cache, strict failures, editable auto/skip and override precedence.
- [x] Implement consumer selection using the producer pin document shape under
  `native_bundle` in build_config.json. Leave that entry absent until publication.
  Temporary source default is explicitly authorized for this part only.
- [x] Preserve native bytes, tags and sdist pins. Add bundle identity and the selected
  Ipopt library hash to existing native-solver provenance without changing discovery.
- [x] Exercise actual local/candidate archive through a test-only HTTP mirror,
  checkout and sdist-to-wheel builds and editable installation.
- [x] Prepare the ordinary workflow switch and tool removal as an unapplied patch.
  Keep unfinished official-default documentation in internal notes until activation.
- [x] Run focused checks, full pre-merge checks and one cross-family review.
- [x] Push a draft PR based on #233 and report achievable criteria and remaining inputs.

Tests precede consumer implementation. Packaging/provenance checks follow it. Workflow
preparation can be done independently, but activation depends on published inputs.
One agent implements this part. The independent reviewer reads the finished diff.

## Part 2, publication gate

- [ ] Receive maintainer-published release identity, repository, five asset names,
  archive SHA-256 values and platform tags in the producer's consumer-pins.json.
- [ ] Receive successful producer run/attempt, reviewed main commit and five acceptance
  reports, including original source controls and numerical comparison.
- [ ] Pin those actual inputs. Remove the temporary source default and ordinary build
  requirements. Keep the explicit producer usable. Apply/review the workflow patch.
- [ ] Run final installed-consumer gates on all five targets and established Python
  coverage, including clean Linux/macOS installs and Windows 3.12 through 3.14.
- [ ] Gather three independent cold-download-cache runs per target, with no producer
  commands, excluding queue time. Target under five minutes per Ipopt wheel job.
- [ ] Measure the editable native installation separately from environment creation.
  Target under two minutes. Report missed targets as unresolved evidence.
- [ ] Publish accurate install/developer docs for official pins, offline reuse and
  rollback. Delete this temporary plan only when the full task is complete.

No fabricated release URL or hash, source fallback after activation, release changes,
merges, tags, publication dispatch or pushes to main are authorized.

Part 1 evidence and exact activation inputs are in `internal/notes/ipopt_native_consumer.md`.
The real local Windows archive passes preserved-byte, checkout/sdist/editable, C/C++, sparse
MUMPS, coexistence, source-control numerical comparison and missing-runtime probes. The
prepared activation also builds with only Hatchling and refuses absent pins. Focused tests
report 93 passed. Type/Ruff pass. Full pre-merge reports 2772 passed, 107 skipped and the
known missing-bare-cc failure also recorded by both prerequisite PRs. Benchmark smoke and
the documentation build pass separately. No builder/producer defect has been found.
Draft PR: https://github.com/PREDICT-EPFL/scaly/pull/234. Fable 5.1 at medium effort
approved part 1. Both actionable notes and cheap diagnostic/prose notes are fixed.
Fourteen focused checks pass for those fixes. The producer selection run confirms that
consumer-only changes skip native compilation. Fable 5.1 at low effort confirms both
findings resolved with no adjacent regression. Own CI remains pending.
Keep the PR draft and source default active at the publication gate.

Rebased onto the producer's macOS import-resolution and versioned CasADi identity fixes.
The only conflict was the two appended plan sections, both preserved. Consumer files are
unchanged by the rebase. After checkout exposed CRLF conversion of the activation patch,
a file-scoped Git attribute preserves LF. A Windows checkout test and patch-application
check pass. All 93 consumer/provenance checks and 24 producer checks pass after the rebase,
as do type, lint and format checks.

Ordinary CI built all five Ipopt wheels and passed clean installs, then exposed a test-only
host mismatch in the new pinned-install test on Linux/macOS. The fixture now sets the selector
OS/libc and validator target consistently and exercises installation/override on all five
layouts. The mismatch reproduces locally before the fixture fix. No production code changes.
The collection baseline includes the four added cases. All 97 focused tests, type/lint/format
and full collection checks pass locally. Own CI reruns after the test-only fix.

Coordinator update: rebase onto the final producer branch and retain only this issue's plan.
The completed producer plan remains removed. The final PR #233 pin-document example has been
parsed through the actual consumer on all five fixture targets, including exact asset names
and platform tags. Its PR candidate hashes cannot become official pins. Part 2 needs a new
reviewed main producer run, trusted attestations and the maintainer-published pin document.
No parser or producer change is needed. All 97 consumer/provenance and 27 producer tests pass,
along with type, lint, format and activation-patch checks. The rebased PR's CI remains to run.
