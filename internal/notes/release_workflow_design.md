# Release workflow

`.github/workflows/release.yml` implements this. It is kept out of the published versioning policy
because it describes maintainer work.

The four packages first shipped as `0.1.0a1` on 2026-09-29 through this workflow, tagged
`<package>-v0.1.0a1`. The next release's execution is tracked in
[#85]; the compiler roadmap records its agreed scope.

## Building

`ci-build.yml` builds each platform's native plugin wheels with cibuildwheel and the pure wheels
and all four sdists with `uv build`. CI and the nightly share this workflow and retain its artifacts
for 90 days, the organization's maximum. The native library cache hashes the build hooks, pins and
licenses, together with the platform. Wrapper edits and version bumps reuse the native libraries.
The tests install the plugin wheels rather than editable builds.

The nightly calls the same build, then runs independent toolchain, Python and benchmark groups.
Its Python versions come from the package classifiers. Compile caches and Google Benchmark builds
start cold, and the baseline platform entries save caches that CI can restore. Native builds
restore their cache except on Mondays or a manual `rebuild-native` run, which builds from source.
The nightly saves native libraries and Google Benchmark on main so pull requests can reuse them.

## Releasing

A release is started by hand on `main`, with the target `testpypi` or `pypi`:

1. It chooses packages whose versions have no `<name>-v<version>` tag. A changed package with
   an existing version tag produces a warning and is left out.
2. It looks for completed validation on main for the exact release commit. Completed TestPyPI
   verification takes priority, followed by full nightly validation. It reuses the recorded files
   and skips the checks they already passed. With no usable record, it builds and runs the full
   nightly suite in the release run. A fresh run on Monday also rebuilds the native plugins.
3. A full nightly records its commit, run attempt and immutable artifact IDs only after every
   toolchain, Python and benchmark entry passes. An unchanged nightly that skips the suite writes
   no record. Missing or expired artifacts disqualify a record.
4. It collects the selected packages' files without rebuilding. Unless those files already passed
   TestPyPI verification for this commit, it uploads them to TestPyPI, installs them in clean
   environments and solves with every backend. It records the verified artifact IDs and package
   versions after every installation check passes and TestPyPI's file hashes match the selected
   files. A later publication failure preserves this completed verification for the next attempt.
   A `testpypi` run stops here.
5. A `pypi` run automatically publishes those same files to PyPI, then creates each package's tag
   at the commit together with its GitHub Release and attached files. There is no approval pause.
6. When `scaly` itself is released, it publishes that commit's documentation through `pages.yml`.
   A plugin-only release leaves the site alone.

A release after the overnight nightly therefore skips the full build and suite. A `pypi` dispatch
after a successful `testpypi` dispatch also skips the TestPyPI upload and installation checks.
The run summary links to the earlier validation it reused. Selection reads only this repository's
main runs of `nightly.yml` and `release.yml`; it checks the commit, run attempt, artifact IDs and,
for TestPyPI, the package versions covered by the record. Failed standalone nightlies do not
qualify. A failed release can supply a record from a validation stage it completed successfully.

Tags are created by the release workflow only, after the PyPI upload succeeded, and are never moved
or reused. Every step skips what an earlier attempt already did, so a partly failed release is
finished by running it again.

A tag or release created with the workflow token starts no other workflow, so anything that must
follow a release has to run inside `release.yml`, which is why it calls `pages.yml` directly.

## Documentation

GitHub Pages holds the docs of the latest scaly release, built without a banner. Cloudflare Pages
holds one deployment per branch from `docs.yml`, each with a banner naming its branch and commit.
The `github-pages` environment only accepts `main`, so `pages.yml` always runs on `main` and checks
out the commit or tag it publishes. Run it by hand with a tag to republish an earlier release.

[#85]: https://github.com/PREDICT-EPFL/scaly/issues/85
