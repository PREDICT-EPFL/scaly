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

1. It chooses the packages, then calls `nightly.yml` on the release commit. This runs every
   toolchain, Python and benchmark entry and builds the distributions in the release run itself.
   No previous CI or Docs run is required. A Monday release rebuilds the native plugins from source
   under the same rule as the scheduled nightly.
2. A package is released when its version has no `<name>-v<version>` tag yet, so bumping a version
   is how a release is requested and unchanged packages get no new tag. A package that changed
   since its last tag without a version bump produces a warning.
3. After the full suite passes, it collects this run's artifacts and keeps the released packages'
   files. These files pass unchanged through TestPyPI and PyPI publication.
4. It publishes them to TestPyPI, then installs them from TestPyPI (everything else from PyPI) in
   clean environments and solves with every backend. A `testpypi` run stops here.
5. It publishes to PyPI, then creates each package's tag at the commit together with its GitHub
   Release, with that package's files attached.
6. When `scaly` itself is released, it publishes that commit's documentation to GitHub Pages
   through `pages.yml`. A plugin-only release leaves the site alone.

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
