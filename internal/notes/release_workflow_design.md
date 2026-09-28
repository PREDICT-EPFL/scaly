# Release workflow

`.github/workflows/release.yml` implements this. It is kept out of the published versioning policy
because it describes maintainer work.

## Building

`ci.yml` builds every release artifact on each run: the native plugin wheels with cibuildwheel (one
`py3-none-<platform>` wheel per OS and architecture), the pure wheels and all four sdists with
`uv build`. It keeps them as artifacts for 90 days, the organization's maximum. A plugin wheel is
cached under a hash of its plugin directory, so an unchanged plugin is not rebuilt. The solver
tests install those wheels rather than an editable build, so they test the binaries users get.

## Releasing

A release is started by hand on `main`, with the target `testpypi` or `pypi`. It never rebuilds:

1. It requires a successful `push` run of `ci.yml` and of `docs.yml` for the exact commit. The Docs
   run includes the Cloudflare deployment.
2. A package is released when its version has no `<name>-v<version>` tag yet, so bumping a version
   is how a release is requested and unchanged packages get no new tag. A package that changed
   since its last tag without a version bump produces a warning.
3. It downloads that CI run's artifacts and keeps the released packages' files.
4. It publishes them to TestPyPI, then installs them from TestPyPI (everything else from PyPI) in
   clean environments and solves with every backend. A `testpypi` run stops here.
5. It publishes to PyPI, then creates each package's tag at the commit together with its GitHub
   Release, with that package's files attached.

Tags are created by the release workflow only, after the PyPI upload succeeded, and are never moved
or reused. Every step skips what an earlier attempt already did, so a partly failed release is
finished by running it again.

A tag or release created with the workflow token starts no other workflow, so anything that must
follow a release, such as a tagged documentation deployment, has to run inside `release.yml`.
