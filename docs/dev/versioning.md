# Versioning and releases

This document records the initial versioning and release policy for the core
package and its official solver plugins.

## Initial versions

The first paper-worthy release will be `0.1.0`. If installable wheels are useful
before then, they will be published as Python pre-releases of that version:
`0.1.0a1`, `0.1.0a2`, and so on. A release candidate such as `0.1.0rc1` may be
used when the paper release is being frozen. We will not use a separate
`0.0.x` series merely to signal that the project is young.

The project will remain in the `0.x.y` range while its public interfaces are
still expected to change substantially. We will release `1.0.0` when we are
ready to maintain the documented public API through explicit deprecations
rather than breaking it without a migration period. Feature completeness by
itself is not the criterion for `1.0.0`.

Within the pre-1.0 series, patch releases stay within a compatible release
line. A new minor version may open a new compatibility line and contain
breaking changes.

## Core and plugin versions

The core package and each solver plugin are versioned independently. A release
of one package does not require publishing unchanged packages merely to keep
their version numbers aligned. Their initial versions may happen to be equal,
and a coordinated compatibility change may update several versions in the same
commit, but equality has no compatibility meaning.

Compatibility is enforced in two places:

- Each plugin declares the supported core-package range in its package
  dependencies, for example `core>=0.1.0,<0.2` for the `0.1.x` compatibility
  line.
- The solver registry checks the plugin protocol version at runtime. A breaking
  change to the plugin contract bumps that protocol version and requires
  coordinated releases of the affected official plugins.

## Git tags

Every published package version has exactly one package-qualified Git tag, for
example:

```text
alloy-v0.1.0
alloy-piqp-v0.1.0
alloy-ipopt-v0.1.0
```

There is no additional repository-wide release tag. Several package tags may
point to the same commit when that commit releases several packages, and
packages that are not released receive no new tag.

## Release process

Releases are manually started in continuous integration rather than triggered
by pushing a tag. The operator selects the package or packages to release, and
the versions in their package metadata are the source of truth. The workflow
then:

1. records the exact source commit and validates the selected package versions;
2. builds and tests every source distribution and wheel;
3. publishes the artifacts to PyPI;
4. creates one package tag per released version at the recorded commit; and
5. creates the corresponding GitHub Releases and attaches their artifacts.

A GitHub Release must refer to a tag, so continuous integration creates each
tag immediately before its corresponding GitHub Release, after the PyPI
publication has succeeded. Tags are created by the release workflow rather
than manually and are not moved or reused after publication.
