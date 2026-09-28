# Planned release workflow

This design is not implemented. It was moved out of the published versioning policy because it
describes future maintainer work.

## Release process

Releases start manually in continuous integration; pushing a tag does not trigger one. The workflow
does not exist yet, and this section records its design so the first one is built to it. The
operator selects the package or packages to release, and the versions in their package metadata are
the source of truth. The workflow then:

1. records the exact source commit and validates the selected package versions;
2. builds and tests every source distribution and wheel;
3. publishes the artifacts to PyPI;
4. creates one package tag per released version at the recorded commit; and
5. creates the corresponding GitHub Releases and attaches their artifacts.

A GitHub Release must refer to a tag, so the workflow creates each tag immediately before its GitHub
Release, after the PyPI publication has succeeded. Tags are created by the release workflow only and
are not moved or reused after publication.
