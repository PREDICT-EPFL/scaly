# Merge the pytest node-ID baseline by union

The maintainer requested only a root `.gitattributes` entry and one sentence beside the
regeneration recipe in `docs/dev/contributing.md`. Follow the compiler-run plan and
`~/.scaly-suite/RULES.md`. Base this change on main and use the pull request as its task record.

- [ ] In a scratch clone, reproduce the conflict without the attribute, then prove merge and
      rebase keep both adjacent additions once with the attribute on the base.
- [ ] Prove the existing node-ID check rejects a deletion/addition union result and a missing ID.
- [ ] Add the requested two lines, remove this temporary plan, and pass the detached full
      pre-merge through the suite runner on la015.
- [ ] Push once, open the pull request, and get one low-effort correctness review by another model.
- [ ] Queue after the review verdict and green checks, then verify GitHub's mergeability behavior
      after this pull request merges. Close any throwaway drafts and report all four criteria.

Evidence is kept under `/tmp/scaly-nodeid-union.Ee5yI2` and in the pull request description.
GitHub verification depends on this change merging. No issue, runtime code, or baseline change
belongs in the delivered diff.
