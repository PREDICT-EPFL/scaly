# Project naming

No rename has been decided or scheduled. This note records why `alloy` worked,
why we are reconsidering it, and which alternative currently feels strongest.

## Lineage

The project began inside **anvil**, a tinygrad-based system that forged Python
numerical functions into native code. In May 2026, a separate experiment was
started around a new CasADi-like symbolic system: named functions over one
sparse typed graph, with scalar expressions, block operations, loops, calls,
derivatives, and external solvers able to coexist without becoming separate
worlds.

**Alloy** was chosen as its provisional name because it preserved the metal
lineage while expressing the new idea more precisely. An anvil is a tool;
an alloy is the material created by combining things with different properties.
The original naming note preferred it because it suggested **composability
rather than just tooling**.

The name turned out to carry more than that technical analogy. It explained the
project's upbringing without making it sound like a subordinate compiler
component, and it left room for the experiment to become an independent system.
It was concise, physical, poetic, and legible at several depths: useful on first
contact, more meaningful once the architecture and history were known. That
“if you know, you know” quality is worth preserving.

## Why reconsider it

`alloy` is already occupied on PyPI by a small unrelated package. The name is
also established elsewhere in software, most notably by the MIT Alloy modeling
language, Grafana Alloy, and the Rust Ethereum ecosystem. None of these makes
our use conceptually dishonest, and the audiences only partially overlap, but
together they make the name difficult to own, search for, and publish without a
qualified distribution name.

The PyPI owner may still voluntarily transfer the slug. Until that conversation
is resolved, renaming remains an option rather than a decision.

## What a successor must preserve

A replacement should meet a higher bar than merely being available:

- one vivid, pronounceable name rather than a technical abbreviation;
- distinctive and searchable, with a plausible package slug;
- broad enough not to bind the project to control, optimization, automatic
  differentiation, code generation, Python, or its current IR;
- a name for the whole medium or system, not for one mechanism inside it;
- modern and forceful, with enough physicality to pass the shout test;
- a natural two-letter Python alias, comparable to `import alloy as al`;
- a story connected to Anvil and Alloy without sounding like a forced sequel;
- meaning that rewards discovery instead of explaining the entire product in
  the spelling.

This rules out names built from suffixes such as `-ir`, `-core`, `-ad`,
`-control`, `-opt`, or `-codegen`. It also rules out most acronym-shaped names:
they describe today's feature list, sound like neighboring optimal-control
tools, and constrain tomorrow's project.

Earlier metal-themed candidates included `forge`, `crucible`, `foundry`,
`ferro`, `ore`, `tempera`, `lode`, and `smelt`. They preserve part of the
lineage, but most name a tool, process, or ordinary material rather than giving
the system an identity of its own.

## Leading candidate: Orichal

**Orichal**, pronounced roughly *OR-ih-kal*, is the strongest candidate found
so far.

```python
import orichal as oc
```

The name evokes **orichalcum**, the legendary metal associated with Atlantis.
It retains the material metaphor but moves from a common category—an alloy—to a
singular substance with its own identity. It can represent a symbolic language,
compiler, numerical runtime, optimization system, or something broader without
needing to change meaning as the project grows.

It also gives the lineage a satisfying progression:

```text
Anvil   — the tool
Alloy   — the compositional idea
Orichal — the singular material
```

`oc` is a clean import alias and can quietly nod toward optimal control without
putting that application in the name. The PyPI slug was unclaimed when checked
on August 12, 2026. A Hong Kong digital-assets firm uses Orichal, but no notable
scientific-computing or developer tool surfaced under the name; that overlap is
not currently considered significant.

### The oracle echo

The name's proximity to **oracle** initially looked like a possible source of
confusion, particularly with Oracle Corporation. On further consideration, the
technical meaning makes the echo an advantage. In optimization, an oracle is an
interface queried for information about a problem, such as objective or
constraint values, derivatives, or separating hyperplanes. CasADi itself uses
the `OracleFunction` concept for functions that manage and expose an
optimization problem's evaluation and derivative helpers; that architecture
directly informed Alloy's early design.

Most prospective users will already know this meaning. They can notice the
oracle resonance without being told, while the spelling and material identity
keep Orichal separate from the ordinary term. The resemblance to Oracle
Corporation is therefore not considered an important objection for this
audience. It remains a minor practical possibility that someone hearing the
name without context will transcribe it as “Oracle,” but this does not currently
outweigh the double meaning.

The resulting name has several quiet layers:

- **orichalcum**, the legendary alloy, carries the Anvil and Alloy lineage;
- **oracle** resonates with the mathematical functions the system constructs;
- **`oc`** is a natural import alias with an unobtrusive optimal-control echo.

None of these should become an advertised expansion or restrict what Orichal
can become. The oracle association is a play on words, not a claim that the
library is specifically an oracle abstraction.

### Presentation and visual identity

Orichal does not need a slogan that explains its name. Public descriptions can
state plainly what the library does. Most users may notice the oracle echo; a
smaller number may eventually learn about orichalcum and the Anvil-to-Alloy
lineage, perhaps by asking directly. That asymmetry is desirable: the name's
meaning should reward curiosity rather than arrive with a mandatory footnote.

The existing Alloy logo drafts under [`assets/logo/`](../assets/logo/) already
use copper-like colors. That palette can carry naturally into an Orichal identity
and distinguish it visually from Oracle Corporation without making the logo
literal metallurgy, Atlantis, or fantasy imagery. No logo adaptation is part of
the current naming exploration.

The remaining uncertainties are mostly matters of taste. The mythical
association could still read as fantasy or crypto if a future visual identity
leans too heavily into it, and the unfamiliar spelling may require an
introduction. Its obscurity is also part of its appeal.

### PyPI name

The `orichal` distribution name was still unclaimed on August 12, 2026. PyPI
does not provide a separate name-reservation mechanism, and an empty or
nonfunctional placeholder can be treated as name squatting under PEP 541. The
preferred way to secure the name is therefore a small but genuine pre-alpha
release containing a useful core of the project, not a “coming soon” stub.

Defining that minimum releasable core is deferred to a separate design session.
Nothing has been uploaded to PyPI as part of this exploration.

For now, the project remains **Alloy**. **Orichal is recorded as the best
candidate, not an accepted rename.**

## References and provenance

The history above was reconstructed from the following local sources:

- The initial May 15–16, 2026 design and naming session is archived at
  `~/.pi/agent/sessions/--Users-tudoroancea-dev-anvil-.worktrees-casadi-ir--/2026-05-15T13-33-26-300Z_019e2bd7-899b-75df-a646-5eb25c7859db.jsonl`.
  The May 16 exchange begins with the request to implement the new library as a
  separate package and asks what to call it. The response selects Alloy, gives
  the composability rationale, and lists the other metal-themed candidates.
- The design document produced during that session remains at
  `~/dev/anvil/.worktrees/casadi-ir/casadi-ir.md`. Its opening proposal contains
  the original “deep-learning tooling to control” / “CasADi-like tooling toward
  deep learning” reversal. Its **Follow-up notes from design discussion** section
  records Alloy as a working, explicitly non-final name.
- An earlier May 7, 2026 session in `~/.claude/history.jsonl`, session
  `a96cf66f-5635-4193-80ab-908347d8d7e7`, records the `new-ir2` experiment that
  preceded the CasADi-focused design: preserve the spirit of tinygrad's IR while
  adding functions, calls, control flow, constants, richer math, and a dedicated
  visualizer.
- The anvil Git history for `src/alloy/` and `docs/alloy/` records the experiment
  becoming an implementation on May 16 and its rapid development over the
  following days. Search by those paths rather than citing a branch-local commit
  hash, since the repository's worktree workflow rewrites such hashes on merge.
- The current [IR specification](spec.md) still describes Alloy as a provisional
  name fitting both the metal theme and mixed scalar/block lowering. The
  [README](../README.md#relationship-to-anvil) records the May 2026 extraction
  from anvil into a self-contained project.

These are archaeological references, not permanent public documentation: the
session archive and anvil worktree paths are local to the original development
machine and may eventually be moved or removed.
