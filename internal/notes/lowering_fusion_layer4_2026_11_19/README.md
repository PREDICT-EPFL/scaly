# Lowering-time fusion, layer 4

The base is main after the typed workspace change in #503. The comparison uses the
same benchmark builders and standalone assembly fixture on both revisions. The
snapshot generator rewrote the full 21-file corpus and every regenerated file
matched main byte for byte. The audit records both digests and the empty diff for
each file.

## Seed layout and storage

In the folded matrix-vector derivative, the sparse product used to read a
transpose copy of the seed tensor, store the product in row-major order and then
transpose it back. The seed tensor keeps shape `(seeds, input)`, and the
derivative keeps shape `(seeds, output)`. The reduction now reads
`seeds[seed * input_width + column]` and stores straight into
`out[seed * output_width + row]`. The tensor derivative also covers the folded
reshape around a matrix-valued tangent. Both match NumPy, have no private array
in the lowered procedure, and have no division or remainder in the final
indices.

The neural model predictive control (MPC) Jacobian on #166 had six
1,024-double buffers across the two hidden layers in automatic mode. At
horizon 100 the caller supplied 7,744 doubles and the root declared another
3,512. Layers 1 to 3 and the code that landed between #166 and this layer
already brought caller workspace to zero. This layer fuses pointwise maps
when they traverse the same active seed pairs, then puts the activation
derivative in the product's final store. Adjacent products with the same
active seed planes use the same layout for their intermediate. The two hidden
layers now store three 32-wide active planes per mapped stage, so automatic
mode reserves two 768-double buffers in place of two 1,024-double buffers.

The probes record identical mapped lane capacities on both revisions for
each horizon and lane request. The root declaration counts exclude constants,
callee frames, lane-helper scratch and compiler-managed registers. Caller workspace includes the typed
workspace capacity of the root and its callees. Both are reported separately.
The horizon 6 case with four requested lanes already uses a scalar call for
the six trips and keeps no full hidden-layer array in the root. #71 owns
reuse inside nested loops and packing of the remaining live temporaries.
The horizon-sized arrays hold mapped results that later assembly loops read.

## Karush-Kuhn-Tucker assembly

The fixture ports the sparse interior-point method's `Kernels.kkt_matrix`
from `origin/devrush:src/scaly/opt/ipm/kkt.py`, *PIQP's KKT system as
generated code*, as exercised by the T3-3 and T3-4 drivers. It supplies
sparse objective and constraint values, a missing objective diagonal entry,
and equality and inequality regularizers. It returns the upper-triangle
values in compressed sparse column order. It uses main's Function,
concatenation, gather and selection operations. Scaling, factorization,
permutation for factorization, and solving are outside the fixture.

On main, six loops write the assembled block vector and one loop gathers the
block values and adds the diagonal. The lowered procedure now has one loop
and zero private arrays. Each output entry reads its source block through a
bounded access map and writes to the declared output. NumPy independently
assembles the blocks and extracts the upper triangle.

## Reproduction

Run the commands from the repository root. The compiler is Clang 20.1.8
on la015. Each generation and compilation cell is one observation. Generation
includes building the derivative, lowering, rendering and writing the source
and header. Compilation stops at an object file. No generated kernel is timed.

```sh
notes=internal/notes/lowering_fusion_layer4_2026_11_19
scratch=$(mktemp -d)
mkdir "$scratch/main"
git archive 7986e2ccda9962ef6ebcc13da688890a88bf24af src | tar -x -C "$scratch/main"
export PYTHONHASHSEED=0 SCALY_VECTOR_LIBM=glibc
PYTHONPATH="$scratch/main/src:." uv run "$notes/measure.py" "$scratch/before"
PYTHONPATH="src:." uv run "$notes/measure.py" "$scratch/after"
uv run python -m tests.test_c_snapshot
uv run "$notes/audit_snapshots.py" 7986e2ccda9962ef6ebcc13da688890a88bf24af
uv run "$notes/prove.py" "$scratch/main/src" "$scratch/perturbed"
```

The perturbation script imports a separate source copy and changes one rule
at a time. Main fails all five cases after matching NumPy. Restoring either
transpose copy fails both tensor cases. Disabling seed traversal agreement
brings back a pointwise buffer. Storing the product before the activation or
allocating every logical seed plane fails the two-product storage assertions.
Restoring the KKT block concatenation gives seven loops again.

The first full run caught a reduction in the dense scratch fixture. The broader
rule removed the transpose copy and reduced its dispatch scratch from 21 to 9
doubles at one lane, and from 42 to 18 at two lanes. Both versions match NumPy.
Reading the right transpose in place also changes reduction loads to unit
stride. The dense scheduler switches from a reduction-outer loop with memory
accumulators to a reduction-inner loop with scalar accumulators. The mixed
sparse path keeps its stored-column traversal. Dense scheduling belongs with
the stride-aware loop order work in [#71](https://github.com/PREDICT-EPFL/scaly/issues/71#issuecomment-7011173432).

The chain M=5 source differs only in generated `it_tNNN` names. Its size and
caller workspace are equal on both revisions.
