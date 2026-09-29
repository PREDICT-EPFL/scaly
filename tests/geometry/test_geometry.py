"""Geometry against SciPy's rotations (scalar-last quaternions, as here): the product composes, the
matrix and the rotation agree, exp and log are the rotation vector's, log is the same for q and -q
and through the identity; the cross product and its matrix; the manifolds' retraction and local
coordinates invert each other."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import scaly as sc
from scaly import geometry
from scaly.geometry import quaternion as quat

RNG = np.random.default_rng(9)
QA, QB = Rotation.random(random_state=1).as_quat(), Rotation.random(random_state=2).as_quat()
V = RNG.standard_normal(3)


@sc.function(4, 4, 3, output=sc.G("ab", "rotated", "matrix", "exp", "log", "log_neg"), name="geo_quaternion")
def quaternion_fns(a, b, v):
  return quat.mul(a, b), quat.rotate(a, v), quat.matrix(a), quat.exp(v), quat.log(a), quat.log(-a)


def test_quaternions_are_scipys_rotations() -> None:
  ab, rotated, mat, ex, lg, lg_neg = quaternion_fns(QA, QB, V)
  ra, rb = Rotation.from_quat(QA), Rotation.from_quat(QB)
  np.testing.assert_allclose(ra.as_matrix() @ rb.as_matrix(), Rotation.from_quat(ab).as_matrix(), atol=1e-14)
  np.testing.assert_allclose(rotated, ra.apply(V), atol=1e-14)
  np.testing.assert_allclose(mat, ra.as_matrix(), atol=1e-14)
  np.testing.assert_allclose(Rotation.from_quat(ex).as_matrix(), Rotation.from_rotvec(V).as_matrix(), atol=1e-14)
  np.testing.assert_allclose(lg, ra.as_rotvec(), atol=1e-12)
  np.testing.assert_allclose(lg_neg, lg, atol=1e-14)  # q and -q are one rotation


def test_the_identity_is_smooth() -> None:
  @sc.function(3, output=sc.G("q", "back", "jac"), name="geo_near_identity")
  def near(v):
    q = quat.exp(v)
    return q, quat.log(q), sc.jacobian(quat.log(q), v)

  q, back, jac = near(np.zeros(3))
  np.testing.assert_allclose(q, [0, 0, 0, 1], atol=1e-15)
  np.testing.assert_allclose(back, 0.0, atol=1e-15)
  assert np.isfinite(jac).all()
  np.testing.assert_allclose(jac, np.eye(3), atol=1e-7)


@sc.function(3, 3, output=sc.G("cross", "skew"), name="geo_vectors")
def vector_fns(a, b):
  return geometry.cross(a, b), geometry.skew(a) @ b


def test_the_cross_product_and_its_matrix() -> None:
  a, b = RNG.standard_normal(3), RNG.standard_normal(3)
  cross, skew = vector_fns(a, b)
  np.testing.assert_allclose(cross, np.cross(a, b), rtol=1e-15)
  np.testing.assert_allclose(skew, np.cross(a, b), rtol=1e-15)


def test_retraction_and_local_coordinates_invert_each_other() -> None:
  for name, manifold, point in (
    ("euclidean", geometry.Euclidean(3), RNG.standard_normal(3)),
    ("so3", geometry.SO3(), QA),
    ("pose3", geometry.Pose3(), np.r_[QA, RNG.standard_normal(3)]),
  ):
    delta = 0.3 * RNG.standard_normal(manifold.tangent)

    @sc.function(manifold.size, manifold.tangent, output=sc.G("moved", "back"), name=f"geo_{name}")
    def roundtrip(x, d):
      moved = manifold.retract(x, d)  # noqa: B023
      return moved, manifold.local(x, moved)  # noqa: B023

    moved, back = roundtrip(point, delta)
    np.testing.assert_allclose(back, delta, atol=1e-12, err_msg=name)
    if name == "pose3":  # the rotation on the right, the translation added
      np.testing.assert_allclose(
        Rotation.from_quat(moved[:4]).as_matrix(), Rotation.from_quat(QA).as_matrix() @ Rotation.from_rotvec(delta[:3]).as_matrix(), atol=1e-14
      )
      np.testing.assert_allclose(moved[4:], point[4:] + delta[3:], atol=1e-15)
