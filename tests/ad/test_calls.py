"""Call and map derivative rules have one owner shared by both AD modes."""

from scaly.ad import forward, reverse


def test_call_rules_live_in_shared_module() -> None:
  from scaly.ad import calls

  assert forward._call_jvp is calls._call_jvp
  assert forward._call_jvp_many is calls._call_jvp_many
  assert forward._vmap_jvp_many is calls._vmap_jvp_many
  assert reverse._call_vjp is calls._call_vjp
  assert reverse._vmap_vjp is calls._vmap_vjp
  assert calls.body_tangents.__module__ == "scaly.ad.calls"
  assert calls.body_cotangents.__module__ == "scaly.ad.calls"
