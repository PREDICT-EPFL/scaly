from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from benchmarks.harness.recording import (
  CAR_COLORS,
  CarShape,
  ControlState,
  PlanarVehicleState,
  PointState3D,
  Recorder,
  RunMetadata,
  ScalarTelemetry,
  arena_scene,
  planar_car_scene,
  layout_path,
  write_result_artifacts,
)


def test_recording_schemas_are_strict_json_schemas() -> None:
  for model in (RunMetadata, ScalarTelemetry, PlanarVehicleState, PointState3D, ControlState):
    schema = model.model_json_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]
  with pytest.raises(ValidationError):
    ScalarTelemetry(step=-1, time_s=0.0, success=True, solver_time_ms=1.0)


def test_recorder_writes_custom_and_scene_channels(tmp_path: Path) -> None:
  path = tmp_path / "episode.mcap"
  with Recorder(path) as recorder:
    custom_channels = (recorder._metadata, recorder._telemetry, recorder._planar, recorder._points, recorder._control)
    schemas = {}
    for channel in custom_channels:
      schema = channel.schema()
      assert schema is not None
      schemas[schema.name] = json.loads(schema.data)
    topics = {channel.topic() for channel in (*custom_channels, recorder._scene, recorder._tf)}
    recorder.record_metadata(RunMetadata(run_id="smoke", problem="cars", backend="alloy", seed=42, dt=0.1))
    recorder.record_telemetry(ScalarTelemetry(step=0, time_s=0.0, success=True, solver_time_ms=1.2, objective=3.0))
    recorder.record_arena((0.0, 4.0, 0.0, 3.0))
    recorder.record_planar([PlanarVehicleState(step=0, time_s=0.0, vehicle_id="ego", x=1.0, y=2.0, yaw=0.2)])
    recorder.record_chain(
      [
        PointState3D(step=0, time_s=0.0, point_id="0", x=0.0, y=0.0, z=0.0),
        PointState3D(step=0, time_s=0.0, point_id="1", x=1.0, y=0.0, z=0.5),
      ]
    )
    recorder.record_control([ControlState(step=0, time_s=0.0, entity_id="ego", desired=[1.0, 0.0], applied=[0.8, 0.1])])

  data = path.read_bytes()
  assert len(data) > 8
  assert data.startswith(b"\x89MCAP0\r\n") and data.endswith(b"\x89MCAP0\r\n")
  assert topics == {"/run/metadata", "/telemetry", "/planar/state", "/chain/point", "/control", "/scene", "/tf"}
  assert {"RunMetadata", "ScalarTelemetry", "PlanarVehicleState", "PointState3D", "ControlState"} == schemas.keys()
  assert all(schema["type"] == "object" for schema in schemas.values())


def test_planar_scene_draws_body_safety_circle_and_control_arrows() -> None:
  shape = CarShape(length=1.2, width=0.9, height=0.5, center_offset=0.1, safety_radius=0.95, max_steer=2.0, arrow_length=1.9)
  scene = repr(
    planar_car_scene(
      [
        PlanarVehicleState(step=1, time_s=0.1, vehicle_id="0", x=2.0, y=3.0, yaw=0.0),
        PlanarVehicleState(step=1, time_s=0.1, vehicle_id="1", x=6.0, y=3.0, yaw=0.0),
      ],
      shape=shape,
      controls=[ControlState(step=1, time_s=0.1, entity_id="0", desired=[1.0, 0.5], applied=[0.4, -0.5])],
      trails={"0": [(0.0, 0.0), (1.0, 1.0)]},
    )
  )
  # body cube, sized from the shape and offset forward from the recorded (x, y)
  assert f"size: Some(Vector3 {{ x: {shape.length}, y: {shape.width}, z: {shape.height} }})" in scene
  assert f"position: Some(Vector3 {{ x: {2.0 + shape.center_offset}, y: 3.0, z: {0.5 * shape.height} }})" in scene
  # keep-out circle around the state position, and a trail only for the car that has one
  assert f"x: {2.0 + shape.safety_radius}, y: 3.0, z: 0.01" in scene
  assert scene.count('id: "trail/0"') == 1 and 'id: "trail/1"' not in scene
  # desired arrow in the car colour, applied arrow dark; both start at the car and are steered
  assert scene.count("ArrowPrimitive") == 2
  assert f"r: {CAR_COLORS[0][0]}" in scene and "r: 0.08, g: 0.08, b: 0.08" in scene
  assert f"r: {CAR_COLORS[1][0]}" in scene
  assert f"z: {float(np.sin(0.5 * 0.5 * shape.max_steer))}" in scene


def test_planar_scene_arrow_flips_and_shrinks_with_throttle() -> None:
  shape = CarShape(max_steer=1.0, arrow_length=2.0)
  braking = repr(
    planar_car_scene(
      [PlanarVehicleState(step=0, time_s=0.0, vehicle_id="0", x=0.0, y=0.0, yaw=0.0)],
      shape=shape,
      controls=[ControlState(step=0, time_s=0.0, entity_id="0", applied=[-1.0, 0.0])],
    )
  )
  # full reverse points backwards at full length; the arrow never collapses to nothing
  assert f"shaft_length: {0.75 * shape.arrow_length}" in braking
  assert f"z: {float(np.sin(0.5 * np.pi))}" in braking
  coasting = repr(
    planar_car_scene(
      [PlanarVehicleState(step=0, time_s=0.0, vehicle_id="0", x=0.0, y=0.0, yaw=0.0)],
      shape=shape,
      controls=[ControlState(step=0, time_s=0.0, entity_id="0", applied=[0.0, 0.0])],
    )
  )
  assert f"shaft_length: {0.75 * 0.25 * shape.arrow_length}" in coasting


def test_arena_scene_adds_the_wall_margin_inset() -> None:
  plain = repr(arena_scene((0.0, 15.0, 0.0, 12.0)))
  assert plain.count("LinePrimitive") == 1
  assert f"thickness: {0.006 * 15.0}" in plain
  inset = repr(arena_scene((0.0, 15.0, 0.0, 12.0), margin=1.0))
  assert inset.count("LinePrimitive") == 2
  assert "x: 1.0, y: 1.0, z: 0.005" in inset and "x: 14.0, y: 11.0, z: 0.005" in inset


def test_layouts_live_next_to_their_problem_and_parse_when_present() -> None:
  paths = {problem: layout_path(problem) for problem in ("chain", "tracking", "bumpercars")}
  assert paths["bumpercars"] == Path(__file__).resolve().parents[2] / "benchmarks/problems/bumpercars_filter/foxglove-layout.json"
  for problem, path in paths.items():
    assert path.parent.is_dir(), problem
    if path.is_file():
      layout = json.loads(path.read_text())
      assert isinstance(layout, dict) and layout, problem


def test_result_harvest_selects_midpoint_success_deterministically(tmp_path: Path) -> None:
  fe_inputs = [{"x": np.array([step, step + 0.5]), "p": np.array([step * 2])} for step in range(6)]
  first = write_result_artifacts(
    tmp_path / "first",
    config={"seed": 42},
    summary={"success": True},
    provenance={"commit": "abc"},
    fe_inputs=fe_inputs,
    successful_steps=[4, 0, 2],
  )
  second = write_result_artifacts(
    tmp_path / "second",
    config={"seed": 42},
    summary={"success": True},
    provenance={"commit": "abc"},
    fe_inputs=fe_inputs,
    successful_steps=[0, 2, 4],
  )

  assert json.loads(first["metadata"].read_text())["representative_step"] == 2
  with np.load(first["fe_inputs"]) as arrays:
    np.testing.assert_array_equal(arrays["x"], fe_inputs[2]["x"])
    assert arrays.files == ["p", "x"]
  assert first["fe_inputs"].read_bytes() == second["fe_inputs"].read_bytes()
  for artifact in ("config", "summary", "provenance", "metadata", "fe_inputs"):
    assert first[artifact].is_file()
