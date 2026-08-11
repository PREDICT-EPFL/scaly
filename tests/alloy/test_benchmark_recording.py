from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from benchmarks.harness.recording import (
  ACCEL_COLOR,
  BRAKE_COLOR,
  CAR_COLORS,
  CHAIN_COLORS,
  CHAIN_REFERENCE_COLORS,
  CONE_STYLES,
  HORIZON_STYLES,
  CarShape,
  ChainPlan,
  ControlState,
  HorizonPath,
  PlanarVehicleState,
  PointState3D,
  Recorder,
  RunMetadata,
  ScalarTelemetry,
  arena_scene,
  car_frame,
  chain_plan_scene,
  chain_reference_scene,
  chain_scene,
  horizon_scene,
  planar_car_scene,
  layout_path,
  track_scene,
  write_result_artifacts,
)


def test_recording_schemas_are_strict_json_schemas() -> None:
  for model in (RunMetadata, ScalarTelemetry, PlanarVehicleState, PointState3D, ControlState, HorizonPath, ChainPlan):
    schema = model.model_json_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]
  with pytest.raises(ValidationError):
    ScalarTelemetry(step=-1, time_s=0.0, success=True, solver_time_ms=1.0)


def test_recorder_writes_custom_and_scene_channels(tmp_path: Path) -> None:
  path = tmp_path / "episode.mcap"
  with Recorder(path) as recorder:
    custom_channels = (
      recorder._metadata,
      recorder._telemetry,
      recorder._planar,
      recorder._points,
      recorder._control,
      recorder._horizon,
      recorder._plan,
    )
    schemas = {}
    for channel in custom_channels:
      schema = channel.schema()
      assert schema is not None
      schemas[schema.name] = json.loads(schema.data)
    scene_channels = (recorder._scene, recorder._static_scene, recorder._horizon_scene)
    topics = {channel.topic() for channel in (*custom_channels, *scene_channels, recorder._tf)}
    recorder.record_metadata(RunMetadata(run_id="smoke", problem="cars", backend="alloy", seed=42, dt=0.1))
    recorder.record_telemetry(ScalarTelemetry(step=0, time_s=0.0, success=True, solver_time_ms=1.2, objective=3.0))
    recorder.record_arena((0.0, 4.0, 0.0, 3.0))
    recorder.record_planar([PlanarVehicleState(step=0, time_s=0.0, vehicle_id="ego", x=1.0, y=2.0, yaw=0.2)])
    recorder.record_chain_references({"end-mass reference": (0.75, 0.0, 0.0)})
    recorder.record_chain(
      [
        PointState3D(step=0, time_s=0.0, point_id="0", x=0.0, y=0.0, z=0.0),
        PointState3D(step=0, time_s=0.0, point_id="1", x=1.0, y=0.0, z=0.5),
      ],
      control=ControlState(step=0, time_s=0.0, entity_id="tip", applied=[0.1, 0.2, -0.3]),
    )
    recorder.record_control([ControlState(step=0, time_s=0.0, entity_id="ego", desired=[1.0, 0.0], applied=[0.8, 0.1])])
    recorder.record_track(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), {"blue": np.array([[0.0, 1.0]]), "small_orange": np.zeros((0, 2))})
    recorder.record_horizons([HorizonPath(step=0, time_s=0.0, path_id="prediction", x=[0.0, 1.0], y=[0.0, 0.5])])
    recorder.record_chain_plan(ChainPlan(step=0, time_s=0.0, n_masses=2, nodes=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 0.0, 1.1, 0.0, 0.4]]))

  data = path.read_bytes()
  assert len(data) > 8
  assert data.startswith(b"\x89MCAP0\r\n") and data.endswith(b"\x89MCAP0\r\n")
  assert topics == {
    "/run/metadata",
    "/telemetry",
    "/planar/state",
    "/chain/point",
    "/chain/plan",
    "/control",
    "/horizon",
    "/scene",
    "/scene/static",
    "/scene/horizon",
    "/tf",
  }
  assert {
    "RunMetadata",
    "ScalarTelemetry",
    "PlanarVehicleState",
    "PointState3D",
    "ControlState",
    "HorizonPath",
    "ChainPlan",
  } == schemas.keys()
  assert all(schema["type"] == "object" for schema in schemas.values())


def test_static_geometry_gets_its_own_scene_topic_and_is_logged_once(tmp_path: Path) -> None:
  """Seeking hands a panel only the newest message per topic, so track and arena geometry must be
  alone on theirs — otherwise a jump ahead replaces them with a per-step update and they vanish."""

  class SpyChannel:
    def __init__(self) -> None:
      self.log_times: list[int] = []

    def log(self, _scene, *, log_time: int) -> None:
      self.log_times.append(log_time)

    def close(self) -> None:
      pass

  with Recorder(tmp_path / "episode.mcap") as recorder:
    static, dynamic, horizon = SpyChannel(), SpyChannel(), SpyChannel()
    recorder._static_scene, recorder._scene, recorder._horizon_scene = static, dynamic, horizon  # ty: ignore[invalid-assignment]
    recorder.record_arena((0.0, 4.0, 0.0, 3.0))
    recorder.record_track(np.array([[0.0, 0.0], [1.0, 0.0]]), {"blue": np.array([[0.0, 1.0]])})
    for step in range(3):
      recorder.record_planar([PlanarVehicleState(step=step, time_s=0.1 * step, vehicle_id="ego", x=float(step), y=0.0, yaw=0.0)])
      recorder.record_horizons([HorizonPath(step=step, time_s=0.1 * step, path_id="prediction", x=[0.0, 1.0], y=[0.0, 0.5])])

  # the two static entities are sent once each, at time zero, and never re-sent per step
  assert static.log_times == [0, 0]
  assert dynamic.log_times == [0, 100_000_000, 200_000_000]
  assert horizon.log_times == dynamic.log_times


def test_chain_reference_survives_seeking_while_the_chain_and_plan_update_per_step(tmp_path: Path) -> None:
  """Same seeking rule as the track: the end-mass reference is sent once, so it has to be on the
  static topic or a jump ahead would replace it with a per-step chain update and lose it."""

  class SpyChannel:
    def __init__(self) -> None:
      self.log_times: list[int] = []

    def log(self, _scene, *, log_time: int) -> None:
      self.log_times.append(log_time)

    def close(self) -> None:
      pass

  with Recorder(tmp_path / "episode.mcap") as recorder:
    static, dynamic, horizon = SpyChannel(), SpyChannel(), SpyChannel()
    recorder._static_scene, recorder._scene, recorder._horizon_scene = static, dynamic, horizon  # ty: ignore[invalid-assignment]
    recorder.record_chain_references({"end-mass reference": (0.75, 0.0, 0.0)})
    for step in range(3):
      recorder.record_chain(_chain((0.0, 0.0, 0.0), (float(step), 0.0, 0.0)))
      recorder.record_chain_plan(ChainPlan(step=step, time_s=0.2 * step, n_masses=2, nodes=[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))

  assert static.log_times == [0]
  assert dynamic.log_times == [400_000_000] * 3  # `_chain` stamps one time; what matters is the topic
  assert horizon.log_times == [0, 200_000_000, 400_000_000]


def test_car_bodies_ride_their_own_frame_while_trails_stay_in_the_scene() -> None:
  """The camera can only follow a frame, so the car's pose has to live in a transform rather than
  be baked into its primitives. The trail is a world-space path and must not move with the car."""
  scene = planar_car_scene(
    [
      PlanarVehicleState(step=1, time_s=0.1, vehicle_id="ego", x=2.0, y=3.0, yaw=0.7),
      PlanarVehicleState(step=1, time_s=0.1, vehicle_id="other", x=6.0, y=3.0, yaw=0.0),
    ],
    shape=CarShape(length=1.2, width=0.9, height=0.5),
    trails={"ego": [(0.0, 0.0), (1.0, 1.0)]},
  )
  text = repr(scene)
  frames = re.findall(r'frame_id: "([^"]+)", id: "([^"]+)", lifetime: [^,]+, frame_locked: (\w+)', text)
  assert frames == [(car_frame("ego"), "car/ego", "true"), ("scene", "trail/ego", "false"), (car_frame("other"), "car/other", "true")]
  # nothing on the car body carries the absolute pose any more — that is the transform's job
  for absolute in ("x: 2.0", "x: 6.0", f"z: {float(np.sin(0.5 * 0.7))}"):
    assert absolute not in text, absolute


def test_recorder_publishes_a_car_transform_with_every_planar_scene(tmp_path: Path) -> None:
  class SpyChannel:
    def __init__(self) -> None:
      self.messages: list[tuple[int, object]] = []

    def log(self, message, *, log_time: int) -> None:
      self.messages.append((log_time, message))

    def close(self) -> None:
      pass

  with Recorder(tmp_path / "episode.mcap") as recorder:
    tf, scene = SpyChannel(), SpyChannel()
    recorder._tf, recorder._scene = tf, scene  # ty: ignore[invalid-assignment]
    recorder.record_planar(
      [
        PlanarVehicleState(step=2, time_s=0.2, vehicle_id="ego", x=1.5, y=-2.5, yaw=np.pi / 2),
        PlanarVehicleState(step=2, time_s=0.2, vehicle_id="other", x=0.0, y=0.0, yaw=0.0),
      ]
    )

  # one transform message, at the same log time as the scene it places
  assert len(tf.messages) == 1 and [time for time, _ in tf.messages] == [time for time, _ in scene.messages]
  transforms = repr(tf.messages[0][1])
  assert transforms.count("FrameTransform {") == 2
  assert f'parent_frame_id: "scene", child_frame_id: "{car_frame("other")}"' in transforms
  assert (
    f'parent_frame_id: "scene", child_frame_id: "{car_frame("ego")}", '
    "translation: Some(Vector3 { x: 1.5, y: -2.5, z: 0.0 }), "
    f"rotation: Some(Quaternion {{ x: 0.0, y: 0.0, z: {float(np.sin(np.pi / 4))}, w: {float(np.cos(np.pi / 4))} }})"
  ) in transforms


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
  # body cube, sized from the shape and offset forward from the car frame's origin
  assert f"size: Some(Vector3 {{ x: {shape.length}, y: {shape.width}, z: {shape.height} }})" in scene
  assert f"position: Some(Vector3 {{ x: {shape.center_offset}, y: 0.0, z: {0.5 * shape.height} }})" in scene
  # keep-out circle around the car frame's origin, and a trail only for the car that has one
  assert f"x: {shape.safety_radius}, y: 0.0, z: 0.01" in scene
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


def test_planar_scene_draws_steered_wheels_and_shades_the_body_by_drive_force() -> None:
  shape = CarShape(length=2.8, width=1.5, height=0.55, max_throttle=500.0, max_steer=0.5, wheelbase=1.5706)
  car = PlanarVehicleState(step=0, time_s=0.0, vehicle_id="0", x=0.0, y=0.0, yaw=0.0)

  def scene(applied: list[float]) -> str:
    return repr(planar_car_scene([car], shape=shape, controls=[ControlState(step=0, time_s=0.0, entity_id="0", applied=applied)]))

  turning = scene([250.0, 0.4])
  # no arrow at all, and a wheel pair on the front axle turned by the raw steering angle
  assert "ArrowPrimitive" not in turning
  assert turning.count("CubePrimitive") == 3
  assert f"z: {float(np.sin(0.5 * 0.4))}" in turning
  half_track = 0.5 * (shape.width - 0.12 * shape.width)
  for offset in (-half_track, half_track):
    assert f"x: {0.5 * shape.wheelbase}, y: {offset}, z: {0.5 * 0.30 * shape.height}" in turning
  # steering is an angle in radians here, so it saturates at max_steer rather than being scaled by it
  assert f"z: {float(np.sin(0.5 * shape.max_steer))}" in scene([0.0, 5.0])
  # the body shades from its own colour toward green under drive and red under braking
  blue = CAR_COLORS[0]
  assert f"r: {blue[0]}, g: {blue[1]}, b: {blue[2]}, a: 0.55" in scene([0.0, 0.0])
  assert f"r: {ACCEL_COLOR[0]}, g: {ACCEL_COLOR[1]}, b: {ACCEL_COLOR[2]}, a: 0.55" in scene([500.0, 0.0])
  assert f"r: {BRAKE_COLOR[0]}, g: {BRAKE_COLOR[1]}, b: {BRAKE_COLOR[2]}, a: 0.55" in scene([-500.0, 0.0])


def test_planar_scene_normalizes_a_physical_throttle() -> None:
  shape = CarShape(max_throttle=500.0, max_steer=0.5, arrow_length=2.0)
  scene = repr(
    planar_car_scene(
      [PlanarVehicleState(step=0, time_s=0.0, vehicle_id="0", x=0.0, y=0.0, yaw=0.0)],
      shape=shape,
      controls=[ControlState(step=0, time_s=0.0, entity_id="0", applied=[500.0, 0.0])],
    )
  )
  # full physical throttle reads as full scale rather than saturating at the raw clip of 1.0
  assert f"shaft_length: {0.75 * shape.arrow_length}" in scene


def _chain(*positions: tuple[float, float, float]) -> list[PointState3D]:
  return [PointState3D(step=2, time_s=0.4, point_id=str(i), x=x, y=y, z=z) for i, (x, y, z) in enumerate(positions)]


def test_chain_scene_marks_the_anchor_and_end_mass_and_draws_the_control_vector() -> None:
  radius, scale = 0.1, 2.0
  scene = repr(
    chain_scene(
      _chain((0.0, 0.0, 0.0), (1.0, 0.0, -0.2), (2.0, 0.0, -0.3)),
      radius=radius,
      control=ControlState(step=2, time_s=0.4, entity_id="tip", applied=[0.0, 1.0, 0.0]),
      control_scale=scale,
      trail=[(3.0, 0.0, 0.0), (2.5, 0.0, -0.1)],
    )
  )
  # anchor, spring-driven mass and end mass each in their own colour, the end mass drawn larger
  for role in ("anchor", "mass", "end", "link"):
    assert f"r: {CHAIN_COLORS[role][0]}" in scene, role
  assert scene.count("SpherePrimitive") == 3
  assert f"x: {2 * radius}, y: {2 * radius}, z: {2 * radius}" in scene
  assert f"x: {2 * radius * 1.4}, y: {2 * radius * 1.4}, z: {2 * radius * 1.4}" in scene
  # one arrow, starting on the end mass, `control_scale` metres long per unit command
  assert scene.count("ArrowPrimitive") == 1
  assert "position: Some(Vector3 { x: 2.0, y: 0.0, z: -0.3 })" in scene
  assert f"shaft_length: {0.75 * scale}" in scene and f"head_length: {0.25 * scale}" in scene
  # the end-mass trail is a second entity, so it survives the chain entity being replaced
  assert scene.count('id: "chain/trail"') == 1 and scene.count("LinePrimitive") == 2


def test_chain_scene_points_the_control_arrow_along_the_command() -> None:
  for command in ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.4, -0.5, 0.8]):
    scene = repr(
      chain_scene(
        _chain((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        control=ControlState(step=2, time_s=0.4, entity_id="tip", applied=command),
      )
    )
    # the arrow's own axis is +x, so its pose must rotate +x onto the command direction
    x, y, z, w = (float(value) for value in re.findall(r"Quaternion \{ x: (\S+), y: (\S+), z: (\S+), w: (\S+) \}", scene)[0])
    rotated = np.array([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)])
    np.testing.assert_allclose(rotated, np.array(command) / np.linalg.norm(command), atol=1e-12)


def test_chain_scene_omits_the_arrow_and_trail_when_there_is_nothing_to_draw() -> None:
  points = _chain((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
  assert "ArrowPrimitive" not in repr(chain_scene(points))
  assert "ArrowPrimitive" not in repr(chain_scene(points, control=ControlState(step=2, time_s=0.4, entity_id="tip", applied=[0.0, 0.0, 0.0])))
  assert 'id: "chain/trail"' not in repr(chain_scene(points, trail=[(0.0, 0.0, 0.0)]))
  assert "entities=[]" in repr(chain_scene([])) and "entities=[]" in repr(chain_reference_scene({}))


def test_chain_reference_scene_labels_one_marker_per_target() -> None:
  radius = 0.1
  scene = repr(chain_reference_scene({"end-mass reference": (0.75, 0.0, 0.0), "second": (2.0, 0.0, 0.0)}, radius=radius))
  assert scene.count("SpherePrimitive") == 2 and scene.count("TextPrimitive") == 2
  assert 'id: "chain/reference/end-mass reference"' in scene and 'id: "chain/reference/second"' in scene
  # each marker is translucent, keeps its own colour in hand-over order, and is labelled with its position
  for index, name in enumerate(("end-mass reference", "second")):
    rgb = CHAIN_REFERENCE_COLORS[index]
    assert f"r: {rgb[0]}, g: {rgb[1]}, b: {rgb[2]}, a: 0.4" in scene, name
  assert 'text: "end-mass reference (x=0.75)"' in scene and 'text: "second (x=2.00)"' in scene
  assert f"x: 0.75, y: 0.0, z: {3 * radius}" in scene


def test_chain_plan_scene_draws_every_predicted_shape_and_the_end_mass_path() -> None:
  thickness = 0.05
  nodes = [[0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 2.0, 0.0, -0.2], [0.0, 0.0, 0.0, 0.9, 0.0, -0.2, 1.8, 0.0, -0.4]]
  scene = repr(chain_plan_scene(ChainPlan(step=4, time_s=0.8, n_masses=3, nodes=nodes), thickness=thickness))
  # one faint strip per horizon node, plus one brighter strip through the nodes' end masses
  assert scene.count("LinePrimitive") == len(nodes) + 1
  assert scene.count('id: "chain/plan"') == 1
  assert f"thickness: {thickness}" in scene and f"thickness: {2.0 * thickness}" in scene
  assert "a: 0.25" in scene and "a: 0.9" in scene
  assert "x: 2.0, y: 0.0, z: -0.2 }, Point3 { x: 1.8, y: 0.0, z: -0.4 }" in scene
  assert "entities=[]" in repr(chain_plan_scene(ChainPlan(step=0, time_s=0.0, n_masses=3, nodes=[])))


def test_track_scene_draws_the_center_loop_and_one_cube_per_cone() -> None:
  center_line = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 8.0], [0.0, 8.0]])
  cones = {"blue": np.array([[1.0, 2.0], [2.0, 3.0]]), "big_orange": np.array([[0.0, 0.0]]), "small_orange": np.zeros((0, 2))}
  scene = repr(track_scene(center_line, cones))
  assert "LineLoop" in scene and scene.count("LinePrimitive") == 1
  assert "x: 10.0, y: 8.0, z: 0.01" in scene
  # one entity per non-empty colour, one cube per cone, each sitting on the ground
  assert 'id: "track/cones/blue"' in scene and 'id: "track/cones/big_orange"' in scene
  assert "small_orange" not in scene
  assert scene.count("CubePrimitive") == 3
  blue_size = CONE_STYLES["blue"][1]
  assert f"position: Some(Vector3 {{ x: 1.0, y: 2.0, z: {0.5 * blue_size[2]} }})" in scene
  assert f"r: {CONE_STYLES['big_orange'][0][0]}" in scene


def test_horizon_scene_separates_the_reference_and_prediction_strips() -> None:
  scene = repr(
    horizon_scene(
      [
        HorizonPath(step=3, time_s=0.15, path_id="reference", x=[0.0, 1.0], y=[0.0, 0.0]),
        HorizonPath(step=3, time_s=0.15, path_id="prediction", x=[0.0, 1.0], y=[0.1, 0.2]),
      ]
    )
  )
  assert scene.count("LineStrip") == 2
  assert 'id: "horizon/reference"' in scene and 'id: "horizon/prediction"' in scene
  # each strip gets its own colour and height, so they neither z-fight nor blend
  for path_id, (rgb, z) in HORIZON_STYLES.items():
    assert f"r: {rgb[0]}" in scene, path_id
    assert f"z: {z}" in scene, path_id


def test_arena_scene_adds_the_wall_margin_inset() -> None:
  plain = repr(arena_scene((0.0, 15.0, 0.0, 12.0)))
  assert plain.count("LinePrimitive") == 1
  assert f"thickness: {0.006 * 15.0}" in plain
  inset = repr(arena_scene((0.0, 15.0, 0.0, 12.0), margin=1.0))
  assert inset.count("LinePrimitive") == 2
  assert "x: 1.0, y: 1.0, z: 0.005" in inset and "x: 14.0, y: 11.0, z: 0.005" in inset


def test_layouts_live_next_to_their_problem_and_parse_when_present() -> None:
  paths = {problem: layout_path(problem) for problem in ("chain", "race_cars", "bumpercars")}
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
