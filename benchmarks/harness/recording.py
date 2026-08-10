from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from foxglove import Channel, Context, open_mcap
from foxglove.channels import FrameTransformsChannel, SceneUpdateChannel
from foxglove.messages import (
  ArrowPrimitive,
  Color,
  CubePrimitive,
  FrameTransform,
  FrameTransforms,
  LinePrimitive,
  LinePrimitiveLineType,
  Point3,
  Pose,
  Quaternion,
  SceneEntity,
  SceneUpdate,
  SpherePrimitive,
  Timestamp,
  Vector3,
)
from pydantic import BaseModel, ConfigDict, Field


class _Schema(BaseModel):
  model_config = ConfigDict(extra="forbid")


class RunMetadata(_Schema):
  run_id: str
  problem: str
  backend: str
  seed: int
  dt: float = Field(gt=0.0)
  config: dict[str, Any] = Field(default_factory=dict)


class ScalarTelemetry(_Schema):
  step: int = Field(ge=0)
  time_s: float = Field(ge=0.0)
  success: bool
  solver_time_ms: float = Field(ge=0.0)
  fe_time_ms: float = Field(default=0.0, ge=0.0)
  objective: float | None = None
  constraint_margin: float | None = None
  scalars: dict[str, float] = Field(default_factory=dict)


class PlanarVehicleState(_Schema):
  step: int = Field(ge=0)
  time_s: float = Field(ge=0.0)
  vehicle_id: str
  x: float
  y: float
  yaw: float


class PointState3D(_Schema):
  step: int = Field(ge=0)
  time_s: float = Field(ge=0.0)
  point_id: str
  x: float
  y: float
  z: float


class ControlState(_Schema):
  step: int = Field(ge=0)
  time_s: float = Field(ge=0.0)
  entity_id: str
  applied: list[float]
  desired: list[float] = Field(default_factory=list)


class HorizonPath(_Schema):
  """A planned or predicted path over one control horizon, in scene coordinates."""

  step: int = Field(ge=0)
  time_s: float = Field(ge=0.0)
  path_id: str
  x: list[float]
  y: list[float]
  yaw: list[float] = Field(default_factory=list)


CAR_COLORS = (
  (0.12, 0.47, 0.87),
  (0.89, 0.24, 0.20),
  (0.18, 0.63, 0.31),
  (0.76, 0.22, 0.78),
  (0.85, 0.73, 0.15),
  (0.20, 0.74, 0.78),
  (0.95, 0.55, 0.15),
  (0.55, 0.40, 0.85),
)
APPLIED_COLOR = (0.08, 0.08, 0.08)


@dataclass(frozen=True)
class CarShape:
  """Body geometry and control-arrow scaling used by `planar_car_scene`.

  `center_offset` is how far the body centre sits ahead of the recorded `(x, y)`, so a
  model whose position tracks the centre of gravity rather than the geometric centre
  still draws in the right place. Control vectors are read as (throttle, steering);
  `max_throttle` and `max_steer` normalize them, and `arrow_length` is the arrow length
  at full throttle (0 disables the arrows, as does `safety_radius` for the keep-out
  circle)."""

  length: float = 0.45
  width: float = 0.24
  height: float = 0.15
  center_offset: float = 0.0
  safety_radius: float = 0.0
  max_throttle: float = 1.0
  max_steer: float = 0.0
  arrow_length: float = 0.0


def _timestamp(time_s: float) -> Timestamp:
  nanoseconds = round(time_s * 1_000_000_000)
  return Timestamp(sec=nanoseconds // 1_000_000_000, nsec=nanoseconds % 1_000_000_000)


def _log_time(time_s: float) -> int:
  return round(time_s * 1_000_000_000)


def _color(rgb: tuple[float, float, float], alpha: float = 1.0) -> Color:
  return Color(r=rgb[0], g=rgb[1], b=rgb[2], a=alpha)


def _yaw_quaternion(yaw: float) -> Quaternion:
  return Quaternion(x=0.0, y=0.0, z=float(np.sin(0.5 * yaw)), w=float(np.cos(0.5 * yaw)))


def _circle(x: float, y: float, radius: float, color: Color, *, segments: int = 48) -> LinePrimitive:
  angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
  return LinePrimitive(
    type=LinePrimitiveLineType.LineLoop,
    thickness=0.06 * radius,
    points=[Point3(x=x + radius * float(np.cos(a)), y=y + radius * float(np.sin(a)), z=0.01) for a in angles],
    color=color,
  )


def _control_arrow(car: PlanarVehicleState, shape: CarShape, command: Sequence[float], color: Color, z: float) -> ArrowPrimitive:
  throttle = float(np.clip(command[0] / shape.max_throttle, -1.0, 1.0))
  steer = float(np.clip(command[1], -1.0, 1.0)) * shape.max_steer if len(command) > 1 else 0.0
  heading = car.yaw + steer + (np.pi if throttle < 0.0 else 0.0)
  length = shape.arrow_length * (0.25 + 0.75 * abs(throttle))
  return ArrowPrimitive(
    pose=Pose(position=Vector3(x=car.x, y=car.y, z=z), orientation=_yaw_quaternion(heading)),
    shaft_length=0.75 * length,
    shaft_diameter=0.07 * shape.arrow_length,
    head_length=0.25 * length,
    head_diameter=0.18 * shape.arrow_length,
    color=color,
  )


def planar_car_scene(
  states: Sequence[PlanarVehicleState | Mapping[str, Any]],
  *,
  shape: CarShape = CarShape(),
  controls: Sequence[ControlState | Mapping[str, Any]] = (),
  trails: Mapping[str, Sequence[tuple[float, float]]] | None = None,
) -> SceneUpdate:
  cars = [PlanarVehicleState.model_validate(state) for state in states]
  commands = {command.entity_id: command for command in (ControlState.model_validate(control) for control in controls)}
  entities = []
  for index, car in enumerate(cars):
    rgb = CAR_COLORS[index % len(CAR_COLORS)]
    timestamp = _timestamp(car.time_s)
    cos_yaw, sin_yaw = float(np.cos(car.yaw)), float(np.sin(car.yaw))
    lines = [_circle(car.x, car.y, shape.safety_radius, _color(rgb, 0.9))] if shape.safety_radius else []
    arrows = []
    command = commands.get(car.vehicle_id)
    if command is not None and shape.arrow_length:
      if command.desired:
        arrows.append(_control_arrow(car, shape, command.desired, _color(rgb), shape.height + 0.05))
      arrows.append(_control_arrow(car, shape, command.applied, _color(APPLIED_COLOR), shape.height + 0.15))
    entities.append(
      SceneEntity(
        timestamp=timestamp,
        frame_id="scene",
        id=f"car/{car.vehicle_id}",
        cubes=[
          CubePrimitive(
            pose=Pose(
              position=Vector3(x=car.x + cos_yaw * shape.center_offset, y=car.y + sin_yaw * shape.center_offset, z=0.5 * shape.height),
              orientation=_yaw_quaternion(car.yaw),
            ),
            size=Vector3(x=shape.length, y=shape.width, z=shape.height),
            color=_color(rgb, 0.55),
          )
        ],
        lines=lines,
        arrows=arrows,
      )
    )
    trail = [] if trails is None else trails.get(car.vehicle_id, [])
    if len(trail) > 1:
      entities.append(
        SceneEntity(
          timestamp=timestamp,
          frame_id="scene",
          id=f"trail/{car.vehicle_id}",
          lines=[
            LinePrimitive(
              type=LinePrimitiveLineType.LineStrip,
              thickness=0.05 * shape.length,
              points=[Point3(x=x, y=y, z=0.02) for x, y in trail],
              color=_color(rgb, 0.7),
            )
          ],
        )
      )
  return SceneUpdate(entities=entities)


def chain_scene(points: Sequence[PointState3D | Mapping[str, Any]], *, radius: float = 0.08) -> SceneUpdate:
  states = [PointState3D.model_validate(point) for point in points]
  if not states:
    return SceneUpdate(entities=[])
  xyz = [Point3(x=point.x, y=point.y, z=point.z) for point in states]
  timestamp = _timestamp(states[0].time_s)
  spheres = [
    SpherePrimitive(
      pose=Pose(position=Vector3(x=point.x, y=point.y, z=point.z), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
      size=Vector3(x=2 * radius, y=2 * radius, z=2 * radius),
      color=Color(r=1.0, g=0.48, b=0.1, a=1.0),
    )
    for point in states
  ]
  return SceneUpdate(
    entities=[
      SceneEntity(
        timestamp=timestamp,
        frame_id="scene",
        id="chain",
        lines=[
          LinePrimitive(
            type=LinePrimitiveLineType.LineStrip,
            thickness=radius * 0.4,
            points=xyz,
            color=Color(r=0.2, g=0.72, b=0.95, a=1.0),
          )
        ],
        spheres=spheres,
      )
    ]
  )


CONE_STYLES = {
  "blue": ((0.12, 0.35, 0.92), (0.25, 0.25, 0.32)),
  "yellow": ((0.95, 0.85, 0.12), (0.25, 0.25, 0.32)),
  "big_orange": ((1.0, 0.45, 0.05), (0.30, 0.30, 0.50)),
  "small_orange": ((1.0, 0.55, 0.15), (0.22, 0.22, 0.28)),
}
# (colour, height above the ground) per horizon; the offsets keep the two strips from z-fighting
HORIZON_STYLES = {"reference": ((0.15, 0.80, 0.35), 0.06), "prediction": ((0.95, 0.20, 0.70), 0.10)}


def track_scene(center_line: np.ndarray, cones: Mapping[str, np.ndarray]) -> SceneUpdate:
  """Static track geometry: the closed center line plus one cube per cone, by colour."""
  timestamp = _timestamp(0.0)
  entities = [
    SceneEntity(
      timestamp=timestamp,
      frame_id="scene",
      id="track/center_line",
      lines=[
        LinePrimitive(
          type=LinePrimitiveLineType.LineLoop,
          thickness=0.12,
          points=[Point3(x=float(x), y=float(y), z=0.01) for x, y in np.asarray(center_line, dtype=np.float64)],
          color=Color(r=0.85, g=0.85, b=0.85, a=0.8),
        )
      ],
    )
  ]
  for name, positions in cones.items():
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
    if not len(positions):
      continue
    rgb, size = CONE_STYLES[name]
    entities.append(
      SceneEntity(
        timestamp=timestamp,
        frame_id="scene",
        id=f"track/cones/{name}",
        cubes=[
          CubePrimitive(
            pose=Pose(position=Vector3(x=float(x), y=float(y), z=0.5 * size[2]), orientation=_yaw_quaternion(0.0)),
            size=Vector3(x=size[0], y=size[1], z=size[2]),
            color=_color(rgb),
          )
          for x, y in positions
        ],
      )
    )
  return SceneUpdate(entities=entities)


def horizon_scene(paths: Sequence[HorizonPath | Mapping[str, Any]], *, thickness: float = 0.18) -> SceneUpdate:
  """One line strip per planned/predicted horizon, drawn just above the track."""
  entities = []
  for path in (HorizonPath.model_validate(item) for item in paths):
    rgb, z = HORIZON_STYLES.get(path.path_id, ((0.6, 0.6, 0.6), 0.04))
    entities.append(
      SceneEntity(
        timestamp=_timestamp(path.time_s),
        frame_id="scene",
        id=f"horizon/{path.path_id}",
        lines=[
          LinePrimitive(
            type=LinePrimitiveLineType.LineStrip,
            thickness=thickness,
            points=[Point3(x=x, y=y, z=z) for x, y in zip(path.x, path.y, strict=True)],
            color=_color(rgb),
          )
        ],
      )
    )
  return SceneUpdate(entities=entities)


def arena_scene(bounds: tuple[float, float, float, float], *, margin: float = 0.0) -> SceneUpdate:
  """Arena walls, plus the inset rectangle the wall barriers actually keep the car
  centres inside when `margin` is the filter's wall margin."""
  x_min, x_max, y_min, y_max = bounds
  thickness = 0.006 * max(x_max - x_min, y_max - y_min)

  def loop(inset: float, color: Color, z: float) -> LinePrimitive:
    corners = ((x_min + inset, y_min + inset), (x_max - inset, y_min + inset), (x_max - inset, y_max - inset), (x_min + inset, y_max - inset))
    return LinePrimitive(
      type=LinePrimitiveLineType.LineLoop,
      thickness=thickness,
      points=[Point3(x=x, y=y, z=z) for x, y in corners],
      color=color,
    )

  lines = [loop(0.0, Color(r=0.95, g=0.72, b=0.2, a=1.0), 0.0)]
  if margin > 0.0:
    lines.append(loop(margin, Color(r=0.95, g=0.45, b=0.45, a=0.9), 0.005))
  return SceneUpdate(entities=[SceneEntity(timestamp=_timestamp(0.0), frame_id="scene", id="arena/bounds", lines=lines)])


class Recorder:
  def __init__(
    self,
    path: Path | str,
    *,
    allow_overwrite: bool = False,
    scene_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    car_shape: CarShape = CarShape(),
  ):
    self.car_shape = car_shape
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._context = Context()
    self._writer = open_mcap(self.path, allow_overwrite=allow_overwrite, context=self._context)
    self._metadata = self._channel("/run/metadata", RunMetadata)
    self._telemetry = self._channel("/telemetry", ScalarTelemetry)
    self._planar = self._channel("/planar/state", PlanarVehicleState)
    self._points = self._channel("/chain/point", PointState3D)
    self._control = self._channel("/control", ControlState)
    self._horizon = self._channel("/horizon", HorizonPath)
    self._scene = SceneUpdateChannel("/scene", context=self._context)
    self._tf = FrameTransformsChannel("/tf", context=self._context)
    self._planar_history: dict[str, list[tuple[float, float]]] = {}
    self._tf.log(
      FrameTransforms(
        transforms=[
          FrameTransform(
            timestamp=_timestamp(0.0),
            parent_frame_id="world",
            child_frame_id="scene",
            translation=Vector3(x=-scene_center[0], y=-scene_center[1], z=-scene_center[2]),
            rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
          )
        ]
      ),
      log_time=0,
    )
    self._closed = False

  def _channel(self, topic: str, model: type[BaseModel]) -> Channel:
    return Channel(topic, schema=model.model_json_schema(), message_encoding="json", context=self._context)

  @staticmethod
  def _log(channel: Channel, model: type[BaseModel], payload: BaseModel | Mapping[str, Any], log_time: int) -> BaseModel:
    validated = model.model_validate(payload)
    channel.log(validated.model_dump(mode="json"), log_time=log_time)
    return validated

  def record_metadata(self, metadata: RunMetadata | Mapping[str, Any], *, log_time: int = 0) -> RunMetadata:
    return self._log(self._metadata, RunMetadata, metadata, log_time)  # type: ignore[return-value]

  def record_telemetry(self, telemetry: ScalarTelemetry | Mapping[str, Any]) -> ScalarTelemetry:
    validated = ScalarTelemetry.model_validate(telemetry)
    return self._log(self._telemetry, ScalarTelemetry, validated, _log_time(validated.time_s))  # type: ignore[return-value]

  def record_planar(
    self,
    states: Sequence[PlanarVehicleState | Mapping[str, Any]],
    *,
    controls: Sequence[ControlState | Mapping[str, Any]] = (),
  ) -> list[PlanarVehicleState]:
    validated = [PlanarVehicleState.model_validate(state) for state in states]
    for state in validated:
      self._log(self._planar, PlanarVehicleState, state, _log_time(state.time_s))
      self._planar_history.setdefault(state.vehicle_id, []).append((state.x, state.y))
    if validated:
      scene = planar_car_scene(validated, shape=self.car_shape, controls=controls, trails=self._planar_history)
      self._scene.log(scene, log_time=_log_time(validated[0].time_s))
    return validated

  def record_chain(self, points: Sequence[PointState3D | Mapping[str, Any]]) -> list[PointState3D]:
    validated = [PointState3D.model_validate(point) for point in points]
    for point in validated:
      self._log(self._points, PointState3D, point, _log_time(point.time_s))
    if validated:
      self._scene.log(chain_scene(validated), log_time=_log_time(validated[0].time_s))
    return validated

  def record_arena(self, bounds: tuple[float, float, float, float], *, margin: float = 0.0) -> None:
    self._scene.log(arena_scene(bounds, margin=margin), log_time=0)

  def record_track(self, center_line: np.ndarray, cones: Mapping[str, np.ndarray]) -> None:
    self._scene.log(track_scene(center_line, cones), log_time=0)

  def record_horizons(self, paths: Sequence[HorizonPath | Mapping[str, Any]]) -> list[HorizonPath]:
    validated = [HorizonPath.model_validate(path) for path in paths]
    for path in validated:
      self._log(self._horizon, HorizonPath, path, _log_time(path.time_s))
    if validated:
      self._scene.log(horizon_scene(validated), log_time=_log_time(validated[0].time_s))
    return validated

  def record_control(self, controls: Sequence[ControlState | Mapping[str, Any]]) -> list[ControlState]:
    validated = [ControlState.model_validate(control) for control in controls]
    for control in validated:
      self._log(self._control, ControlState, control, _log_time(control.time_s))
    return validated

  def close(self) -> None:
    if self._closed:
      return
    for channel in (self._metadata, self._telemetry, self._planar, self._points, self._control, self._horizon, self._scene, self._tf):
      channel.close()
    self._writer.close()
    self._closed = True

  def __enter__(self) -> Recorder:
    return self

  def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
    self.close()


PROBLEM_DIRS = {"chain": "chain_of_masses", "race_cars": "race_cars", "bumpercars": "bumpercars_filter"}


def layout_path(problem: str) -> Path:
  """The problem's hand-authored Foxglove layout, exported from Foxglove Desktop. Layouts
  are not generated: Desktop's exported envelope is the only format its importer accepts,
  so each problem keeps one checked-in file next to its runner."""
  return Path(__file__).resolve().parents[1] / "problems" / PROBLEM_DIRS[problem] / "foxglove-layout.json"


def write_result_artifacts(
  output_dir: Path | str,
  *,
  config: Mapping[str, Any],
  summary: Mapping[str, Any],
  provenance: Mapping[str, Any],
  fe_inputs: Sequence[Mapping[str, Any]],
  successful_steps: Sequence[int] | None = None,
) -> dict[str, Path]:
  output = Path(output_dir)
  output.mkdir(parents=True, exist_ok=True)
  if not fe_inputs:
    raise ValueError("fe_inputs must contain at least one step")
  candidates = list(range(len(fe_inputs))) if successful_steps is None else sorted(set(successful_steps))
  candidates = [step for step in candidates if 0 <= step < len(fe_inputs)]
  if not candidates:
    raise ValueError("no successful step has FE inputs")
  representative_step = candidates[len(candidates) // 2]

  paths = {name: output / f"{name}.json" for name in ("config", "summary", "provenance", "metadata")}
  documents = {
    "config": dict(config),
    "summary": dict(summary),
    "provenance": dict(provenance),
    "metadata": {"representative_step": representative_step, "successful_steps": candidates},
  }
  for name, document in documents.items():
    temporary = paths[name].with_suffix(".json.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(paths[name])

  paths["fe_inputs"] = output / "representative_fe_inputs.npz"
  temporary_npz = paths["fe_inputs"].with_suffix(".npz.tmp")
  arrays = {name: np.asarray(value) for name, value in sorted(fe_inputs[representative_step].items())}
  with ZipFile(temporary_npz, "w", compression=ZIP_DEFLATED) as archive:
    for name, array in arrays.items():
      buffer = BytesIO()
      np.lib.format.write_array(buffer, array, allow_pickle=False)
      info = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
      info.compress_type = ZIP_DEFLATED
      info.external_attr = 0o600 << 16
      archive.writestr(info, buffer.getvalue())
  temporary_npz.replace(paths["fe_inputs"])
  return paths
