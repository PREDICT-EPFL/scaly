from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Self
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
  TextPrimitive,
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


class ChainPlan(_Schema):
  """One open-loop plan for a chain: the predicted shape at every node of the control horizon.

  `nodes` holds one entry per horizon node, each the flattened `x, y, z` of every mass from the
  anchor outwards, so a plan of `H + 1` nodes over `M` masses is `H + 1` lists of `3 * M` floats.
  """

  step: int = Field(ge=0)
  time_s: float = Field(ge=0.0)
  n_masses: int = Field(ge=2)
  nodes: list[list[float]]


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
ACCEL_COLOR = (0.15, 0.80, 0.30)
BRAKE_COLOR = (0.90, 0.15, 0.15)
WHEEL_COLOR = (0.10, 0.10, 0.12)
# Chain scene: one colour per role, so the fixed wall anchor and the actuated end mass that
# carries the control arrow are told apart from the masses that only obey the springs.
CHAIN_COLORS = {
  "link": (0.20, 0.72, 0.95),
  "mass": (1.00, 0.48, 0.10),
  "anchor": (0.55, 0.58, 0.62),
  "end": (0.96, 0.26, 0.21),
  "control": (0.18, 0.80, 0.44),
  "trail": (0.60, 0.62, 0.70),
  "plan": (0.95, 0.20, 0.70),
}
# Marker colours for the labelled reference points, taken in the order they are handed over.
CHAIN_REFERENCE_COLORS = ((0.95, 0.80, 0.20), (0.62, 0.45, 0.92), (0.20, 0.80, 0.85))


@dataclass(frozen=True)
class CarShape:
  """Body geometry and control depiction used by `planar_car_scene`.

  `center_offset` is how far the body centre sits ahead of the recorded `(x, y)`, so a
  model whose position tracks the centre of gravity rather than the geometric centre
  still draws in the right place.

  `safety_radius` is the body disc — half the centre distance at which two cars touch.
  `keep_out_radius` is the larger disc a filter actually enforces, drawn fainter so the
  margin between the two is visible; either is disabled by setting it to 0.

  There are two ways to draw the control, and a shape picks one by setting its length:

  - `arrow_length` draws arrows out in front of the car at full throttle. The command is
    read as (throttle, steering normalized to [-1, 1]), which `max_throttle` and
    `max_steer` scale into physical units.
  - `wheelbase` draws the two front wheels turned by the steering angle and shades the
    body green under drive, red under braking. The command is read as (drive force,
    steering angle in radians); `max_throttle` normalizes the force and `max_steer`
    clamps the angle. This suits a car whose steering is already a physical angle, where
    an arrow sticking out ahead of the body just reads as a heading vector."""

  length: float = 0.45
  width: float = 0.24
  height: float = 0.15
  center_offset: float = 0.0
  safety_radius: float = 0.0
  keep_out_radius: float = 0.0
  max_throttle: float = 1.0
  max_steer: float = 0.0
  arrow_length: float = 0.0
  wheelbase: float = 0.0


def _timestamp(time_s: float) -> Timestamp:
  nanoseconds = round(time_s * 1_000_000_000)
  return Timestamp(sec=nanoseconds // 1_000_000_000, nsec=nanoseconds % 1_000_000_000)


def _log_time(time_s: float) -> int:
  return round(time_s * 1_000_000_000)


def _color(rgb: tuple[float, float, float], alpha: float = 1.0) -> Color:
  return Color(r=rgb[0], g=rgb[1], b=rgb[2], a=alpha)


def _yaw_quaternion(yaw: float) -> Quaternion:
  return Quaternion(x=0.0, y=0.0, z=float(np.sin(0.5 * yaw)), w=float(np.cos(0.5 * yaw)))


def car_frame(vehicle_id: str) -> str:
  """The per-vehicle coordinate frame, a child of `scene` that carries the car's pose.

  Everything drawn on the car is expressed in this frame rather than in absolute track
  coordinates, so pointing the 3D panel's display frame at it makes the camera ride
  along with the car."""
  return f"car/{vehicle_id}"


def _direction_quaternion(vector: np.ndarray) -> Quaternion:
  """Rotation taking +X onto `vector`, since an arrow points along its pose's +X axis."""
  direction = vector / np.linalg.norm(vector)
  axis = np.array([0.0, -direction[2], direction[1]])  # x_hat x direction
  norm = float(np.linalg.norm(axis))
  if norm < 1e-12:  # (anti)parallel to +X, where the cross product carries no axis
    return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0) if direction[0] > 0.0 else Quaternion(x=0.0, y=0.0, z=1.0, w=0.0)
  half = 0.5 * float(np.arccos(np.clip(direction[0], -1.0, 1.0)))
  axis = axis * (float(np.sin(half)) / norm)
  return Quaternion(x=float(axis[0]), y=float(axis[1]), z=float(axis[2]), w=float(np.cos(half)))


def _circle(radius: float, color: Color, *, segments: int = 48) -> LinePrimitive:
  angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
  return LinePrimitive(
    type=LinePrimitiveLineType.LineLoop,
    thickness=0.06 * radius,
    points=[Point3(x=radius * float(np.cos(a)), y=radius * float(np.sin(a)), z=0.01) for a in angles],
    color=color,
  )


def _control_arrow(shape: CarShape, command: Sequence[float], color: Color, z: float) -> ArrowPrimitive:
  throttle = float(np.clip(command[0] / shape.max_throttle, -1.0, 1.0))
  steer = float(np.clip(command[1], -1.0, 1.0)) * shape.max_steer if len(command) > 1 else 0.0
  heading = steer + (np.pi if throttle < 0.0 else 0.0)
  length = shape.arrow_length * (0.25 + 0.75 * abs(throttle))
  return ArrowPrimitive(
    pose=Pose(position=Vector3(x=0.0, y=0.0, z=z), orientation=_yaw_quaternion(heading)),
    shaft_length=0.75 * length,
    shaft_diameter=0.07 * shape.arrow_length,
    head_length=0.25 * length,
    head_diameter=0.18 * shape.arrow_length,
    color=color,
  )


def _drive_color(rgb: tuple[float, float, float], throttle: float) -> Color:
  """The car's own colour, shaded toward green under drive and red under braking."""
  target = ACCEL_COLOR if throttle >= 0.0 else BRAKE_COLOR
  weight = abs(throttle)
  return _color(tuple(base * (1.0 - weight) + tip * weight for base, tip in zip(rgb, target, strict=True)), 0.55)  # type: ignore[arg-type]


def _steered_wheels(shape: CarShape, steer: float) -> list[CubePrimitive]:
  """The pair of front wheels, sitting on the front axle and turned by the steering angle."""
  length, width, height = 0.20 * shape.length, 0.12 * shape.width, 0.30 * shape.height
  axle, half_track = shape.center_offset + 0.5 * shape.wheelbase, 0.5 * (shape.width - width)
  return [
    CubePrimitive(
      pose=Pose(position=Vector3(x=axle, y=offset, z=0.5 * height), orientation=_yaw_quaternion(steer)),
      size=Vector3(x=length, y=width, z=height),
      color=_color(WHEEL_COLOR),
    )
    for offset in (-half_track, half_track)
  ]


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
    lines = [_circle(shape.safety_radius, _color(rgb, 0.9))] if shape.safety_radius else []
    if shape.keep_out_radius:
      lines.append(_circle(shape.keep_out_radius, _color(rgb, 0.35)))
    arrows, wheels, body_color = [], [], _color(rgb, 0.55)
    command = commands.get(car.vehicle_id)
    if command is not None and shape.arrow_length:
      if command.desired:
        arrows.append(_control_arrow(shape, command.desired, _color(rgb), shape.height + 0.05))
      arrows.append(_control_arrow(shape, command.applied, _color(APPLIED_COLOR), shape.height + 0.15))
    if command is not None and shape.wheelbase:
      steer = float(np.clip(command.applied[1], -shape.max_steer, shape.max_steer)) if len(command.applied) > 1 else 0.0
      body_color = _drive_color(rgb, float(np.clip(command.applied[0] / shape.max_throttle, -1.0, 1.0)))
      wheels = _steered_wheels(shape, steer)
    entities.append(
      SceneEntity(
        timestamp=timestamp,
        frame_id=car_frame(car.vehicle_id),
        frame_locked=True,
        id=f"car/{car.vehicle_id}",
        cubes=[
          CubePrimitive(
            pose=Pose(position=Vector3(x=shape.center_offset, y=0.0, z=0.5 * shape.height), orientation=_yaw_quaternion(0.0)),
            size=Vector3(x=shape.length, y=shape.width, z=shape.height),
            color=body_color,
          ),
          *wheels,
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
              thickness=0.02 * shape.length,
              points=[Point3(x=x, y=y, z=0.02) for x, y in trail],
              color=_color(rgb, 0.7),
            )
          ],
        )
      )
  return SceneUpdate(entities=entities)


def _vector_arrow(origin: Sequence[float], vector: Sequence[float], color: Color, *, scale: float, shaft_diameter: float) -> ArrowPrimitive | None:
  """Arrow `scale * |vector|` long, from `origin` along `vector`; `None` when there is nothing to draw."""
  values = np.asarray(vector, dtype=np.float64)
  length = scale * float(np.linalg.norm(values))
  if length < 1e-9:
    return None
  return ArrowPrimitive(
    pose=Pose(
      position=Vector3(x=float(origin[0]), y=float(origin[1]), z=float(origin[2])),
      orientation=_direction_quaternion(values),
    ),
    shaft_length=0.75 * length,
    shaft_diameter=shaft_diameter,
    head_length=0.25 * length,
    head_diameter=2.0 * shaft_diameter,
    color=color,
  )


def chain_scene(
  points: Sequence[PointState3D | Mapping[str, Any]],
  *,
  radius: float = 0.08,
  control: ControlState | Mapping[str, Any] | None = None,
  control_scale: float = 0.6,
  trail: Sequence[tuple[float, float, float]] = (),
) -> SceneUpdate:
  """The chain as a strip through its masses, the first drawn as the fixed wall anchor and the
  last, larger, as the actuated end mass. The end mass's control *is* its velocity, so `control`
  is drawn as an arrow there, `control_scale` metres long per unit command. `trail` is the path
  the end mass has taken so far."""
  states = [PointState3D.model_validate(point) for point in points]
  if not states:
    return SceneUpdate(entities=[])
  timestamp = _timestamp(states[0].time_s)
  spheres = []
  for index, point in enumerate(states):
    role = "anchor" if index == 0 else ("end" if index == len(states) - 1 else "mass")
    size = 2 * radius * (1.4 if role == "end" else 1.0)
    spheres.append(
      SpherePrimitive(
        pose=Pose(position=Vector3(x=point.x, y=point.y, z=point.z), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        size=Vector3(x=size, y=size, z=size),
        color=_color(CHAIN_COLORS[role]),
      )
    )
  arrows = []
  if control is not None:
    end = states[-1]
    arrow = _vector_arrow(
      (end.x, end.y, end.z),
      ControlState.model_validate(control).applied,
      _color(CHAIN_COLORS["control"]),
      scale=control_scale,
      shaft_diameter=0.5 * radius,
    )
    if arrow is not None:
      arrows.append(arrow)
  entities = [
    SceneEntity(
      timestamp=timestamp,
      frame_id="scene",
      id="chain",
      lines=[
        LinePrimitive(
          type=LinePrimitiveLineType.LineStrip,
          thickness=radius * 0.4,
          points=[Point3(x=point.x, y=point.y, z=point.z) for point in states],
          color=_color(CHAIN_COLORS["link"]),
        )
      ],
      spheres=spheres,
      arrows=arrows,
    )
  ]
  if len(trail) > 1:
    entities.append(
      SceneEntity(
        timestamp=timestamp,
        frame_id="scene",
        id="chain/trail",
        lines=[
          LinePrimitive(
            type=LinePrimitiveLineType.LineStrip,
            thickness=radius * 0.25,
            points=[Point3(x=x, y=y, z=z) for x, y, z in trail],
            color=_color(CHAIN_COLORS["trail"], 0.7),
          )
        ],
      )
    )
  return SceneUpdate(entities=entities)


def chain_plan_scene(plan: ChainPlan | Mapping[str, Any], *, thickness: float = 0.02) -> SceneUpdate:
  """The open-loop plan behind the applied control: the predicted chain at every node of the
  horizon as a faint strip, plus the path the plan takes the end mass along."""
  validated = ChainPlan.model_validate(plan)
  nodes = [np.asarray(node, dtype=np.float64).reshape(validated.n_masses, 3) for node in validated.nodes]
  if not nodes:
    return SceneUpdate(entities=[])
  rgb = CHAIN_COLORS["plan"]
  shapes = [
    LinePrimitive(
      type=LinePrimitiveLineType.LineStrip,
      thickness=thickness,
      points=[Point3(x=float(x), y=float(y), z=float(z)) for x, y, z in node],
      color=_color(rgb, 0.25),
    )
    for node in nodes
  ]
  shapes.append(
    LinePrimitive(
      type=LinePrimitiveLineType.LineStrip,
      thickness=2.0 * thickness,
      points=[Point3(x=float(node[-1][0]), y=float(node[-1][1]), z=float(node[-1][2])) for node in nodes],
      color=_color(rgb, 0.9),
    )
  )
  return SceneUpdate(entities=[SceneEntity(timestamp=_timestamp(validated.time_s), frame_id="scene", id="chain/plan", lines=shapes)])


def chain_reference_scene(references: Mapping[str, Sequence[float]], *, radius: float = 0.08) -> SceneUpdate:
  """The end-mass positions the cost pulls towards, one labelled translucent marker each; the
  label carries the name it was handed over under and the position itself."""
  entities = []
  for index, (name, position) in enumerate(references.items()):
    x, y, z = (float(value) for value in position)
    rgb = CHAIN_REFERENCE_COLORS[index % len(CHAIN_REFERENCE_COLORS)]
    entities.append(
      SceneEntity(
        timestamp=_timestamp(0.0),
        frame_id="scene",
        id=f"chain/reference/{name}",
        spheres=[
          SpherePrimitive(
            pose=Pose(position=Vector3(x=x, y=y, z=z), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
            size=Vector3(x=3 * radius, y=3 * radius, z=3 * radius),
            color=_color(rgb, 0.4),
          )
        ],
        texts=[
          TextPrimitive(
            pose=Pose(position=Vector3(x=x, y=y, z=z + 3 * radius), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
            billboard=True,
            font_size=12.0,
            scale_invariant=True,
            color=_color(rgb),
            text=f"{name} (x={x:.2f})",
          )
        ],
      )
    )
  return SceneUpdate(entities=entities)


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


def horizon_scene(paths: Sequence[HorizonPath | Mapping[str, Any]], *, thickness: float = 0.06) -> SceneUpdate:
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
  """What every episode records whatever the problem: the MCAP writer, the run metadata, the
  solver telemetry, the applied control, and the two scene topics all three problems draw on.

  A problem records the rest through its own subclass — `ChainRecorder`, `RaceCarRecorder`,
  `UnbumpercarsRecorder` — which opens its extra channels in `_open_channels`. Channels register
  themselves as they are opened, so `close()` needs no per-subclass list, and an episode's MCAP
  offers Foxglove only the topics its problem actually writes."""

  def __init__(
    self,
    path: Path | str,
    *,
    allow_overwrite: bool = False,
    scene_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
  ):
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._context = Context()
    self._writer = open_mcap(self.path, allow_overwrite=allow_overwrite, context=self._context)
    self._channels: list[Any] = []
    self._metadata = self._channel("/run/metadata", RunMetadata)
    self._telemetry = self._channel("/telemetry", ScalarTelemetry)
    self._control = self._channel("/control", ControlState)
    # Seeking makes Foxglove hand each panel the *single* newest message per subscribed topic, so
    # anything that must survive a jump has to be the last message on a topic of its own. Geometry
    # logged once at time zero therefore gets `/scene/static` to itself, and stays visible however
    # far ahead you jump without being re-sent every step.
    self._scene = self._scene_channel("/scene")
    self._static_scene = self._scene_channel("/scene/static")
    self._tf = FrameTransformsChannel("/tf", context=self._context)
    self._channels.append(self._tf)
    self._open_channels()
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

  def _open_channels(self) -> None:
    """Open the channels only this problem writes. Subclasses override; the base opens none."""

  def _channel(self, topic: str, model: type[BaseModel]) -> Channel:
    channel = Channel(topic, schema=model.model_json_schema(), message_encoding="json", context=self._context)
    self._channels.append(channel)
    return channel

  def _scene_channel(self, topic: str) -> SceneUpdateChannel:
    channel = SceneUpdateChannel(topic, context=self._context)
    self._channels.append(channel)
    return channel

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

  def record_control(self, controls: Sequence[ControlState | Mapping[str, Any]]) -> list[ControlState]:
    validated = [ControlState.model_validate(control) for control in controls]
    for control in validated:
      self._log(self._control, ControlState, control, _log_time(control.time_s))
    return validated

  def close(self) -> None:
    if self._closed:
      return
    for channel in self._channels:
      channel.close()
    self._writer.close()
    self._closed = True

  def __enter__(self) -> Self:
    return self

  def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
    self.close()


class PlanarRecorder(Recorder):
  """Shared by the two planar-vehicle problems: one `/planar/state` row per car per step, the
  per-car transform its frame-locked scene entities hang off, and the trail each car leaves."""

  def __init__(
    self,
    path: Path | str,
    *,
    allow_overwrite: bool = False,
    scene_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    car_shape: CarShape = CarShape(),
  ):
    super().__init__(path, allow_overwrite=allow_overwrite, scene_center=scene_center)
    self.car_shape = car_shape

  def _open_channels(self) -> None:
    self._planar = self._channel("/planar/state", PlanarVehicleState)
    self._planar_history: dict[str, list[tuple[float, float]]] = {}

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
      log_time = _log_time(validated[0].time_s)
      # The car frames go out alongside the scene they carry: the entities are frame-locked, so a
      # pose that arrived without its transform would draw the car at the previous step's place.
      self._tf.log(
        FrameTransforms(
          transforms=[
            FrameTransform(
              timestamp=_timestamp(state.time_s),
              parent_frame_id="scene",
              child_frame_id=car_frame(state.vehicle_id),
              translation=Vector3(x=state.x, y=state.y, z=0.0),
              rotation=_yaw_quaternion(state.yaw),
            )
            for state in validated
          ]
        ),
        log_time=log_time,
      )
      scene = planar_car_scene(validated, shape=self.car_shape, controls=controls, trails=self._planar_history)
      self._scene.log(scene, log_time=log_time)
    return validated


class RaceCarRecorder(PlanarRecorder):
  """Race cars: the static track, plus the reference and predicted horizons redrawn every step."""

  def _open_channels(self) -> None:
    super()._open_channels()
    self._horizon = self._channel("/horizon", HorizonPath)
    self._horizon_scene = self._scene_channel("/scene/horizon")

  def record_track(self, center_line: np.ndarray, cones: Mapping[str, np.ndarray]) -> None:
    self._static_scene.log(track_scene(center_line, cones), log_time=0)

  def record_horizons(self, paths: Sequence[HorizonPath | Mapping[str, Any]]) -> list[HorizonPath]:
    validated = [HorizonPath.model_validate(path) for path in paths]
    for path in validated:
      self._log(self._horizon, HorizonPath, path, _log_time(path.time_s))
    if validated:
      self._horizon_scene.log(horizon_scene(validated), log_time=_log_time(validated[0].time_s))
    return validated


class UnbumpercarsRecorder(PlanarRecorder):
  """Unbumpercars: a persistent arena boundary on top of the shared planar channels. The filter
  only ever commits the next input, so there is no horizon to draw and no topic to carry one."""

  def record_arena(self, bounds: tuple[float, float, float, float], *, margin: float = 0.0) -> None:
    self._static_scene.log(arena_scene(bounds, margin=margin), log_time=0)


class ChainRecorder(Recorder):
  """Chain of masses: the mass positions and the end mass's trail, the open-loop plan on the
  horizon topic, and the end-mass reference markers pinned to the static one."""

  def _open_channels(self) -> None:
    self._points = self._channel("/chain/point", PointState3D)
    self._plan = self._channel("/chain/plan", ChainPlan)
    self._horizon_scene = self._scene_channel("/scene/horizon")
    self._chain_history: list[tuple[float, float, float]] = []

  def record_chain(
    self, points: Sequence[PointState3D | Mapping[str, Any]], *, control: ControlState | Mapping[str, Any] | None = None
  ) -> list[PointState3D]:
    validated = [PointState3D.model_validate(point) for point in points]
    for point in validated:
      self._log(self._points, PointState3D, point, _log_time(point.time_s))
    if validated:
      end = validated[-1]
      self._chain_history.append((end.x, end.y, end.z))
      self._scene.log(chain_scene(validated, control=control, trail=self._chain_history), log_time=_log_time(validated[0].time_s))
    return validated

  def record_chain_plan(self, plan: ChainPlan | Mapping[str, Any]) -> ChainPlan:
    validated = ChainPlan.model_validate(plan)
    self._log(self._plan, ChainPlan, validated, _log_time(validated.time_s))
    # The plan is this problem's horizon, so it rides the horizon topic the race-car horizons use.
    self._horizon_scene.log(chain_plan_scene(validated), log_time=_log_time(validated.time_s))
    return validated

  def record_chain_references(self, references: Mapping[str, Sequence[float]]) -> None:
    self._static_scene.log(chain_reference_scene(references), log_time=0)


PROBLEM_DIRS = {problem: problem for problem in ("chain", "race_cars", "unbumpercars")}


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
