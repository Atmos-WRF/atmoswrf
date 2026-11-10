# Chase vehicles for atmos-wrf chase mode — integration notes

Four storm chase vehicles with mobile mesonet roof racks, ready to drop into a game.
All files are glTF 2.0 binary (`.glb`): textures and materials are embedded, and there are no external dependencies.

| File | Vehicle | Triangles |
|---|---|---|
| `PROBE_1_SUV_LOD0.glb` / `_LOD1.glb` | Full-size SUV | 305k / 96k |
| `PROBE_2_Pickup_LOD0.glb` / `_LOD1.glb` | Crew-cab pickup | 289k / 91k |
| `PROBE_3_Sedan_LOD0.glb` / `_LOD1.glb` | Full-size sedan | 238k / 77k |
| `PROBE_4_Minivan_LOD0.glb` / `_LOD1.glb` | Minivan | 282k / 87k |

Use LOD0 for the player's vehicle and close cameras. Use LOD1 for AI or other vehicles and at distance. Both versions have the same node names and materials, so the same code drives either one.
Each vehicle also has a `*_README.txt` with exact wheel hub positions and dimensions.

## Conventions
- Units: meters. Up is +Y. **The vehicle front faces +Z** (glTF standard).
- The root node holds a child called `Model` with a -90° Y rotation. The vehicle was authored with its front at +X, and that rotation turns it to +Z. Leave it in place, or bake it into the transform.
- Origin: on the ground, centered on the body's length.
- Root-node extras (glTF `extras`): `wheelRadius`, `wheelWidth`, `wheelbase`, `track`, `units`, `up`, `forward`.

## Nodes to drive
| Node | What to do |
|---|---|
| `Wheel_FL`, `Wheel_FR` | Steering pivots at the hub centers. Rotate about local Y. |
| `Wheel_RL`, `Wheel_RR` | Rear hub positions. They don't steer. |
| `WheelSpin_FL/FR/RL/RR` (children of `Wheel_*`) | Wheel roll. Rotate about local Z, at angle += speed / wheelRadius × dt. The brake calipers sit on `Wheel_*`, so they steer but don't spin. |
| `SteeringWheel` (under `Interior`) | Rotate about local X, about 2–3× the road-wheel steer angle. |
| `Anemometer_Vane` | Wind vane. Yaw about Y toward the relative wind. You can feed it the model's wind field minus the vehicle's velocity. |
| `Anemometer_Prop` (child of the vane) | Propeller. Spin about local X in proportion to relative wind speed. |
| `COL_Hull` | Low-poly collision mesh. Hide it and use it as a convex collider, or ignore it and use wheel colliders plus a box. |

Static parts: `Body`, `Glass`, `Exterior_Trim`, `Lights`, `Decals`, `Interior`, `Chassis`, `Roof_Rack`.

## Materials you can change at runtime (by name)
- **Recolor:** `Paint_Primary`, `Paint_Roof`, `Livery_Accent`
- **Lights:** raise the emissive intensity to switch them on.
  - `Light_Head`, `Light_DRL`, `Light_Tail` (running), `Light_Brake`, `Light_Turn` (alternate for blinkers or hazards), `Light_Reverse`
- **Light bar:** alternate `LightBar_A` and `LightBar_B` (or `LightBar_B_White`) at about 2 Hz.
- **Glass:** `Glass_Front` and `Glass_Privacy` are alpha-blended, and the interior is modeled behind them.

## Suggested physics values (if the engine wants them)
| Vehicle | Mass | Wheel radius | Wheelbase | Drive |
|---|---|---|---|---|
| SUV | ~2600 kg | 0.40 m | 3.15 m | 4WD |
| Pickup | ~2700 kg | 0.41 m | 3.70 m | 4WD |
| Sedan | ~1900 kg | 0.345 m | 2.91 m | RWD |
| Minivan | ~2100 kg | 0.35 m | 3.03 m | FWD |

Put the center of mass about 0.15 m above the wheel centers. The mesonet rack adds height.

## Engine import
- **Three.js / web:** `GLTFLoader`. Find nodes with `getObjectByName('Wheel_FL')` and materials by `material.name`.
- **Godot 4:** drag the .glb into the project. For physics, use the `Wheel_*` hub positions for `VehicleWheel3D` nodes.
- **Unreal 5:** use the native glTF import.
- **Unity:** install `com.unity.cloud.gltfast`.

## Notes
- The designs are unbranded on purpose: no manufacturer names or logos.
- UVs are projection-based. They work for colors and decal-style liveries but aren't an artist unwrap.
- To get a different paint, livery, wheels or equipment, re-export from the configurator page with the "Download this vehicle" button. Node and material names stay the same.
