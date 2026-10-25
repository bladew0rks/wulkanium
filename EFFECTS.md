# Screen effects

wulkanium loads screen effects from `garrysmod/shaders/wulkanium/*.wgsl`. They run after the
3D view (and DLSS), before the HUD, so they never smear the UI. Unlike ReShade they don't have to
guess anything: they get the renderer's own depth buffer, the camera matrices and, on DLSS
frames, motion vectors.

Files are checked about once a second, so you can edit an effect while the game runs. A broken
edit is logged to `shaderapiwgpu.log` with the line number and the last working version keeps
running.


## Turning effects on

With the addon installed, each effect gets convars and a page under **Options > wulkanium >
Screen effects**:

| Convar | |
|---|---|
| `wk_fx_<effect>` | 1 = on |
| `wk_fx_<effect>_<param>` | parameter value |
| `wk_fx_list` | print every effect and its values |
| `wk_fx_reload` | pick up newly added effects |

`<effect>` is the effect name in lowercase with anything other than letters and digits turned
into `_`. Without the addon, effects use `//! enabled:` and their parameter defaults.

## File format

The header is `//!` lines; everything else is plain WGSL.

```wgsl
//! effect: My Effect          display name (default: the file name)
//! order: 100                 lower runs first (default 0)
//! enabled: 1                 on without the addon / before its convar exists (default off)
//! texture: half rgba16float 0.5   name, format, size relative to the screen (default 1);
                                add `depth` to size it from the depth buffer instead (smaller
                                than the screen under DLSS upscaling)
//! pass: blur_h -> half       @fragment entry point -> target texture
//! pass: combine -> screen    "screen" is the back buffer
//! param: strength 1 0 4      name, default, min, max (add `int` for whole numbers)

@fragment
fn blur_h(i: WkVsOut) -> @location(0) vec4f { ... }
```

- Passes run in order. A pass sees every texture of the effect except the one it draws to.
- Effect textures keep their contents between frames, so an effect can keep history.
- Texture formats: `r8unorm rg8unorm rgba8unorm r16float rg16float rgba16float r32float
  rg32float rgba32float`. The 32-bit float ones can't be filtered: read them with
  `textureLoad` or the `wk_point` sampler.
- Parameters are `f32`s, read as `params.<name>`.
- Names starting with `wk_` are reserved.

## What an effect can use

| Name | |
|---|---|
| `wk_color` | the screen so far (earlier effects included); `texture_2d<f32>` |
| `wk_depth_tex` | the depth buffer, 0 = near, 1 = far; `texture_depth_2d`. Under DLSS upscaling the scene covers only its top-left `wk.depth_size` pixels, so read it through `wk_depth` / `wk_pixel` |
| `wk_motion` | motion vectors (DLSS frames only, see `wk_has_motion()`) |
| `wk_linear`, `wk_point` | clamped samplers |
| `wk.view`, `wk.proj`, `wk.view_proj` | the main 3D camera; `wk.proj * wk.view * vec4f(world, 1.0)` |
| `wk.inv_view`, `wk.inv_proj`, `wk.inv_view_proj` | their inverses |
| `wk.prev_view_proj` | last frame's `view_proj` |
| `wk.camera_pos` | camera position in world units (`.xyz`) |
| `wk.resolution` | width, height, 1/width, 1/height |
| `wk.time`, `wk.delta`, `wk.frame` | seconds since start, frame time, frame counter |
| `params.<name>` | this effect's parameters |
| `WkVsOut` | fragment input: `.pos` (pixel position), `.uv` (0..1, y down) |

Helpers:

| Function | |
|---|---|
| `wk_depth(uv) -> f32` | depth at a uv |
| `wk_is_sky(depth)` | nothing was drawn there |
| `wk_is_viewmodel(depth)` | the viewmodel (drawn with its depth squeezed in front) |
| `wk_view_pos_at(uv) -> vec3f` | view-space position of the surface at a uv |
| `wk_world_pos_at(uv) -> vec3f` | world position of the surface at a uv |
| `wk_view_pos(uv, depth) -> vec3f` | view-space position for a given depth (use a pixel centre: `wk_snap(uv)`) |
| `wk_world_pos(uv, depth) -> vec3f` | world position |
| `wk_project(world) -> vec3f` | world position to (uv, depth), unjittered like `wk_color` |
| `wk_distance(uv) -> f32` | distance from the camera |
| `wk_world_normal(uv) -> vec3f` | world normal rebuilt from depth |
| `wk_view_normal(uv) -> vec3f` | the same in view space (consistent with `wk_view_pos`) |
| `wk_motion_uv(uv) -> vec2f` | how far this pixel moved since last frame, in uv |
| `wk_has_camera()`, `wk_has_motion()` | whether those inputs are valid this frame |

Under DLSS the depth buffer is jittered by a fraction of a pixel each frame; the position helpers
remove that, so reconstructed positions stay still.

The "main camera" is the last 3D view drawn outside the viewmodel's depth range: the world view,
not the 3D skybox or the viewmodel.

## Shipped effects

`effects/` in this repo; copy them to `garrysmod/shaders/wulkanium/`.

| File | |
|---|---|
| `gtao.wgsl` | ambient occlusion (ground truth AO), half resolution with a depth-aware blur |
| `rcas.wgsl` | contrast adaptive sharpening, after AMD FidelityFX RCAS |
| `debug_view.wgsl` | shows distance, normals, a world grid or motion vectors; useful for checking inputs |
