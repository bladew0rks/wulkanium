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
| `wk_depth_tex` | the solid scene's depth: where water covers the ground it holds the ground, not the water surface. 0 = near, 1 = far; `texture_depth_2d`. Under DLSS upscaling the scene covers only its top-left `wk.depth_size` pixels, so read it through `wk_depth` / `wk_pixel` |
| `wk_final_depth_tex` | the depth buffer as drawn, water included; `texture_depth_2d` |
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
| `wk_covered(uv) -> bool` | water (or anything drawn over the solid scene) covers this pixel; `wk_depth` is what's under it |
| `wk_has_viewmodel() -> bool` | a viewmodel was drawn this frame |
| `wk_vm_view_pos(uv, depth) -> vec3f` | a viewmodel pixel's position in the viewmodel's own view space (it has its own projection and a squeezed depth range, so `wk_view_pos` is wrong for it) |
| `wk_vm_project(p) -> vec3f` | viewmodel view position -> (uv, raw depth) |
| `wk_vm_light_dir(p) -> vec3f` | towards the viewmodel's lights from a viewmodel view position: all of them blended into one direction by brightness and falloff, eased over a few frames; zero without lights |
| `wk_vm_light_agreement() -> f32` | how much those lights agree on that direction: 1 for a single light, towards 0 when lights from opposite sides cancel out |
| `wk_is_dynamic(pixel) -> bool` | the surface at a depth pixel is a moving object (prop, player, npc, moved brush entity) rather than something the lightmaps baked; true everywhere under msaa. using it makes scene draws mark stencil bit 0x80 |
| `wk_sun_shadow(pixel) -> f32` | sun shadow at a depth pixel (1 lit, 0 shadowed) from a screen-space shadow pass (a port of bend studio's); 1 without a sun. using it runs the pass; the effect's params `sun_thickness`, `sun_contrast`, `all_casters`, `ignore_edges` tune it |
| `wk_has_sun_shadow() -> bool` | the sun shadow mask is valid this frame |
| `wk_has_sun() -> bool` | the map has a sun (light_environment) that models have seen in the last 10 seconds |
| `wk_sun_dir() -> vec3f` | world-space direction towards the sun; zero without one |
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
