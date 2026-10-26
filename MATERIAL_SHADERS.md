# Material shaders

Material shaders are shaders that VMTs name, like `VertexLitGeneric` or `SurfaceGGX`. You can write new ones while the game runs: put a `.wgsl` file in
`garrysmod/shaders/wulkanium/materials/` and materials can use it by name.

```wgsl
//! shader: Toon
//! param: $bands float 3 "number of light bands"
//! param: $shadowtint color [0.35 0.35 0.5]
//! param: $ramp texture

fn shade(s: Surface) -> vec4f {
    let m = material();
    let base = base_texture(s.uv) * tint();
    let light = diffuse_lighting(s, s.normal);
    return vec4f(base.rgb * floor(light * m.bands) / m.bands, base.a);
}
```

```
"Toon"
{
	"$basetexture" "models/props_c17/oil_drum001a"
	"$bands" "4"
}
```

## While the game runs

- **Editing the body** reloads the shader within half a second. Materials keep working.
- **A new file, or a changed `//! shader:` / `//! param:` header** registers the shader with the
  game again. Materials that are already loaded keep the old version until they reload
  (`mat_reloadmaterial <name>` or `mat_reloadallmaterials`). A material that named the shader
  before it existed has to reload too.
- **Errors** show up on screen and in `shaderapiwgpu.log`. The last working version stays in use.

## Header

| line | meaning |
|---|---|
| `//! shader: Name` | the name VMTs use (letters, digits, `_`) |
| `//! param: $name type [default] ["help"]` | a material param |

Param types:
- `float`, `int`, `bool`, `vec2`, `vec3`, `vec4`, `color`: up to 16 in total. `color` arrives
  linear (gamma 2.2 removed, like `$color`).
- `texture` (linear), `texture_srgb`, `cubemap`: up to 7, plus `$basetexture`.

The engine's standard params always exist: `$basetexture`, `$color`, `$color2`, `$alpha`,
`$basetexturetransform`, `$frame` and the `$flags` (`$translucent`, `$additive`,
`$alphatest` + `$alphatestreference`, `$nocull`, `$ignorez`, `$decal`). The flags set blending,
culling and depth as they do for stock shaders.

## What the script gets

`fn shade(s: Surface) -> vec4f` returns linear color and alpha. Tonemapping, fog, the alpha test,
sRGB output and motion vectors are applied afterwards.

`Surface`, with every direction in world space and normalized:

| field | |
|---|---|
| `uv`, `uv_raw` | texcoord with and without `$basetexturetransform` |
| `world` | position |
| `normal`, `tangent`, `binormal` | tangents only when the body mentions them |
| `view`, `eye` | towards the eye, eye position |
| `vertex_light`, `has_vertex_light` | baked per-vertex lighting of static props |
| `screen` | pixel position |
| `front_facing` | |

Functions:

| function | |
|---|---|
| `material()` | the number params, as fields named without `$` (`m.bands`) |
| `base_texture(uv)` | `$basetexture`, sRGB decoded |
| `sample_<param>(uv)`, `sample_<param>_level(uv, lod)` | a texture param (a direction for cubemaps) |
| `tint()` | `$color * $color2` (linear) and `$alpha` |
| `time()` | seconds |
| `ambient(n)` | the ambient light cube along `n` |
| `diffuse_lighting(s, n)` | Source-style diffuse: baked vertex light or ambient, plus the dynamic lights |
| `light_count()`, `light_color(i)`, `light_dir(i, world)`, `light_atten(i, world)` | the dynamic lights, up to 4 |

## How registering works

materialsystem reads a shader DLL's shader list only when it loads that DLL. wulkanium asks
materialsystem to load its own DLL again, which reads the current list. A shader whose params
changed gets a new shader object, while the old one stays alive for the materials that still use it.
