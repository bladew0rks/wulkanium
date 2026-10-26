<p align="center"><img src="logo.png" alt="wulkanium logo" width="192"></p>

<h1 align="center">wulkanium</h1>

<p align="center">Drop-in Vulkan renderer for Garry's Mod, built on wgpu.</p>

<br>

> [!WARNING]
> **AI-generated code.** Most of this project was written by an AI coding assistant
> (Anthropic's Claude) under my direction, review and testing. Expect rough edges, read the
> code before relying on it, and report problems you find.

## About

A drop-in shaderapi DLL for Garry's Mod that renders through wgpu / Vulkan instead of
DirectX9.

## Build

```sh
# Plain build (MinGW)
cargo build --release --target x86_64-pc-windows-gnu

# With DLSS (MSVC via cargo-xwin; needs the DLSS SDK and Vulkan headers)
DLSS_SDK=/path/to/DLSS VULKAN_SDK=/path/to/vulkan XWIN_ACCEPT_LICENSE=1 \
  cargo xwin build --release --features dlss --target x86_64-pc-windows-msvc
```

## Install

Copy `target/<triple>/release/shaderapiwgpu.dll` to `GarrysMod/bin/win64/` and launch with:

```
-shaderapi shaderapiwgpu
```

DLSS builds also need NVIDIA's `nvngx_dlss.dll` in `GarrysMod/bin/win64/`: the release one from
the DLSS SDK (`lib/Windows_x86_64/rel`), or a newer version dropped in its place.

## Keys

| Key | Action |
|---|---|
| Insert | Overlay (fps, frame stats) |
| Home | Log one frame of draws (capture) |
| Shift+Home | Sampling profiler on/off (results go to the log) |
| End | DLSS on/off (DLSS builds; mode via `wk_dlss_mode` or Options > wulkanium > DLSS) |
| Page Up / Page Down | DLSS input view / jitter pattern |
| Scroll Lock | Shadow map view |

## Screen effects

Effects written in WGSL, loaded from `garrysmod/shaders/wulkanium/`, with access to the real
depth buffer, the camera and motion vectors, and drawn before the HUD. Ambient occlusion and
sharpening ship in `effects/`. See [EFFECTS.md](EFFECTS.md) for writing your own.

## Addon

`addon/` is a small Lua addon (copy it to `garrysmod/addons/wulkanium/`). It adds the screen
effects settings page (Options > wulkanium) and benchmark commands for comparing renderers from
the same spot:

| Command | Action |
|---|---|
| `bench_save` | save your position and view for this map |
| `bench_lock` / `bench_unlock` | teleport there and freeze, or release |
| `bench_run [seconds] [warmup]` | time frames: avg fps, 1% low, worst frame |

## Tests

```sh
SHADERS_DIR=/path/to/extracted/shaders/fxc cargo test --release
```

`SHADERS_DIR` holds `.vcs` files from `garrysmod_dir.vpk` (`shaders/fxc`); without it the
shader tests are skipped.

## License

See [LICENSE](LICENSE.md).

Third-party code:

- `effects/gtao.wgsl` ports Intel's [XeGTAO](https://github.com/GameTechDev/XeGTAO) (MIT).
- `src/sss.rs` ports Bend Studio's [screen space shadows](https://www.bendstudio.com/blog/inside-bend-screen-space-shadows/)
  (Copyright 2023 Sony Interactive Entertainment, Apache License 2.0).

---

<sub>wulkanium is an unofficial fan project and is not affiliated with, endorsed by or sponsored
by Valve, Facepunch Studios, NVIDIA or the wgpu project. Source, Garry's Mod, DLSS and all
related names and logos are trademarks of their respective owners.</sub>
