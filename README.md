<p align="center"><img src="logo.png" alt="wulkanium logo" width="160"></p>

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

## Keys

| Key | Action |
|---|---|
| Insert | Overlay (fps, frame stats) |
| Home | Log one frame of draws (capture) |
| End | DLAA on/off (DLSS builds) |
| Page Up / Page Down | DLSS input view / jitter pattern |
| Scroll Lock | Shadow map view |

## Tests

```sh
SHADERS_DIR=/path/to/extracted/shaders/fxc cargo test --release
```

`SHADERS_DIR` holds `.vcs` files from `garrysmod_dir.vpk` (`shaders/fxc`); without it the
shader tests are skipped.

## License

Copyright 2026 Bladeworks. Licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE.md):
free for personal, hobby, research and other noncommercial use. **Commercial use of any kind is
not permitted.**

DLSS builds use NVIDIA's DLSS SDK, which is covered by NVIDIA's own license.

---

<sub>wulkanium is an unofficial fan project and is not affiliated with, endorsed by or sponsored
by Valve, Facepunch Studios, NVIDIA or the wgpu project. Source, Garry's Mod, DLSS and all
related names and logos are trademarks of their respective owners.</sub>
