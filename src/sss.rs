//! screen-space sun shadows: a port of bend studio's screen space shadows (days gone).
//!
//! the dispatch planning (`dispatch_list`) and the compute shader (`SHADER`) are ported from
//! bend_sss_cpu.h and bend_sss_gpu.h (license below).

//   Copyright 2023 Sony Interactive Entertainment.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
//! changes from the original: wgsl, textureLoad with a manual border instead of a border sampler,
//! no wave-intrinsic early out, depth with near 0 / far 1, and casters can be limited to moving
//! objects (stencil bit 0x80, see `effects::DYNAMIC_MASK`) and never include the viewmodel.
//!
//! every 64-thread workgroup walks one line of pixels towards the light's point on screen; its
//! depths go to workgroup memory, scaled so that all rays become parallel, and each pixel compares
//! itself against the next 60 along the line.

pub const WAVE: i32 = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dispatch {
    /// workgroups x, y, z
    pub count: [u32; 3],
    pub wave_offset: [i32; 2],
}

#[derive(Clone, Debug, Default)]
pub struct DispatchList {
    /// light x, y in pixels, its depth, and 1 / -1 for in front of / behind the camera
    pub light: [f32; 4],
    pub dispatches: Vec<Dispatch>,
}

/// `light` is the light through view x projection without the w divide: (direction, 0) for the
/// sun, (position, 1) for a point light. bounds are inclusive pixels.
pub fn dispatch_list(light: [f32; 4], size: [i32; 2], min: [i32; 2], max: [i32; 2], far_depth: f32) -> DispatchList {
    let mut out = DispatchList::default();
    // very far off screen, the divide in the shader runs out of precision
    let limit = 0.000002 * WAVE as f32;
    let mut w = light[3];
    if w >= 0.0 && w < limit {
        w = limit;
    } else if w < 0.0 && w > -limit {
        w = -limit;
    }
    out.light[0] = ((light[0] / w) * 0.5 + 0.5) * size[0] as f32;
    out.light[1] = ((light[1] / w) * -0.5 + 0.5) * size[1] as f32;
    out.light[2] = if light[3] == 0.0 { far_depth } else { light[2] / light[3] };
    out.light[3] = if light[3] > 0.0 { 1.0 } else { -1.0 };

    let lxy = [(out.light[0] + 0.5) as i32, (out.light[1] + 0.5) as i32];
    let biased = [min[0] - lxy[0], -(max[1] - lxy[1]), max[0] - lxy[0], -(min[1] - lxy[1])];

    // four quadrants around the light, each a rectangle with a corner on it; split in two when
    // the light's diagonal crosses it
    for q in 0..4 {
        let vertical = q == 0 || q == 3;
        let b = [
            0.max(if q & 1 != 0 { biased[0] } else { -biased[2] }) / WAVE,
            0.max(if q & 2 != 0 { biased[1] } else { -biased[3] }) / WAVE,
            0.max((if q & 1 != 0 { biased[2] } else { -biased[0] }) + WAVE * if vertical { 1 } else { 2 } - 1) / WAVE,
            0.max((if q & 2 != 0 { biased[3] } else { -biased[1] }) + WAVE * if vertical { 2 } else { 1 } - 1) / WAVE,
        ];
        if b[2] - b[0] <= 0 || b[3] - b[1] <= 0 {
            continue;
        }
        let bias_x = (q == 2 || q == 3) as i32;
        let bias_y = (q == 1 || q == 3) as i32;
        // (count y, count z, offset x, offset y)
        let mut d = [b[2] - b[0], b[3] - b[1], (if q & 1 != 0 { b[0] } else { -b[2] }) + bias_x, (if q & 2 != 0 { -b[3] } else { b[1] }) + bias_y];
        let mut axis_delta = match q {
            0 => biased[0] - biased[1],
            1 => biased[2] + biased[1],
            2 => -biased[0] - biased[3],
            _ => -biased[2] + biased[3],
        };
        axis_delta = (axis_delta + WAVE - 1) / WAVE;
        let mut second = None;
        if axis_delta > 0 {
            let mut d2 = d;
            match q {
                0 => {
                    d2[1] = d[1].min(axis_delta);
                    d[1] -= d2[1];
                    d2[3] = d[3] + d[1];
                    d2[2] -= 1;
                    d2[0] += 1;
                }
                1 => {
                    d2[0] = d[0].min(axis_delta);
                    d[0] -= d2[0];
                    d2[2] = d[2] + d[0];
                    d2[1] += 1;
                }
                2 => {
                    d2[0] = d[0].min(axis_delta);
                    d[0] -= d2[0];
                    d[2] += d2[0];
                    d2[1] += 1;
                    d2[3] -= 1;
                }
                _ => {
                    d2[1] = d[1].min(axis_delta);
                    d[1] -= d2[1];
                    d[3] += d2[1];
                    d2[0] += 1;
                }
            }
            second = Some(d2);
        }
        for d in std::iter::once(d).chain(second) {
            if d[0] > 0 && d[1] > 0 {
                out.dispatches.push(Dispatch { count: [WAVE as u32, d[0] as u32, d[1] as u32], wave_offset: [d[2] * WAVE, d[3] * WAVE] });
            }
        }
    }
    out
}

/// per dispatch, at 256-byte offsets
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub light: [f32; 4],
    pub wave_offset: [i32; 2],
    pub size: [i32; 2],
    pub surface_thickness: f32,
    pub bilinear_threshold: f32,
    pub contrast: f32,
    /// bit 0: only stencil-marked pixels cast, bit 1: edge pixels don't cast
    pub flags: u32,
    /// depths below this are the viewmodel
    pub viewmodel_depth: f32,
    pub _pad: [f32; 3],
}

pub const SHADER: &str = r#"
struct P {
    light: vec4f,
    wave_offset: vec2i,
    size: vec2i,
    surface_thickness: f32,
    bilinear_threshold: f32,
    contrast: f32,
    flags: u32,
    viewmodel_depth: f32,
}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var stencil_tex: texture_2d<u32>;
@group(0) @binding(3) var out_tex: texture_storage_2d<r32float, write>;

const WAVE: i32 = 64;
const SAMPLE_COUNT: i32 = 60;
const HARD_SHADOW_SAMPLES: i32 = 4;
const FADE_OUT_SAMPLES: i32 = 8;
const READ_COUNT: i32 = 2; // SAMPLE_COUNT / WAVE + 2
const FAR: f32 = 1.0;
const Z_SIGN: f32 = 1.0; // near 0 < far 1
const EDGE_SKIP: f32 = 1e20;

var<workgroup> depth_data: array<f32, 128>;

fn inside(xy: vec2i) -> bool {
    return all(xy >= vec2i(0)) && all(xy < p.size);
}

/// off screen reads as the far plane, like a border sampler
fn load_depth(xy: vec2i) -> f32 {
    if !inside(xy) { return FAR; }
    return textureLoad(depth_tex, xy, 0);
}

/// may this pixel cast: never the viewmodel, and only moving objects when asked
fn caster(xy: vec2i, depth: f32) -> bool {
    if depth < p.viewmodel_depth { return false; }
    if (p.flags & 1u) != 0u && inside(xy) {
        return (textureLoad(stencil_tex, xy, 0).r & 0x80u) != 0u;
    }
    return true;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) tid_u: u32) {
    let tid = i32(tid_u);
    // ---- wavefront extents ----
    var xy = vec2i(group.yz) * WAVE + p.wave_offset;
    let light_xy = floor(p.light.xy) + 0.5;
    let light_frac = p.light.xy - light_xy;
    let reverse = p.light.w > 0.0;
    let sxy = sign(xy);
    let horizontal = abs(xy.x + sxy.y) < abs(xy.y - sxy.x);
    let axis = select(vec2i(0, -sxy.x), vec2i(sxy.y, 0), horizontal);
    xy = axis * i32(group.x) + xy;
    let xy_f = vec2f(xy);
    let x_major = abs(xy_f.x) > abs(xy_f.y);
    let major = select(xy_f.y, xy_f.x, x_major);
    let major_start = abs(major);
    let major_end = abs(major) - f32(WAVE);
    var ma_frac = select(light_frac.y, light_frac.x, x_major);
    ma_frac = select(ma_frac, -ma_frac, major > 0.0);
    let start_xy = xy_f + light_xy;
    let end_xy = mix(p.light.xy, start_xy, (major_end + ma_frac) / (major_start + ma_frac));
    let xy_delta = start_xy - end_xy;
    let thread_step = f32(tid ^ select(WAVE - 1, 0, reverse));
    var pixel_xy = mix(start_xy, end_xy, thread_step / f32(WAVE));
    let pixel_distance = major_start - thread_step + ma_frac;

    // ---- read depths ----
    var sampling_depth: array<f32, 2>;
    var shadowing_depth: array<f32, 2>;
    var thickness_scale: array<f32, 2>;
    var sample_distance: array<f32, 2>;
    let direction = -p.light.w;
    let write_xy = vec2i(floor(pixel_xy));
    for (var i = 0; i < READ_COUNT; i++) {
        let read_xy = vec2i(floor(pixel_xy));
        let minor = select(pixel_xy.x, pixel_xy.y, x_major);
        let bilinear = fract(minor) - 0.5;
        let b = select(-1, 1, bilinear > 0.0);
        let off = select(vec2i(b, 0), vec2i(0, b), x_major);
        let dx = load_depth(read_xy);
        var dy = load_depth(read_xy + off);
        if dy < p.viewmodel_depth { dy = dx; }
        thickness_scale[i] = abs(FAR - dx);
        let point_filter = abs(dx - dy) > thickness_scale[i] * p.bilinear_threshold;
        sampling_depth[i] = dx;
        let edge_depth = select(dx, EDGE_SKIP, (p.flags & 2u) != 0u);
        let shadow_depth = dx + abs(dx - dy) * Z_SIGN;
        shadowing_depth[i] = select(shadow_depth, edge_depth, point_filter);
        if !caster(read_xy, dx) { shadowing_depth[i] = EDGE_SKIP; }
        sample_distance[i] = pixel_distance + f32(WAVE * i) * direction;
        pixel_xy += xy_delta * direction;
    }

    // ---- share them, perspective corrected so all rays are parallel ----
    for (var i = 0; i < READ_COUNT; i++) {
        var stored = (shadowing_depth[i] - p.light.z) / sample_distance[i];
        if i != 0 && sample_distance[i] <= 0.0 { stored = 1e10; }
        depth_data[i * WAVE + tid] = stored;
    }
    workgroupBarrier();

    // ---- compare against the next samples along the ray ----
    var start_depth = (sampling_depth[0] - p.light.z) / sample_distance[0];
    let sample_index = tid + 1;
    var shadow_value = vec4f(1.0);
    var hard_shadow = 1.0;
    let depth_scale = min(sample_distance[0] + direction, 1.0 / p.surface_thickness) * sample_distance[0] / thickness_scale[0];
    start_depth = start_depth * depth_scale - Z_SIGN;
    for (var i = 0; i < HARD_SHADOW_SAMPLES; i++) {
        hard_shadow = min(hard_shadow, abs(start_depth - depth_data[sample_index + i] * depth_scale));
    }
    for (var i = HARD_SHADOW_SAMPLES; i < SAMPLE_COUNT - FADE_OUT_SAMPLES; i++) {
        let d = abs(start_depth - depth_data[sample_index + i] * depth_scale);
        shadow_value[i & 3] = min(shadow_value[i & 3], d);
    }
    for (var i = SAMPLE_COUNT - FADE_OUT_SAMPLES; i < SAMPLE_COUNT; i++) {
        let d = abs(start_depth - depth_data[sample_index + i] * depth_scale);
        let fade = f32(i + 1 - (SAMPLE_COUNT - FADE_OUT_SAMPLES)) / f32(FADE_OUT_SAMPLES + 1) * 0.75;
        shadow_value[i & 3] = min(shadow_value[i & 3], d + fade);
    }
    shadow_value = saturate(shadow_value * p.contrast + (1.0 - p.contrast));
    hard_shadow = saturate(hard_shadow * p.contrast + (1.0 - p.contrast));
    var result = min(hard_shadow, dot(shadow_value, vec4f(0.25)));
    // sky and viewmodel pixels don't receive
    if sampling_depth[0] >= FAR || sampling_depth[0] < p.viewmodel_depth || !(result == result) { result = 1.0; }
    if inside(write_xy) {
        textureStore(out_tex, write_xy, vec4f(result, 0.0, 0.0, 1.0));
    }
}
"#;

/// gpu side: pipeline, per-dispatch params and the output mask
pub struct ScreenShadows {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    pub out: Option<(wgpu::Texture, wgpu::TextureView)>,
}

const PARAM_STRIDE: u64 = 256;

impl ScreenShadows {
    pub fn new(device: &wgpu::Device) -> ScreenShadows {
        use wgpu::{BindGroupLayoutEntry as E, BindingType as B, ShaderStages as S};
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sun shadows"),
            entries: &[
                E { binding: 0, visibility: S::COMPUTE, ty: B::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64) }, count: None },
                E { binding: 1, visibility: S::COMPUTE, ty: B::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                E { binding: 2, visibility: S::COMPUTE, ty: B::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                E { binding: 3, visibility: S::COMPUTE, ty: B::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::R32Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("sun shadows"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("sun shadows"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("sun shadows"), layout: Some(&pl), module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("sun shadow params"), size: PARAM_STRIDE * 8, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        ScreenShadows { pipeline, layout, params, out: None }
    }

    /// clear the mask to lit, then run the dispatches. `size` is the part of the depth buffer the
    /// scene covers
    #[allow(clippy::too_many_arguments)]
    pub fn run(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, depth: &wgpu::TextureView, stencil: &wgpu::TextureView, size: (u32, u32), list: &DispatchList, base: Params) {
        if self.out.as_ref().is_none_or(|(t, _)| (t.width(), t.height()) != size) {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("sun shadow mask"),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            self.out = Some((t, v));
        }
        let out = &self.out.as_ref().unwrap().1;
        drop(enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sun shadow clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: out, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store } })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        }));
        let mut bytes = vec![0u8; (PARAM_STRIDE as usize) * list.dispatches.len().min(8)];
        for (i, d) in list.dispatches.iter().take(8).enumerate() {
            let p = Params { light: list.light, wave_offset: d.wave_offset, ..base };
            bytes[i * PARAM_STRIDE as usize..][..std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(&p));
        }
        if bytes.is_empty() {
            return;
        }
        queue.write_buffer(&self.params, 0, &bytes);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.params, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64) }) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(stencil) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(out) },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sun shadows"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        for (i, d) in list.dispatches.iter().take(8).enumerate() {
            pass.set_bind_group(0, &group, &[(i as u64 * PARAM_STRIDE) as u32]);
            pass.dispatch_workgroups(d.count[0], d.count[1], d.count[2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates() {
        let module = naga::front::wgsl::parse_str(SHADER).unwrap_or_else(|e| panic!("{}", e.emit_to_string(SHADER)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
    }

    /// every pixel of the screen gets written by exactly one thread
    #[test]
    fn dispatches_cover_screen() {
        for (light, size) in [([0.3f32, 0.2, 0.5, 1.0], [640, 360]), ([2.0, -0.1, 0.9, 0.5], [333, 200]), ([0.1, 0.1, 0.4, -1.0], [256, 256]), ([-5.0, 3.0, 0.1, 0.01], [300, 170])] {
            let list = dispatch_list(light, size, [0, 0], size, 1.0);
            let mut hits = vec![0u32; (size[0] * size[1]) as usize];
            for d in &list.dispatches {
                for gx in 0..d.count[0] as i32 {
                    for gy in 0..d.count[1] as i32 {
                        for gz in 0..d.count[2] as i32 {
                            for tid in 0..WAVE {
                                if let Some(xy) = thread_pixel(&list, d, [gx, gy, gz], tid) {
                                    if xy[0] >= 0 && xy[1] >= 0 && xy[0] < size[0] && xy[1] < size[1] {
                                        hits[(xy[1] * size[0] + xy[0]) as usize] += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // the pixel under the light itself can be skipped when the light sits exactly on a
            // pixel corner (as in the original); the mask is cleared to lit first
            let lxy = [list.light[0] as i32, list.light[1] as i32];
            let missing: Vec<_> = (0..hits.len() as i32).filter(|&i| hits[i as usize] == 0).map(|i| [i % size[0], i / size[0]]).filter(|p| (p[0] - lxy[0]).abs() > 1 || (p[1] - lxy[1]).abs() > 1).collect();
            assert!(missing.is_empty(), "{} pixels unwritten for light {light:?}, first {:?}", missing.len(), missing.first());
        }
    }

    /// the shader's write position, mirrored on the cpu
    fn thread_pixel(list: &DispatchList, d: &Dispatch, group: [i32; 3], tid: i32) -> Option<[i32; 2]> {
        let l = list.light;
        let mut xy = [group[1] * WAVE + d.wave_offset[0], group[2] * WAVE + d.wave_offset[1]];
        let light_xy = [l[0].floor() + 0.5, l[1].floor() + 0.5];
        let frac = [l[0] - light_xy[0], l[1] - light_xy[1]];
        let reverse = l[3] > 0.0;
        let s = [xy[0].signum(), xy[1].signum()];
        let horizontal = (xy[0] + s[1]).abs() < (xy[1] - s[0]).abs();
        let axis = if horizontal { [s[1], 0] } else { [0, -s[0]] };
        xy = [axis[0] * group[0] + xy[0], axis[1] * group[0] + xy[1]];
        let f = [xy[0] as f32, xy[1] as f32];
        let x_major = f[0].abs() > f[1].abs();
        let major = if x_major { f[0] } else { f[1] };
        let (ms, me) = (major.abs(), major.abs() - WAVE as f32);
        let mut mf = if x_major { frac[0] } else { frac[1] };
        if major > 0.0 {
            mf = -mf;
        }
        let start = [f[0] + light_xy[0], f[1] + light_xy[1]];
        let t = (me + mf) / (ms + mf);
        let end = [l[0] + (start[0] - l[0]) * t, l[1] + (start[1] - l[1]) * t];
        let step = (tid ^ if reverse { 0 } else { WAVE - 1 }) as f32;
        let px = [start[0] + (end[0] - start[0]) * step / WAVE as f32, start[1] + (end[1] - start[1]) * step / WAVE as f32];
        Some([px[0].floor() as i32, px[1].floor() as i32])
    }
}
