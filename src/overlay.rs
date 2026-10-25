//! egui overlay on top of the frame (Insert toggles)

use std::collections::VecDeque;
use std::time::Instant;

pub struct Stats {
    /// renderer cpu time before submit
    pub cpu_ms: f32,
    /// recording on the game thread
    pub record_ms: f32,
    pub dlss: String,
    /// from timestamp queries, 0 = unavailable
    pub gpu_ms: f32,
    pub vertex_bytes: usize,
    pub indices: usize,
    pub vs_blocks: usize,
    pub ps_blocks: usize,
    pub copies: usize,
    pub mesh_uploads: usize,
    pub mesh_upload_bytes: usize,
    pub tex_uploads: usize,
    pub tex_bytes: usize,
    pub new_pipelines: usize,
    pub bind_groups: usize,
    pub targets_drawn: usize,
    pub skipped: usize,
    pub no_shader: usize,
    pub adapter: String,
    pub backend: String,
    pub present_mode: String,
    pub size: (u32, u32),
    pub draws: usize,
    pub failed_draws: usize,
    pub passes: usize,
    pub pipelines: usize,
    pub shader_modules: usize,
    pub textures: usize,
    pub meshes: usize,
    pub targets: usize,
}

pub struct Overlay {
    ctx: egui::Context,
    renderer: egui_wgpu::Renderer,
    start: Instant,
    last: Instant,
    times: VecDeque<f32>,
    visible: bool,
    key_down: bool,
}

impl Overlay {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Overlay {
        let ctx = egui::Context::default();
        ctx.set_visuals(egui::Visuals::dark());
        let renderer = egui_wgpu::Renderer::new(device, format, egui_wgpu::RendererOptions::default());
        let now = Instant::now();
        Overlay { ctx, renderer, start: now, last: now, times: VecDeque::new(), visible: true, key_down: false }
    }

    fn poll_toggle(&mut self) {
        #[cfg(windows)]
        {
            use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_INSERT};
            let down = unsafe { GetAsyncKeyState(VK_INSERT as i32) } as u16 & 0x8000 != 0;
            if down && !self.key_down {
                self.visible = !self.visible;
            }
            self.key_down = down;
        }
    }

    pub fn draw(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, stats: &Stats) {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32() * 1000.0;
        self.last = now;
        self.times.push_back(dt);
        while self.times.len() > 240 {
            self.times.pop_front();
        }
        self.poll_toggle();
        if !self.visible {
            return;
        }
        let (w, h) = stats.size;
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(w as f32, h as f32))),
            time: Some((now - self.start).as_secs_f64()),
            ..Default::default()
        };
        let times = &self.times;
        let avg = times.iter().sum::<f32>() / times.len().max(1) as f32;
        let worst = times.iter().copied().fold(0.0f32, f32::max);
        let output = self.ctx.run_pass(input, |ctx| {
            egui::Window::new("wgpu").title_bar(false).resizable(false).anchor(egui::Align2::RIGHT_TOP, egui::vec2(-8.0, 8.0)).show(ctx, |ui| {
                ui.label(egui::RichText::new(format!("shaderapiwgpu · wgpu 30 · {}", stats.backend)).strong().color(egui::Color32::from_rgb(120, 200, 255)));
                ui.label(&stats.adapter);
                ui.label(format!("{:.0} fps · {avg:.2} ms avg · {worst:.1} ms worst · {}", 1000.0 / avg.max(0.001), stats.present_mode));
                // last 240 frames, scaled to max(33 ms, worst)
                let (rect, _) = ui.allocate_exact_size(egui::vec2(260.0, 48.0), egui::Sense::hover());
                let p = ui.painter();
                p.rect_filled(rect, 2.0, egui::Color32::from_black_alpha(120));
                let top = worst.max(33.3);
                let y = |t: f32| rect.bottom() - (t / top).min(1.0) * rect.height();
                for line in [16.7f32, 33.3] {
                    p.hline(rect.x_range(), y(line), egui::Stroke::new(1.0, egui::Color32::from_white_alpha(30)));
                }
                let pts: Vec<egui::Pos2> = times.iter().enumerate().map(|(i, t)| egui::pos2(rect.left() + i as f32 * rect.width() / 239.0, y(*t))).collect();
                p.add(egui::Shape::line(pts, egui::Stroke::new(1.0, egui::Color32::from_rgb(120, 220, 120))));
                ui.label(format!("record {:.2} ms · submit {:.2} ms · GPU {:.2} ms", stats.record_ms, stats.cpu_ms, stats.gpu_ms));
                let mib = |b: usize| b as f64 / 1048576.0;
                let section = |ui: &mut egui::Ui, id: &str, rows: Vec<(&str, String)>| {
                    ui.separator();
                    egui::Grid::new(id).num_columns(2).spacing([16.0, 2.0]).show(ui, |ui| {
                        for (k, v) in rows {
                            ui.label(k);
                            ui.label(v);
                            ui.end_row();
                        }
                    });
                };
                let mut draws = stats.draws.to_string();
                if stats.failed_draws > 0 {
                    draws += &format!(" ({} no pipeline)", stats.failed_draws);
                }
                if stats.skipped + stats.no_shader > 0 {
                    draws += &format!(" (+{} skipped, {} without shaders)", stats.skipped, stats.no_shader);
                }
                section(ui, "frame", vec![
                    ("resolution", format!("{} × {}", stats.size.0, stats.size.1)),
                    ("draws", draws),
                    ("passes / targets", format!("{} / {}", stats.passes, stats.targets_drawn)),
                    ("copies", stats.copies.to_string()),
                    ("dynamic geometry", format!("{:.2} MiB, {} indices", mib(stats.vertex_bytes), stats.indices)),
                    ("constant blocks", format!("{} vs / {} ps ({:.2} MiB)", stats.vs_blocks, stats.ps_blocks, mib(stats.vs_blocks * crate::api::VS_BLOCK + stats.ps_blocks * crate::api::PS_BLOCK))),
                    ("mesh uploads", format!("{} ({:.2} MiB)", stats.mesh_uploads, mib(stats.mesh_upload_bytes))),
                    ("texture uploads", format!("{} ({:.2} MiB)", stats.tex_uploads, mib(stats.tex_bytes))),
                    ("new pipelines", stats.new_pipelines.to_string()),
                    ("DLSS", stats.dlss.clone()),
                ]);
                section(ui, "caches", vec![
                    ("pipelines", stats.pipelines.to_string()),
                    ("shader modules", stats.shader_modules.to_string()),
                    ("bind groups", stats.bind_groups.to_string()),
                    ("textures", stats.textures.to_string()),
                    ("static meshes", stats.meshes.to_string()),
                    ("render targets", stats.targets.to_string()),
                ]);
            });
        });
        let ppp = output.pixels_per_point;
        let jobs = self.ctx.tessellate(output.shapes, ppp);
        for (id, deltas) in &output.textures_delta.set {
            for delta in deltas.iter() {
                self.renderer.update_texture(device, queue, *id, delta);
            }
        }
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: ppp };
        self.renderer.update_buffers(device, queue, encoder, &jobs, &screen);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: target, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
        }
        for id in &output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}
