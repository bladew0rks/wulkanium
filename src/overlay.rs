//! egui overlay on top of the frame (Insert toggles)

use std::collections::VecDeque;
use std::time::Instant;

pub struct Stats {
    /// renderer cpu time before submit
    pub cpu_ms: f32,
    /// recording on the game thread
    pub record_ms: f32,
    pub dlss: String,
    /// built-in shader draws this frame: (shader, draws, distinct static combos)
    pub builtin_draws: Vec<(String, usize, usize)>,
    /// (shader, our materials, stock materials)
    pub materials: Vec<(String, usize, usize)>,
    /// (effect, on, passes)
    pub effects: Vec<(String, bool, usize)>,
    /// from timestamp queries, 0 = unavailable
    pub gpu_ms: f32,
    pub vertex_bytes: usize,
    pub indices: usize,
    pub vs_blocks: usize,
    pub ps_blocks: usize,
    pub const_bytes: usize,
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

/// short messages shown top left for a few seconds, overlay open or not: (text, is error, since)
static NOTICES: std::sync::Mutex<Vec<(String, bool, Instant)>> = std::sync::Mutex::new(Vec::new());
const NOTICE_SECS: f32 = 5.0;

pub fn notify(text: impl Into<String>, error: bool) {
    let mut n = NOTICES.lock().unwrap_or_else(|e| e.into_inner());
    n.push((text.into(), error, Instant::now()));
    let len = n.len();
    if len > 8 {
        n.drain(..len - 8);
    }
}

pub struct Overlay {
    ctx: egui::Context,
    renderer: egui_wgpu::Renderer,
    start: Instant,
    last: Instant,
    times: VecDeque<f32>,
    visible: bool,
    #[allow(dead_code)]
    key_down: bool,
    /// egui ids of the buffer previews and the texture generation they were registered for
    previews: Vec<egui::TextureId>,
    previews_generation: u32,
}

const BG: egui::Color32 = egui::Color32::from_rgba_premultiplied(8, 9, 11, 225);
const BORDER: egui::Color32 = egui::Color32::from_rgb(52, 56, 62);
const ACCENT: egui::Color32 = egui::Color32::from_rgb(120, 200, 255);
const DIM: egui::Color32 = egui::Color32::from_rgb(140, 146, 154);

/// square, borderline, no shadow
fn panel() -> egui::Frame {
    egui::Frame::new().fill(BG).stroke(egui::Stroke::new(1.0, BORDER)).inner_margin(8.0)
}

/// monospace everywhere, a little smaller and tighter than egui's defaults
fn style(ctx: &egui::Context) {
    use egui::{FontFamily::Monospace, FontId, TextStyle};
    ctx.set_visuals(egui::Visuals::dark());
    let mut st = (*ctx.global_style()).clone();
    st.text_styles = [
        (TextStyle::Small, FontId::new(10.0, Monospace)),
        (TextStyle::Body, FontId::new(12.0, Monospace)),
        (TextStyle::Button, FontId::new(12.0, Monospace)),
        (TextStyle::Heading, FontId::new(13.0, Monospace)),
        (TextStyle::Monospace, FontId::new(12.0, Monospace)),
    ]
    .into();
    st.spacing.item_spacing = egui::vec2(6.0, 2.0);
    st.visuals.window_shadow = egui::Shadow::NONE;
    st.visuals.popup_shadow = egui::Shadow::NONE;
    st.visuals.window_corner_radius = egui::CornerRadius::ZERO;
    st.visuals.widgets.noninteractive.fg_stroke.color = egui::Color32::from_rgb(205, 210, 216);
    st.visuals.widgets.noninteractive.bg_stroke.color = BORDER;
    ctx.set_global_style(st);
}

impl Overlay {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Overlay {
        let ctx = egui::Context::default();
        style(&ctx);
        let renderer = egui_wgpu::Renderer::new(device, format, egui_wgpu::RendererOptions::default());
        let now = Instant::now();
        Overlay { ctx, renderer, start: now, last: now, times: VecDeque::new(), visible: true, key_down: false, previews: Vec::new(), previews_generation: 0 }
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

    pub fn draw(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView, stats: &Stats, previews: Option<(u32, Vec<&wgpu::TextureView>)>) {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32() * 1000.0;
        self.last = now;
        self.times.push_back(dt);
        while self.times.len() > 240 {
            self.times.pop_front();
        }
        self.poll_toggle();
        crate::effects::set_previews(self.visible);
        // (re)register the preview textures when the effects recreated them
        if let Some((generation, views)) = &previews {
            if *generation != self.previews_generation || self.previews.len() != views.len() {
                for (i, v) in views.iter().enumerate() {
                    match self.previews.get(i) {
                        Some(&id) => self.renderer.update_egui_texture_from_wgpu_texture(device, v, wgpu::FilterMode::Linear, id),
                        None => self.previews.push(self.renderer.register_native_texture(device, v, wgpu::FilterMode::Linear)),
                    }
                }
                self.previews_generation = *generation;
            }
        }
        let preview_ids: Vec<(egui::TextureId, [f32; 2])> = match &previews {
            Some((_, views)) => self.previews.iter().zip(views).map(|(id, v)| (*id, [v.texture().width() as f32, v.texture().height() as f32])).collect(),
            None => Vec::new(),
        };
        let notices: Vec<(String, bool, f32)> = {
            let mut n = NOTICES.lock().unwrap_or_else(|e| e.into_inner());
            n.retain(|x| x.2.elapsed().as_secs_f32() < NOTICE_SECS);
            n.iter().map(|x| (x.0.clone(), x.1, x.2.elapsed().as_secs_f32())).collect()
        };
        if !self.visible && notices.is_empty() {
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
        let visible = self.visible;
        let output = self.ctx.run_pass(input, |ctx| {
            // notices go under the buffer previews when those are up
            let notice_y = if visible && !preview_ids.is_empty() { 8.0 + preview_ids[0].1[1] + 40.0 } else { 8.0 };
            if !notices.is_empty() {
                egui::Area::new(egui::Id::unique("notices")).anchor(egui::Align2::LEFT_TOP, egui::vec2(8.0, notice_y)).show(ctx, |ui| {
                    egui::Frame::new().fill(BG).inner_margin(6.0).show(ui, |ui| {
                        for (text, error, age) in &notices {
                            // fade over the last second
                            let alpha = ((NOTICE_SECS - age).min(1.0) * 255.0) as u8;
                            let color = if *error { egui::Color32::from_rgba_unmultiplied(255, 120, 110, alpha) } else { egui::Color32::from_rgba_unmultiplied(140, 230, 140, alpha) };
                            ui.label(egui::RichText::new(text).monospace().color(color));
                        }
                    });
                });
            }
            if !visible {
                return;
            }
            if !preview_ids.is_empty() {
                egui::Area::new(egui::Id::unique("previews")).anchor(egui::Align2::LEFT_TOP, egui::vec2(8.0, 8.0)).show(ctx, |ui| {
                    panel().show(ui, |ui| {
                        ui.horizontal(|ui| {
                            for ((id, [w, h]), name) in preview_ids.iter().zip(crate::effects::PREVIEWS) {
                                ui.vertical(|ui| {
                                    ui.label(egui::RichText::new(name).color(DIM));
                                    ui.image(egui::load::SizedTexture::new(*id, egui::vec2(*w, *h)));
                                });
                            }
                        });
                    });
                });
            }
            egui::Window::new("wgpu").title_bar(false).resizable(false).frame(panel()).anchor(egui::Align2::RIGHT_TOP, egui::vec2(-8.0, 8.0)).show(ctx, |ui| {
                ui.label(egui::RichText::new(format!("{} / wgpu", stats.backend)).strong().color(ACCENT));
                ui.label(egui::RichText::new(&stats.adapter).color(DIM));
                ui.label(format!("{:.0} fps · {avg:.2} ms avg · {worst:.1} ms worst · {}", 1000.0 / avg.max(0.001), stats.present_mode));
                // last 240 frames, scaled to max(33 ms, worst)
                let (rect, _) = ui.allocate_exact_size(egui::vec2(260.0, 48.0), egui::Sense::hover());
                let p = ui.painter();
                p.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(140));
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
                            ui.label(egui::RichText::new(k).color(DIM));
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
                    ("constant blocks", format!("{} vs / {} ps ({:.2} MiB)", stats.vs_blocks, stats.ps_blocks, mib(stats.const_bytes))),
                    ("mesh uploads", format!("{} ({:.2} MiB)", stats.mesh_uploads, mib(stats.mesh_upload_bytes))),
                    ("texture uploads", format!("{} ({:.2} MiB)", stats.tex_uploads, mib(stats.tex_bytes))),
                    ("new pipelines", stats.new_pipelines.to_string()),
                    ("DLSS", stats.dlss.clone()),
                ]);
                let mut custom: Vec<(String, String)> = Vec::new();
                for (name, ours, stock) in &stats.materials {
                    let (draws, combos) = stats.builtin_draws.iter().find(|b| b.0.eq_ignore_ascii_case(name)).map_or((0, 0), |b| (b.1, b.2));
                    let mut v = format!("{ours} materials · {draws} draws · {combos} combos");
                    if *stock > 0 {
                        v += &format!(" · {stock} on stock");
                    }
                    custom.push((name.clone(), v));
                }
                for (name, on, passes) in &stats.effects {
                    custom.push((name.clone(), format!("{} · {passes} pass{}", if *on { "on" } else { "off" }, if *passes == 1 { "" } else { "es" })));
                }
                if stats.effects.is_empty() {
                    custom.push(("effects".into(), "none loaded".into()));
                }
                section(ui, "custom", custom.iter().map(|(k, v)| (k.as_str(), v.clone())).collect());
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
