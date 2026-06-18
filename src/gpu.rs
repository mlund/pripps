// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

//! GPU (wgpu/WGSL) implementation of the [`Direct`](crate::IntensityScheme::Direct)
//! Fourier transform, for the vacuum path only.
//!
//! The kernel evaluates, per reciprocal-lattice q-vector,
//!
//! ```text
//!     F(q) = Σ_i f_i · exp(i q·r_i),    I(q) = |F(q)|²
//! ```
//!
//! exactly as the CPU [`DirectTransform`](crate::explicit) does, but on the
//! GPU in `f32`. The structure-factor sum has no catastrophic cancellation, so
//! `f32` keeps the relative error well below experimental SAXS noise (see
//! `WGPU-REPORT.md`); a golden-reference test pins it against the CPU path.
//!
//! The win is **trajectory averaging**: the q-vectors and atom ids stay
//! resident on the device across frames, and only the atom positions (and the
//! per-frame box / form-factor table) are re-uploaded each frame.

use std::sync::Mutex;

use wgpu::util::DeviceExt;

use crate::explicit::{average_duplicates, directions};
use crate::formfactor::FormFactor;
use crate::{Error, FormFactorMap, Intensity, IntensityCalculator, Result, Structure, Vector3};

/// One workgroup reduces over all atoms for a single q-vector. The thread
/// count per workgroup is fixed at 256 in the WGSL `@workgroup_size` and the
/// matching shared-memory array sizes and reduction stride below.
const SHADER: &str = r#"
struct Params {
    inv_box: vec4<f32>,   // 1/box_sides; .xyz used
    n_atoms: u32,
    n_q: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> directions: array<vec4<f32>>; // n_q (.xyz used)
@group(0) @binding(2) var<storage, read> positions: array<vec4<f32>>;  // n_atoms (.xyz used)
@group(0) @binding(3) var<storage, read> atom_ids: array<u32>;         // n_atoms
@group(0) @binding(4) var<storage, read> ff_table: array<f32>;         // n_kinds * n_q (kind-major)
@group(0) @binding(5) var<storage, read_write> out_iq: array<f32>;     // n_q

var<workgroup> sh_re: array<f32, 256>;
var<workgroup> sh_im: array<f32, 256>;

@compute @workgroup_size(256)
fn structure_factor(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let qi = wg.x;
    let q = directions[qi].xyz * params.inv_box.xyz;

    // Strided walk over atoms; each thread accumulates its own partial sum.
    var re = 0.0;
    var im = 0.0;
    var a = lid.x;
    loop {
        if (a >= params.n_atoms) { break; }
        let r = positions[a].xyz;
        let f = ff_table[atom_ids[a] * params.n_q + qi];
        let qr = dot(q, r);
        re = re + f * cos(qr);
        im = im + f * sin(qr);
        a = a + 256u;
    }
    sh_re[lid.x] = re;
    sh_im[lid.x] = im;
    workgroupBarrier();

    // Shared-memory tree reduction.
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) {
            sh_re[lid.x] = sh_re[lid.x] + sh_re[lid.x + stride];
            sh_im[lid.x] = sh_im[lid.x] + sh_im[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid.x == 0u) {
        let rr = sh_re[0];
        let ii = sh_im[0];
        out_iq[qi] = rr * rr + ii * ii;
    }
}
"#;

/// Per-frame uniform block. `#[repr(C)]` with explicit padding so the layout
/// matches the WGSL `Params` struct (vec4 + two u32 + two pad u32 = 32 bytes,
/// the size rounding required by the vec4's 16-byte alignment).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    inv_box: [f32; 4],
    n_atoms: u32,
    n_q: u32,
    _pad: [u32; 2],
}

/// GPU device context plus the static, structure-independent resources
/// (q-directions, pipeline). Built once in [`GpuDirectTransform::new`].
struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// q-directions as `vec4<f32>` (xyz used); divided by the box in-shader.
    dir_buf: wgpu::Buffer,
    n_q: u32,
}

/// Structure-sized resources, (re)built when the atom count first becomes
/// known or changes. Positions, the form-factor table, and the box are
/// rewritten every frame; ids stay fixed across a trajectory.
struct FrameState {
    n_atoms: u32,
    pos_buf: wgpu::Buffer,
    ff_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// GPU Direct Fourier transform (vacuum path). Implements
/// [`IntensityCalculator`] and slots in alongside the CPU
/// [`DirectTransform`](crate::explicit::DirectTransform).
pub(crate) struct GpuDirectTransform {
    formfactors: FormFactorMap,
    /// q-directions in host (f64) form, for per-frame `|q|` and box division.
    directions: Vec<Vector3>,
    ctx: GpuContext,
    /// Lazily built on the first frame; the trait is `&self`, so interior
    /// mutability is required. The trajectory loop is serial, so contention
    /// is nil.
    frame: Mutex<Option<FrameState>>,
}

impl GpuDirectTransform {
    /// Initialise wgpu and compile the kernel. Returns `Err` if no suitable
    /// GPU adapter is available, so the caller can fall back to the CPU path.
    pub(crate) fn new(formfactors: FormFactorMap, pmax: usize) -> Result<Self> {
        let directions = directions(pmax);
        let n_q = directions.len() as u32;

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| Error::internal(format!("no GPU adapter available: {e}")))?;

        log::info!("GPU adapter: {}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pripps-direct"),
            ..Default::default()
        }))
        .map_err(|e| Error::internal(format!("failed to create GPU device: {e}")))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("direct-structure-factor"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("direct-bind-layout"),
            entries: &bind_layout_entries(),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("direct-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("direct-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("structure_factor"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Static q-direction buffer (xyz packed into vec4 for alignment).
        let dir_data: Vec<[f32; 4]> = directions
            .iter()
            .map(|d| [d.x as f32, d.y as f32, d.z as f32, 0.0])
            .collect();
        let dir_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("directions"),
            contents: bytemuck::cast_slice(&dir_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Ok(Self {
            formfactors,
            directions,
            ctx: GpuContext {
                device,
                queue,
                pipeline,
                layout,
                dir_buf,
                n_q,
            },
            frame: Mutex::new(None),
        })
    }
}

impl IntensityCalculator for GpuDirectTransform {
    fn calc_intensities(&self, structure: &Structure) -> Result<Intensity> {
        let box_sides = structure.box_sides.ok_or_else(|| {
            Error::usage("direct Fourier transform requires box side lengths on the structure")
        })?;

        // |q| for every direction; a pure function of the (per-frame) box.
        let qmags: Vec<f64> = self
            .directions
            .iter()
            .map(|d| d.component_div(&box_sides).norm())
            .collect();

        // Degenerate sizes would create zero-length GPU buffers (a hard wgpu
        // validation error), so handle them on the host exactly as the CPU
        // path does: no q-vectors → empty curve; no atoms → all-zero I(q).
        if qmags.is_empty() {
            return Ok(Intensity::default());
        }
        if structure.pos.is_empty() {
            let table = qmags.into_iter().map(|q| (q, 0.0)).collect();
            let (q, total): (Vec<f64>, Vec<f64>) = average_duplicates(table).into_iter().unzip();
            return Ok(Intensity {
                q,
                total,
                ..Default::default()
            });
        }

        let n_atoms = structure.pos.len() as u32;
        let n_kinds = self.formfactors.len() as u32;
        let ctx = &self.ctx;

        // (Re)build the structure-sized buffers if this is the first frame or
        // the atom count changed (it doesn't within a trajectory).
        let mut guard = self.frame.lock().map_err(|_| {
            Error::internal("GPU frame state mutex was poisoned by an earlier panic")
        })?;
        if guard.as_ref().map(|f| f.n_atoms) != Some(n_atoms) {
            *guard = Some(ctx.build_frame_state(structure, n_kinds));
        }
        let frame = guard
            .as_ref()
            .ok_or_else(|| Error::internal("GPU frame state missing after initialization"))?;

        // Per-frame host work: form-factor table, packed positions, box.
        let ff_table = self.form_factor_table(&qmags)?;

        let positions: Vec<[f32; 4]> = structure
            .pos
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32, 0.0])
            .collect();

        let params = Params {
            inv_box: [
                (1.0 / box_sides.x) as f32,
                (1.0 / box_sides.y) as f32,
                (1.0 / box_sides.z) as f32,
                0.0,
            ],
            n_atoms,
            n_q: ctx.n_q,
            _pad: [0, 0],
        };

        ctx.queue
            .write_buffer(&frame.pos_buf, 0, bytemuck::cast_slice(&positions));
        ctx.queue
            .write_buffer(&frame.ff_buf, 0, bytemuck::cast_slice(&ff_table));
        ctx.queue
            .write_buffer(&frame.params_buf, 0, bytemuck::bytes_of(&params));

        let intensities = ctx.dispatch(frame)?;

        // Pair each |q| with its I(q) and average duplicate magnitudes, exactly
        // as the CPU vacuum path does.
        let table: Vec<(f64, f64)> = qmags
            .into_iter()
            .zip(intensities.into_iter().map(f64::from))
            .collect();
        let (q, total): (Vec<f64>, Vec<f64>) = average_duplicates(table).into_iter().unzip();

        Ok(Intensity {
            q,
            total,
            ..Default::default()
        })
    }
}

impl GpuDirectTransform {
    /// Build the `[n_kinds × n_q]` (kind-major) form-factor table in `f32`,
    /// evaluating each kind's form factor at every q-magnitude.
    fn form_factor_table(&self, qmags: &[f64]) -> Result<Vec<f32>> {
        let ffs: Vec<&FormFactor> = self.formfactors.form_factors().collect();
        let mut table = Vec::with_capacity(ffs.len() * qmags.len());
        for ff in &ffs {
            for &q in qmags {
                table.push(ff.eval(q)? as f32);
            }
        }
        Ok(table)
    }
}

impl GpuContext {
    /// Allocate the structure-sized buffers and the bind group. `atom_ids` is
    /// uploaded once here (constant across a trajectory); positions, the
    /// form-factor table, and the box are written per frame.
    fn build_frame_state(&self, structure: &Structure, n_kinds: u32) -> FrameState {
        let n_atoms = structure.pos.len() as u32;
        let device = &self.device;

        let ids: Vec<u32> = structure.ids.iter().map(|&id| id as u32).collect();
        let ids_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("atom-ids"),
            contents: bytemuck::cast_slice(&ids),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let pos_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("positions"),
            size: u64::from(n_atoms) * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ff_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ff-table"),
            size: u64::from(n_kinds) * u64::from(self.n_q) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_size = u64::from(self.n_q) * 4;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out-iq"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("direct-bind-group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.dir_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ids_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ff_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        FrameState {
            n_atoms,
            pos_buf,
            ff_buf,
            params_buf,
            out_buf,
            read_buf,
            bind_group,
        }
    }

    /// Dispatch one workgroup per q-vector, copy the result to the readback
    /// buffer, and block until it is host-visible. Returns I(q) per q-vector.
    ///
    /// Any device-side failure (lost device, failed map) is returned as an
    /// [`Error`] rather than panicking, so a transient GPU fault mid-trajectory
    /// surfaces as a recoverable error instead of aborting the process.
    fn dispatch(&self, frame: &FrameState) -> Result<Vec<f32>> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("direct-encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("direct-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &frame.bind_group, &[]);
            pass.dispatch_workgroups(self.n_q, 1, 1);
        }
        let out_size = u64::from(self.n_q) * 4;
        encoder.copy_buffer_to_buffer(&frame.out_buf, 0, &frame.read_buf, 0, out_size);
        self.queue.submit([encoder.finish()]);

        let slice = frame.read_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::internal(format!("GPU poll failed: {e}")))?;
        rx.recv()
            .map_err(|_| Error::internal("GPU readback callback was dropped"))?
            .map_err(|e| Error::internal(format!("GPU buffer map failed: {e}")))?;

        let data = slice.get_mapped_range();
        let out = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        frame.read_buf.unmap();
        Ok(out)
    }
}

/// Bind-group layout: one uniform (params), four read-only storage inputs,
/// one read-write storage output.
fn bind_layout_entries() -> [wgpu::BindGroupLayoutEntry; 6] {
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        storage(1, true),
        storage(2, true),
        storage(3, true),
        storage(4, true),
        storage(5, false),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pripps;
    use crate::explicit::DirectTransform;

    /// The GPU (f32) vacuum transform must agree with the CPU (f64)
    /// `DirectTransform` to within SAXS-grade tolerance. Skips gracefully when
    /// no GPU adapter is available (e.g. headless CI), so the suite still passes.
    #[test]
    fn gpu_matches_cpu_direct_vacuum() {
        let yaml = std::path::Path::new("assets/poly_amino_acid.yaml");
        let xyz = std::path::Path::new("tests/lysozyme/4lzt.xyz");
        let pripps = Pripps::from_yaml(yaml).unwrap();

        let gpu = match GpuDirectTransform::new(pripps.formfactors.clone(), 4) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("skipping GPU golden-reference test: {e}");
                return;
            }
        };
        let cpu = DirectTransform::new(pripps.formfactors.clone(), 4, None);

        let mut structure = Structure::from_file(xyz, &pripps.atomkinds()).unwrap();
        structure.box_sides = Some(Vector3::new(200.0, 200.0, 200.0));

        let cpu_i = cpu.calc_intensities(&structure).unwrap();
        let gpu_i = gpu.calc_intensities(&structure).unwrap();

        assert_eq!(cpu_i.q, gpu_i.q, "q grids must match exactly");
        for i in 0..cpu_i.total.len() {
            let (a, b) = (cpu_i.total[i], gpu_i.total[i]);
            let rel = (a - b).abs() / a.abs().max(1e-30);
            assert!(
                rel < 1e-3,
                "q={}: cpu={a} gpu={b} rel_err={rel}",
                cpu_i.q[i]
            );
        }
    }
}
