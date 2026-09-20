use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

pub const MAX_N: usize = 256;

#[cuda_module]
mod kernels {
    use super::*;

    /// Edge order: (w, lo, hi). Returns true if a < b.
    #[inline(always)]
    fn edge_lt(wa: f32, la: u32, ha: u32, wb: f32, lb: u32, hb: u32) -> bool {
        if wa != wb { return wa < wb; }
        if la != lb { return la < lb; }
        ha < hb
    }

    /// One block per batch item, one thread per vertex. Prim from vertex 0.
    /// d: [B * n * n] row-major. out: [B * (n-1) * 2] as (i, j) pairs, i < j, in
    /// insertion order (host sorts by (death, i, j) afterwards).
    #[kernel]
    pub fn h0_prim(d: &[f32], n: u32, mut out: DisjointSlice<u32>) {
        static mut DIST: SharedArray<f32, MAX_N> = SharedArray::UNINIT;
        static mut PAR: SharedArray<u32, MAX_N> = SharedArray::UNINIT;
        static mut DONE: SharedArray<u32, MAX_N> = SharedArray::UNINIT;
        static mut RW: SharedArray<f32, MAX_N> = SharedArray::UNINIT;
        static mut RI: SharedArray<u32, MAX_N> = SharedArray::UNINIT;
        static mut CUR: SharedArray<u32, 1> = SharedArray::UNINIT;

        let b = thread::blockIdx_x() as usize;
        let i = thread::threadIdx_x() as usize;
        let n = n as usize;
        let base = b * n * n;
        let obase = b * (n - 1) * 2;

        unsafe {
            if i < n {
                DIST[i] = if i == 0 { 0.0 } else { d[base + i] }; // row 0
                PAR[i] = 0;
                DONE[i] = (i == 0) as u32;
            }
        }
        thread::sync_threads();

        for step in 0..(n - 1) {
            // candidates: every vertex not in the tree, keyed by its best edge
            unsafe {
                if i < n {
                    RW[i] = if DONE[i] == 1 { f32::INFINITY } else { DIST[i] };
                    RI[i] = i as u32;
                }
            }
            thread::sync_threads();

            // block min-reduce on (w, lo, hi)
            let mut s = MAX_N / 2;
            while s > 0 {
                if i < s && i + s < n {
                    unsafe {
                        let (wa, va) = (RW[i], RI[i]);
                        let (wb, vb) = (RW[i + s], RI[i + s]);
                        let (pa, pb) = (PAR[va as usize], PAR[vb as usize]);
                        let (la, ha) = (va.min(pa), va.max(pa));
                        let (lb, hb) = (vb.min(pb), vb.max(pb));
                        if edge_lt(wb, lb, hb, wa, la, ha) {
                            RW[i] = wb;
                            RI[i] = vb;
                        }
                    }
                }
                thread::sync_threads();
                s /= 2;
            }

            // thread 0 commits the winner
            if i == 0 {
                unsafe {
                    let v = RI[0];
                    let p = PAR[v as usize];
                    CUR[0] = v;
                    DONE[v as usize] = 1;
                    *out.get_unchecked_mut(obase + 2 * step) = v.min(p);
                    *out.get_unchecked_mut(obase + 2 * step + 1) = v.max(p);
                }
            }
            thread::sync_threads();

            // relax against the new vertex (row read is coalesced: d[v, i])
            unsafe {
                let v = CUR[0] as usize;
                if i < n && DONE[i] == 0 {
                    let w = d[base + v * n + i];
                    let (lo, hi) = ((i as u32).min(v as u32), (i as u32).max(v as u32));
                    let p = PAR[i];
                    let (plo, phi) = ((i as u32).min(p), (i as u32).max(p));
                    if edge_lt(w, lo, hi, DIST[i], plo, phi) {
                        DIST[i] = w;
                        PAR[i] = v as u32;
                    }
                }
            }
            thread::sync_threads();
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("cuda context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let mut failures = 0;

    for &n in &[8usize, 32, 64, 128, 256] {
        let b = 64usize;
        // random symmetric distance matrices with zero diagonal
        let mut d = vec![0f32; b * n * n];
        for k in 0..b {
            for i in 0..n {
                for j in (i + 1)..n {
                    let w: f32 = rand::random_range(0.0..1.0);
                    d[k * n * n + i * n + j] = w;
                    d[k * n * n + j * n + i] = w;
                }
            }
        }

        let d_dev = DeviceBuffer::from_host(&stream, &d).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, b * (n - 1) * 2).unwrap();
        let cfg = LaunchConfig {
            grid_dim: (b as u32, 1, 1),
            block_dim: (n as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { module.h0_prim(stream.as_ref(), cfg, &d_dev, n as u32, &mut out_dev) }
            .expect("launch");
        stream.synchronize().unwrap();
        let gpu = out_dev.to_host_vec(&stream).unwrap();

        // compare edge SETS (as sorted lists) against CPU Kruskal
        for k in 0..b {
            let cpu = persist0::h0::h0_single(&d[k * n * n..(k + 1) * n * n], n);
            let mut g: Vec<(u32, u32)> = gpu[k * (n - 1) * 2..(k + 1) * (n - 1) * 2]
                .chunks(2)
                .map(|e| (e[0], e[1]))
                .collect();
            let mut c = cpu.clone();
            g.sort_unstable();
            c.sort_unstable();
            if g != c {
                failures += 1;
                if failures <= 3 {
                    eprintln!("mismatch n={n} item={k}");
                }
            }
        }
        println!("n={n:3} B={b}: {}", if failures == 0 { "✓" } else { "✗" });
    }

    if failures > 0 {
        std::process::exit(1);
    }
    println!("✓ SUCCESS: GPU Prim matches CPU Kruskal on all batches");
}