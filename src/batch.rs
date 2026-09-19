use crate::h0::h0_single;
use rayon::prelude::*;

/// d: [B * N * N] row-major. Returns [B * (N-1) * 2] flattened as i, j, i, j, ...
pub fn h0_batched(d: &[f32], b: usize, n: usize) -> Vec<u32> {
    let stride = n * n;
    let m = n.saturating_sub(1);
    let mut out = vec![0u32; b * m * 2];

    out.par_chunks_mut(m * 2)
        .zip(d.par_chunks(stride))
        .for_each(|(dst, src)| {
            for (k, (i, j)) in h0_single(src, n).into_iter().enumerate() {
                dst[2 * k] = i;
                dst[2 * k + 1] = j;
            }
        });
    out
}
