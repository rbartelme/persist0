use criterion::{Criterion, criterion_group, criterion_main};
use persist0::h0::h0_single;

fn bench_h0(c: &mut Criterion) {
    for &n in &[32usize, 64, 128, 256] {
        let d: Vec<f32> = (0..n * n)
            .map(|k| ((k * 7919) % 1000) as f32 / 1000.0)
            .collect();
        c.bench_function(&format!("h0_single_n{n}"), |b| b.iter(|| h0_single(&d, n)));
    }
}

criterion_group!(benches, bench_h0);
criterion_main!(benches);
