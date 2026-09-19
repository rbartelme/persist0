# persist0

Batched, differentiable H₀ persistence for PyTorch. Rust core, PyO3 bindings. CUDA kernel planned.

Given a batch of distance matrices, `persist0` returns the minimum-spanning-tree death edges of
the H₀ persistence pairs as *indices*, not values. Gathering `D[b, i, j]` at those indices in
torch gives death times with gradients attached, so a topological loss term costs one MST per
batch item and no custom autograd.

## Install

The PyPI distribution is `persist0-tda` (PyPI rejects `persist0` as too similar to an existing
project); the import name is `persist0`.

```bash
pip install persist0-tda              # kernel + numpy API
pip install "persist0-tda[torch]"     # + torch wrapper and loss
```

Rust crate: `cargo add persist0`.

From source:

```bash
pip install maturin
maturin develop --release
```

## Usage

```python
import torch
from persist0 import h0_persistence, TopoH0Loss

z = torch.nn.functional.normalize(encoder(x), dim=-1)   # [B, N, d]
D = 1 - z @ z.transpose(1, 2)                            # cosine distance, [B, N, N]

idx, deaths = h0_persistence(D)   # idx: [B, N-1, 2] long; deaths: [B, N-1], carries grad

# match the H0 death vector of a live space to a fixed reference space
loss = TopoH0Loss(top_k=32)(D_live, D_ref)
```

`h0_persistence` returns finite pairs only; the infinite bar is implicit. Births in H₀ are all
zero and are not returned. The numpy-level kernel is also exposed as `persist0.h0_batched`.

## Design

- **Indices, not values.** The kernel is a pure index oracle. All differentiability lives in torch
  advanced indexing, which keeps the Rust side backend-agnostic and makes the gradient exact
  almost everywhere (the pairing is piecewise constant in the filtration values).
- **Distance matrix in, not points.** Metric choice, subsampling, and any temperature or scaling
  happen in torch before the call.
- **Deterministic tie-break.** Edges sort by `(death, i, j)` using `f32::total_cmp`, so output
  is reproducible across runs and, once it exists, across CPU and GPU backends.
- **CPU path is the specification.** Kruskal with union-find, rayon across the batch. Any future
  GPU path must match it exactly on indices in the test suite.

## Correctness

`tests/test_parity_ripser.py` checks death values against `ripser` H₀ on random Euclidean and
cosine point clouds (N up to 128), the MST total against `scipy.sparse.csgraph.minimum_spanning_tree`,
and batched vs. per-item output. `tests/test_gradients.py` runs `torch.autograd.gradcheck` on the
gathered deaths and checks that `TopoH0Loss` decreases under optimization.

```bash
pip install "persist0-tda[test]"
pytest
cargo test
```

## Benchmarks

`benches/h0_bench.rs` (criterion) covers the CPU path over N ∈ {32, 64, 128, 256}. Numbers TBD.

## Roadmap

- CUDA backend behind a `cuda` feature: one thread block per batch item, Borůvka in shared memory,
  targeting NVIDIA's CUDA Rust toolchain. Will ship with a CPU/GPU parity test and a crossover
  benchmark.
- H₁ and above, which need a real persistence algorithm (column reduction with clearing and
  apparent pairs); the plan is to integrate [lophat](https://crates.io/crates/lophat) for the
  reduction step rather than reimplement it.
- manylinux x86_64 and aarch64 wheels via CI.

Tested on x86_64 Linux. aarch64 untested.

## License

MIT OR Apache-2.0.
