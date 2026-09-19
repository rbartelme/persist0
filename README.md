# persist0

Batched, differentiable H₀ persistence for PyTorch. Rust core, PyO3 bindings, optional CUDA kernel.

Given a batch of distance matrices, `persist0` returns the minimum-spanning-tree death edges of
the H₀ persistence pairs as *indices*, not values. Gathering `D[b, i, j]` at those indices in
torch gives death times with gradients attached, so a topological loss term costs one MST per
batch item and no custom autograd.

## Install

```bash
pip install persist0            # CPU
pip install "persist0[torch]"   # + torch wrapper
```

From source:

```bash
pip install maturin
maturin develop --release                       # CPU
maturin develop --release --features cuda       # + CUDA (see toolchain note below)
```

## Usage

```python
import torch
from persist0 import h0_persistence, TopoH0Loss

z = encoder(x)                                  # [B, N, d]
D = 1 - torch.nn.functional.cosine_similarity(z[:, :, None], z[:, None, :], dim=-1)

idx, deaths = h0_persistence(D)                 # idx: [B, N-1, 2] long, deaths: [B, N-1] with grad

# preserve H0 structure of a reference space
loss = TopoH0Loss(top_k=32)(D_live, D_ref)
```

`h0_persistence` returns finite pairs only; the infinite bar is implicit. Births in H₀ are all
zero and are not returned.

## Design

- **Indices, not values.** The kernel is a pure index oracle. All differentiability lives in torch's
  `gather`, which makes the Rust side backend-agnostic and the gradient exact almost everywhere
  (the pairing is piecewise constant in the filtration values).
- **Distance matrix in, not points.** Metric choice, subsampling, and any temperature or scaling
  happen in torch before the call.
- **Deterministic tie-break.** Edges sort by death value, then lexicographically by `(i, j)`.
  CPU and CUDA paths produce identical output.
- **CPU path is the specification.** The CUDA path (one thread block per batch item, Borůvka in
  shared memory) must match the CPU path bit-for-bit on indices in the test suite.

## Correctness

`tests/test_parity_ripser.py` checks death values against `ripser` H₀ on random Euclidean and
cosine point clouds; `tests/test_mst_weight.py` checks the MST total against
`scipy.sparse.csgraph.minimum_spanning_tree`; `tests/test_gradients.py` runs `gradcheck` on the loss.

```bash
pip install "persist0[test]"
pytest
cargo test
```

## Benchmarks

TBD. `benches/h0_bench.rs` (criterion) covers the CPU path over N ∈ {32, 64, 128, 256} and
B ∈ {1, 64}; `examples/bench_gpu.py` finds the CPU/GPU crossover batch size.

## Toolchain

- Rust stable for the CPU path (`rust-toolchain.toml`).
- The `cuda` feature targets NVIDIA's CUDA Rust toolchain (cutile-rs), which is early-stage.
  Expect a pinned nightly and a pinned CUDA version; see `docs/cuda.md` once it exists.
- Tested on x86_64 Linux with an 8 GB CUDA GPU. aarch64 (DGX Spark) untested.

## Scope

H₀ only. H₁ and above need a real persistence algorithm (column reduction with clearing and
apparent pairs); the plan is to integrate with [lophat](https://crates.io/crates/lophat) for the
reduction step rather than reimplement it. Tracked in the issues.

## License

MIT OR Apache-2.0.
