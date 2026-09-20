# persist0

Batched, differentiable H₀ persistence for PyTorch. Rust core, PyO3 bindings, CUDA kernel
written in Rust via [cuda-oxide](https://github.com/NVlabs/cuda-oxide).

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
- **Deterministic tie-break.** Edges are ordered by `(death, i, j)` using `f32::total_cmp`. Under
  a strict total order on edges the MST is unique, so every MST algorithm returns the same edge
  set — which is what lets the CPU and GPU paths use different algorithms and still agree exactly.
- **CPU path is the specification.** Kruskal with union-find, rayon across the batch. The GPU
  path must match it edge-for-edge in the test suite.

## GPU kernel

`gpu/` is a standalone crate containing the CUDA kernel, written in Rust and compiled to PTX by
cuda-oxide. One thread block per batch item, one thread per vertex, Prim's algorithm with a
shared-memory min-reduction per round; no atomics. `D` is read directly from global memory (row
reads are coalesced since `D` is symmetric), so N is limited by block size (≤ 1024), not shared
memory. The crate's `main` verifies the GPU edge set against the CPU Kruskal on 64 random
matrices at each of N ∈ {8, 32, 64, 128, 256} and then prints the timing table below.

```bash
cd gpu && cargo oxide run     # needs the cuda-oxide toolchain; see below
```

The kernel is not yet reachable from Python; wiring it behind a `cuda` feature is the next step.

## Correctness

`tests/test_parity_ripser.py` checks death values against `ripser` H₀ on random Euclidean and
cosine point clouds (N up to 128), the MST total against `scipy.sparse.csgraph.minimum_spanning_tree`,
and batched vs. per-item output. `tests/test_gradients.py` runs `torch.autograd.gradcheck` on the
gathered deaths and checks that `TopoH0Loss` decreases under optimization. `gpu/src/main.rs`
checks the GPU kernel's edge set against the CPU kernel.

```bash
pip install "persist0-tda[test]"
pytest
cargo test
```

## Benchmarks

RTX 4070 Laptop (sm_89, 36 SMs) vs. one core of the same laptop. GPU times are kernel + sync with
`D` already resident; CPU times are single-threaded `h0_single` from criterion.

| N   | CPU, 1 core (Kruskal) | GPU, B = 1 | GPU per item, B = 64 |
| --- | --------------------: | ---------: | -------------------: |
| 64  | 62 µs                 | 94 µs      | 1.5 µs               |
| 128 | 988 µs                | 196 µs     | 3.2 µs               |
| 256 | 4.6 ms                | 427 µs     | 7.1 µs               |

GPU time is nearly flat in B until the grid exceeds one wave (~216 resident blocks of 256
threads on this card), because every batch item is its own block. It is latency-bound in N:
N−1 rounds of ~9 block barriers each. The CPU path is superlinear in N because Kruskal on a
complete graph sorts O(N²) edges; a dense Prim on the CPU would be O(N²) and is a planned
improvement.

Takeaway: for the training-loop regime (B ≥ 16, N ≥ 64) the GPU kernel is 30–60× faster than
the batched CPU path; at N = 64 the crossover is roughly B ≈ 12 on 8 cores.

## Toolchain (GPU crate only)

The main crate builds on stable Rust. `gpu/` pins `nightly-2026-08-28` and depends on
`cuda-device` / `cuda-host` from the cuda-oxide repository at a fixed revision. It needs:

- CUDA toolkit 13.1 (driver ≥ 13.1), compute capability ≥ 8.0
- clang 21 + `libclang-common-21-dev` (bindgen)
- `cargo-oxide` installed from the cuda-oxide repo; `cargo oxide doctor` checks everything

Tested on x86_64 Linux under WSL2. aarch64 untested.

## Roadmap

- Expose the GPU kernel from Python behind a `cuda` feature with a CPU/GPU parity test.
- Dense Prim on the CPU path.
- Borůvka on the GPU if N ≫ 256 ever matters (fewer rounds; the Prim kernel is round-bound).
- H₁ and above, which need a real persistence algorithm (column reduction with clearing and
  apparent pairs); the plan is to integrate [lophat](https://crates.io/crates/lophat) for the
  reduction step rather than reimplement it.
- manylinux x86_64 and aarch64 wheels via CI.

## License

MIT OR Apache-2.0.
