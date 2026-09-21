# Changelog

## 0.2.0 — 2026-09-21
- CPU path: dense Prim replaces Kruskal; O(N²), no edge sort.
- `gpu/`: standalone crate with a Prim H₀ kernel written in Rust (cuda-oxide), CPU/GPU parity check
  at N ∈ {8, 32, 64, 128, 256}, and a timing table.
- CI: fmt, clippy, cargo test, pytest on every push.
- Release: manylinux x86_64 + aarch64 wheels and sdist to PyPI, `cargo publish` to crates.io, on tag.
- README: design notes, benchmarks, toolchain requirements.

## 0.1.0 — 2026-09-19
- Initial release: batched H₀ via Kruskal, pyo3 bindings, torch wrapper, `TopoH0Loss`, ripser parity tests.
