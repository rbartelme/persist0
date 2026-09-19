/// H0 persistence of one NxN distance matrix 9row-major, uppper triangle read).
/// Returns the N-1 Minimum Spanning Tree edges as (i, j), i < j, sorted by (death, i, j).
pub fn h0_single(d: &[f32], n: usize) -> Vec<(u32, u32)> {
    debug_assert_eq!(d.len(), n * n);
    if n < 2 {
        return Vec::new();
    }

    // All upper-triangle edges, sorted by (weight, i, j) with deterministic tie-breaking.
    let mut edges: Vec<(f32, u32, u32)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((d[i * n + j], i as u32, j as u32));
        }
    }
    edges.sort_unstable_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.2.cmp(&b.2))
    });

    let mut uf = UnionFind::new(n);
    let mut out = Vec::with_capacity(n - 1);
    for (_, i, j) in edges {
        if uf.union(i as usize, j as usize) {
            out.push((i, j));
            if out.len() == n - 1 {
                break;
            }
        }
    }
    out
}

struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] as usize != x {
            let p = self.parent[x] as usize;
            self.parent[x] = self.parent[p]; // path halving
            x = p;
        }
        x
    }

    /// Returns true if merged happened (i.e. edge is in the minimum spanning tree)
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb as u32,
            std::cmp::Ordering::Greater => self.parent[rb] = ra as u32,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra as u32;
                self.rank[ra] += 1;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn square_with_diagonal() {
        // unit square 0-1-2-3, diagonal 0-2 = sqrt2
        let s = 2f32.sqrt();
        #[rustfmt::skip]
            let d = [
                0., 1., s, 1.,
                1., 0., 1., s,
                s, 1., 0., 1.,
                1., s, 1., 0.,
            ];
        let e = h0_single(&d, 4);
        assert_eq!(e, vec![(0, 1), (0, 3), (1, 2)]); // three unit edges, lexographic
        // tie-broken
    }
}
