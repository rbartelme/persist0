use std::cmp::Ordering;

/// H0 persistence of one NxN distance matrix (row-major, symmetric, zero diagonal).
/// Returns the N-1 minimum spanning tree edges as (i, j), i < j, sorted by (death, i, j).
///
/// Dense Prim from vertex 0. Edges are totally ordered by `(weight, min(i, j), max(i, j))`
/// with `f32::total_cmp` on the weight, so the MST is unique and the result is
/// deterministic. Each round selects the cheapest edge out of the tree and relaxes every
/// remaining vertex through the new tree vertex; the selection for the next round is done
/// in the same pass, so the whole thing is one row read per round and O(N²) total.
pub fn h0_single(d: &[f32], n: usize) -> Vec<(u32, u32)> {
    debug_assert_eq!(d.len(), n * n);
    if n < 2 {
        return Vec::new();
    }

    // dist[i], par[i]: best known edge from vertex i into the tree (i not yet in the tree).
    // rem: vertices not yet in the tree.
    let mut dist: Vec<f32> = d[..n].to_vec();
    let mut par: Vec<u32> = vec![0; n];
    let mut rem: Vec<u32> = (1..n as u32).collect();
    let mut out: Vec<(u32, u32)> = Vec::with_capacity(n - 1);

    // Initial selection from the edges out of vertex 0.
    let mut best_k = 0usize;
    let mut best_key = edge_key(dist[1], 1, 0);
    for (k, &i) in rem.iter().enumerate().skip(1) {
        let key = edge_key(dist[i as usize], i, par[i as usize]);
        if edge_lt(key, best_key) {
            best_k = k;
            best_key = key;
        }
    }

    loop {
        let v = rem.swap_remove(best_k);
        out.push((best_key.1, best_key.2));
        let Some((&first, rest)) = rem.split_first() else {
            break;
        };

        // Relax every remaining vertex through v and pick next round's winner as we go.
        let row = &d[v as usize * n..(v as usize + 1) * n];
        best_k = 0;
        best_key = relax(row, v, first, &mut dist, &mut par);
        for (k, &i) in rest.iter().enumerate() {
            let key = relax(row, v, i, &mut dist, &mut par);
            if edge_lt(key, best_key) {
                best_k = k + 1;
                best_key = key;
            }
        }
    }

    out.sort_unstable_by(|&(i0, j0), &(i1, j1)| {
        let w0 = d[i0 as usize * n + j0 as usize];
        let w1 = d[i1 as usize * n + j1 as usize];
        w0.total_cmp(&w1).then(i0.cmp(&i1)).then(j0.cmp(&j1))
    });
    out
}

/// Sort key of the edge {a, b} with weight w.
#[inline(always)]
fn edge_key(w: f32, a: u32, b: u32) -> (f32, u32, u32) {
    (w, a.min(b), a.max(b))
}

#[inline(always)]
fn edge_lt(x: (f32, u32, u32), y: (f32, u32, u32)) -> bool {
    x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)) == Ordering::Less
}

/// Offer vertex i the edge (i, v) with weight row[i]; keep it if it beats the current best.
/// Returns the key of i's best edge afterwards.
#[inline(always)]
fn relax(row: &[f32], v: u32, i: u32, dist: &mut [f32], par: &mut [u32]) -> (f32, u32, u32) {
    let w = row[i as usize];
    let cand = edge_key(w, i, v);
    let cur = edge_key(dist[i as usize], i, par[i as usize]);
    if edge_lt(cand, cur) {
        dist[i as usize] = w;
        par[i as usize] = v;
        cand
    } else {
        cur
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
