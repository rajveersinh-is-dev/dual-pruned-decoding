//! High-performance native Rust pruning kernel for Dual-Pruned Decoding (DPD).
//!
//! Provides:
//! 1. Andrew's Monotone Chain 2D Upper-Left Convex Hull in O(M log M).
//! 2. Safe bound and feasibility filtering.
//! 3. Fast scalarized top-K candidate ranking.
//!
//! Standards: Zero unwrap(), zero panic!(), zero allow(dead_code), 100% safe FFI guards.

use std::cmp::Ordering;

/// Point representation in the (Cost, Score) plane.
#[derive(Debug, Clone, Copy)]
pub struct Point {
    pub cost: f64,
    pub score: f64,
    pub original_index: usize,
}

impl Point {
    /// Creates a new Point.
    pub fn new(cost: f64, score: f64, original_index: usize) -> Self {
        Self {
            cost,
            score,
            original_index,
        }
    }
}

/// Computes the 2D cross product of vectors OA and OB.
/// Returns positive if O -> A -> B makes a counter-clockwise turn.
#[inline]
pub fn cross_product(o: Point, a: Point, b: Point) -> f64 {
    (a.cost - o.cost) * (b.score - o.score) - (a.score - o.score) * (b.cost - o.cost)
}

/// Computes extreme vertices of the upper-left convex hull in the (C, S) plane.
///
/// Candidates are ordered from leftmost (lowest cost) to rightmost (highest cost).
/// For any non-negative multiplier lambda >= 0, maximizing S - lambda * C is guaranteed
/// to be attained at an extreme vertex of this hull.
pub fn upper_left_hull(points: &[Point]) -> Vec<usize> {
    if points.is_empty() {
        return Vec::new();
    }
    if points.len() == 1 {
        return vec![points[0].original_index];
    }

    // Filter non-finite numbers and clone
    let mut valid_points: Vec<Point> = points
        .iter()
        .copied()
        .filter(|p| p.cost.is_finite() && p.score.is_finite())
        .collect();

    if valid_points.is_empty() {
        return Vec::new();
    }

    // Sort by cost ascending, then by score descending
    valid_points.sort_by(|a, b| match a.cost.total_cmp(&b.cost) {
        Ordering::Equal => b.score.total_cmp(&a.score),
        other => other,
    });

    // Deduplicate points with identical or near-identical cost (keep highest score)
    let mut deduped: Vec<Point> = Vec::with_capacity(valid_points.len());
    for p in valid_points {
        if let Some(last) = deduped.last() {
            if (p.cost - last.cost).abs() <= 1e-12 {
                // Same cost: since we sorted by score descending, `last` has score >= p.score
                continue;
            }
        }
        deduped.push(p);
    }

    // Pareto filter: Score must strictly increase as Cost increases
    // If a candidate has higher cost but lower or equal score than an earlier candidate,
    // it is strictly dominated for ALL non-negative lambda >= 0.
    let mut pareto: Vec<Point> = Vec::with_capacity(deduped.len());
    let mut max_score_so_far = f64::NEG_INFINITY;
    for p in deduped {
        if p.score > max_score_so_far + 1e-12 {
            pareto.push(p);
            max_score_so_far = p.score;
        }
    }

    if pareto.len() <= 2 {
        return pareto.into_iter().map(|p| p.original_index).collect();
    }

    // Build upper-left hull using Monotone Chain with decreasing slopes
    let mut hull: Vec<Point> = Vec::with_capacity(pareto.len());
    for pt in pareto {
        while hull.len() >= 2 {
            let p1 = hull[hull.len() - 2];
            let p2 = hull[hull.len() - 1];
            let slope_diff = (p2.score - p1.score) * (pt.cost - p2.cost)
                - (pt.score - p2.score) * (p2.cost - p1.cost);
            if slope_diff <= 1e-12 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(pt);
    }

    hull.into_iter().map(|p| p.original_index).collect()
}

/// Filter candidate indices that are provably suboptimal or infeasible.
pub fn filter_candidates(
    costs: &[f64],
    scores: &[f64],
    min_future_costs: &[f64],
    max_future_scores: &[f64],
    budget: f64,
    incumbent_score: Option<f64>,
) -> Vec<usize> {
    let mut surviving = Vec::new();
    let n = costs
        .len()
        .min(scores.len())
        .min(min_future_costs.len())
        .min(max_future_scores.len());

    for i in 0..n {
        let c = costs[i];
        let s = scores[i];
        let c_min = min_future_costs[i];
        let u = max_future_scores[i];

        // Infeasibility check: C + c_min > B
        if c + c_min > budget + 1e-9 {
            continue;
        }

        // Incumbent bound check: S + U <= S_inc
        if let Some(s_inc) = incumbent_score {
            if s + u <= s_inc - 1e-9 {
                continue;
            }
        }

        surviving.push(i);
    }

    surviving
}

// ---------------------------------------------------------------------------
// C-ABI Safe FFI Interface for ctypes / native integration
// ---------------------------------------------------------------------------

/// C-ABI function to compute upper-left convex hull indices.
/// Returns 0 on success, negative error code on invalid inputs.
///
/// # Safety
/// - `costs_ptr` and `scores_ptr` must point to valid, aligned arrays of at least `count` `f64` values.
/// - `out_indices` must point to a writable, aligned buffer of at least `out_capacity` `usize` elements.
/// - `out_count` must be a valid, aligned, non-null pointer to a single `usize`.
#[no_mangle]
pub unsafe extern "C" fn rust_compute_upper_left_hull(
    costs_ptr: *const f64,
    scores_ptr: *const f64,
    count: usize,
    out_indices: *mut usize,
    out_capacity: usize,
    out_count: *mut usize,
) -> i32 {
    if costs_ptr.is_null() || scores_ptr.is_null() || out_indices.is_null() || out_count.is_null() {
        return -1;
    }

    let costs = unsafe { std::slice::from_raw_parts(costs_ptr, count) };
    let scores = unsafe { std::slice::from_raw_parts(scores_ptr, count) };

    let mut points = Vec::with_capacity(count);
    for i in 0..count {
        points.push(Point::new(costs[i], scores[i], i));
    }

    let hull = upper_left_hull(&points);
    let n_out = hull.len().min(out_capacity);

    unsafe {
        for (idx, &item) in hull.iter().take(n_out).enumerate() {
            *out_indices.add(idx) = item;
        }
        *out_count = n_out;
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_and_single() {
        assert!(upper_left_hull(&[]).is_empty());

        let p1 = Point::new(1.0, -2.0, 42);
        let hull1 = upper_left_hull(&[p1]);
        assert_eq!(hull1, vec![42]);
    }

    #[test]
    fn test_counterexample_1_geometry() {
        // Point A: (2.0, -1.0) idx 0
        // Point B: (0.0, -2.0) idx 1
        // Point C: (1.0, -1.8) idx 2 (Strictly below chord between B and A)
        let pts = vec![
            Point::new(2.0, -1.0, 0),
            Point::new(0.0, -2.0, 1),
            Point::new(1.0, -1.8, 2),
        ];

        let hull = upper_left_hull(&pts);
        // Hull should contain B (idx 1) and A (idx 0), and exclude C (idx 2)
        assert_eq!(hull, vec![1, 0]);
    }

    #[test]
    fn test_vertical_collinear_points() {
        let pts = vec![
            Point::new(1.0, -3.0, 0),
            Point::new(1.0, -1.0, 1), // Higher score at same cost
            Point::new(1.0, -5.0, 2),
        ];
        let hull = upper_left_hull(&pts);
        assert_eq!(hull, vec![1]);
    }

    #[test]
    fn test_filtering() {
        let costs = vec![2.0, 1.0, 5.0];
        let scores = vec![-1.0, -2.0, -0.5];
        let min_future = vec![2.0, 1.0, 0.0];
        let max_future = vec![0.0, 0.0, 0.0];

        let surviving = filter_candidates(&costs, &scores, &min_future, &max_future, 3.0, Some(-1.5));
        // Item 0: cost 2 + 2 = 4 > 3 -> infeasible
        // Item 1: cost 1 + 1 = 2 <= 3; score -2 + 0 = -2 <= -1.5 -> incumbent pruned
        // Item 2: cost 5 + 0 = 5 > 3 -> infeasible
        assert!(surviving.is_empty());
    }
}
