use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// Compute the intersection matrix (co-occurrence matrix) for categories.
///
/// For each variant, generates all category index combinations and increments
/// matrix[i][j] for every pair of categories that the variant belongs to.
/// The result is a symmetric matrix.
///
/// Arguments:
///   annotations: list of 5 groups, each a list of n_variants index lists
///                annotations[g][v] = Vec<usize> of active term indices for group g, variant v
///   group_sizes: list of 5 values, the number of terms in each annotation group
///
/// Returns:
///   2D numpy array of shape (n_categories, n_categories) with f64 counts
#[pyfunction]
pub fn compute_intersection_matrix<'py>(
    py: Python<'py>,
    annotations: Vec<Vec<Vec<usize>>>,
    group_sizes: Vec<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n_variants = annotations[0].len();

    // Compute strides
    let mut strides = [0usize; 5];
    strides[4] = 1;
    for i in (0..4).rev() {
        strides[i] = strides[i + 1] * group_sizes[i + 1];
    }
    let n_categories = strides[0] * group_sizes[0];

    // Build per-variant index slices
    let variant_data: Vec<[&[usize]; 5]> = (0..n_variants)
        .map(|v| {
            [
                annotations[0][v].as_slice(),
                annotations[1][v].as_slice(),
                annotations[2][v].as_slice(),
                annotations[3][v].as_slice(),
                annotations[4][v].as_slice(),
            ]
        })
        .collect();

    // Process in parallel with thread-local upper-triangle matrices
    let chunk_size = std::cmp::max(1, n_variants / rayon::current_num_threads());
    let partial_matrices: Vec<Vec<f64>> = variant_data
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_matrix = vec![0.0f64; n_categories * n_categories];

            for bits in chunk {
                // Collect all category indices for this variant
                let mut cat_indices = Vec::new();
                for &b0 in bits[0] {
                    let idx0 = b0 * strides[0];
                    for &b1 in bits[1] {
                        let idx1 = idx0 + b1 * strides[1];
                        for &b2 in bits[2] {
                            let idx2 = idx1 + b2 * strides[2];
                            for &b3 in bits[3] {
                                let idx3 = idx2 + b3 * strides[3];
                                for &b4 in bits[4] {
                                    cat_indices.push(idx3 + b4);
                                }
                            }
                        }
                    }
                }

                // Update co-occurrence matrix (upper triangle including diagonal)
                for (ii, &ci) in cat_indices.iter().enumerate() {
                    for &cj in &cat_indices[ii..] {
                        let (r, c) = if ci <= cj { (ci, cj) } else { (cj, ci) };
                        local_matrix[r * n_categories + c] += 1.0;
                    }
                }
            }

            local_matrix
        })
        .collect();

    // Merge partial results
    let mut matrix = Array2::<f64>::zeros((n_categories, n_categories));
    for local in &partial_matrices {
        for i in 0..n_categories {
            for j in i..n_categories {
                let val = local[i * n_categories + j];
                if val != 0.0 {
                    matrix[[i, j]] += val;
                    if i != j {
                        matrix[[j, i]] += val; // Mirror for symmetry
                    }
                }
            }
        }
    }

    Ok(matrix.into_pyarray(py))
}

/// Compute a sparse intersection matrix (co-occurrence) in COO format.
///
/// Memory-efficient alternative to `compute_intersection_matrix` for large
/// category counts (e.g. C=20,000). Uses FxHashMap instead of dense arrays,
/// reducing per-thread memory from O(C^2) to O(nnz).
///
/// 3-phase algorithm:
///   Phase 1 (parallel): For each variant, compute its category indices.
///   Phase 2 (sequential): Accumulate upper-triangle (row<=col) pairs in a
///                          single FxHashMap<u64, f64>.
///   Phase 3: Convert HashMap to COO arrays (row, col, data).
///
/// Arguments:
///   annotations: list of 5 groups, each a list of n_variants index lists
///                annotations[g][v] = Vec<usize> of active term indices for group g, variant v
///   group_sizes: list of 5 values, the number of terms in each annotation group
///
/// Returns:
///   dict with keys "row" (u32[]), "col" (u32[]), "data" (f64[]), "shape" (n, n)
#[pyfunction]
pub fn compute_intersection_matrix_sparse<'py>(
    py: Python<'py>,
    annotations: Vec<Vec<Vec<usize>>>,
    group_sizes: Vec<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let n_variants = annotations[0].len();

    // Compute strides (same as dense version)
    let mut strides = [0usize; 5];
    strides[4] = 1;
    for i in (0..4).rev() {
        strides[i] = strides[i + 1] * group_sizes[i + 1];
    }
    let n_categories = strides[0] * group_sizes[0];

    // Build per-variant index slices
    let variant_data: Vec<[&[usize]; 5]> = (0..n_variants)
        .map(|v| {
            [
                annotations[0][v].as_slice(),
                annotations[1][v].as_slice(),
                annotations[2][v].as_slice(),
                annotations[3][v].as_slice(),
                annotations[4][v].as_slice(),
            ]
        })
        .collect();

    // Phase 1 (parallel): compute category indices for each variant
    let chunk_size = std::cmp::max(1, n_variants / rayon::current_num_threads());
    let per_variant_cats: Vec<Vec<u32>> = variant_data
        .par_chunks(chunk_size)
        .flat_map_iter(|chunk| {
            chunk.iter().map(|bits| {
                let mut cat_indices = Vec::new();
                for &b0 in bits[0] {
                    let idx0 = b0 * strides[0];
                    for &b1 in bits[1] {
                        let idx1 = idx0 + b1 * strides[1];
                        for &b2 in bits[2] {
                            let idx2 = idx1 + b2 * strides[2];
                            for &b3 in bits[3] {
                                let idx3 = idx2 + b3 * strides[3];
                                for &b4 in bits[4] {
                                    cat_indices.push((idx3 + b4) as u32);
                                }
                            }
                        }
                    }
                }
                cat_indices
            })
        })
        .collect();

    // Phase 2 (sequential): accumulate upper-triangle pairs in FxHashMap
    // Key encoding: (row << 32) | col  (row <= col)
    let mut map: FxHashMap<u64, f64> = FxHashMap::default();
    for cat_indices in &per_variant_cats {
        let n = cat_indices.len();
        for ii in 0..n {
            let ci = cat_indices[ii] as u64;
            for jj in ii..n {
                let cj = cat_indices[jj] as u64;
                let (r, c) = if ci <= cj { (ci, cj) } else { (cj, ci) };
                let key = (r << 32) | c;
                *map.entry(key).or_insert(0.0) += 1.0;
            }
        }
    }

    // Phase 3: convert HashMap to COO arrays
    let nnz = map.len();
    let mut rows = Vec::with_capacity(nnz);
    let mut cols = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);

    for (&key, &val) in &map {
        rows.push((key >> 32) as u32);
        cols.push((key & 0xFFFF_FFFF) as u32);
        data.push(val);
    }

    // Build return dict
    let dict = PyDict::new(py);
    dict.set_item("row", PyArray1::from_vec(py, rows))?;
    dict.set_item("col", PyArray1::from_vec(py, cols))?;
    dict.set_item("data", PyArray1::from_vec(py, data))?;
    dict.set_item("shape", (n_categories, n_categories))?;

    Ok(dict)
}
