use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Categorize variants using index-list-based annotation groups.
///
/// For each variant, takes 5 annotation index lists and generates all category
/// combinations as flat indices. Counts are accumulated per sample.
///
/// Arguments:
///   annotations: list of 5 groups, each a list of n_variants index lists
///                annotations[g][v] = Vec<usize> of active term indices for group g, variant v
///   group_sizes: list of 5 values, the number of terms in each annotation group
///   sample_ids: array of length n_variants, the sample index for each variant
///   n_samples: total number of distinct samples
///
/// Returns:
///   2D numpy array of shape (n_samples, n_categories) with i32 counts
#[pyfunction]
pub fn categorize_variants<'py>(
    py: Python<'py>,
    annotations: Vec<Vec<Vec<usize>>>,
    group_sizes: Vec<usize>,
    sample_ids: PyReadonlyArray1<'py, usize>,
    n_samples: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let sample_ids = sample_ids.as_slice()?;
    let n_variants = sample_ids.len();

    // Compute strides for flat category index
    // category_idx = b0 * stride[0] + b1 * stride[1] + ... + b4 * stride[4]
    let mut strides = [0usize; 5];
    strides[4] = 1;
    for i in (0..4).rev() {
        strides[i] = strides[i + 1] * group_sizes[i + 1];
    }
    let n_categories = strides[0] * group_sizes[0];

    // Build per-variant tuples: (indices for each group, sample_id)
    let variant_data: Vec<([&[usize]; 5], usize)> = (0..n_variants)
        .map(|v| {
            let idx = [
                annotations[0][v].as_slice(),
                annotations[1][v].as_slice(),
                annotations[2][v].as_slice(),
                annotations[3][v].as_slice(),
                annotations[4][v].as_slice(),
            ];
            (idx, sample_ids[v])
        })
        .collect();

    // Use rayon to process variants in parallel with thread-local accumulators
    let chunk_size = std::cmp::max(1, n_variants / rayon::current_num_threads());
    let result_vec: Vec<Vec<i32>> = variant_data
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut local_counts = vec![0i32; n_samples * n_categories];

            for &(ref bits, sample_id) in chunk {
                // Generate all combinations via nested iteration (product of 5 groups)
                let base_offset = sample_id * n_categories;
                for &b0 in bits[0] {
                    let idx0 = b0 * strides[0];
                    for &b1 in bits[1] {
                        let idx1 = idx0 + b1 * strides[1];
                        for &b2 in bits[2] {
                            let idx2 = idx1 + b2 * strides[2];
                            for &b3 in bits[3] {
                                let idx3 = idx2 + b3 * strides[3];
                                for &b4 in bits[4] {
                                    let cat_idx = idx3 + b4;
                                    local_counts[base_offset + cat_idx] += 1;
                                }
                            }
                        }
                    }
                }
            }

            local_counts
        })
        .collect();

    // Merge thread-local results
    let mut final_counts = Array2::<i32>::zeros((n_samples, n_categories));
    for local in &result_vec {
        for s in 0..n_samples {
            let row_offset = s * n_categories;
            let mut row = final_counts.row_mut(s);
            for c in 0..n_categories {
                row[c] += local[row_offset + c];
            }
        }
    }

    Ok(final_counts.into_pyarray(py))
}

/// Map flat category indices back to category name strings.
///
/// Arguments:
///   group_terms: list of 5 lists, each containing the annotation term strings
///
/// Returns:
///   list of category name strings, ordered by flat index
#[pyfunction]
#[pyo3(name = "build_category_names")]
pub fn build_category_names(
    group_terms: Vec<Vec<String>>,
) -> PyResult<Vec<String>> {
    if group_terms.len() != 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Expected exactly 5 annotation groups",
        ));
    }

    let sizes: Vec<usize> = group_terms.iter().map(|g| g.len()).collect();
    let n_categories: usize = sizes.iter().product();
    let mut names = Vec::with_capacity(n_categories);

    // Generate names in the same order as flat indices
    for b0 in 0..sizes[0] {
        for b1 in 0..sizes[1] {
            for b2 in 0..sizes[2] {
                for b3 in 0..sizes[3] {
                    for b4 in 0..sizes[4] {
                        let name = format!(
                            "{}_{}_{}_{}_{}",
                            group_terms[0][b0],
                            group_terms[1][b1],
                            group_terms[2][b2],
                            group_terms[3][b3],
                            group_terms[4][b4],
                        );
                        names.push(name);
                    }
                }
            }
        }
    }

    Ok(names)
}
