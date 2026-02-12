use pyo3::prelude::*;

mod categorizer;
mod intersection;
mod vcf_parser;

/// Native acceleration module for CWAS-Plus.
#[pymodule]
fn cwas_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(categorizer::categorize_variants, m)?)?;
    m.add_function(wrap_pyfunction!(categorizer::build_category_names, m)?)?;
    m.add_function(wrap_pyfunction!(intersection::compute_intersection_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(intersection::compute_intersection_matrix_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(vcf_parser::parse_vcf, m)?)?;
    Ok(())
}
