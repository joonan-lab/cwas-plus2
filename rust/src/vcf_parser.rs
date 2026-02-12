use flate2::read::GzDecoder;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

/// Parse a gzip-compressed VCF file with VEP annotations.
///
/// Performs streaming gzip decompression and fast field extraction
/// using memchr for delimiter scanning.
///
/// Arguments:
///   path: path to the gzip-compressed VCF file
///
/// Returns:
///   A dict with keys:
///     "columns": list of column name strings (VCF columns + CSQ fields + extra INFO fields)
///     "data": list of lists, each inner list is a row of string values
#[pyfunction]
pub fn parse_vcf<'py>(
    py: Python<'py>,
    path: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let file = File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot open file: {}", e)))?;
    let decoder = GzDecoder::new(file);
    let reader = BufReader::with_capacity(256 * 1024, decoder);

    let mut variant_col_names: Vec<String> = Vec::new();
    let mut csq_field_names: Vec<String> = Vec::new();
    let mut annot_field_names: Vec<String> = Vec::new();
    let mut info_keys: Vec<String> = Vec::new();
    let mut info_keys_initialized = false;
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut info_idx: usize = 7;

    for line_result in reader.lines() {
        let line = line_result
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Read error: {}", e)))?;

        if line.starts_with('#') {
            if line.starts_with("##INFO=<ID=CSQ") {
                csq_field_names = parse_vcf_info_field(&line);
            } else if line.starts_with("##INFO=<ID=ANNOT") {
                annot_field_names = parse_annot_field(&line);
            } else if line.starts_with("#CHROM") {
                variant_col_names = line[1..].split('\t').map(|s| s.to_string()).collect();
                info_idx = variant_col_names
                    .iter()
                    .position(|s| s == "INFO")
                    .unwrap_or(7);
            }
            continue;
        }

        // Data row
        if variant_col_names.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "The VCF does not have column names.",
            ));
        }
        if csq_field_names.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "The VCF does not have CSQ information.",
            ));
        }
        if annot_field_names.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "The VCF does not have annotation information.",
            ));
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < variant_col_names.len() {
            continue;
        }

        // Parse INFO field
        let info_str = fields[info_idx];
        let mut info_map: HashMap<&str, &str> = HashMap::new();
        let mut current_info_keys: Vec<String> = Vec::new();
        for kv in info_str.split(';') {
            if let Some(eq_pos) = memchr::memchr(b'=', kv.as_bytes()) {
                let key = &kv[..eq_pos];
                let value = &kv[eq_pos + 1..];
                info_map.insert(key, value);
                if key != "CSQ" {
                    current_info_keys.push(key.to_string());
                }
            }
        }

        // Initialize info_keys from first data row
        if !info_keys_initialized {
            info_keys = current_info_keys;
            info_keys_initialized = true;
        }

        // Parse CSQ (pipe-delimited)
        let csq_str = info_map.get("CSQ").copied().unwrap_or("");
        let csq_values: Vec<&str> = csq_str.split('|').collect();

        // Build output row: non-INFO VCF columns + INFO fields (except CSQ) + CSQ fields
        let mut row = Vec::with_capacity(
            variant_col_names.len() - 1 + info_keys.len() + csq_field_names.len(),
        );

        // Add non-INFO VCF columns
        for (i, field) in fields.iter().enumerate() {
            if i != info_idx {
                row.push(field.to_string());
            }
        }

        // Add INFO key-value pairs except CSQ (in consistent order)
        for key in &info_keys {
            let value = info_map.get(key.as_str()).copied().unwrap_or("");
            row.push(value.to_string());
        }

        // Add CSQ values
        for i in 0..csq_field_names.len() {
            if i < csq_values.len() {
                row.push(csq_values[i].to_string());
            } else {
                row.push(String::new());
            }
        }

        rows.push(row);
    }

    // Build output column names
    let mut all_col_names: Vec<String> = Vec::new();
    for name in &variant_col_names {
        if name != "INFO" {
            all_col_names.push(name.clone());
        }
    }
    for key in &info_keys {
        all_col_names.push(key.clone());
    }
    for name in &csq_field_names {
        all_col_names.push(name.clone());
    }

    let result = PyDict::new(py);

    let columns_py = PyList::new(py, &all_col_names)?;
    result.set_item("columns", &columns_py)?;

    // Convert rows to Python lists
    let data_py = PyList::empty(py);
    for row in &rows {
        let row_py = PyList::new(py, row)?;
        data_py.append(&row_py)?;
    }
    result.set_item("data", &data_py)?;

    let csq_py = PyList::new(py, &csq_field_names)?;
    result.set_item("csq_field_names", &csq_py)?;
    let annot_py = PyList::new(py, &annot_field_names)?;
    result.set_item("annot_field_names", &annot_py)?;

    Ok(result)
}

/// Parse CSQ field names from VCF header line.
fn parse_vcf_info_field(line: &str) -> Vec<String> {
    if let Some(idx) = line.find("Format: ") {
        let rest = &line[idx + 8..];
        let trimmed = rest.trim_end_matches(|c: char| c == '"' || c == '>' || c == '\n');
        trimmed.split('|').map(|s| s.to_string()).collect()
    } else {
        Vec::new()
    }
}

/// Parse ANNOT field names from VCF header line.
fn parse_annot_field(line: &str) -> Vec<String> {
    if let Some(idx) = line.find("Key=") {
        let rest = &line[idx + 4..];
        let trimmed = rest.trim_end_matches(|c: char| c == '"' || c == '>' || c == '\n');
        trimmed.split('|').map(|s| s.to_string()).collect()
    } else {
        Vec::new()
    }
}
