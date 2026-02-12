//! Canvas export command for persisting pipeline layout from Studio.

use crate::models::{
    PipelineCanvasExportResult, PipelineEdgeSnapshot, PipelineNodeSnapshot,
};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const CANVAS_EXPORT_DIR: &str = "outputs/canvas";
const CANVAS_EXPORT_FORMAT_VERSION: u32 = 1;

#[tauri::command]
pub fn export_pipeline_canvas(
    data_root: String,
    nodes: Vec<PipelineNodeSnapshot>,
    edges: Vec<PipelineEdgeSnapshot>,
    start_node_id: Option<String>,
    output_path: Option<String>,
) -> Result<PipelineCanvasExportResult, String> {
    validate_canvas_payload(&nodes, &edges)?;
    let output_path = resolve_output_path(&data_root, output_path)?;
    create_parent_dir(&output_path)?;
    let payload = build_canvas_payload(nodes, edges, start_node_id)?;
    write_export_file(&output_path, &payload)?;
    Ok(PipelineCanvasExportResult {
        output_path: output_path.display().to_string(),
    })
}

fn validate_canvas_payload(
    nodes: &[PipelineNodeSnapshot],
    edges: &[PipelineEdgeSnapshot],
) -> Result<(), String> {
    for node in nodes {
        if node.id.trim().is_empty() {
            return Err("Canvas export failed: node id cannot be empty.".to_string());
        }
    }
    for edge in edges {
        if edge.source_node_id.trim().is_empty() || edge.target_node_id.trim().is_empty() {
            return Err("Canvas export failed: edge source/target ids cannot be empty.".to_string());
        }
    }
    Ok(())
}

fn create_parent_dir(output_path: &Path) -> Result<(), String> {
    let Some(parent_dir) = output_path.parent() else {
        return Err(format!(
            "Canvas export failed: output path {} is invalid.",
            output_path.display()
        ));
    };
    fs::create_dir_all(parent_dir).map_err(|error| {
        format!(
            "Canvas export failed: could not create export directory {}: {error}",
            parent_dir.display()
        )
    })
}

fn build_default_output_path(export_dir: &Path) -> Result<PathBuf, String> {
    let epoch_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("Canvas export failed: system clock is invalid: {error}"))?
        .as_secs();
    Ok(export_dir.join(format!("forge-canvas-{epoch_seconds}.json")))
}

fn resolve_output_path(data_root: &str, output_path: Option<String>) -> Result<PathBuf, String> {
    if let Some(path_value) = output_path {
        let trimmed_path = path_value.trim();
        if !trimmed_path.is_empty() {
            let requested_path = PathBuf::from(trimmed_path);
            let normalized_path = if requested_path.is_absolute() {
                requested_path
            } else {
                Path::new(data_root).join(requested_path)
            };
            return Ok(append_json_extension_if_missing(normalized_path));
        }
    }
    let export_dir = Path::new(data_root).join(CANVAS_EXPORT_DIR);
    build_default_output_path(&export_dir)
}

fn append_json_extension_if_missing(mut output_path: PathBuf) -> PathBuf {
    if output_path.extension().is_none() {
        output_path.set_extension("json");
    }
    output_path
}

fn build_canvas_payload(
    nodes: Vec<PipelineNodeSnapshot>,
    edges: Vec<PipelineEdgeSnapshot>,
    start_node_id: Option<String>,
) -> Result<Value, String> {
    let exported_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("Canvas export failed: system clock is invalid: {error}"))?
        .as_secs();
    Ok(json!({
        "format_version": CANVAS_EXPORT_FORMAT_VERSION,
        "exported_unix_seconds": exported_unix_seconds,
        "start_node_id": start_node_id,
        "nodes": nodes,
        "edges": edges
    }))
}

fn write_export_file(output_path: &Path, payload: &Value) -> Result<(), String> {
    let serialized = serde_json::to_string_pretty(payload).map_err(|error| {
        format!("Canvas export failed: could not serialize canvas payload: {error}")
    })?;
    fs::write(output_path, serialized).map_err(|error| {
        format!(
            "Canvas export failed: could not write export file {}: {error}",
            output_path.display()
        )
    })
}
