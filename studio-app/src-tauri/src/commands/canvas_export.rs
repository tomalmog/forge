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
) -> Result<PipelineCanvasExportResult, String> {
    validate_canvas_payload(&nodes, &edges)?;
    let export_dir = Path::new(&data_root).join(CANVAS_EXPORT_DIR);
    create_export_dir(&export_dir)?;
    let output_path = build_output_path(&export_dir)?;
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

fn create_export_dir(export_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(export_dir).map_err(|error| {
        format!(
            "Canvas export failed: could not create export directory {}: {error}",
            export_dir.display()
        )
    })
}

fn build_output_path(export_dir: &Path) -> Result<PathBuf, String> {
    let epoch_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("Canvas export failed: system clock is invalid: {error}"))?
        .as_secs();
    Ok(export_dir.join(format!("forge-canvas-{epoch_seconds}.json")))
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
