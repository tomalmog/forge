//! Forge command execution helpers for Studio.

use crate::commands::forge_task_store::CommandTaskStore;
use crate::models::{CommandTaskStart, CommandTaskStatus};
use tauri::State;

const ALLOWED_COMMANDS: [&str; 5] = ["ingest", "filter", "train", "export-training", "versions"];

#[tauri::command]
pub fn start_forge_command(
    data_root: String,
    args: Vec<String>,
    task_store: State<'_, CommandTaskStore>,
) -> Result<CommandTaskStart, String> {
    validate_args(&args)?;
    Ok(task_store.start_task(data_root, args))
}

#[tauri::command]
pub fn get_forge_command_status(
    task_id: String,
    task_store: State<'_, CommandTaskStore>,
) -> Result<CommandTaskStatus, String> {
    task_store.get_task_status(&task_id)
}

fn validate_args(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("Forge args must include a command".to_string());
    }
    let command = args[0].as_str();
    if ALLOWED_COMMANDS.contains(&command) {
        Ok(())
    } else {
        Err(format!("Unsupported command '{command}' for Studio execution"))
    }
}

#[cfg(test)]
mod tests {
    use super::validate_args;

    #[test]
    fn validate_args_accepts_supported_command() {
        let args = vec!["train".to_string(), "--dataset".to_string(), "demo".to_string()];
        assert!(validate_args(&args).is_ok());
    }

    #[test]
    fn validate_args_rejects_empty_args() {
        let args: Vec<String> = Vec::new();
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn validate_args_rejects_unsupported_command() {
        let args = vec!["shell".to_string()];
        assert!(validate_args(&args).is_err());
    }
}
