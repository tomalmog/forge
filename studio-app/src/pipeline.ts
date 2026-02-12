import { PipelineNode, PipelineNodeType } from "./types";

const DEFAULT_NODE_DURATION_SECONDS: Record<PipelineNodeType, number> = {
  ingest: 60,
  filter: 30,
  train: 240,
  export: 60,
  chat: 20,
  custom: 30,
};

export function buildDefaultNode(
  type: PipelineNodeType,
  nodeIndex: number = 0,
): PipelineNode {
  const position = computeDefaultNodePosition(nodeIndex);
  const id = `${type}-${Math.random().toString(16).slice(2, 10)}`;
  if (type === "ingest") {
    return {
      id,
      type,
      title: "Ingest",
      canvas_x: position.x,
      canvas_y: position.y,
      config: {
        source: "./datasets/my_source",
        dataset: "demo",
        quality_model: "perplexity",
        incremental: "false",
        resume: "false",
      },
    };
  }
  if (type === "filter") {
    return {
      id,
      type,
      title: "Filter",
      canvas_x: position.x,
      canvas_y: position.y,
      config: { dataset: "demo", language: "en", min_quality: "0.35" },
    };
  }
  if (type === "train") {
    return {
      id,
      type,
      title: "Train",
      canvas_x: position.x,
      canvas_y: position.y,
      config: {
        dataset: "demo",
        output_dir: "./outputs/train/demo",
        epochs: "2",
        learning_rate: "0.001",
        batch_size: "16",
        max_token_length: "256",
        validation_split: "0.1",
        hidden_dim: "256",
        num_layers: "2",
        attention_heads: "8",
        mlp_hidden_dim: "1024",
        mlp_layers: "2",
        dropout: "0.1",
        position_embedding_type: "learned",
        vocabulary_size: "",
        architecture_file: "",
        custom_loop_file: "",
        initial_weights_path: "",
      },
    };
  }
  if (type === "export") {
    return {
      id,
      type,
      title: "Export Training",
      canvas_x: position.x,
      canvas_y: position.y,
      config: {
        dataset: "demo",
        output_dir: "./outputs/export/demo",
        shard_size: "500",
      },
    };
  }
  if (type === "chat") {
    return {
      id,
      type,
      title: "Chat",
      canvas_x: position.x,
      canvas_y: position.y,
      config: {
        dataset: "demo",
        model_path: "./outputs/train/demo/model.pt",
        prompt: "hello",
        max_new_tokens: "80",
        temperature: "0.8",
        top_k: "40",
        position_embedding_type: "learned",
      },
    };
  }
  return {
    id,
    type,
    title: "Custom Step",
    canvas_x: position.x,
    canvas_y: position.y,
    config: { args: "versions --dataset demo" },
  };
}

export function toForgeArgs(node: PipelineNode): string[] {
  if (node.type === "ingest") {
    const args = [
      "ingest",
      node.config.source,
      "--dataset",
      node.config.dataset,
    ];
    args.push("--quality-model", node.config.quality_model || "perplexity");
    if (isTrue(node.config.incremental)) {
      args.push("--incremental");
    }
    if (isTrue(node.config.resume)) {
      args.push("--resume");
    }
    return args;
  }
  if (node.type === "filter") {
    const args = ["filter", "--dataset", node.config.dataset];
    if (node.config.language) {
      args.push("--language", node.config.language);
    }
    if (node.config.min_quality) {
      args.push("--min-quality", node.config.min_quality);
    }
    return args;
  }
  if (node.type === "train") {
    const args = [
      "train",
      "--dataset",
      node.config.dataset,
      "--output-dir",
      node.config.output_dir,
    ];
    appendOptionalArg(args, "--epochs", node.config.epochs);
    appendOptionalArg(args, "--learning-rate", node.config.learning_rate);
    appendOptionalArg(args, "--batch-size", node.config.batch_size);
    appendOptionalArg(args, "--max-token-length", node.config.max_token_length);
    appendOptionalArg(args, "--validation-split", node.config.validation_split);
    appendOptionalArg(args, "--hidden-dim", node.config.hidden_dim);
    appendOptionalArg(args, "--num-layers", node.config.num_layers);
    appendOptionalArg(args, "--attention-heads", node.config.attention_heads);
    appendOptionalArg(args, "--mlp-hidden-dim", node.config.mlp_hidden_dim);
    appendOptionalArg(args, "--mlp-layers", node.config.mlp_layers);
    appendOptionalArg(args, "--dropout", node.config.dropout);
    appendOptionalArg(
      args,
      "--position-embedding-type",
      node.config.position_embedding_type,
    );
    appendOptionalArg(args, "--vocabulary-size", node.config.vocabulary_size);
    appendOptionalArg(
      args,
      "--architecture-file",
      node.config.architecture_file,
    );
    appendOptionalArg(args, "--custom-loop-file", node.config.custom_loop_file);
    appendOptionalArg(
      args,
      "--initial-weights-path",
      node.config.initial_weights_path,
    );
    return args;
  }
  if (node.type === "export") {
    return [
      "export-training",
      "--dataset",
      node.config.dataset,
      "--output-dir",
      node.config.output_dir,
      "--shard-size",
      node.config.shard_size,
      "--include-metadata",
    ];
  }
  if (node.type === "chat") {
    const args = [
      "chat",
      "--dataset",
      node.config.dataset,
      "--model-path",
      node.config.model_path,
      "--prompt",
      node.config.prompt,
    ];
    appendOptionalArg(args, "--max-new-tokens", node.config.max_new_tokens);
    appendOptionalArg(args, "--temperature", node.config.temperature);
    appendOptionalArg(args, "--top-k", node.config.top_k);
    appendOptionalArg(args, "--version-id", node.config.version_id);
    appendOptionalArg(
      args,
      "--architecture-file",
      node.config.architecture_file,
    );
    appendOptionalArg(args, "--max-token-length", node.config.max_token_length);
    appendOptionalArg(args, "--vocabulary-size", node.config.vocabulary_size);
    appendOptionalArg(args, "--hidden-dim", node.config.hidden_dim);
    appendOptionalArg(args, "--num-layers", node.config.num_layers);
    appendOptionalArg(args, "--attention-heads", node.config.attention_heads);
    appendOptionalArg(args, "--mlp-hidden-dim", node.config.mlp_hidden_dim);
    appendOptionalArg(args, "--mlp-layers", node.config.mlp_layers);
    appendOptionalArg(args, "--dropout", node.config.dropout);
    appendOptionalArg(
      args,
      "--position-embedding-type",
      node.config.position_embedding_type,
    );
    return args;
  }
  return splitArgs(node.config.args);
}

export function estimateNodeDurationSeconds(
  nodeType: PipelineNodeType,
): number {
  return DEFAULT_NODE_DURATION_SECONDS[nodeType] ?? 30;
}

function isTrue(value: string | undefined): boolean {
  return value?.toLowerCase() === "true";
}

function splitArgs(rawArgs: string | undefined): string[] {
  if (!rawArgs) {
    return ["versions"];
  }
  return rawArgs
    .split(" ")
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

function appendOptionalArg(
  args: string[],
  flag: string,
  value: string | undefined,
): void {
  if (value && value.trim().length > 0) {
    args.push(flag, value.trim());
  }
}

function computeDefaultNodePosition(nodeIndex: number): {
  x: number;
  y: number;
} {
  const columnCount = 3;
  const xSpacing = 230;
  const ySpacing = 160;
  const baseX = 20;
  const baseY = 20;
  const row = Math.floor(nodeIndex / columnCount);
  const column = nodeIndex % columnCount;
  return {
    x: baseX + column * xSpacing,
    y: baseY + row * ySpacing,
  };
}
