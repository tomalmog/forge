import { FormEvent, useEffect, useMemo, useState } from "react";
import { getForgeCommandStatus, startForgeCommand } from "../api/studioApi";

interface ChatRoomViewProps {
  dataRoot: string;
  selectedDataset: string | null;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

const CHAT_POLL_MS = 700;
const SAMPLING_PRESETS = {
  deterministic: { maxNewTokens: "80", temperature: "0", topK: "0" },
  balanced: { maxNewTokens: "120", temperature: "0.7", topK: "40" },
  creative: { maxNewTokens: "160", temperature: "1.0", topK: "80" },
} as const;
type SamplingPreset = keyof typeof SAMPLING_PRESETS | "custom";

export function ChatRoomView(props: ChatRoomViewProps) {
  const [datasetName, setDatasetName] = useState(
    props.selectedDataset ?? "demo",
  );
  const [modelPath, setModelPath] = useState("./outputs/train/demo/model.pt");
  const [versionId, setVersionId] = useState("");
  const [maxNewTokens, setMaxNewTokens] = useState("120");
  const [temperature, setTemperature] = useState("0.7");
  const [topK, setTopK] = useState("40");
  const [maxTokenLength, setMaxTokenLength] = useState("256");
  const [positionEmbeddingType, setPositionEmbeddingType] = useState("learned");
  const [samplingPreset, setSamplingPreset] =
    useState<SamplingPreset>("balanced");
  const [draftMessage, setDraftMessage] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  useEffect(() => {
    if (!props.selectedDataset) {
      return;
    }
    setDatasetName(props.selectedDataset);
  }, [props.selectedDataset]);

  const canSend = useMemo(
    () =>
      datasetName.trim().length > 0 &&
      modelPath.trim().length > 0 &&
      draftMessage.trim().length > 0 &&
      !isSending,
    [datasetName, modelPath, draftMessage, isSending],
  );

  async function onSendMessage(event: FormEvent) {
    event.preventDefault();
    if (!canSend) {
      return;
    }
    const userText = draftMessage.trim();
    setDraftMessage("");
    setChatError(null);
    setMessages((current) => [...current, { role: "user", content: userText }]);
    setIsSending(true);
    try {
      const prompt = buildPromptText(messages, userText);
      const args = buildChatArgs({
        datasetName,
        modelPath,
        prompt,
        versionId,
        maxNewTokens,
        temperature,
        topK,
        maxTokenLength,
        positionEmbeddingType,
      });
      const taskStart = await startForgeCommand(props.dataRoot, args);
      const taskStatus = await waitForChatTask(taskStart.task_id);
      if (taskStatus.status !== "completed" || taskStatus.exit_code !== 0) {
        throw new Error(taskStatus.stderr || "Chat command failed.");
      }
      const responseText = parseResponseText(taskStatus.stdout);
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: responseText || "(no response generated)",
        },
      ]);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setChatError(message);
    } finally {
      setIsSending(false);
    }
  }

  return (
    <section className="panel">
      <h3>Chat Room</h3>
      <div className="chat-config-grid">
        <label>
          dataset
          <input
            value={datasetName}
            onChange={(event) => setDatasetName(event.currentTarget.value)}
          />
        </label>
        <label>
          model_path
          <input
            value={modelPath}
            onChange={(event) => setModelPath(event.currentTarget.value)}
          />
        </label>
        <label>
          version_id
          <input
            value={versionId}
            onChange={(event) => setVersionId(event.currentTarget.value)}
            placeholder="latest"
          />
        </label>
        <label>
          sampling_preset
          <select
            value={samplingPreset}
            onChange={(event) =>
              applyPreset(event.currentTarget.value as SamplingPreset)
            }
          >
            <option value="deterministic">deterministic</option>
            <option value="balanced">balanced</option>
            <option value="creative">creative</option>
            <option value="custom">custom</option>
          </select>
        </label>
        <label>
          max_new_tokens
          <input
            value={maxNewTokens}
            onChange={(event) =>
              updateSamplingValue("max_new_tokens", event.currentTarget.value)
            }
          />
        </label>
        <label>
          temperature
          <input
            value={temperature}
            onChange={(event) =>
              updateSamplingValue("temperature", event.currentTarget.value)
            }
          />
        </label>
        <label>
          top_k
          <input
            value={topK}
            onChange={(event) =>
              updateSamplingValue("top_k", event.currentTarget.value)
            }
          />
        </label>
        <label>
          max_token_length
          <input
            value={maxTokenLength}
            onChange={(event) => setMaxTokenLength(event.currentTarget.value)}
          />
        </label>
        <label>
          position_embedding_type
          <select
            value={positionEmbeddingType}
            onChange={(event) =>
              setPositionEmbeddingType(event.currentTarget.value)
            }
          >
            <option value="learned">learned</option>
            <option value="sinusoidal">sinusoidal</option>
          </select>
        </label>
      </div>
      <div className="chat-thread">
        {messages.length === 0 ? (
          <p className="chat-empty">
            Send a message to evaluate your trained model response quality.
          </p>
        ) : (
          messages.map((message, index) => (
            <article
              key={`${message.role}-${index}`}
              className={`chat-message ${message.role}`}
            >
              <header>{message.role === "user" ? "You" : "Model"}</header>
              <p>{message.content}</p>
            </article>
          ))
        )}
      </div>
      {chatError ? <p className="chat-error">chat error: {chatError}</p> : null}
      <form className="chat-input-row" onSubmit={onSendMessage}>
        <input
          value={draftMessage}
          onChange={(event) => setDraftMessage(event.currentTarget.value)}
          placeholder="Type a prompt..."
        />
        <button className="button action" type="submit" disabled={!canSend}>
          {isSending ? "Sending..." : "Send"}
        </button>
        <button
          className="button"
          type="button"
          disabled={isSending}
          onClick={() => setMessages([])}
        >
          Clear
        </button>
      </form>
    </section>
  );

  function applyPreset(preset: SamplingPreset): void {
    setSamplingPreset(preset);
    if (preset === "custom") {
      return;
    }
    const values = SAMPLING_PRESETS[preset];
    setMaxNewTokens(values.maxNewTokens);
    setTemperature(values.temperature);
    setTopK(values.topK);
  }

  function updateSamplingValue(
    field: "max_new_tokens" | "temperature" | "top_k",
    value: string,
  ): void {
    setSamplingPreset("custom");
    if (field === "max_new_tokens") {
      setMaxNewTokens(value);
      return;
    }
    if (field === "temperature") {
      setTemperature(value);
      return;
    }
    setTopK(value);
  }
}

async function waitForChatTask(taskId: string) {
  while (true) {
    const status = await getForgeCommandStatus(taskId);
    if (status.status !== "running") {
      return status;
    }
    await sleep(CHAT_POLL_MS);
  }
}

function buildPromptText(
  messages: ChatMessage[],
  currentUserText: string,
): string {
  const historyRows = messages.slice(-6).map((message) => {
    if (message.role === "user") {
      return `User: ${message.content}`;
    }
    return `Assistant: ${message.content}`;
  });
  historyRows.push(`User: ${currentUserText}`);
  historyRows.push("Assistant:");
  return historyRows.join("\n");
}

interface ChatArgOptions {
  datasetName: string;
  modelPath: string;
  prompt: string;
  versionId: string;
  maxNewTokens: string;
  temperature: string;
  topK: string;
  maxTokenLength: string;
  positionEmbeddingType: string;
}

function buildChatArgs(options: ChatArgOptions): string[] {
  const args = [
    "chat",
    "--dataset",
    options.datasetName.trim(),
    "--model-path",
    options.modelPath.trim(),
    "--prompt",
    options.prompt,
  ];
  appendOptionalArg(args, "--version-id", options.versionId);
  appendOptionalArg(args, "--max-new-tokens", options.maxNewTokens);
  appendOptionalArg(args, "--temperature", options.temperature);
  appendOptionalArg(args, "--top-k", options.topK);
  appendOptionalArg(args, "--max-token-length", options.maxTokenLength);
  appendOptionalArg(
    args,
    "--position-embedding-type",
    options.positionEmbeddingType,
  );
  return args;
}

function parseResponseText(stdout: string): string {
  return stdout.trim();
}

function appendOptionalArg(args: string[], flag: string, value: string): void {
  const trimmed = value.trim();
  if (trimmed.length > 0) {
    args.push(flag, trimmed);
  }
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}
