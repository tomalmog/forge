interface StatusConsoleProps {
  output: string;
}

export function StatusConsole({ output }: StatusConsoleProps) {
  return (
    <section className="panel">
      <h3>Run Console</h3>
      <pre className="console">{output || "No command output yet."}</pre>
    </section>
  );
}
