import "./StatusBadge.css";

export function StatusBadge({ status }: { status: string }) {
  const error = status === "failed" || status === "cancelled";
  return <span className={error ? "status status--error" : "status"}>{status}</span>;
}
