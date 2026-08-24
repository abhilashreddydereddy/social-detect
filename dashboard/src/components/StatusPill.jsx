import React, { useEffect, useState } from "react";
import { getStatus } from "../api/client.js";

export default function StatusPill() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getStatus()
      .then(setStatus)
      .catch(() => setError(true));
  }, []);

  if (error) {
    return (
      <span className="font-mono text-xs border border-signal-red/40 text-signal-red rounded-full px-3 py-1">
        backend unreachable
      </span>
    );
  }
  if (!status) {
    return (
      <span className="font-mono text-xs border border-ink-700 text-slate400 rounded-full px-3 py-1">
        connecting…
      </span>
    );
  }

  const activeCount = status.detectors.filter((d) => d.available).length;
  return (
    <span className="font-mono text-xs border border-signal-cyan/40 text-signal-cyan rounded-full px-3 py-1">
      {activeCount}/{status.detectors.length} detectors online · {status.gpu_available ? "GPU" : "CPU"}
    </span>
  );
}
