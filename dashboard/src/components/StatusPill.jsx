import React, { useEffect, useState } from "react";
import { getStatus } from "../api/client.js";

export default function StatusPill() {
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;

    function refresh() {
      getStatus()
        .then((data) => {
          if (cancelled) return;
          setStatus(data);
          setError(false);
        })
        .catch(() => {
          if (cancelled) return;
          setError(true);
        });
    }

    refresh();
    const id = setInterval(refresh, 10000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
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
  const cifake = status.detectors.find((d) => d.name === "image_branch_cifake");
  const modelLabel = cifake?.available ? "CIFake online" : "heuristics only";
  const modelClass = cifake?.available
    ? "border-signal-cyan/40 text-signal-cyan"
    : "border-signal-amber/40 text-signal-amber";

  return (
    <span className={`font-mono text-xs border rounded-full px-3 py-1 ${modelClass}`}>
      {modelLabel} · {activeCount}/{status.detectors.length} detectors · {status.gpu_available ? "GPU" : "CPU"}
    </span>
  );
}
