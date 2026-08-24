import React, { useState } from "react";

function barColor(prob) {
  if (prob >= 0.7) return "bg-signal-red";
  if (prob >= 0.5) return "bg-signal-amber";
  return "bg-signal-cyan";
}

export default function FrameTimeline({ frames }) {
  const [active, setActive] = useState(null);
  if (!frames || frames.length === 0) return null;

  const activeFrame = active !== null ? frames[active] : frames[frames.length - 1];

  return (
    <div>
      <div className="flex items-end gap-1 h-24 mb-3">
        {frames.map((f, i) => (
          <button
            key={i}
            onMouseEnter={() => setActive(i)}
            onFocus={() => setActive(i)}
            className={`flex-1 rounded-t-sm ${barColor(f.ai_probability)} transition-opacity hover:opacity-100 ${
              active === i ? "opacity-100" : "opacity-60"
            }`}
            style={{ height: `${Math.max(8, f.ai_probability * 100)}%` }}
            aria-label={`Frame at ${f.timestamp_seconds}s, ${Math.round(f.ai_probability * 100)}% AI probability`}
          />
        ))}
      </div>

      <div className="flex items-center gap-4 border border-ink-700 rounded-md p-3 bg-ink-900">
        {activeFrame.thumbnail_base64 ? (
          <img
            src={`data:image/jpeg;base64,${activeFrame.thumbnail_base64}`}
            alt={`Frame at ${activeFrame.timestamp_seconds}s`}
            className="w-24 h-16 object-cover rounded border border-ink-700"
          />
        ) : (
          <div className="w-24 h-16 rounded border border-ink-700 bg-ink-800" />
        )}
        <div className="font-mono text-xs text-slate400 space-y-1">
          <div>t = {activeFrame.timestamp_seconds}s</div>
          <div>
            AI probability:{" "}
            <span className="text-paper">{Math.round(activeFrame.ai_probability * 100)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
