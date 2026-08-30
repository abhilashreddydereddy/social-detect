import React from "react";

const CATEGORY_LABELS = {
  frequency_artifact: "Frequency artifact",
  texture_repetition: "Texture repetition",
  noise_residual: "Sensor noise",
  compression_artifact: "Compression artifact",
  lighting_inconsistency: "Lighting inconsistency",
  metadata: "Metadata",
  semantic: "Semantic model",
  temporal_inconsistency: "Temporal consistency",
  face_artifact: "Face artifact",
  audio_artifact: "Audio authenticity",
};

function severityColor(score) {
  if (score >= 0.7) return "text-signal-red border-signal-red/40 bg-signal-red/10";
  if (score >= 0.5) return "text-signal-amber border-signal-amber/40 bg-signal-amber/10";
  return "text-signal-cyan border-signal-cyan/40 bg-signal-cyan/10";
}

export default function EvidenceList({ evidence }) {
  if (!evidence || evidence.length === 0) {
    return <p className="text-sm text-slate400">No specific evidence signals were returned.</p>;
  }

  const visible = evidence.filter((e) => e.detector === "image_branch_cifake");
  const rows = visible.length ? visible : evidence;

  return (
    <ul className="space-y-2">
      {rows.map((e, i) => (
        <li
          key={i}
          className={`border rounded-md px-3 py-2.5 text-sm flex gap-3 items-start ${severityColor(e.score)}`}
        >
          <span className="font-mono text-xs mt-0.5 shrink-0 opacity-80">
            {Math.round(e.score * 100)}%
          </span>
          <div>
            <div className="font-display text-xs uppercase tracking-wide opacity-90 mb-0.5">
              {CATEGORY_LABELS[e.category] || e.category}
              <span className="text-slate400 normal-case tracking-normal font-mono ml-2 opacity-70">
                {e.detector}
              </span>
            </div>
            <p className="text-paper/90 leading-snug">{e.summary}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}
