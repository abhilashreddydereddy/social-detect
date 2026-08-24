import React from "react";
import ProbabilityGauge from "./ProbabilityGauge.jsx";
import EvidenceList from "./EvidenceList.jsx";
import FrameTimeline from "./FrameTimeline.jsx";

const CLASSIFICATION_STYLE = {
  "Likely Authentic": "text-signal-cyan border-signal-cyan/50",
  "Possibly Manipulated": "text-signal-amber border-signal-amber/50",
  "Likely AI Generated": "text-signal-red border-signal-red/50",
  Inconclusive: "text-slate400 border-slate400/50",
};

export default function ResultPanel({ result }) {
  if (!result) return null;

  const caseId = result.request_id.split("-")[0];
  const classStyle = CLASSIFICATION_STYLE[result.classification] || CLASSIFICATION_STYLE.Inconclusive;

  return (
    <div className="relative bg-ink-900 border border-ink-700 rounded-b-lg rounded-t-sm shadow-xl">
      {/* torn / perforated top edge, evidence-tag signature */}
      <div className="h-3 perforated-top" />

      <div className="p-6 border-b border-ink-700 flex items-start justify-between gap-4">
        <div>
          <div className="font-mono text-[11px] text-slate400 tracking-widest uppercase mb-1">
            Case #{caseId} — {result.media_type}
          </div>
          <h2 className="font-display text-2xl">{result.classification}</h2>
        </div>
        <span className={`font-mono text-xs border rounded-full px-3 py-1 ${classStyle} shrink-0`}>
          {Math.round(result.confidence * 100)}% confidence
        </span>
      </div>

      <div className="p-6 border-b border-ink-700">
        <div className="flex items-baseline justify-between mb-2">
          <span className="font-mono text-xs text-slate400 uppercase tracking-wide">AI Probability</span>
          <span className="font-mono text-3xl">{Math.round(result.ai_probability * 100)}%</span>
        </div>
        <ProbabilityGauge probability={result.ai_probability} confidence={result.confidence} />
      </div>

      {result.frame_results && result.frame_results.length > 0 && (
        <div className="p-6 border-b border-ink-700">
          <h3 className="font-display text-sm uppercase tracking-wide text-slate400 mb-3">
            Frame-by-frame breakdown
          </h3>
          <FrameTimeline frames={result.frame_results} />
        </div>
      )}

      <div className="p-6 border-b border-ink-700">
        <h3 className="font-display text-sm uppercase tracking-wide text-slate400 mb-3">Evidence</h3>
        <EvidenceList evidence={result.evidence} />
      </div>

      <div className="p-6 border-b border-ink-700">
        <h3 className="font-display text-sm uppercase tracking-wide text-slate400 mb-3">
          Detector comparison
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm font-mono">
            <thead>
              <tr className="text-left text-slate400 text-xs uppercase tracking-wide">
                <th className="pb-2 font-normal">Detector</th>
                <th className="pb-2 font-normal">AI prob.</th>
                <th className="pb-2 font-normal">Confidence</th>
                <th className="pb-2 font-normal">Status</th>
              </tr>
            </thead>
            <tbody>
              {result.detector_results.map((d, i) => (
                <tr key={i} className="border-t border-ink-800">
                  <td className="py-2 pr-3">{d.detector}</td>
                  <td className="py-2 pr-3">
                    {d.error ? "—" : `${Math.round(d.ai_probability * 100)}%`}
                  </td>
                  <td className="py-2 pr-3">{d.error ? "—" : `${Math.round(d.confidence * 100)}%`}</td>
                  <td className="py-2 text-xs">
                    {d.error ? (
                      <span className="text-signal-amber">unavailable</span>
                    ) : (
                      <span className="text-signal-cyan">ok</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="p-6 flex items-start justify-between gap-6">
        <div className="font-mono text-xs text-slate400 space-y-1">
          {Object.entries(result.metadata || {}).map(([k, v]) => (
            <div key={k}>
              <span className="opacity-70">{k}:</span> {String(v)}
            </div>
          ))}
          <div className="opacity-70">processed in {result.processing_time_ms}ms</div>
        </div>
        <p className="text-xs text-slate400 max-w-xs text-right leading-relaxed">{result.disclaimer}</p>
      </div>
    </div>
  );
}
