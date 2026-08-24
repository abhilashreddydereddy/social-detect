import React, { useState } from "react";
import UploadImage from "./components/UploadImage.jsx";
import UploadVideo from "./components/UploadVideo.jsx";
import UploadAuto from "./components/UploadAuto.jsx";
import UrlAnalyze from "./components/UrlAnalyze.jsx";
import ResultPanel from "./components/ResultPanel.jsx";
import StatusPill from "./components/StatusPill.jsx";

const TABS = [
  { id: "auto", label: "Auto-detect" },
  { id: "image", label: "Upload image" },
  { id: "video", label: "Upload video" },
  { id: "url", label: "Paste URL" },
];

export default function App() {
  const [tab, setTab] = useState("auto");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  function handleResult(r) {
    setResult(r);
    setHistory((h) => [{ id: r.request_id, classification: r.classification, ai_probability: r.ai_probability, media_type: r.media_type }, ...h].slice(0, 8));
  }

  return (
    <div className="min-h-screen bg-lab-grid">
      <header className="border-b border-ink-800 bg-ink-950/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
          <div>
            <h1 className="font-display text-xl tracking-tight">
              Social Detect <span className="text-slate400 font-normal">/ Evidence Lab</span>
            </h1>
            <p className="text-xs text-slate400 mt-0.5 font-mono">
              probability, not verdicts — demo &amp; testing console
            </p>
          </div>
          <StatusPill />
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-5 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <div className="flex gap-1 bg-ink-900 border border-ink-700 rounded-lg p-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex-1 font-mono text-xs py-2 rounded-md transition-colors ${
                  tab === t.id ? "bg-ink-700 text-paper" : "text-slate400 hover:text-paper"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {tab === "auto" && <UploadAuto onResult={handleResult} onLoading={setLoading} onError={setError} />}
          {tab === "image" && <UploadImage onResult={handleResult} onLoading={setLoading} onError={setError} />}
          {tab === "video" && <UploadVideo onResult={handleResult} onLoading={setLoading} onError={setError} />}
          {tab === "url" && <UrlAnalyze onResult={handleResult} onLoading={setLoading} onError={setError} />}

          {error && (
            <div className="border border-signal-red/40 bg-signal-red/10 text-signal-red text-sm rounded-md px-4 py-3 font-mono">
              {error}
            </div>
          )}

          {history.length > 0 && (
            <div>
              <h3 className="font-display text-sm uppercase tracking-wide text-slate400 mb-3">
                Recent analyses
              </h3>
              <ul className="space-y-1.5 font-mono text-xs">
                {history.map((h) => (
                  <li
                    key={h.id}
                    className="flex justify-between border border-ink-800 rounded px-3 py-2 text-slate400"
                  >
                    <span>
                      {h.media_type} · {h.id.split("-")[0]}
                    </span>
                    <span>
                      {h.classification} ({Math.round(h.ai_probability * 100)}%)
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="lg:col-span-3">
          {loading && (
            <div className="border border-ink-700 rounded-lg p-10 text-center font-mono text-sm text-slate400 relative overflow-hidden scanline">
              detecting media · cutting frames · scoring audio…
            </div>
          )}
          {!loading && result && <ResultPanel result={result} />}
          {!loading && !result && (
            <div className="border border-dashed border-ink-700 rounded-lg p-10 text-center font-mono text-sm text-slate400">
              Analysis results will appear here.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
