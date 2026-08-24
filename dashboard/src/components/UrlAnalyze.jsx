import React, { useState } from "react";
import { analyzeUrl } from "../api/client.js";

const PLATFORMS = ["auto", "instagram", "x", "reddit", "facebook", "tiktok", "youtube"];

export default function UrlAnalyze({ onResult, onLoading, onError }) {
  const [url, setUrl] = useState("");
  const [platform, setPlatform] = useState("auto");

  async function handleSubmit(e) {
    e.preventDefault();
    if (!url.trim()) return;
    onError(null);
    onLoading(true);
    try {
      const result = await analyzeUrl(url.trim(), platform === "auto" ? null : platform);
      onResult(result);
    } catch (err) {
      onError(err.message);
    } finally {
      onLoading(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="border-2 border-dashed border-ink-700 rounded-lg p-8">
      <label className="block font-mono text-xs uppercase tracking-wide text-slate400 mb-2">
        Direct media URL
      </label>
      <input
        type="url"
        required
        placeholder="https://example.com/photo.jpg"
        value={url}
        onChange={(e) => setUrl(e.target.value)}
        className="w-full bg-ink-800 border border-ink-700 rounded-md px-3 py-2.5 font-mono text-sm mb-4 focus:border-signal-cyan/60 outline-none"
      />

      <label className="block font-mono text-xs uppercase tracking-wide text-slate400 mb-2">
        Platform hint (optional)
      </label>
      <select
        value={platform}
        onChange={(e) => setPlatform(e.target.value)}
        className="w-full bg-ink-800 border border-ink-700 rounded-md px-3 py-2.5 font-mono text-sm mb-4 outline-none"
      >
        {PLATFORMS.map((p) => (
          <option key={p} value={p}>
            {p}
          </option>
        ))}
      </select>

      <button
        type="submit"
        className="w-full bg-signal-cyan/90 hover:bg-signal-cyan text-ink-950 font-display font-medium rounded-md py-2.5 transition-colors"
      >
        Analyze
      </button>

      <p className="text-xs text-slate400 mt-3 leading-relaxed">
        This works best with a direct link to an image or video file. Social post page URLs
        (e.g. an Instagram permalink) generally can't be scraped server-side — use the browser
        extension on the actual post instead, which reads the media straight from the page.
      </p>
    </form>
  );
}
