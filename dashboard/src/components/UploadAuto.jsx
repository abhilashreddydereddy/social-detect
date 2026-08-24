import React, { useRef, useState } from "react";
import { analyzeMedia } from "../api/client.js";

/**
 * Unified upload: auto-detects whether the file is an image or a video.
 * Videos are frame-cut and scored as images while audio is analyzed in parallel.
 */
export default function UploadAuto({ onResult, onLoading, onError }) {
  const [preview, setPreview] = useState(null);
  const [previewKind, setPreviewKind] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [detected, setDetected] = useState(null);
  const inputRef = useRef(null);

  function sniffKind(file) {
    if (!file) return null;
    if (file.type.startsWith("image/")) return "image";
    if (file.type.startsWith("video/")) return "video";
    const name = (file.name || "").toLowerCase();
    if (/\.(jpe?g|png|webp|gif|bmp|tiff?)$/.test(name)) return "image";
    if (/\.(mp4|webm|mov|mkv|avi|m4v)$/.test(name)) return "video";
    return "auto";
  }

  async function handleFile(file) {
    if (!file) return;
    const kind = sniffKind(file);
    setFileName(file.name);
    setDetected(kind);
    setPreviewKind(kind === "video" ? "video" : "image");
    setPreview(URL.createObjectURL(file));
    onError(null);
    onLoading(true);
    try {
      const result = await analyzeMedia(file);
      onResult(result);
    } catch (err) {
      onError(err.message);
    } finally {
      onLoading(false);
    }
  }

  return (
    <div
      className="border-2 border-dashed border-ink-700 rounded-lg p-8 text-center hover:border-signal-cyan/50 transition-colors cursor-pointer"
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        handleFile(e.dataTransfer.files?.[0]);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*,video/mp4,video/webm,video/quicktime,video/x-matroska,.mp4,.webm,.mov,.mkv,.jpg,.jpeg,.png,.webp,.gif"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      {preview && previewKind === "video" ? (
        <video src={preview} className="max-h-48 mx-auto rounded mb-3" controls />
      ) : preview ? (
        <img src={preview} alt="preview" className="max-h-48 mx-auto rounded mb-3" />
      ) : (
        <div className="text-4xl mb-3 text-slate400">⬆</div>
      )}
      <p className="font-mono text-sm text-slate400">
        {fileName ? fileName : "Drop an image or video, or click to browse"}
      </p>
      <p className="text-xs text-slate400 mt-1">
        {detected
          ? `Detected: ${detected} — backend confirms via content sniffing`
          : "Auto-detects image vs video. Videos: frames + parallel audio authenticity."}
      </p>
    </div>
  );
}
