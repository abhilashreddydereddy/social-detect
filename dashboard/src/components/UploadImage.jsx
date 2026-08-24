import React, { useRef, useState } from "react";
import { analyzeImage } from "../api/client.js";

export default function UploadImage({ onResult, onLoading, onError }) {
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState(null);
  const inputRef = useRef(null);

  async function handleFile(file) {
    if (!file) return;
    setFileName(file.name);
    setPreview(URL.createObjectURL(file));
    onError(null);
    onLoading(true);
    try {
      const result = await analyzeImage(file);
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
        accept="image/jpeg,image/png,image/webp,image/gif,image/bmp"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      {preview ? (
        <img src={preview} alt="preview" className="max-h-48 mx-auto rounded mb-3 object-contain" />
      ) : (
        <div className="text-4xl mb-3 text-slate400">◎</div>
      )}
      <p className="font-mono text-sm text-slate400">
        {fileName ? fileName : "Drop an image here, or click to browse"}
      </p>
      <p className="text-xs text-slate400 mt-1">JPG, PNG, WebP, GIF, BMP — up to 50MB</p>
    </div>
  );
}
