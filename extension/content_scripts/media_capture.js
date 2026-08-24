/**
 * Shared media capture for Social Detect content scripts.
 *
 * Social CDNs (Instagram/YouTube) almost never allow the backend to download
 * media by URL. So we ALWAYS extract pixels in the page (or via a tab
 * screenshot) and upload a JPEG data URL to the backend.
 *
 * Strategy (in order):
 *   1. Draw <img>/<video> onto a canvas → JPEG data URL
 *   2. Fetch the media URL from the page context → blob → data URL
 *   3. chrome.tabs.captureVisibleTab + crop to the element rect
 */
(() => {
  function loadImage(dataUrl) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Failed to decode image"));
      img.src = dataUrl;
    });
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error || new Error("FileReader failed"));
      reader.readAsDataURL(blob);
    });
  }

  function canvasFromElement(el) {
    try {
      const isVideo = el.tagName === "VIDEO";
      const width = isVideo
        ? (el.videoWidth || el.clientWidth || 0)
        : (el.naturalWidth || el.clientWidth || 0);
      const height = isVideo
        ? (el.videoHeight || el.clientHeight || 0)
        : (el.naturalHeight || el.clientHeight || 0);
      if (width < 2 || height < 2) return null;

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(el, 0, 0, width, height);
      // Throws SecurityError if canvas is tainted (CORS).
      const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
      if (!dataUrl || dataUrl.length < 100) return null;
      return dataUrl;
    } catch (err) {
      console.warn("[Social Detect] canvas capture blocked:", err?.message || err);
      return null;
    }
  }

  async function fetchAsDataUrl(url) {
    if (!url || url.startsWith("blob:") || url.startsWith("data:") || url.startsWith("mediasource:")) {
      return null;
    }
    try {
      const resp = await fetch(url, {
        credentials: "include",
        mode: "cors",
        cache: "force-cache",
      });
      if (!resp.ok) return null;
      const blob = await resp.blob();
      if (!blob || blob.size < 32) return null;
      // Only accept image-ish payloads for the image pipeline.
      if (blob.type && !blob.type.startsWith("image/") && !blob.type.startsWith("video/")) {
        // Some CDNs omit content-type; still try.
        if (blob.type && blob.type !== "application/octet-stream") return null;
      }
      if (blob.type.startsWith("video/")) {
        // Backend video path needs a real clip; skip URL-fetched videos here.
        return null;
      }
      return await blobToDataUrl(blob);
    } catch (err) {
      console.warn("[Social Detect] page fetch of media URL failed:", err?.message || err);
      return null;
    }
  }

  async function captureViewportCrop(el) {
    const rect = el.getBoundingClientRect();
    if (rect.width < 20 || rect.height < 20) return null;

    const shot = await chrome.runtime.sendMessage({ type: "CAPTURE_VISIBLE_TAB" });
    if (!shot?.ok || !shot.dataUrl) {
      console.warn("[Social Detect] captureVisibleTab failed:", shot?.error);
      return null;
    }

    try {
      const img = await loadImage(shot.dataUrl);
      const dpr = window.devicePixelRatio || 1;
      const sx = Math.max(0, Math.round(rect.left * dpr));
      const sy = Math.max(0, Math.round(rect.top * dpr));
      const sw = Math.min(img.width - sx, Math.max(1, Math.round(rect.width * dpr)));
      const sh = Math.min(img.height - sy, Math.max(1, Math.round(rect.height * dpr)));
      if (sw < 2 || sh < 2) return null;

      const canvas = document.createElement("canvas");
      canvas.width = sw;
      canvas.height = sh;
      canvas.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
      return canvas.toDataURL("image/jpeg", 0.92);
    } catch (err) {
      console.warn("[Social Detect] viewport crop failed:", err?.message || err);
      return null;
    }
  }

  /**
   * Extract analyzable media from an <img> or <video>.
   * Always returns { kind: "frame", dataUrl, sourceUrl, method } or null.
   */
  async function extractForBackend(mediaEl) {
    if (!mediaEl || !mediaEl.isConnected) return null;
    const tag = mediaEl.tagName;
    if (tag !== "IMG" && tag !== "VIDEO") return null;

    const sourceUrl = location.href;

    // 1) Direct canvas read of the rendered element
    const fromCanvas = canvasFromElement(mediaEl);
    if (fromCanvas) {
      return { kind: "frame", dataUrl: fromCanvas, sourceUrl, method: "canvas", mediaTag: tag };
    }

    // 2) Fetch the resource URL in-page (has cookies / referrer Instagram needs)
    if (tag === "IMG") {
      const url = mediaEl.currentSrc || mediaEl.src;
      const fromFetch = await fetchAsDataUrl(url);
      if (fromFetch) {
        return { kind: "frame", dataUrl: fromFetch, sourceUrl, method: "fetch", mediaTag: tag };
      }
    }

    // 3) Tab screenshot + crop (works even when CORS taints canvas)
    const fromViewport = await captureViewportCrop(mediaEl);
    if (fromViewport) {
      return { kind: "frame", dataUrl: fromViewport, sourceUrl, method: "viewport", mediaTag: tag };
    }

    return null;
  }

  window.SocialDetectMedia = {
    extractForBackend,
    canvasFromElement,
    captureViewportCrop,
  };
})();
