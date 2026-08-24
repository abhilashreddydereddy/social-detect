/**
 * TikTok adapter — STUB. Not yet wired into manifest.json.
 *
 * TikTok's feed items live in elements matching `[data-e2e="recommend-list-item-container"]`
 * (For You feed) or `[data-e2e="user-post-item"]` (profile grid), each
 * wrapping a <video>. TikTok almost always streams via MSE (blob: URLs),
 * so the canvas-frame-capture fallback (see instagram.js) will be the
 * primary path here, not direct video URL analysis.
 *
 * To activate: fill in the TODOs, then add to manifest.json:
 *   {
 *     "matches": ["https://www.tiktok.com/*"],
 *     "js": ["content_scripts/core.js", "content_scripts/platforms/tiktok.js"],
 *     "run_at": "document_idle"
 *   }
 */
(() => {
  function findMediaElement(post) {
    return post.querySelector("video");
  }

  function captureVideoFrame(video) {
    try {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || video.clientWidth || 576;
      canvas.height = video.videoHeight || video.clientHeight || 1024;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return { kind: "frame", dataUrl: canvas.toDataURL("image/jpeg", 0.92), sourceUrl: location.href };
    } catch (err) {
      console.warn("[Social Detect] Could not capture TikTok frame:", err);
      return null;
    }
  }

  async function extractMedia(mediaEl) {
    if (mediaEl.tagName !== "VIDEO") return null;
    const src = mediaEl.currentSrc || mediaEl.src;
    if (src && !src.startsWith("blob:")) {
      return { kind: "url", mediaType: "video", url: src };
    }
    return captureVideoFrame(mediaEl);
  }

  if (window.SocialDetectCore) {
    window.SocialDetectCore.start({
      name: "tiktok",
      postSelector: '[data-e2e="recommend-list-item-container"], [data-e2e="user-post-item"]',
      findMediaElement,
      extractMedia,
    });
  }
})();
