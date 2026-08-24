/**
 * YouTube adapter — STUB. Not yet wired into manifest.json.
 *
 * Two very different surfaces to handle:
 *   - Watch pages (`/watch?v=...`): single `<video>` inside `#movie_player`.
 *   - Shorts (`/shorts/...`): a vertically-swiping feed of `<video>` elements
 *     inside `ytd-reel-video-renderer`, similar in spirit to TikTok.
 * Both stream via MSE, so this is another primarily frame-capture-based
 * adapter. Long-form video also benefits most from *temporal* analysis
 * (sample frames across the whole video, not just one) -- consider adding
 * a "sample N frames while paused/seeking" flow here rather than a single
 * snapshot, using the video element's `currentTime` + a seek-and-capture
 * loop, then POSTing multiple frames as a synthetic clip.
 *
 * To activate: fill in the TODOs, then add to manifest.json:
 *   {
 *     "matches": ["https://www.youtube.com/*"],
 *     "js": ["content_scripts/core.js", "content_scripts/platforms/youtube.js"],
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
      canvas.width = video.videoWidth || video.clientWidth || 1280;
      canvas.height = video.videoHeight || video.clientHeight || 720;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return { kind: "frame", dataUrl: canvas.toDataURL("image/jpeg", 0.92), sourceUrl: location.href };
    } catch (err) {
      console.warn("[Social Detect] Could not capture YouTube frame:", err);
      return null;
    }
  }

  async function extractMedia(mediaEl) {
    if (mediaEl.tagName !== "VIDEO") return null;
    return captureVideoFrame(mediaEl); // YouTube is effectively always MSE
  }

  if (window.SocialDetectCore) {
    window.SocialDetectCore.start({
      name: "youtube",
      postSelector: "#movie_player, ytd-reel-video-renderer",
      findMediaElement,
      extractMedia,
    });
  }
})();
