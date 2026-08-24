/**
 * Facebook adapter — STUB. Not yet wired into manifest.json.
 *
 * Facebook's feed markup uses deeply nested, obfuscated div soup with no
 * stable class names; `[role="article"]` is the most durable structural
 * selector for a feed post. Expect to spend real time here validating
 * against A/B-tested layout variants Facebook runs concurrently.
 *
 * To activate: fill in the TODOs, then add to manifest.json:
 *   {
 *     "matches": ["https://www.facebook.com/*"],
 *     "js": ["content_scripts/core.js", "content_scripts/platforms/facebook.js"],
 *     "run_at": "document_idle"
 *   }
 */
(() => {
  function findMediaElement(post) {
    // TODO: Facebook lazy-loads images as background-image on divs in some
    // layouts, not just <img> tags -- may need a background-image extraction
    // fallback in addition to querySelector("img, video").
    return post.querySelector("img, video");
  }

  async function extractMedia(mediaEl) {
    if (mediaEl.tagName === "IMG") {
      const url = mediaEl.currentSrc || mediaEl.src;
      return url ? { kind: "url", mediaType: "image", url } : null;
    }
    if (mediaEl.tagName === "VIDEO") {
      const src = mediaEl.currentSrc || mediaEl.src;
      return src && !src.startsWith("blob:") ? { kind: "url", mediaType: "video", url: src } : null;
    }
    return null;
  }

  if (window.SocialDetectCore) {
    window.SocialDetectCore.start({
      name: "facebook",
      postSelector: '[role="article"]',
      findMediaElement,
      extractMedia,
    });
  }
})();
