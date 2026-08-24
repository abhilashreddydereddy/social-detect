/**
 * X (twitter.com / x.com) adapter — STUB.
 *
 * Not yet wired into manifest.json. To activate:
 *   1. Fill in the selectors below (X renders tweets as
 *      `article[data-testid="tweet"]`, media inside
 *      `div[data-testid="tweetPhoto"]` / <video>, as of late 2025 --
 *      verify against the current DOM before shipping, X changes markup
 *      periodically).
 *   2. Add a content_scripts entry to manifest.json:
 *      {
 *        "matches": ["https://x.com/*", "https://twitter.com/*"],
 *        "js": ["content_scripts/core.js", "content_scripts/platforms/x.js"],
 *        "run_at": "document_idle"
 *      }
 *   3. X's images are served from pbs.twimg.com with predictable
 *      `?format=jpg&name=orig` query params for full-resolution originals --
 *      prefer that over the thumbnail URL rendered in the DOM.
 */
(() => {
  function findMediaElement(post) {
    // TODO: replace with real selectors, e.g.:
    // return post.querySelector('div[data-testid="tweetPhoto"] img, video');
    return post.querySelector("img, video");
  }

  async function extractMedia(mediaEl) {
    if (mediaEl.tagName === "IMG") {
      const url = mediaEl.currentSrc || mediaEl.src;
      return url ? { kind: "url", mediaType: "image", url } : null;
    }
    if (mediaEl.tagName === "VIDEO") {
      const src = mediaEl.currentSrc || mediaEl.src;
      if (src && !src.startsWith("blob:")) {
        return { kind: "url", mediaType: "video", url: src };
      }
      // TODO: add the same canvas-frame-capture fallback used in instagram.js
      // for MSE-streamed video.
      return null;
    }
    return null;
  }

  if (window.SocialDetectCore) {
    window.SocialDetectCore.start({
      name: "x",
      postSelector: 'article[data-testid="tweet"], article',
      findMediaElement,
      extractMedia,
    });
  }
})();
