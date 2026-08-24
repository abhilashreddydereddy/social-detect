/**
 * Reddit adapter — STUB. Not yet wired into manifest.json.
 *
 * Reddit's new UI renders posts inside `shreddit-post` custom elements,
 * with media inside a nested `<img>`/`<video>` (sometimes inside its own
 * shadow root, which requires piercing with `.shadowRoot?.querySelector`).
 * Old Reddit (old.reddit.com) uses plain `.thing` post containers instead
 * -- you likely want two selector strategies gated on hostname.
 *
 * To activate: fill in the TODOs, then add to manifest.json:
 *   {
 *     "matches": ["https://www.reddit.com/*", "https://old.reddit.com/*"],
 *     "js": ["content_scripts/core.js", "content_scripts/platforms/reddit.js"],
 *     "run_at": "document_idle"
 *   }
 */
(() => {
  function findMediaElement(post) {
    // TODO: handle shreddit-post's shadow root:
    // const shredditPost = post.shadowRoot ? post : post.closest("shreddit-post");
    // return shredditPost?.shadowRoot?.querySelector("img, video") ?? post.querySelector("img, video");
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
      name: "reddit",
      postSelector: "shreddit-post, .thing",
      findMediaElement,
      extractMedia,
    });
  }
})();
