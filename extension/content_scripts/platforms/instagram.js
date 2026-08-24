/**
 * Instagram adapter.
 *
 * Instagram's DOM/class names are obfuscated and change often, so this
 * adapter deliberately avoids relying on generated class names and instead
 * uses structural/semantic signals: <article> as the post container, and
 * "the largest non-avatar <img>, or a <video>" as the media element.
 *
 * Video handling: Instagram often streams video via Media Source Extensions,
 * meaning `video.src`/`currentSrc` is a `blob:` URL that only exists inside
 * the page and can't be downloaded server-side. In that case we fall back
 * to analyzing a single captured frame (canvas snapshot) instead of the
 * full clip, and the resulting overlay is built from an image analysis --
 * still explainable, just frame-level rather than temporal.
 */
(() => {
  const MIN_MEDIA_DIMENSION = 150; // px, filters out avatars/icons
  const REEL_SURFACE_SELECTORS = [
    "article",
    "main section",
    "main div[role='presentation']",
  ];

  function isValidMedia(el) {
    if (!el) return false;
    if (el.tagName === "VIDEO") {
      return (el.videoWidth || el.clientWidth || 0) >= MIN_MEDIA_DIMENSION;
    }
    if (el.tagName === "IMG") {
      return el.naturalWidth >= MIN_MEDIA_DIMENSION || el.clientWidth >= MIN_MEDIA_DIMENSION;
    }
    return false;
  }

  function getPostElements() {
    const posts = new Set();

    document.querySelectorAll("article").forEach((post) => posts.add(post));

    const mediaCandidates = document.querySelectorAll("main video, main img");
    mediaCandidates.forEach((mediaEl) => {
      if (!isValidMedia(mediaEl)) return;
      if (mediaEl.closest("article")) return;

      const container = mediaEl.closest(REEL_SURFACE_SELECTORS.join(", ")) || mediaEl.parentElement;
      if (container) {
        posts.add(container);
      }
    });

    return [...posts];
  }

  function findMediaElement(post) {
    const video = post.querySelector("video");
    if (isValidMedia(video)) return video;

    const imgs = Array.from(post.querySelectorAll("img")).filter((img) => isValidMedia(img));
    if (imgs.length === 0) return null;

    return imgs.reduce((largest, current) => {
      const largestArea = largest.clientWidth * largest.clientHeight;
      const currentArea = current.clientWidth * current.clientHeight;
      return currentArea > largestArea ? current : largest;
    });
  }

  function shouldShowControl(_post, mediaEl, rect) {
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;

    if (rect.bottom <= 0 || rect.top >= viewportHeight || rect.right <= 0 || rect.left >= viewportWidth) {
      return false;
    }

    const visibleWidth = Math.min(rect.right, viewportWidth) - Math.max(rect.left, 0);
    const visibleHeight = Math.min(rect.bottom, viewportHeight) - Math.max(rect.top, 0);
    if (visibleWidth <= 0 || visibleHeight <= 0) {
      return false;
    }

    const visibleArea = visibleWidth * visibleHeight;
    const totalArea = Math.max(rect.width * rect.height, 1);
    if ((visibleArea / totalArea) < 0.35) {
      return false;
    }

    if (mediaEl.tagName === "VIDEO" && rect.height > viewportHeight * 0.7) {
      const verticalCenter = rect.top + rect.height / 2;
      if (verticalCenter < viewportHeight * 0.2 || verticalCenter > viewportHeight * 0.8) {
        return false;
      }
    }

    return true;
  }

  function captureVideoFrame(video) {
    try {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || video.clientWidth || 640;
      canvas.height = video.videoHeight || video.clientHeight || 360;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
      return { kind: "frame", dataUrl, sourceUrl: location.href };
    } catch (err) {
      // Canvas is "tainted" if the video element lacks CORS headers --
      // can't read pixel data in that case.
      console.warn("[Social Detect] Could not capture video frame:", err);
      return null;
    }
  }

  async function extractMedia(mediaEl) {
    if (mediaEl.tagName === "IMG") {
      const url = mediaEl.currentSrc || mediaEl.src;
      if (!url) return null;
      return { kind: "url", mediaType: "image", url };
    }

    if (mediaEl.tagName === "VIDEO") {
      const src = mediaEl.currentSrc || mediaEl.src;
      if (src && !src.startsWith("blob:")) {
        return { kind: "url", mediaType: "video", url: src };
      }
      return captureVideoFrame(mediaEl);
    }

    return null;
  }

  window.SocialDetectCore.start({
    name: "instagram",
    postSelector: "article",
    getPostElements,
    findMediaElement,
    shouldShowControl,
    extractMedia,
  });
})();
