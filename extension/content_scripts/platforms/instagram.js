/**
 * Instagram adapter.
 *
 * Finds posts/media structurally, then extracts PIXELS for the backend.
 * We do not send Instagram CDN URLs to the server — they are blocked.
 */
(() => {
  const MIN_MEDIA_DIMENSION = 150;
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

    document.querySelectorAll("main video, main img").forEach((mediaEl) => {
      if (!isValidMedia(mediaEl)) return;
      if (mediaEl.closest("article")) return;
      const container = mediaEl.closest(REEL_SURFACE_SELECTORS.join(", ")) || mediaEl.parentElement;
      if (container) posts.add(container);
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
    if (visibleWidth <= 0 || visibleHeight <= 0) return false;

    const visibleArea = visibleWidth * visibleHeight;
    const totalArea = Math.max(rect.width * rect.height, 1);
    if ((visibleArea / totalArea) < 0.35) return false;

    if (mediaEl.tagName === "VIDEO" && rect.height > viewportHeight * 0.7) {
      const verticalCenter = rect.top + rect.height / 2;
      if (verticalCenter < viewportHeight * 0.2 || verticalCenter > viewportHeight * 0.8) {
        return false;
      }
    }

    return true;
  }

  async function extractMedia(mediaEl) {
    if (!mediaEl) return null;
    if (window.SocialDetectMedia?.extractForBackend) {
      return window.SocialDetectMedia.extractForBackend(mediaEl);
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
