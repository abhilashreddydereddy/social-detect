/**
 * YouTube adapter — finds the player / Shorts reel and extracts pixels
 * for backend analysis (canvas → fetch → viewport screenshot).
 */
(() => {
  const MIN_MEDIA_DIMENSION = 80;

  function isShortsPage() {
    return location.pathname.startsWith("/shorts");
  }

  function isValidMedia(el) {
    if (!el) return false;
    if (el.tagName === "VIDEO") {
      const w = el.videoWidth || el.clientWidth || el.offsetWidth || 0;
      return w >= MIN_MEDIA_DIMENSION || el.clientWidth >= MIN_MEDIA_DIMENSION;
    }
    if (el.tagName === "IMG") {
      return el.naturalWidth >= MIN_MEDIA_DIMENSION || el.clientWidth >= MIN_MEDIA_DIMENSION;
    }
    return false;
  }

  function getPostElements() {
    const posts = new Set();

    if (isShortsPage()) {
      document.querySelectorAll("ytd-reel-video-renderer").forEach((el) => posts.add(el));
    }

    document.querySelectorAll("#movie_player, ytd-player#ytd-player, ytd-player, #player").forEach((el) => {
      const rect = el.getBoundingClientRect();
      if (rect.width >= 160 || rect.height >= 90 || el.querySelector("video")) {
        posts.add(el);
      }
    });

    document.querySelectorAll("video.html5-main-video, video").forEach((video) => {
      if (!video) return;
      const container =
        video.closest("#movie_player, ytd-player, ytd-reel-video-renderer, ytd-watch-flexy") ||
        video.parentElement;
      if (container) posts.add(container);
    });

    return [...posts];
  }

  function findMediaElement(post) {
    if (!post) return null;
    if (post.tagName === "VIDEO") return post;

    const main = post.querySelector("video.html5-main-video");
    if (main) return main;

    const videos = Array.from(post.querySelectorAll("video"));
    if (!videos.length) return null;
    return videos.reduce((largest, current) => {
      const a = (largest.clientWidth || 0) * (largest.clientHeight || 0);
      const b = (current.clientWidth || 0) * (current.clientHeight || 0);
      return b > a ? current : largest;
    });
  }

  function shouldShowControl(_post, mediaEl, rect) {
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;

    if (rect.width < 80 || rect.height < 45) return false;
    if (rect.bottom <= 0 || rect.top >= viewportHeight || rect.right <= 0 || rect.left >= viewportWidth) {
      return false;
    }

    const visibleWidth = Math.min(rect.right, viewportWidth) - Math.max(rect.left, 0);
    const visibleHeight = Math.min(rect.bottom, viewportHeight) - Math.max(rect.top, 0);
    if (visibleWidth < 40 || visibleHeight < 40) return false;

    if (isShortsPage() && rect.height > viewportHeight * 0.45) {
      const verticalCenter = rect.top + rect.height / 2;
      if (verticalCenter < viewportHeight * 0.1 || verticalCenter > viewportHeight * 0.9) {
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

  const adapter = {
    name: "youtube",
    postSelector: "#movie_player, ytd-player, ytd-reel-video-renderer, video.html5-main-video",
    getPostElements,
    findMediaElement,
    shouldShowControl,
    extractMedia,
    navigationEvents: ["yt-navigate-finish", "yt-page-data-updated"],
  };

  function boot() {
    if (!window.SocialDetectCore) {
      setTimeout(boot, 200);
      return;
    }
    window.SocialDetectCore.start(adapter);
    console.info("[Social Detect] YouTube adapter ready");
  }

  boot();
})();
