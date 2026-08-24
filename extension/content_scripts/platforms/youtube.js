/**
 * YouTube adapter — same capture contract as Instagram.
 *
 * Finds the player / Shorts reel, grabs the main <video>, and for MSE/blob
 * streams captures a single frame (canvas, then viewport screenshot fallback).
 *
 * Also self-boots if injected programmatically by the background worker
 * (YouTube SPA soft-nav often misses declarative content_scripts).
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
      const h = el.videoHeight || el.clientHeight || el.offsetHeight || 0;
      // Accept early: YouTube often mounts <video> before metadata arrives.
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

    const players = document.querySelectorAll(
      "#movie_player, ytd-player#ytd-player, ytd-player, #player-container-inner, #player",
    );
    players.forEach((el) => {
      const rect = el.getBoundingClientRect();
      // Allow zero-size briefly; scan loop will hide until ready.
      if (rect.width >= 160 || rect.height >= 90 || el.querySelector("video")) {
        posts.add(el);
      }
    });

    // Always consider the largest page video as a post container.
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
    if (videos.length) {
      return videos.reduce((largest, current) => {
        const a = (largest.clientWidth || 0) * (largest.clientHeight || 0);
        const b = (current.clientWidth || 0) * (current.clientHeight || 0);
        return b > a ? current : largest;
      });
    }
    return null;
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

  function captureVideoFrame(video) {
    try {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || video.clientWidth || 640;
      canvas.height = video.videoHeight || video.clientHeight || 360;
      if (canvas.width < 2 || canvas.height < 2) return null;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return { kind: "frame", dataUrl: canvas.toDataURL("image/jpeg", 0.92), sourceUrl: location.href };
    } catch (err) {
      console.warn("[Social Detect] YouTube canvas capture failed:", err);
      return null;
    }
  }

  function loadImage(dataUrl) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Failed to load screenshot"));
      img.src = dataUrl;
    });
  }

  async function captureViewportFrame(mediaEl) {
    const rect = mediaEl.getBoundingClientRect();
    if (rect.width < 40 || rect.height < 40) return null;

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
      const sw = Math.min(img.width - sx, Math.round(rect.width * dpr));
      const sh = Math.min(img.height - sy, Math.round(rect.height * dpr));
      if (sw < 2 || sh < 2) return null;

      const canvas = document.createElement("canvas");
      canvas.width = sw;
      canvas.height = sh;
      canvas.getContext("2d").drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
      return { kind: "frame", dataUrl: canvas.toDataURL("image/jpeg", 0.92), sourceUrl: location.href };
    } catch (err) {
      console.warn("[Social Detect] Viewport crop failed:", err);
      return null;
    }
  }

  async function extractMedia(mediaEl) {
    if (!mediaEl) return null;

    if (mediaEl.tagName === "IMG") {
      const url = mediaEl.currentSrc || mediaEl.src;
      if (!url) return null;
      return { kind: "url", mediaType: "image", url };
    }

    if (mediaEl.tagName === "VIDEO") {
      const src = mediaEl.currentSrc || mediaEl.src;
      if (src && !src.startsWith("blob:") && !src.startsWith("mediasource:")) {
        return { kind: "url", mediaType: "video", url: src };
      }
      // Instagram-style frame capture first, then screenshot fallback for CORS.
      return captureVideoFrame(mediaEl) || captureViewportFrame(mediaEl);
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
      console.warn("[Social Detect] core not ready yet, retrying…");
      setTimeout(boot, 200);
      return;
    }
    window.SocialDetectCore.start(adapter);
    console.info("[Social Detect] YouTube adapter ready");
  }

  boot();
})();
