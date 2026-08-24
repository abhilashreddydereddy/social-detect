/**
 * YouTube adapter — mirrors the Instagram approach.
 *
 * Instagram works because it:
 *   1. Finds media-bearing containers from the live DOM
 *   2. Picks the primary <video>/<img>
 *   3. For MSE/blob: videos, captures ONE frame and POSTs it as an image
 *
 * YouTube is the same MSE/blob situation. Canvas readback is often blocked
 * by CORS on YouTube, so if the Instagram-style canvas snapshot fails we
 * fall back to chrome.tabs.captureVisibleTab + crop to the video rect
 * (triggered by the Analyze click = user gesture / activeTab).
 *
 * Surfaces:
 *   - Watch pages (`/watch?v=...`): `#movie_player` / `ytd-player` + <video>
 *   - Shorts (`/shorts/...`): `ytd-reel-video-renderer` + <video>
 */
(() => {
  const MIN_MEDIA_DIMENSION = 120;

  function isShortsPage() {
    return location.pathname.startsWith("/shorts");
  }

  function isValidMedia(el) {
    if (!el) return false;
    if (el.tagName === "VIDEO") {
      const w = el.videoWidth || el.clientWidth || 0;
      const h = el.videoHeight || el.clientHeight || 0;
      return w >= MIN_MEDIA_DIMENSION && h >= 40;
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

    document.querySelectorAll("#movie_player, ytd-player").forEach((el) => {
      const rect = el.getBoundingClientRect();
      if (rect.width >= 200 && rect.height >= 120) posts.add(el);
    });

    document.querySelectorAll("video.html5-main-video, video").forEach((video) => {
      if (!isValidMedia(video)) return;
      if (video.closest("#movie_player, ytd-player, ytd-reel-video-renderer")) return;
      const container = video.closest("ytd-watch-flexy, ytd-reel-video-renderer") || video.parentElement;
      if (container) posts.add(container);
    });

    return [...posts];
  }

  function findMediaElement(post) {
    if (!post) return null;
    if (post.tagName === "VIDEO" && isValidMedia(post)) return post;

    const main = post.querySelector("video.html5-main-video");
    if (isValidMedia(main)) return main;

    const videos = Array.from(post.querySelectorAll("video")).filter(isValidMedia);
    if (videos.length) {
      return videos.reduce((largest, current) => {
        const a = largest.clientWidth * largest.clientHeight;
        const b = current.clientWidth * current.clientHeight;
        return b > a ? current : largest;
      });
    }

    const imgs = Array.from(post.querySelectorAll("img")).filter(isValidMedia);
    if (!imgs.length) return null;
    return imgs.reduce((largest, current) => {
      const a = largest.clientWidth * largest.clientHeight;
      const b = current.clientWidth * current.clientHeight;
      return b > a ? current : largest;
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
    if ((visibleArea / totalArea) < 0.25) return false;

    if (isShortsPage() && mediaEl.tagName === "VIDEO" && rect.height > viewportHeight * 0.5) {
      const verticalCenter = rect.top + rect.height / 2;
      if (verticalCenter < viewportHeight * 0.15 || verticalCenter > viewportHeight * 0.85) {
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
      const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
      return { kind: "frame", dataUrl, sourceUrl: location.href };
    } catch (err) {
      console.warn("[Social Detect] Canvas frame capture failed (likely CORS):", err);
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
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
      return {
        kind: "frame",
        dataUrl: canvas.toDataURL("image/jpeg", 0.92),
        sourceUrl: location.href,
      };
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

      // 1) Same path as Instagram: canvas snapshot of the <video>
      const frame = captureVideoFrame(mediaEl);
      if (frame) return frame;

      // 2) YouTube CORS often blocks canvas — screenshot + crop instead
      return captureViewportFrame(mediaEl);
    }

    return null;
  }

  if (window.SocialDetectCore) {
    window.SocialDetectCore.start({
      name: "youtube",
      postSelector: "#movie_player, ytd-player, ytd-reel-video-renderer",
      getPostElements,
      findMediaElement,
      shouldShowControl,
      extractMedia,
      navigationEvents: ["yt-navigate-finish", "yt-page-data-updated"],
    });
  }
})();
