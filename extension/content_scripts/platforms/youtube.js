/**
 * YouTube adapter.
 *
 * Surfaces:
 *   - Watch pages (`/watch?v=...`): `#movie_player video.html5-main-video`
 *   - Shorts (`/shorts/...`): `ytd-reel-video-renderer video`
 *
 * YouTube streams via MSE (blob: URLs), so we cannot hand the backend a CDN
 * URL. Instead we prefer recording a short clip via `captureStream()` +
 * MediaRecorder (video+audio) and POSTing it to /analyze/video — the backend
 * then cuts frames and scores the soundtrack in parallel. If recording is
 * blocked, we fall back to seeking across the timeline and capturing several
 * canvas frames (visual-only).
 */
(() => {
  const RECORD_SECONDS = 6;
  const FRAME_SAMPLES = 8;

  function isWatchPage() {
    return location.pathname === "/watch" || location.pathname.startsWith("/watch");
  }

  function isShortsPage() {
    return location.pathname.startsWith("/shorts");
  }

  function getPostElements() {
    if (isShortsPage()) {
      const reels = document.querySelectorAll("ytd-reel-video-renderer");
      if (reels.length) return [...reels];
    }
    const player = document.querySelector("#movie_player");
    if (player) return [player];
    const video = document.querySelector("video.html5-main-video, ytd-player video, video");
    return video ? [video.closest("#movie_player, ytd-player, ytd-reel-video-renderer") || video.parentElement] : [];
  }

  function findMediaElement(post) {
    if (!post) return null;
    if (post.tagName === "VIDEO") return post;
    return (
      post.querySelector("video.html5-main-video") ||
      post.querySelector("video")
    );
  }

  function shouldShowControl(_post, mediaEl, rect) {
    const vw = window.innerWidth || document.documentElement.clientWidth || 0;
    const vh = window.innerHeight || document.documentElement.clientHeight || 0;
    if (rect.bottom <= 0 || rect.top >= vh || rect.right <= 0 || rect.left >= vw) return false;
    if (rect.width < 120 || rect.height < 80) return false;
    // On Shorts, only the mostly-visible reel.
    if (isShortsPage()) {
      const visibleH = Math.min(rect.bottom, vh) - Math.max(rect.top, 0);
      if (visibleH / Math.max(rect.height, 1) < 0.45) return false;
    }
    return mediaEl.readyState >= 1 || mediaEl.videoWidth > 0;
  }

  function captureVideoFrame(video) {
    try {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || video.clientWidth || 1280;
      canvas.height = video.videoHeight || video.clientHeight || 720;
      if (canvas.width < 2 || canvas.height < 2) return null;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.9);
    } catch (err) {
      console.warn("[Social Detect] Could not capture YouTube frame:", err);
      return null;
    }
  }

  function pickRecorderMime() {
    const candidates = [
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/webm",
      "video/mp4",
    ];
    for (const mime of candidates) {
      if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported?.(mime)) {
        return mime;
      }
    }
    return "";
  }

  async function recordClip(video, seconds = RECORD_SECONDS) {
    if (typeof video.captureStream !== "function" && typeof video.mozCaptureStream !== "function") {
      return null;
    }
    const mime = pickRecorderMime();
    if (!mime || typeof MediaRecorder === "undefined") return null;

    const stream = (video.captureStream || video.mozCaptureStream).call(video);
    if (!stream || stream.getTracks().length === 0) return null;

    const hadAudio = stream.getAudioTracks().length > 0;
    const recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 2_500_000 });
    const chunks = [];

    const stopped = new Promise((resolve, reject) => {
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      };
      recorder.onerror = (e) => reject(e.error || new Error("MediaRecorder failed"));
      recorder.onstop = () => resolve();
    });

    const wasPaused = video.paused;
    try {
      if (wasPaused) {
        await video.play().catch(() => {});
      }
      recorder.start(250);
      await new Promise((r) => setTimeout(r, Math.max(1500, seconds * 1000)));
      if (recorder.state !== "inactive") recorder.stop();
      await stopped;
    } catch (err) {
      try { if (recorder.state !== "inactive") recorder.stop(); } catch { /* ignore */ }
      stream.getTracks().forEach((t) => t.stop());
      throw err;
    } finally {
      stream.getTracks().forEach((t) => t.stop());
      if (wasPaused) {
        try { video.pause(); } catch { /* ignore */ }
      }
    }

    if (!chunks.length) return null;
    const blob = new Blob(chunks, { type: mime.split(";")[0] });
    if (blob.size < 1000) return null;

    const dataUrl = await blobToDataUrl(blob);
    return {
      kind: "clip",
      dataUrl,
      mediaType: "video",
      sourceUrl: location.href,
      hasAudio: hadAudio,
      note: hadAudio
        ? "Recorded short YouTube clip (video+audio) for frame+audio analysis"
        : "Recorded short YouTube clip (video only; no audio track in captureStream)",
    };
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error || new Error("FileReader failed"));
      reader.readAsDataURL(blob);
    });
  }

  async function sampleFrames(video, count = FRAME_SAMPLES) {
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    const originalTime = video.currentTime;
    const wasPaused = video.paused;
    const frames = [];

    const seekTo = (t) => new Promise((resolve) => {
      const onSeeked = () => {
        video.removeEventListener("seeked", onSeeked);
        resolve();
      };
      video.addEventListener("seeked", onSeeked);
      try {
        video.currentTime = t;
      } catch {
        video.removeEventListener("seeked", onSeeked);
        resolve();
      }
      // Safety timeout if seeked never fires.
      setTimeout(() => {
        video.removeEventListener("seeked", onSeeked);
        resolve();
      }, 800);
    });

    try {
      if (!wasPaused) {
        try { video.pause(); } catch { /* ignore */ }
      }

      if (duration > 1.5) {
        for (let i = 0; i < count; i++) {
          const t = (duration * (i + 0.5)) / count;
          await seekTo(Math.min(t, Math.max(duration - 0.05, 0)));
          const dataUrl = captureVideoFrame(video);
          if (dataUrl) frames.push({ timestamp: video.currentTime, dataUrl });
        }
      } else {
        const dataUrl = captureVideoFrame(video);
        if (dataUrl) frames.push({ timestamp: video.currentTime || 0, dataUrl });
      }
    } finally {
      try {
        await seekTo(originalTime);
        if (!wasPaused) await video.play().catch(() => {});
      } catch { /* ignore restore errors */ }
    }

    if (!frames.length) return null;
    if (frames.length === 1) {
      return { kind: "frame", dataUrl: frames[0].dataUrl, sourceUrl: location.href };
    }
    return {
      kind: "frames",
      frames: frames.map((f) => f.dataUrl),
      timestamps: frames.map((f) => f.timestamp),
      sourceUrl: location.href,
      mediaType: "video",
      note: "Sampled YouTube frames (visual-only; audio unavailable without MediaRecorder)",
    };
  }

  async function extractMedia(mediaEl) {
    if (!mediaEl || mediaEl.tagName !== "VIDEO") return null;

    const src = mediaEl.currentSrc || mediaEl.src;
    if (src && !src.startsWith("blob:") && !src.startsWith("mediasource:")) {
      return { kind: "url", mediaType: "video", url: src };
    }

    // Prefer a short recorded clip so the backend can run frame + audio pipelines.
    try {
      const clip = await recordClip(mediaEl, RECORD_SECONDS);
      if (clip) return clip;
    } catch (err) {
      console.warn("[Social Detect] YouTube clip capture failed, falling back to frames:", err);
    }

    return sampleFrames(mediaEl, FRAME_SAMPLES);
  }

  if (window.SocialDetectCore) {
    window.SocialDetectCore.start({
      name: "youtube",
      postSelector: "#movie_player, ytd-reel-video-renderer",
      getPostElements,
      findMediaElement,
      shouldShowControl,
      extractMedia,
    });
  }
})();
