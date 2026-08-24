/**
 * Social Detect content-script core.
 *
 * Platform-agnostic engine. Adapters (instagram.js / youtube.js) call
 * SocialDetectCore.start(...). Safe to inject more than once (idempotent).
 */

(() => {
  if (window.__SOCIAL_DETECT_CORE_LOADED__) {
    return;
  }
  window.__SOCIAL_DETECT_CORE_LOADED__ = true;

  const PROCESSED_ATTR = "data-social-detect-processed";
  const HOST_ATTR = "data-social-detect-host";
  const POST_ID_ATTR = "data-social-detect-id";
  const STARTED = new Set();
  let postCounter = 0;

  const STYLES = `
    :host { all: initial; }
    * { box-sizing: border-box; font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; }

    .sd-button {
      background: rgba(11,14,17,0.85);
      color: #E9EDF1;
      border: 1px solid rgba(233,237,241,0.25);
      border-radius: 999px;
      padding: 5px 12px;
      font-size: 11px;
      letter-spacing: 0.02em;
      cursor: pointer;
      backdrop-filter: blur(4px);
      transition: border-color 0.15s ease, background 0.15s ease;
    }
    .sd-button:hover { border-color: #4CC9C0; background: rgba(11,14,17,0.95); }
    .sd-button[disabled] { opacity: 0.6; cursor: progress; }

    .sd-panel {
      position: absolute;
      top: 32px;
      right: 0;
      width: 280px;
      background: #12161B;
      border: 1px solid #252D37;
      border-radius: 10px;
      color: #E9EDF1;
      box-shadow: 0 12px 32px rgba(0,0,0,0.45);
      padding: 14px;
      font-size: 12px;
      line-height: 1.4;
      z-index: 2;
    }
    .sd-hidden { display: none; }

    .sd-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
    .sd-classification { font-size: 13px; font-weight: 600; }
    .sd-confidence { font-size: 10px; color: #7C8896; border: 1px solid #252D37; border-radius: 999px; padding: 2px 8px; }

    .sd-gauge-track { position: relative; height: 6px; border-radius: 3px; margin: 10px 0 4px;
      background: linear-gradient(90deg, #4CC9C0, #E8A33D, #E2574C); }
    .sd-gauge-marker { position: absolute; top: -4px; width: 2px; height: 14px; background: #E9EDF1; transform: translateX(-1px); }
    .sd-gauge-labels { display: flex; justify-content: space-between; font-size: 9px; color: #7C8896; margin-bottom: 10px; }

    .sd-evidence { margin: 0; padding: 0; list-style: none; }
    .sd-evidence li { border-left: 2px solid #252D37; padding: 4px 0 4px 8px; margin-bottom: 6px; color: #c7cdd4; }
    .sd-evidence li.high { border-left-color: #E2574C; }
    .sd-evidence li.mid { border-left-color: #E8A33D; }
    .sd-evidence li.low { border-left-color: #4CC9C0; }

    .sd-disclaimer { color: #7C8896; font-size: 9.5px; margin-top: 10px; padding-top: 8px; border-top: 1px solid #252D37; }
    .sd-error { color: #E2574C; }
    .sd-loading { color: #7C8896; }
  `;

  function severityClass(score) {
    if (score >= 0.7) return "high";
    if (score >= 0.5) return "mid";
    return "low";
  }

  function buildPanelContent(shadow, state, payload) {
    const panel = shadow.querySelector(".sd-panel");
    panel.innerHTML = "";

    if (state === "loading") {
      panel.innerHTML = `<div class="sd-loading">Running detectors…</div>`;
      return;
    }
    if (state === "error") {
      panel.innerHTML = `<div class="sd-error">Couldn't analyze this media.<br/>${escapeHtml(payload)}</div>`;
      return;
    }

    const result = payload;
    const pct = Math.round(result.ai_probability * 100);
    const topEvidence = [...(result.evidence || [])].slice(0, 3);

    const row = document.createElement("div");
    row.className = "sd-row";
    row.innerHTML = `
      <span class="sd-classification">${escapeHtml(result.classification)}</span>
      <span class="sd-confidence">${Math.round(result.confidence * 100)}% conf.</span>
    `;
    panel.appendChild(row);

    const gaugeTrack = document.createElement("div");
    gaugeTrack.className = "sd-gauge-track";
    const marker = document.createElement("div");
    marker.className = "sd-gauge-marker";
    marker.style.left = `${result.ai_probability * 100}%`;
    gaugeTrack.appendChild(marker);
    panel.appendChild(gaugeTrack);

    const gaugeLabels = document.createElement("div");
    gaugeLabels.className = "sd-gauge-labels";
    gaugeLabels.innerHTML = `<span>${pct}% AI probability</span>`;
    panel.appendChild(gaugeLabels);

    const list = document.createElement("ul");
    list.className = "sd-evidence";
    if (topEvidence.length === 0) {
      list.innerHTML = `<li>No specific evidence signals returned.</li>`;
    } else {
      for (const e of topEvidence) {
        const li = document.createElement("li");
        li.className = severityClass(e.score);
        li.textContent = e.summary;
        list.appendChild(li);
      }
    }
    panel.appendChild(list);

    if (result.audio_result) {
      const audio = document.createElement("div");
      audio.className = "sd-disclaimer";
      if (result.audio_result.available && result.audio_result.error == null) {
        const ap = Math.round((result.audio_result.ai_probability || 0) * 100);
        audio.textContent = `Audio: ${ap}% AI probability (parallel soundtrack check)`;
      } else {
        audio.textContent = `Audio: ${result.audio_result.error || "unavailable"}`;
      }
      panel.appendChild(audio);
    }

    const disclaimer = document.createElement("div");
    disclaimer.className = "sd-disclaimer";
    disclaimer.textContent = result.disclaimer;
    panel.appendChild(disclaimer);
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = String(str);
    return div.innerHTML;
  }

  async function runAnalysis(extracted, platform) {
    if (!extracted) {
      return {
        ok: false,
        error: "No media pixels could be extracted. Try again after the image/video finishes loading.",
      };
    }

    console.info("[Social Detect] sending to backend:", {
      kind: extracted.kind,
      method: extracted.method || null,
      platform,
      hasDataUrl: Boolean(extracted.dataUrl),
      urlHost: extracted.url ? (() => { try { return new URL(extracted.url).host; } catch { return "?"; } })() : null,
    });

    if (extracted.kind === "url") {
      return chrome.runtime.sendMessage({
        type: "ANALYZE_URL",
        url: extracted.url,
        platform,
      });
    }
    if (extracted.kind === "frame") {
      return chrome.runtime.sendMessage({
        type: "ANALYZE_DATA_URL",
        dataUrl: extracted.dataUrl,
        mediaKind: "image",
        sourceUrl: extracted.sourceUrl || null,
        platform,
      });
    }
    if (extracted.kind === "clip") {
      return chrome.runtime.sendMessage({
        type: "ANALYZE_DATA_URL",
        dataUrl: extracted.dataUrl,
        mediaKind: "video",
        sourceUrl: extracted.sourceUrl || null,
        platform,
      });
    }
    if (extracted.kind === "frames") {
      return chrome.runtime.sendMessage({
        type: "ANALYZE_FRAMES",
        frames: extracted.frames,
        timestamps: extracted.timestamps || [],
        sourceUrl: extracted.sourceUrl || null,
        platform,
      });
    }
    return { ok: false, error: `Unsupported extraction kind: ${extracted.kind}` };
  }

  function findExistingHost(post) {
    const postId = post.getAttribute(POST_ID_ATTR);
    return postId ? document.querySelector(`[${HOST_ATTR}="${postId}"]`) : null;
  }

  function ensurePostId(post) {
    let postId = post.getAttribute(POST_ID_ATTR);
    if (!postId) {
      postId = `sd-${++postCounter}`;
      post.setAttribute(POST_ID_ATTR, postId);
    }
    return postId;
  }

  function positionHost(host, post, adapter) {
    const mediaEl = adapter.findMediaElement(post);
    if (!mediaEl || !mediaEl.isConnected) {
      host.style.display = "none";
      return false;
    }

    const rect = mediaEl.getBoundingClientRect();
    if (rect.width < 40 || rect.height < 40) {
      host.style.display = "none";
      return false;
    }

    if (typeof adapter.shouldShowControl === "function" && !adapter.shouldShowControl(post, mediaEl, rect)) {
      host.style.display = "none";
      return false;
    }

    host.style.display = "block";
    host.style.top = `${Math.max(8, rect.top + 8)}px`;
    host.style.left = `${Math.max(8, rect.right - 108)}px`;
    return true;
  }

  function injectControl(post, adapter) {
    const mediaEl = adapter.findMediaElement(post);
    if (!mediaEl) return;
    const postId = ensurePostId(post);

    const host = document.createElement("div");
    host.setAttribute(HOST_ATTR, postId);
    host.style.cssText = "position:fixed; top:8px; left:8px; z-index:2147483647; pointer-events:auto;";
    const shadow = host.attachShadow({ mode: "open" });

    const style = document.createElement("style");
    style.textContent = STYLES;
    shadow.appendChild(style);

    const button = document.createElement("button");
    button.className = "sd-button";
    button.textContent = "🔍 Analyze";
    shadow.appendChild(button);

    const panel = document.createElement("div");
    panel.className = "sd-panel sd-hidden";
    shadow.appendChild(panel);

    let panelOpen = false;
    const syncPosition = () => positionHost(host, post, adapter);

    button.addEventListener("click", async (evt) => {
      evt.preventDefault();
      evt.stopPropagation();
      evt.stopImmediatePropagation();

      panelOpen = true;
      panel.classList.remove("sd-hidden");
      buildPanelContent(shadow, "loading");
      button.disabled = true;

      try {
        // Hide our UI so viewport screenshots don't include the Analyze button.
        document.querySelectorAll(`[${HOST_ATTR}]`).forEach((h) => {
          h.style.visibility = "hidden";
        });
        // Two RAFs so the browser paints without our overlay.
        await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

        const currentMediaEl = adapter.findMediaElement(post);
        let extracted = currentMediaEl ? await adapter.extractMedia(currentMediaEl) : null;

        // Hard fallback: shared pixel extractor (never relies on backend CDN fetch).
        if ((!extracted || extracted.kind === "url") && currentMediaEl && window.SocialDetectMedia) {
          const pixel = await window.SocialDetectMedia.extractForBackend(currentMediaEl);
          if (pixel) extracted = pixel;
        }

        // If adapter still returned a social CDN URL, do not send it to the
        // backend — Instagram/YouTube CDNs block server downloads.
        if (extracted?.kind === "url") {
          const u = String(extracted.url || "");
          const isDirectFile = /\.(jpe?g|png|webp|gif|mp4|webm)(\?|$)/i.test(u)
            && !/instagram\.|cdninstagram\.|fbcdn\.|ytimg\.|googlevideo\./i.test(u);
          if (!isDirectFile && currentMediaEl && window.SocialDetectMedia) {
            extracted = await window.SocialDetectMedia.extractForBackend(currentMediaEl);
          }
        }

        const response = await runAnalysis(extracted, adapter.name);
        if (response?.ok) {
          buildPanelContent(shadow, "result", response.data);
        } else {
          buildPanelContent(shadow, "error", response?.error || "Unknown error");
        }
      } catch (err) {
        buildPanelContent(shadow, "error", err?.message || String(err));
      } finally {
        document.querySelectorAll(`[${HOST_ATTR}]`).forEach((h) => {
          h.style.visibility = "visible";
        });
        button.disabled = false;
      }
    }, true);

    document.addEventListener("click", (evt) => {
      if (panelOpen && !host.contains(evt.target)) {
        panel.classList.add("sd-hidden");
        panelOpen = false;
      }
    });

    (document.documentElement || document.body).appendChild(host);
    syncPosition();

    const reposition = () => syncPosition();
    window.addEventListener("scroll", reposition, true);
    window.addEventListener("resize", reposition);
    host._sdReposition = reposition;
  }

  function scan(adapter) {
    const posts = typeof adapter.getPostElements === "function"
      ? adapter.getPostElements()
      : [...document.querySelectorAll(adapter.postSelector)];
    const activePostIds = new Set();

    posts.forEach((post) => {
      if (!post || !post.isConnected) return;
      const postId = ensurePostId(post);
      activePostIds.add(postId);
      const mediaEl = adapter.findMediaElement(post);
      const existingHost = findExistingHost(post);

      if (!mediaEl) {
        existingHost?.remove();
        post.removeAttribute(PROCESSED_ATTR);
        return;
      }

      if (existingHost) {
        post.setAttribute(PROCESSED_ATTR, "1");
        positionHost(existingHost, post, adapter);
        return;
      }

      post.setAttribute(PROCESSED_ATTR, "1");
      injectControl(post, adapter);
    });

    document.querySelectorAll(`[${HOST_ATTR}]`).forEach((host) => {
      const postId = host.getAttribute(HOST_ATTR);
      if (!activePostIds.has(postId)) {
        host.remove();
      }
    });
  }

  function debounce(fn, ms) {
    let t = null;
    return (...args) => {
      if (t) clearTimeout(t);
      t = setTimeout(() => {
        t = null;
        fn(...args);
      }, ms);
    };
  }

  function whenBodyReady(cb) {
    if (document.body) {
      cb();
      return;
    }
    const obs = new MutationObserver(() => {
      if (document.body) {
        obs.disconnect();
        cb();
      }
    });
    obs.observe(document.documentElement, { childList: true });
  }

  function start(adapter) {
    if (!adapter || !adapter.name) return;
    if (STARTED.has(adapter.name)) {
      // Already running for this platform — just force a rescan.
      try { scan(adapter); } catch (err) {
        console.warn("[Social Detect] rescan failed:", err);
      }
      return;
    }
    STARTED.add(adapter.name);
    console.info(`[Social Detect] starting adapter: ${adapter.name} on ${location.href}`);

    chrome.storage.sync.get({ enabled: true }, ({ enabled }) => {
      if (!enabled) {
        console.info("[Social Detect] disabled in extension settings");
        return;
      }

      const runScan = debounce(() => scan(adapter), 150);

      whenBodyReady(() => {
        scan(adapter);
        const observer = new MutationObserver(runScan);
        observer.observe(document.documentElement, { childList: true, subtree: true });
        setInterval(() => scan(adapter), 1500);
      });

      const navEvents = Array.isArray(adapter.navigationEvents) ? adapter.navigationEvents : [];
      for (const evtName of navEvents) {
        document.addEventListener(evtName, () => {
          setTimeout(() => scan(adapter), 200);
          setTimeout(() => scan(adapter), 1000);
        });
      }
      window.addEventListener("popstate", () => setTimeout(() => scan(adapter), 300));
    });
  }

  // Allow the popup / background to ask "is the content script alive?"
  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "SD_PING") {
      sendResponse({
        ok: true,
        href: location.href,
        adapters: [...STARTED],
        coreLoaded: true,
      });
      return true;
    }
    return false;
  });

  window.SocialDetectCore = { start, scan, started: STARTED };
})();
