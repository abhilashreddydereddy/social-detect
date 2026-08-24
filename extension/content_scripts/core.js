/**
 * Social Detect content-script core.
 *
 * This file is platform-agnostic. Each platform (Instagram, and later X,
 * Reddit, Facebook, TikTok, YouTube) provides a small "adapter" object
 * describing how to find posts and media on that site; this engine handles
 * everything else (scanning the DOM, injecting the control, talking to the
 * background service worker, and rendering the result overlay).
 *
 * Platform adapter shape (see platforms/instagram.js for a full example):
 *   {
 *     name: "instagram",
 *     postSelector: string,                 // CSS selector matching each post container
 *     findMediaElement(postEl): HTMLElement | null,
 *     extractMedia(mediaEl): Promise<{ kind: "url", mediaType: "image"|"video", url: string }
 *                                   | { kind: "frame", dataUrl: string } | null>
 *   }
 *
 * Everything renders inside a Shadow DOM so host-page CSS can never bleed
 * into our UI (and vice versa) -- important on sites with aggressive,
 * frequently-changing utility-class stylesheets.
 */

(() => {
  const PROCESSED_ATTR = "data-social-detect-processed";
  const HOST_ATTR = "data-social-detect-host";
  const POST_ID_ATTR = "data-social-detect-id";
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
    const topEvidence = [...result.evidence].slice(0, 3);

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
    if (!extracted) return { ok: false, error: "No media could be extracted from this post." };

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

      panelOpen = true;
      panel.classList.remove("sd-hidden");
      buildPanelContent(shadow, "loading");
      button.disabled = true;

      try {
        const currentMediaEl = adapter.findMediaElement(post);
        const extracted = currentMediaEl ? await adapter.extractMedia(currentMediaEl) : null;
        const response = await runAnalysis(extracted, adapter.name);
        if (response?.ok) {
          buildPanelContent(shadow, "result", response.data);
        } else {
          buildPanelContent(shadow, "error", response?.error || "Unknown error");
        }
      } catch (err) {
        buildPanelContent(shadow, "error", err?.message || String(err));
      } finally {
        button.disabled = false;
      }
    });

    document.addEventListener("click", (evt) => {
      if (panelOpen && !host.contains(evt.target)) {
        panel.classList.add("sd-hidden");
        panelOpen = false;
      }
    });

    document.body.appendChild(host);
    syncPosition();

    const reposition = () => syncPosition();
    window.addEventListener("scroll", reposition, true);
    window.addEventListener("resize", reposition);
  }

  function scan(adapter) {
    const posts = typeof adapter.getPostElements === "function"
      ? adapter.getPostElements()
      : document.querySelectorAll(adapter.postSelector);
    const activePostIds = new Set();

    posts.forEach((post) => {
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

  function start(adapter) {
    chrome.storage.sync.get({ enabled: true }, ({ enabled }) => {
      if (!enabled) return;
      scan(adapter);
      const observer = new MutationObserver(() => scan(adapter));
      observer.observe(document.body, { childList: true, subtree: true });
    });
  }

  window.SocialDetectCore = { start };
})();
