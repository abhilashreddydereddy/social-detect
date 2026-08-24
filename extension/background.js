/**
 * Background service worker.
 *
 * Content scripts never call the backend directly -- they send a message
 * here instead. This also actively injects the YouTube adapter into YouTube
 * tabs, because YouTube's SPA soft-navigation often skips declarative
 * content_script reinjection (Instagram full page loads do not have this issue).
 */

import { getSettings } from "./utils/config.js";

const YT_HOST_RE = /^https?:\/\/([a-z0-9-]+\.)?(youtube\.com|youtu\.be)\//i;
const IG_HOST_RE = /^https?:\/\/([a-z0-9-]+\.)?instagram\.com\//i;

async function analyzeUrl(url, platform) {
  const { backendUrl } = await getSettings();
  const resp = await fetch(`${backendUrl}/analyze/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, platform_hint: platform || null }),
  });
  return finish(resp);
}

async function analyzeDataUrl(dataUrl, mediaKind, sourceUrl) {
  const { backendUrl } = await getSettings();
  const blob = await (await fetch(dataUrl)).blob();
  const form = new FormData();

  // Pixel captures from the page are always JPEG frames → image pipeline.
  // Full video clips (rare) still go to /analyze/video.
  let endpoint = "/analyze/image";
  let filename = "capture.jpg";
  if (mediaKind === "video") {
    endpoint = "/analyze/video";
    filename = blob.type.includes("mp4") ? "clip.mp4" : "clip.webm";
  } else if (mediaKind === "media") {
    endpoint = "/analyze/media";
    filename = "media.bin";
  }

  // Ensure the backend accepts the upload as an image even if the blob
  // type is empty/octet-stream after data-URL decoding.
  const typedBlob = blob.type
    ? blob
    : new Blob([blob], { type: mediaKind === "video" ? "video/webm" : "image/jpeg" });

  form.append("file", typedBlob, filename);
  console.info("[Social Detect] POST", `${backendUrl}${endpoint}`, {
    bytes: typedBlob.size,
    type: typedBlob.type,
    filename,
  });
  const resp = await fetch(`${backendUrl}${endpoint}`, { method: "POST", body: form });
  return finish(resp);
}

async function analyzeFrames(frames, timestamps, sourceUrl, platform) {
  if (!frames || !frames.length) {
    return { ok: false, error: "No frames to analyze" };
  }

  const { backendUrl } = await getSettings();
  const form = new FormData();
  for (let i = 0; i < frames.length; i++) {
    const blob = await (await fetch(frames[i])).blob();
    form.append("files", blob, `frame_${String(i).padStart(3, "0")}.jpg`);
  }
  if (timestamps && timestamps.length) {
    form.append("timestamps", JSON.stringify(timestamps));
  }
  if (platform) {
    form.append("platform", platform);
  }
  if (sourceUrl) {
    form.append("source_url", sourceUrl);
  }

  const resp = await fetch(`${backendUrl}/analyze/frames`, { method: "POST", body: form });
  return finish(resp);
}

async function captureVisibleTab() {
  try {
    const dataUrl = await chrome.tabs.captureVisibleTab(null, { format: "jpeg", quality: 92 });
    return { ok: true, dataUrl };
  } catch (err) {
    return { ok: false, error: err?.message || String(err) };
  }
}

async function getStatus() {
  const { backendUrl } = await getSettings();
  const resp = await fetch(`${backendUrl}/status`);
  return finish(resp);
}

async function finish(resp) {
  let body;
  try {
    body = await resp.json();
  } catch {
    body = null;
  }
  if (!resp.ok) {
    const detail = body?.detail || resp.statusText || "Request failed";
    return { ok: false, error: detail };
  }
  return { ok: true, data: body };
}

async function injectPlatform(tabId, platform) {
  const files =
    platform === "youtube"
      ? [
          "content_scripts/media_capture.js",
          "content_scripts/core.js",
          "content_scripts/platforms/youtube.js",
        ]
      : [
          "content_scripts/media_capture.js",
          "content_scripts/core.js",
          "content_scripts/platforms/instagram.js",
        ];

  try {
    await chrome.scripting.executeScript({
      target: { tabId, allFrames: false },
      files,
    });
    return { ok: true, platform };
  } catch (err) {
    return { ok: false, error: err?.message || String(err), platform };
  }
}

function platformForUrl(url) {
  if (!url) return null;
  if (YT_HOST_RE.test(url)) return "youtube";
  if (IG_HOST_RE.test(url)) return "instagram";
  return null;
}

async function maybeInject(tabId, url, reason) {
  const platform = platformForUrl(url);
  if (!platform) return;
  const result = await injectPlatform(tabId, platform);
  if (!result.ok) {
    console.warn(`[Social Detect] inject (${reason}) failed:`, result.error);
  } else {
    console.info(`[Social Detect] injected ${platform} into tab ${tabId} (${reason})`);
  }
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // Inject on load complete and when the URL changes (YouTube SPA).
  if (changeInfo.status === "complete" || changeInfo.url) {
    const url = changeInfo.url || tab.url;
    maybeInject(tabId, url, changeInfo.url ? "url-change" : "complete");
  }
});

chrome.runtime.onInstalled.addListener(async () => {
  // Re-inject into already-open YouTube/Instagram tabs after install/update.
  try {
    const tabs = await chrome.tabs.query({
      url: [
        "*://*.youtube.com/*",
        "*://youtube.com/*",
        "*://youtu.be/*",
        "*://*.instagram.com/*",
        "*://instagram.com/*",
      ],
    });
    for (const tab of tabs) {
      if (tab.id != null) await maybeInject(tab.id, tab.url, "onInstalled");
    }
  } catch (err) {
    console.warn("[Social Detect] onInstalled inject failed:", err);
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    try {
      switch (message.type) {
        case "ANALYZE_URL":
          sendResponse(await analyzeUrl(message.url, message.platform));
          break;
        case "ANALYZE_DATA_URL":
          sendResponse(await analyzeDataUrl(message.dataUrl, message.mediaKind, message.sourceUrl));
          break;
        case "ANALYZE_FRAMES":
          sendResponse(await analyzeFrames(
            message.frames,
            message.timestamps,
            message.sourceUrl,
            message.platform,
          ));
          break;
        case "CAPTURE_VISIBLE_TAB":
          sendResponse(await captureVisibleTab());
          break;
        case "GET_STATUS":
          sendResponse(await getStatus());
          break;
        case "INJECT_ACTIVE_TAB": {
          const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
          if (!tab?.id) {
            sendResponse({ ok: false, error: "No active tab" });
            break;
          }
          const platform = message.platform || platformForUrl(tab.url) || "youtube";
          sendResponse(await injectPlatform(tab.id, platform));
          break;
        }
        case "PING_ACTIVE_TAB": {
          const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
          if (!tab?.id) {
            sendResponse({ ok: false, error: "No active tab" });
            break;
          }
          try {
            const resp = await chrome.tabs.sendMessage(tab.id, { type: "SD_PING" });
            sendResponse({ ok: true, tabUrl: tab.url, ping: resp });
          } catch (err) {
            sendResponse({ ok: false, tabUrl: tab.url, error: err?.message || String(err) });
          }
          break;
        }
        default:
          sendResponse({ ok: false, error: `Unknown message type: ${message.type}` });
      }
    } catch (err) {
      sendResponse({ ok: false, error: err?.message || String(err) });
    }
  })();
  return true;
});
