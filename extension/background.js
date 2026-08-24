/**
 * Background service worker.
 *
 * Content scripts never call the backend directly -- they send a message
 * here instead. This keeps backend-URL configuration, auth headers (future),
 * and CORS/host-permission concerns in one place, and means the popup and
 * every content script share identical request logic.
 *
 * Message contract (all messages are { type, ...payload }):
 *   ANALYZE_URL        { url, platform }        -> POSTs to /analyze/url
 *   ANALYZE_DATA_URL    { dataUrl, mediaKind, sourceUrl, platform }
 *                                                -> decodes a data: URL
 *                                                   (captured video frame or
 *                                                   fetched blob) and POSTs
 *                                                   multipart to
 *                                                   /analyze/image
 *   GET_STATUS          {}                       -> GET /status
 */

import { getSettings } from "./utils/config.js";

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
  const filename = mediaKind === "video" ? "clip.mp4" : "frame.jpg";
  form.append("file", blob, filename);

  const endpoint = mediaKind === "video" ? "/analyze/video" : "/analyze/image";
  const resp = await fetch(`${backendUrl}${endpoint}`, { method: "POST", body: form });
  return finish(resp);
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

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    try {
      switch (message.type) {
        case "ANALYZE_URL":
          sendResponse(await analyzeUrl(message.url, message.platform));
          break;
        case "ANALYZE_DATA_URL":
          sendResponse(await analyzeDataUrl(message.dataUrl, message.mediaKind, message.sourceUrl));
          break;
        case "GET_STATUS":
          sendResponse(await getStatus());
          break;
        default:
          sendResponse({ ok: false, error: `Unknown message type: ${message.type}` });
      }
    } catch (err) {
      sendResponse({ ok: false, error: err?.message || String(err) });
    }
  })();
  return true; // keep the message channel open for the async response
});
