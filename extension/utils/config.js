/**
 * Shared configuration helpers. Loaded by both background.js and popup.js
 * (as a classic script via <script src> in the popup, and imported as an
 * ES module in the background service worker).
 */

export const DEFAULT_SETTINGS = {
  backendUrl: "http://localhost:8000",
  enabled: true,
};

export async function getSettings() {
  const stored = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  return { ...DEFAULT_SETTINGS, ...stored };
}

export async function setSettings(partial) {
  await chrome.storage.sync.set(partial);
}
