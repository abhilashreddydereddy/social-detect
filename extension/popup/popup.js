const DEFAULT_SETTINGS = { backendUrl: "http://localhost:8000", enabled: true };

const statusDot = document.getElementById("statusDot");
const statusBadge = document.getElementById("statusBadge");
const enabledToggle = document.getElementById("enabledToggle");
const backendUrlInput = document.getElementById("backendUrl");
const saveBtn = document.getElementById("saveBtn");
const injectBtn = document.getElementById("injectBtn");
const tabStatus = document.getElementById("tabStatus");

async function loadSettings() {
  const stored = await chrome.storage.sync.get(DEFAULT_SETTINGS);
  return { ...DEFAULT_SETTINGS, ...stored };
}

async function refreshStatus(backendUrl) {
  statusBadge.textContent = "checking…";
  statusDot.className = "dot";
  try {
    const resp = await fetch(`${backendUrl}/status`);
    if (!resp.ok) throw new Error("bad response");
    const data = await resp.json();
    const active = data.detectors.filter((d) => d.available).length;
    statusBadge.textContent = `${active}/${data.detectors.length} detectors online`;
    statusDot.className = "dot ok";
  } catch (err) {
    statusBadge.textContent = "backend unreachable";
    statusDot.className = "dot error";
  }
}

async function refreshTabStatus() {
  if (!tabStatus) return;
  tabStatus.textContent = "checking…";
  try {
    const resp = await chrome.runtime.sendMessage({ type: "PING_ACTIVE_TAB" });
    if (resp?.ok && resp.ping?.ok) {
      const adapters = (resp.ping.adapters || []).join(", ") || "none";
      tabStatus.textContent = `Active: ${adapters} · ${resp.tabUrl || ""}`;
    } else {
      tabStatus.textContent = `Not injected on this tab. ${resp?.error || "Open YouTube/Instagram, then click Inject."}`;
    }
  } catch (err) {
    tabStatus.textContent = err?.message || String(err);
  }
}

(async function init() {
  const settings = await loadSettings();
  enabledToggle.checked = settings.enabled;
  backendUrlInput.value = settings.backendUrl;
  refreshStatus(settings.backendUrl);
  refreshTabStatus();
})();

enabledToggle.addEventListener("change", async () => {
  await chrome.storage.sync.set({ enabled: enabledToggle.checked });
});

saveBtn.addEventListener("click", async () => {
  const url = backendUrlInput.value.trim().replace(/\/$/, "");
  await chrome.storage.sync.set({ backendUrl: url });
  refreshStatus(url);
});

if (injectBtn) {
  injectBtn.addEventListener("click", async () => {
    injectBtn.disabled = true;
    injectBtn.textContent = "Injecting…";
    try {
      const resp = await chrome.runtime.sendMessage({ type: "INJECT_ACTIVE_TAB" });
      if (resp?.ok) {
        injectBtn.textContent = `Injected ${resp.platform || ""}`;
      } else {
        injectBtn.textContent = "Inject failed";
        if (tabStatus) tabStatus.textContent = resp?.error || "Injection failed";
      }
      setTimeout(refreshTabStatus, 400);
    } catch (err) {
      injectBtn.textContent = "Inject failed";
      if (tabStatus) tabStatus.textContent = err?.message || String(err);
    } finally {
      setTimeout(() => {
        injectBtn.disabled = false;
        injectBtn.textContent = "Inject on this tab";
      }, 1500);
    }
  });
}
