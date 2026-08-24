const DEFAULT_SETTINGS = { backendUrl: "http://localhost:8000", enabled: true };

const statusDot = document.getElementById("statusDot");
const statusBadge = document.getElementById("statusBadge");
const enabledToggle = document.getElementById("enabledToggle");
const backendUrlInput = document.getElementById("backendUrl");
const saveBtn = document.getElementById("saveBtn");

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

(async function init() {
  const settings = await loadSettings();
  enabledToggle.checked = settings.enabled;
  backendUrlInput.value = settings.backendUrl;
  refreshStatus(settings.backendUrl);
})();

enabledToggle.addEventListener("change", async () => {
  await chrome.storage.sync.set({ enabled: enabledToggle.checked });
});

saveBtn.addEventListener("click", async () => {
  const url = backendUrlInput.value.trim().replace(/\/$/, "");
  await chrome.storage.sync.set({ backendUrl: url });
  refreshStatus(url);
});
