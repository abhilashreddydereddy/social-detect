const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function handle(resp) {
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = body.detail || detail;
    } catch {
      /* ignore parse failure */
    }
    throw new Error(detail);
  }
  return resp.json();
}

export async function getStatus() {
  const resp = await fetch(`${API_BASE_URL}/status`);
  return handle(resp);
}

export async function analyzeImage(file) {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_BASE_URL}/analyze/image`, { method: "POST", body: form });
  return handle(resp);
}

export async function analyzeVideo(file) {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_BASE_URL}/analyze/video`, { method: "POST", body: form });
  return handle(resp);
}

export async function analyzeMedia(file) {
  const form = new FormData();
  form.append("file", file);
  const resp = await fetch(`${API_BASE_URL}/analyze/media`, { method: "POST", body: form });
  return handle(resp);
}

export async function analyzeUrl(url, platformHint) {
  const resp = await fetch(`${API_BASE_URL}/analyze/url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, platform_hint: platformHint || null }),
  });
  return handle(resp);
}

export { API_BASE_URL };
