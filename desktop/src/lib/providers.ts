// Mirror of workflows.is_external_provider (src/workflows.py) for inline UI hints.

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "::1"]);

export function isExternalProvider(baseUrl: string | null, provider: string): boolean {
  if (provider === "openai" && !baseUrl) return true;
  if (!baseUrl) return false;
  try {
    const host = new URL(baseUrl).hostname;
    return !LOCAL_HOSTS.has(host);
  } catch {
    return false;
  }
}
