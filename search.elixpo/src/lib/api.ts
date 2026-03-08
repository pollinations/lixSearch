const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:9002';
const INTERNAL_KEY = process.env.INTERNAL_API_KEY || '';
const API_KEY = process.env.API_KEY || '';
const XID = process.env.XID || '';

export function backendHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (INTERNAL_KEY) headers['X-Internal-Key'] = INTERNAL_KEY;
  if (API_KEY) headers['X-API-Key'] = API_KEY;
  return headers;
}

export function backendUrl(path: string): string {
  return `${BACKEND_URL}${path}`;
}

/**
 * Validate XID from request headers.
 * Returns true if valid, false otherwise.
 */
export function validateXID(requestXID: string | null): boolean {
  if (!XID) return true; // no XID configured = open access
  return requestXID === XID;
}
