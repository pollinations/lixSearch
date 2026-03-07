interface CloudflareEnv {
  DB: D1Database;
  SESSIONS_KV: KVNamespace;
  RATE_LIMIT_KV: KVNamespace;
  BACKEND_URL: string;
  INTERNAL_API_KEY: string;
  XID: string;
  API_KEY: string;
  GUEST_REQUEST_LIMIT: string;
}

declare module '@cloudflare/next-on-pages' {
  export function getRequestContext(): {
    env: CloudflareEnv;
    ctx: ExecutionContext;
  };
}
