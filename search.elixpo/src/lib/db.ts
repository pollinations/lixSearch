import { getRequestContext } from '@cloudflare/next-on-pages';

function cuid(): string {
  const ts = Date.now().toString(36);
  const rand = Math.random().toString(36).slice(2, 10);
  return `c${ts}${rand}`;
}

function getBindings() {
  const ctx = getRequestContext();
  return {
    DB: ctx.env.DB as D1Database,
    SESSIONS_KV: ctx.env.SESSIONS_KV as KVNamespace,
    RATE_LIMIT_KV: ctx.env.RATE_LIMIT_KV as KVNamespace,
  };
}

// ── Session ──────────────────────────────────────────────────────────────────

export async function upsertSession(id: string, clientId: string, title?: string | null) {
  const { DB } = getBindings();
  const now = new Date().toISOString();

  const existing = await DB.prepare('SELECT id FROM Session WHERE id = ?').bind(id).first();

  if (existing) {
    const parts = ['updatedAt = ?'];
    const vals: (string | null)[] = [now];
    if (title !== undefined) {
      parts.push('title = ?');
      vals.push(title ?? null);
    }
    vals.push(id);
    await DB.prepare(`UPDATE Session SET ${parts.join(', ')} WHERE id = ?`).bind(...vals).run();
  } else {
    await DB.prepare(
      'INSERT INTO Session (id, clientId, title, createdAt, updatedAt) VALUES (?, ?, ?, ?, ?)'
    ).bind(id, clientId, title ?? null, now, now).run();
  }

  return { id, clientId, title };
}

export async function getSessionWithMessages(sessionId: string) {
  const { DB, SESSIONS_KV } = getBindings();

  // Try KV cache first
  const cached = await SESSIONS_KV.get(`session:${sessionId}`, 'json');
  if (cached) return cached as { title: string | null; messages: Array<Record<string, unknown>> };

  const session = await DB.prepare('SELECT id, title FROM Session WHERE id = ?')
    .bind(sessionId).first<{ id: string; title: string | null }>();

  if (!session) return null;

  const { results: messages } = await DB.prepare(
    'SELECT id, role, content, sources, images FROM Message WHERE sessionId = ? ORDER BY createdAt ASC'
  ).bind(sessionId).all();

  const data = {
    title: session.title,
    messages: messages.map((m: Record<string, unknown>) => ({
      id: m.id,
      role: m.role,
      content: m.content,
      sources: m.sources ? JSON.parse(m.sources as string) : undefined,
      images: m.images ? JSON.parse(m.images as string) : undefined,
    })),
  };

  // Cache in KV for 10 minutes (sessions are mostly read)
  await SESSIONS_KV.put(`session:${sessionId}`, JSON.stringify(data), { expirationTtl: 600 });

  return data;
}

export async function listSessions(clientId: string, limit: number = 20) {
  const { DB } = getBindings();

  const { results } = await DB.prepare(
    `SELECT s.id, s.title, s.createdAt, s.updatedAt,
            (SELECT COUNT(*) FROM Message m WHERE m.sessionId = s.id) as messageCount
     FROM Session s WHERE s.clientId = ? ORDER BY s.updatedAt DESC LIMIT ?`
  ).bind(clientId, limit).all();

  return results;
}

// ── Message ──────────────────────────────────────────────────────────────────

export async function createMessage(
  sessionId: string,
  role: string,
  content: string,
  sources?: unknown,
  images?: unknown
) {
  const { DB, SESSIONS_KV } = getBindings();
  const id = cuid();
  const now = new Date().toISOString();

  await DB.prepare(
    'INSERT INTO Message (id, sessionId, role, content, sources, images, createdAt) VALUES (?, ?, ?, ?, ?, ?, ?)'
  ).bind(
    id, sessionId, role, content,
    sources ? JSON.stringify(sources) : null,
    images ? JSON.stringify(images) : null,
    now
  ).run();

  // Invalidate KV cache for this session (new message = stale cache)
  await SESSIONS_KV.delete(`session:${sessionId}`);

  return { id, sessionId, role, content };
}

// ── Bookmark ─────────────────────────────────────────────────────────────────

export async function createBookmark(sessionId: string, clientId: string) {
  const { DB } = getBindings();
  const id = cuid();
  const now = new Date().toISOString();

  await DB.prepare(
    'INSERT INTO Bookmark (id, sessionId, clientId, createdAt) VALUES (?, ?, ?, ?)'
  ).bind(id, sessionId, clientId, now).run();

  return { id };
}

// ── Discover ─────────────────────────────────────────────────────────────────

export async function getDiscoverArticles(category?: string, dayKey?: string) {
  const { DB } = getBindings();
  const day = dayKey || new Date().toISOString().slice(0, 10);

  if (category) {
    const { results } = await DB.prepare(
      'SELECT * FROM DiscoverArticle WHERE category = ? AND dayKey = ? ORDER BY generatedAt DESC'
    ).bind(category, day).all();
    return results;
  }

  const { results } = await DB.prepare(
    'SELECT * FROM DiscoverArticle WHERE dayKey = ? ORDER BY generatedAt DESC'
  ).bind(day).all();
  return results;
}

export async function saveDiscoverArticles(
  dayKey: string,
  categories: string[],
  categoryArticles: Record<string, Array<{
    title: string; excerpt: string; sourceUrl?: string; sourceTitle?: string;
  }>>
) {
  const { DB } = getBindings();

  // Delete existing for these categories on this day
  for (const cat of categories) {
    await DB.prepare('DELETE FROM DiscoverArticle WHERE category = ? AND dayKey = ?')
      .bind(cat, dayKey).run();
  }

  let totalSaved = 0;
  for (const [category, articles] of Object.entries(categoryArticles)) {
    if (!Array.isArray(articles)) continue;
    for (const article of articles) {
      const id = cuid();
      await DB.prepare(
        'INSERT INTO DiscoverArticle (id, category, title, excerpt, sourceUrl, sourceTitle, dayKey) VALUES (?, ?, ?, ?, ?, ?, ?)'
      ).bind(id, category, article.title, article.excerpt, article.sourceUrl || null, article.sourceTitle || null, dayKey).run();
      totalSaved++;
    }
  }

  return totalSaved;
}

export async function cleanupOldArticles(retentionDays: number = 30) {
  const { DB } = getBindings();
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - retentionDays);
  const cutoffDay = cutoff.toISOString().slice(0, 10);

  const result = await DB.prepare('DELETE FROM DiscoverArticle WHERE dayKey < ?')
    .bind(cutoffDay).run();

  return result.meta?.changes || 0;
}

// ── Rate Limiting ────────────────────────────────────────────────────────────

export async function checkGuestRateLimit(ip: string, limit: number = 15): Promise<{ allowed: boolean; remaining: number }> {
  const { RATE_LIMIT_KV } = getBindings();
  const key = `guest:${ip}`;

  const current = parseInt(await RATE_LIMIT_KV.get(key) || '0', 10);

  if (current >= limit) {
    return { allowed: false, remaining: 0 };
  }

  // Increment with 24h TTL
  await RATE_LIMIT_KV.put(key, String(current + 1), { expirationTtl: 86400 });

  return { allowed: true, remaining: limit - current - 1 };
}
