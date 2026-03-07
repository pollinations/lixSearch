import { NextRequest } from 'next/server';
import { backendUrl, backendHeaders, validateXID } from '@/lib/api';
import { checkGuestRateLimit } from '@/lib/db';

export const runtime = 'edge';

export async function POST(req: NextRequest) {
  try {
    const xid = req.headers.get('x-xid');
    if (!validateXID(xid)) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Guest rate limiting by IP
    const ip = req.headers.get('cf-connecting-ip') || req.headers.get('x-forwarded-for') || 'unknown';
    const { allowed, remaining } = await checkGuestRateLimit(ip);
    if (!allowed) {
      return Response.json(
        { error: 'Guest request limit reached. Sign in for unlimited access.' },
        { status: 429, headers: { 'X-RateLimit-Remaining': '0' } }
      );
    }

    const body = await req.json();
    const { query, session_id, stream = true, deep_search = false, image } = body;

    if (!session_id || (!query && !image)) {
      return Response.json({ error: 'Missing session_id, and query or image' }, { status: 400 });
    }

    const backendPayload: Record<string, unknown> = { query: query || '', session_id, stream, deep_search };
    if (image) backendPayload.image = image;

    const backendRes = await fetch(backendUrl('/api/search'), {
      method: 'POST',
      headers: backendHeaders(),
      body: JSON.stringify(backendPayload),
    });

    if (!backendRes.ok) {
      const text = await backendRes.text();
      return new Response(text, { status: backendRes.status });
    }

    // For streaming, pass through the SSE stream
    if (stream && backendRes.body) {
      return new Response(backendRes.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'X-RateLimit-Remaining': String(remaining),
        },
      });
    }

    // Non-streaming: return JSON
    const data = await backendRes.json();
    return Response.json(data, {
      headers: { 'X-RateLimit-Remaining': String(remaining) },
    });
  } catch (err) {
    console.error('[API/search] Proxy error:', err);
    return Response.json({ error: 'Internal proxy error' }, { status: 500 });
  }
}
