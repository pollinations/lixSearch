import { NextRequest } from 'next/server';
import { validateXID } from '@/lib/api';
import { upsertSession, createMessage, listSessions } from '@/lib/db';

export const runtime = 'edge';

export async function POST(req: NextRequest) {
  try {
    const xid = req.headers.get('x-xid');
    if (!validateXID(xid)) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json();
    const { sessionId, query, content, sources, images } = body;

    if (!sessionId || !content) {
      return Response.json({ error: 'Missing required fields' }, { status: 400 });
    }

    // Upsert session
    const session = await upsertSession(
      sessionId,
      body.clientId || 'anonymous',
      query?.slice(0, 100) || null
    );

    // Save user message
    if (query) {
      await createMessage(session.id, 'user', query);
    }

    // Save assistant message
    const message = await createMessage(session.id, 'assistant', content, sources, images);

    return Response.json({ id: message.id, sessionId: session.id });
  } catch (err) {
    console.error('[API/conversations] Save error:', err);
    return Response.json({ error: 'Failed to save conversation' }, { status: 500 });
  }
}

export async function GET(req: NextRequest) {
  try {
    const xid = req.headers.get('x-xid');
    if (!validateXID(xid)) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const clientId = req.nextUrl.searchParams.get('clientId') || 'anonymous';
    const limit = parseInt(req.nextUrl.searchParams.get('limit') || '20');

    const sessions = await listSessions(clientId, limit);
    return Response.json(sessions);
  } catch (err) {
    console.error('[API/conversations] List error:', err);
    return Response.json({ error: 'Failed to list conversations' }, { status: 500 });
  }
}
