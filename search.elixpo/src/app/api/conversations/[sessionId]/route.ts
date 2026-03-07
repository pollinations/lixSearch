import { NextRequest } from 'next/server';
import { validateXID } from '@/lib/api';
import { getSessionWithMessages } from '@/lib/db';

export const runtime = 'edge';

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ sessionId: string }> }
) {
  try {
    const xid = req.headers.get('x-xid');
    if (!validateXID(xid)) {
      return Response.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { sessionId } = await params;
    const data = await getSessionWithMessages(sessionId);

    if (!data) {
      return Response.json({ messages: [] });
    }

    return Response.json(data);
  } catch (err) {
    console.error('[API/conversations] Load error:', err);
    return Response.json({ error: 'Failed to load conversation' }, { status: 500 });
  }
}
