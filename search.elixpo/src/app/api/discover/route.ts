import { NextRequest } from 'next/server';
import { getDiscoverArticles } from '@/lib/db';

export const runtime = 'edge';

export async function GET(req: NextRequest) {
  try {
    const category = req.nextUrl.searchParams.get('category') || undefined;
    const day = req.nextUrl.searchParams.get('day') || undefined;

    const articles = await getDiscoverArticles(category, day);
    return Response.json(articles);
  } catch (err) {
    console.error('[API/discover] Read error:', err);
    return Response.json({ error: 'Failed to read discover articles' }, { status: 500 });
  }
}
