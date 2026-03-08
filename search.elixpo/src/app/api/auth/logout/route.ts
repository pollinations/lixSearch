import { NextRequest } from 'next/server';
import { clearAuthCookies } from '@/lib/auth';

export const runtime = 'edge';

/**
 * POST /api/auth/logout
 * Clears auth cookies and redirects to home.
 */
export async function POST(req: NextRequest) {
  const headers = new Headers();
  clearAuthCookies(headers);

  return new Response(JSON.stringify({ ok: true }), {
    status: 200,
    headers,
  });
}

/**
 * GET /api/auth/logout
 * For convenience — clears cookies and redirects to home.
 */
export async function GET(req: NextRequest) {
  const headers = new Headers({ Location: '/' });
  clearAuthCookies(headers);

  return new Response(null, { status: 302, headers });
}
