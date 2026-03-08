import { NextRequest } from 'next/server';
import { getRefreshToken, refreshAccessToken, setAuthCookies, clearAuthCookies } from '@/lib/auth';

export const runtime = 'edge';

/**
 * POST /api/auth/refresh
 * Refreshes the access token using the refresh token cookie.
 * Sets new auth cookies on success.
 */
export async function POST(req: NextRequest) {
  const refreshToken = getRefreshToken(req);

  if (!refreshToken) {
    return Response.json({ error: 'No refresh token' }, { status: 401 });
  }

  try {
    const tokens = await refreshAccessToken(refreshToken);
    const headers = new Headers();
    setAuthCookies(headers, tokens);

    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers,
    });
  } catch (err) {
    // Refresh failed — clear cookies and force re-login
    const headers = new Headers();
    clearAuthCookies(headers);

    return new Response(
      JSON.stringify({ error: 'Refresh failed, please login again' }),
      { status: 401, headers }
    );
  }
}
