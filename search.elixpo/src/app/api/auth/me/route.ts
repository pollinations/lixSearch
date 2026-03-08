import { NextRequest } from 'next/server';
import { getAuthUser } from '@/lib/auth';
import { getUserById } from '@/lib/db';

export const runtime = 'edge';

/**
 * GET /api/auth/me
 * Returns the current user's profile (SSO data + local D1 profile).
 * Returns 401 if not authenticated.
 */
export async function GET(req: NextRequest) {
  const ssoUser = await getAuthUser(req);
  if (!ssoUser) {
    return Response.json({ error: 'Not authenticated' }, { status: 401 });
  }

  // Fetch local profile from D1 (has preferences, bio, etc.)
  const localUser = await getUserById(ssoUser.id);

  return Response.json({
    // SSO fields
    id: ssoUser.id,
    email: ssoUser.email,
    displayName: ssoUser.displayName,
    avatar: ssoUser.avatar,
    provider: ssoUser.provider,
    emailVerified: ssoUser.emailVerified,

    // Local profile fields
    bio: localUser?.bio ?? null,
    location: localUser?.location ?? null,
    website: localUser?.website ?? null,
    company: localUser?.company ?? null,
    jobTitle: localUser?.jobTitle ?? null,

    // Preferences
    theme: localUser?.theme ?? 'system',
    language: localUser?.language ?? 'en',
    searchRegion: localUser?.searchRegion ?? 'auto',
    safeSearch: localUser?.safeSearch ?? 1,
    deepSearchDefault: localUser?.deepSearchDefault ?? 0,

    // Usage
    tier: localUser?.tier ?? 'free',
    totalSearches: localUser?.totalSearches ?? 0,
    totalSessions: localUser?.totalSessions ?? 0,
    memberSince: localUser?.createdAt ?? null,
  });
}
