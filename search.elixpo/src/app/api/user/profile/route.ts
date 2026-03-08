import { NextRequest } from 'next/server';
import { getAuthUser } from '@/lib/auth';
import { getUserById, updateUserProfile, deleteUserAccount } from '@/lib/db';

export const runtime = 'edge';

/**
 * GET /api/user/profile
 * Full user profile with preferences and usage stats.
 */
export async function GET(req: NextRequest) {
  const ssoUser = await getAuthUser(req);
  if (!ssoUser) {
    return Response.json({ error: 'Not authenticated' }, { status: 401 });
  }

  const user = await getUserById(ssoUser.id);
  if (!user) {
    return Response.json({ error: 'User not found' }, { status: 404 });
  }

  return Response.json({
    id: user.id,
    email: user.email,
    displayName: user.displayName,
    avatar: user.avatar,
    provider: user.provider,
    emailVerified: !!user.emailVerified,

    profile: {
      bio: user.bio,
      location: user.location,
      website: user.website,
      company: user.company,
      jobTitle: user.jobTitle,
    },

    preferences: {
      theme: user.theme,
      language: user.language,
      searchRegion: user.searchRegion,
      safeSearch: user.safeSearch,
      deepSearchDefault: !!user.deepSearchDefault,
    },

    usage: {
      tier: user.tier,
      totalSearches: user.totalSearches,
      totalSessions: user.totalSessions,
    },

    memberSince: user.createdAt,
    lastLoginAt: user.lastLoginAt,
  });
}

/**
 * PATCH /api/user/profile
 * Update profile fields and/or preferences.
 * Body: { bio?, location?, website?, company?, jobTitle?, theme?, language?, searchRegion?, safeSearch?, deepSearchDefault? }
 */
export async function PATCH(req: NextRequest) {
  const ssoUser = await getAuthUser(req);
  if (!ssoUser) {
    return Response.json({ error: 'Not authenticated' }, { status: 401 });
  }

  const body = await req.json();

  // Validate website URL if provided
  if (body.website !== undefined && body.website !== null && body.website !== '') {
    try {
      new URL(body.website);
    } catch {
      return Response.json({ error: 'Invalid website URL' }, { status: 400 });
    }
  }

  // Validate bio length
  if (body.bio !== undefined && body.bio !== null && body.bio.length > 500) {
    return Response.json({ error: 'Bio must be under 500 characters' }, { status: 400 });
  }

  // Validate theme
  if (body.theme !== undefined && !['system', 'light', 'dark'].includes(body.theme)) {
    return Response.json({ error: 'Invalid theme' }, { status: 400 });
  }

  // Validate safeSearch
  if (body.safeSearch !== undefined && ![0, 1, 2].includes(body.safeSearch)) {
    return Response.json({ error: 'safeSearch must be 0, 1, or 2' }, { status: 400 });
  }

  const updated = await updateUserProfile(ssoUser.id, {
    bio: body.bio,
    location: body.location,
    website: body.website,
    company: body.company,
    jobTitle: body.jobTitle,
    theme: body.theme,
    language: body.language,
    searchRegion: body.searchRegion,
    safeSearch: body.safeSearch,
    deepSearchDefault: body.deepSearchDefault !== undefined
      ? (body.deepSearchDefault ? 1 : 0) : undefined,
  });

  if (!updated) {
    return Response.json({ error: 'User not found' }, { status: 404 });
  }

  return Response.json({ ok: true, profile: updated });
}

/**
 * DELETE /api/user/profile
 * Delete the user's account and all associated data.
 */
export async function DELETE(req: NextRequest) {
  const ssoUser = await getAuthUser(req);
  if (!ssoUser) {
    return Response.json({ error: 'Not authenticated' }, { status: 401 });
  }

  await deleteUserAccount(ssoUser.id);
  return Response.json({ ok: true, message: 'Account deleted' });
}
