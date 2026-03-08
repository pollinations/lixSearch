-- Guest session TTL and incognito mode support

-- isGuest: 1 for guest sessions (no userId), 0 for authenticated
-- expiresAt: ISO timestamp; guest sessions expire 24h after creation
-- incognito: 1 = don't persist message content, only metadata
ALTER TABLE Session ADD COLUMN isGuest INTEGER DEFAULT 1;
ALTER TABLE Session ADD COLUMN expiresAt TEXT;
ALTER TABLE Session ADD COLUMN incognito INTEGER DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_session_expiresAt ON Session(expiresAt);
CREATE INDEX IF NOT EXISTS idx_session_isGuest ON Session(isGuest);
