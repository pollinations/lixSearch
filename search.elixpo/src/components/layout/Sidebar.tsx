'use client';

import { useState, useRef, useEffect } from 'react';
import { Plus, Home, Globe, BookOpen, Settings, LogIn, LogOut, User } from 'lucide-react';
import { usePathname, useRouter } from 'next/navigation';
import IconButton from '@/components/ui/IconButton';
import { useAuth } from '@/hooks/useAuth';

interface SidebarProps {
  onNewSearch: () => void;
}

export default function Sidebar({ onNewSearch }: SidebarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, loading, login, logout } = useAuth();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const isActive = (path: string) => pathname === path;

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    if (menuOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [menuOpen]);

  return (
    <aside className="w-[60px] h-full bg-[#18191a] border-r border-[#333] flex flex-col items-center py-6 shrink-0">
      <div className="w-5 h-5 mb-8 opacity-80">
        <svg viewBox="0 0 24 24" fill="none" className="w-full h-full invert">
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" />
          <path d="M12 6v12M6 12h12" stroke="currentColor" strokeWidth="2" />
        </svg>
      </div>

      <div className="flex flex-col items-center gap-4">
        <IconButton
          icon={Plus}
          onClick={onNewSearch}
          title="New Search"
          variant="default"
          className="rounded-full"
        />

        <div className="w-8 h-px bg-[#333] my-2" />

        <IconButton
          icon={Home}
          onClick={() => router.push('/')}
          active={isActive('/')}
          title="Home"
          variant="ghost"
        />
        <IconButton
          icon={Globe}
          onClick={() => router.push('/discover')}
          active={isActive('/discover')}
          title="Discover"
          variant="ghost"
        />
        <IconButton
          icon={BookOpen}
          onClick={() => router.push('/library')}
          active={isActive('/library')}
          title="Library"
          variant="ghost"
        />
      </div>

      {/* Bottom: settings + user avatar */}
      <div className="mt-auto flex flex-col items-center gap-3 relative" ref={menuRef}>
        <IconButton
          icon={Settings}
          onClick={() => router.push('/settings')}
          active={pathname.startsWith('/settings')}
          title="Settings"
          variant="ghost"
        />

        {!loading && (
          user ? (
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              title={user.displayName || user.email}
              className="w-9 h-9 rounded-full overflow-hidden border-2 border-transparent hover:border-lime-main transition-colors cursor-pointer"
            >
              {user.avatar ? (
                <img src={user.avatar} alt="" className="w-full h-full object-cover" />
              ) : (
                <div className="w-full h-full bg-[#333] flex items-center justify-center text-white text-sm font-medium">
                  {(user.displayName || user.email)[0].toUpperCase()}
                </div>
              )}
            </button>
          ) : (
            <IconButton
              icon={LogIn}
              onClick={() => login(pathname)}
              title="Sign in"
              variant="ghost"
            />
          )
        )}

        {/* User dropdown */}
        {menuOpen && user && (
          <div className="absolute bottom-12 left-[60px] w-56 bg-[#232425] border border-[#333] rounded-xl shadow-card-lg py-2 z-50">
            <div className="px-4 py-3 border-b border-[#333]">
              <p className="text-white text-sm font-medium truncate">{user.displayName || 'User'}</p>
              <p className="text-[#888] text-xs truncate">{user.email}</p>
            </div>
            <button
              onClick={() => { setMenuOpen(false); router.push('/settings'); }}
              className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-[#ccc] hover:text-white hover:bg-[#333] transition-colors"
            >
              <User size={16} />
              Profile & Settings
            </button>
            <div className="h-px bg-[#333] mx-3" />
            <button
              onClick={() => { setMenuOpen(false); logout(); }}
              className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-red-400 hover:text-red-300 hover:bg-[#333] transition-colors"
            >
              <LogOut size={16} />
              Sign out
            </button>
          </div>
        )}
      </div>
    </aside>
  );
}
