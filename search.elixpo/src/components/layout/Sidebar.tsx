'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import {
  Plus, Home, Globe, BookOpen, Clock, MoreHorizontal,
  LogIn, LogOut, User, Bell, ChevronLeft,
} from 'lucide-react';
import { usePathname, useRouter } from 'next/navigation';
import { useAuth } from '@/hooks/useAuth';

interface SidebarProps {
  onNewSearch: () => void;
}

interface RecentSession {
  id: string;
  title: string | null;
  updatedAt: string;
}

export default function Sidebar({ onNewSearch }: SidebarProps) {
  const pathname = usePathname();
  const router = useRouter();
  const { user, loading, login, logout } = useAuth();
  const [recents, setRecents] = useState<RecentSession[]>([]);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const isActive = (path: string) =>
    path === '/' ? pathname === '/' : pathname.startsWith(path);

  // Fetch recent searches
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const clientId = window.localStorage.getItem('elixpo_client_id') || '';
    if (!clientId) return;
    fetch(`/api/conversations?clientId=${encodeURIComponent(clientId)}&limit=8`)
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setRecents(data);
      })
      .catch(() => {});
  }, []);

  // Close menu on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    }
    if (menuOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [menuOpen]);

  const navItems = [
    { icon: Home, label: 'Home', href: '/' },
    { icon: Globe, label: 'Discover', href: '/discover' },
    { icon: BookOpen, label: 'Library', href: '/library' },
  ];

  return (
    <aside className="w-[240px] h-full bg-[#171717] flex flex-col shrink-0 border-r border-[#2a2b2d]">
      {/* New Thread */}
      <div className="px-3 pt-4 pb-2">
        <button
          onClick={onNewSearch}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm text-[#ccc] hover:text-white hover:bg-[#2a2b2d] transition-colors"
        >
          <Plus size={18} />
          <span>New Thread</span>
        </button>
      </div>

      {/* Navigation */}
      <nav className="px-3 flex flex-col gap-0.5">
        {navItems.map((item) => (
          <button
            key={item.href}
            onClick={() => router.push(item.href)}
            className={`
              w-full flex items-center gap-2.5 px-3 py-2 rounded-xl text-sm transition-colors
              ${isActive(item.href)
                ? 'text-white bg-[#2a2b2d]'
                : 'text-[#999] hover:text-white hover:bg-[#222]'}
            `}
          >
            <item.icon size={18} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      {/* Divider */}
      <div className="mx-4 my-3 h-px bg-[#2a2b2d]" />

      {/* Recent searches */}
      <div className="flex-1 overflow-y-auto custom-scrollbar px-3">
        <p className="px-3 text-[#666] text-xs font-medium mb-2">Recent</p>
        <div className="flex flex-col gap-0.5">
          {recents.length > 0 ? (
            recents.map((s) => (
              <button
                key={s.id}
                onClick={() => router.push(`/c/${s.id}`)}
                className={`
                  w-full flex items-center gap-2.5 px-3 py-1.5 rounded-lg text-sm transition-colors text-left group
                  ${pathname === `/c/${s.id}`
                    ? 'text-white bg-[#2a2b2d]'
                    : 'text-[#999] hover:text-white hover:bg-[#222]'}
                `}
              >
                <span className="truncate flex-1">{s.title || 'Untitled'}</span>
              </button>
            ))
          ) : (
            <p className="px-3 text-[#444] text-xs">No recent searches</p>
          )}
        </div>
      </div>

      {/* Bottom: user section */}
      <div className="px-3 pb-4 pt-2 border-t border-[#2a2b2d] relative" ref={menuRef}>
        {!loading && (
          user ? (
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className="w-full flex items-center gap-2.5 px-2 py-2 rounded-xl hover:bg-[#222] transition-colors"
            >
              {/* Avatar */}
              <div className="w-8 h-8 rounded-full overflow-hidden shrink-0">
                {user.avatar ? (
                  <img src={user.avatar} alt="" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full bg-[#333] flex items-center justify-center text-white text-xs font-medium">
                    {(user.displayName || user.email)[0].toUpperCase()}
                  </div>
                )}
              </div>
              <span className="text-sm text-[#ccc] truncate flex-1 text-left">
                {user.displayName || user.email.split('@')[0]}
              </span>
            </button>
          ) : (
            <button
              onClick={() => login(pathname)}
              className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-sm text-[#999] hover:text-white hover:bg-[#222] transition-colors"
            >
              <LogIn size={18} />
              <span>Sign in</span>
            </button>
          )
        )}

        {/* User dropdown - pops upward */}
        {menuOpen && user && (
          <div className="absolute bottom-full left-3 right-3 mb-1 bg-[#232425] border border-[#333] rounded-xl shadow-card-lg py-1.5 z-50">
            <div className="px-4 py-2.5 border-b border-[#2a2b2d]">
              <p className="text-white text-sm font-medium truncate">{user.displayName || 'User'}</p>
              <p className="text-[#666] text-xs truncate">{user.email}</p>
            </div>
            <button
              onClick={() => { setMenuOpen(false); router.push('/profile'); }}
              className="w-full flex items-center gap-2.5 px-4 py-2 text-sm text-[#ccc] hover:text-white hover:bg-[#2a2b2d] transition-colors"
            >
              <User size={15} />
              Profile & Settings
            </button>
            <div className="h-px bg-[#2a2b2d] mx-3" />
            <button
              onClick={() => { setMenuOpen(false); logout(); }}
              className="w-full flex items-center gap-2.5 px-4 py-2 text-sm text-red-400 hover:text-red-300 hover:bg-[#2a2b2d] transition-colors"
            >
              <LogOut size={15} />
              Sign out
            </button>
          </div>
        )}
      </div>
    </aside>
  );
}
