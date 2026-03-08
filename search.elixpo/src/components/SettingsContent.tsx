'use client';

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import {
  User, Shield, Palette, Search, Globe, Zap,
  ChevronRight, LogIn, Trash2, ExternalLink,
} from 'lucide-react';
import Sidebar from '@/components/layout/Sidebar';
import SidePanel from '@/components/layout/SidePanel';
import { useAuth, type AuthUser } from '@/hooks/useAuth';

type Tab = 'profile' | 'preferences' | 'account';

export default function SettingsContent() {
  const router = useRouter();
  const { user, loading, login, logout, refetch } = useAuth();
  const [activeTab, setActiveTab] = useState<Tab>('profile');

  if (loading) {
    return (
      <div className="h-screen flex bg-[#18191a]">
        <Sidebar onNewSearch={() => router.push('/')} />
        <SidePanel />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-[#555] text-sm">Loading...</div>
        </main>
      </div>
    );
  }

  const tabs: { id: Tab; label: string; icon: typeof User }[] = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'preferences', label: 'Preferences', icon: Palette },
    { id: 'account', label: 'Account', icon: Shield },
  ];

  return (
    <div className="h-screen flex bg-[#18191a]">
      <Sidebar onNewSearch={() => router.push('/')} />
      <SidePanel />

      <main className="flex-1 flex flex-col h-full overflow-hidden">
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          <div className="max-w-2xl mx-auto px-6 py-10">
            <h1 className="text-2xl font-bold text-white font-display mb-8">Settings</h1>

            {!user ? (
              <GuestPrompt onLogin={() => login('/settings')} />
            ) : (
              <>
                {/* Tab navigation */}
                <div className="flex gap-1 bg-[#1a1a1a] rounded-xl p-1 mb-8">
                  {tabs.map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`
                        flex-1 flex items-center justify-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all
                        ${activeTab === tab.id
                          ? 'bg-[#2a2b2d] text-white shadow-sm'
                          : 'text-[#888] hover:text-white'}
                      `}
                    >
                      <tab.icon size={16} />
                      {tab.label}
                    </button>
                  ))}
                </div>

                {activeTab === 'profile' && <ProfileTab user={user} onUpdate={refetch} />}
                {activeTab === 'preferences' && <PreferencesTab user={user} onUpdate={refetch} />}
                {activeTab === 'account' && <AccountTab user={user} onLogout={logout} />}
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

// ── Guest prompt ────────────────────────────────────────────────────────────

function GuestPrompt({ onLogin }: { onLogin: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <div className="w-16 h-16 rounded-full bg-[#232425] flex items-center justify-center mb-6">
        <User size={28} className="text-[#555]" />
      </div>
      <h2 className="text-xl font-semibold text-white mb-2">Sign in to access settings</h2>
      <p className="text-[#888] text-sm mb-8 max-w-sm">
        Create an account to save your preferences, access your library across devices, and get unlimited searches.
      </p>
      <button
        onClick={onLogin}
        className="flex items-center gap-2 px-6 py-3 bg-lime-main text-black rounded-xl font-medium text-sm hover:bg-lime-light transition-colors"
      >
        <LogIn size={18} />
        Sign in with Elixpo
      </button>
    </div>
  );
}

// ── Profile tab ─────────────────────────────────────────────────────────────

function ProfileTab({ user, onUpdate }: { user: AuthUser; onUpdate: () => void }) {
  const [saving, setSaving] = useState(false);
  const [bio, setBio] = useState(user.bio || '');
  const [location, setLocation] = useState(user.location || '');
  const [website, setWebsite] = useState(user.website || '');
  const [company, setCompany] = useState(user.company || '');
  const [jobTitle, setJobTitle] = useState(user.jobTitle || '');

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      await fetch('/api/user/profile', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bio, location, website, company, jobTitle }),
      });
      onUpdate();
    } finally {
      setSaving(false);
    }
  }, [bio, location, website, company, jobTitle, onUpdate]);

  return (
    <div className="space-y-6">
      {/* Avatar + name header */}
      <div className="flex items-center gap-4 p-5 bg-[#1a1a1a] rounded-2xl border border-[#2a2b2d]">
        <div className="w-16 h-16 rounded-full overflow-hidden shrink-0">
          {user.avatar ? (
            <img src={user.avatar} alt="" className="w-full h-full object-cover" />
          ) : (
            <div className="w-full h-full bg-[#333] flex items-center justify-center text-white text-xl font-semibold">
              {(user.displayName || user.email)[0].toUpperCase()}
            </div>
          )}
        </div>
        <div className="min-w-0">
          <h3 className="text-white font-semibold text-lg truncate">{user.displayName || 'User'}</h3>
          <p className="text-[#888] text-sm truncate">{user.email}</p>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs px-2 py-0.5 rounded-full bg-lime-dim text-lime-main border border-lime-border">
              {user.tier}
            </span>
            {user.emailVerified && (
              <span className="text-xs text-sage-main">Verified</span>
            )}
          </div>
        </div>
      </div>

      {/* Profile fields */}
      <Section title="About you">
        <Field label="Bio" value={bio} onChange={setBio} placeholder="Tell us about yourself" multiline maxLength={500} />
        <Field label="Location" value={location} onChange={setLocation} placeholder="City, Country" />
        <Field label="Website" value={website} onChange={setWebsite} placeholder="https://..." />
        <Field label="Company" value={company} onChange={setCompany} placeholder="Where you work" />
        <Field label="Job title" value={jobTitle} onChange={setJobTitle} placeholder="What you do" />
      </Section>

      <button
        onClick={handleSave}
        disabled={saving}
        className="w-full py-3 bg-lime-main hover:bg-lime-light text-black font-medium rounded-xl transition-colors disabled:opacity-50"
      >
        {saving ? 'Saving...' : 'Save changes'}
      </button>
    </div>
  );
}

// ── Preferences tab ─────────────────────────────────────────────────────────

function PreferencesTab({ user, onUpdate }: { user: AuthUser; onUpdate: () => void }) {
  const [saving, setSaving] = useState(false);

  const updatePref = useCallback(async (field: string, value: string | number | boolean) => {
    setSaving(true);
    try {
      await fetch('/api/user/profile', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ [field]: value }),
      });
      onUpdate();
    } finally {
      setSaving(false);
    }
  }, [onUpdate]);

  return (
    <div className="space-y-6">
      <Section title="Search">
        <SelectField
          label="Default search region"
          icon={Globe}
          value={user.searchRegion}
          options={[
            { value: 'auto', label: 'Auto-detect' },
            { value: 'us', label: 'United States' },
            { value: 'gb', label: 'United Kingdom' },
            { value: 'de', label: 'Germany' },
            { value: 'fr', label: 'France' },
            { value: 'jp', label: 'Japan' },
            { value: 'in', label: 'India' },
          ]}
          onChange={(v) => updatePref('searchRegion', v)}
        />
        <SelectField
          label="Safe search"
          icon={Shield}
          value={String(user.safeSearch)}
          options={[
            { value: '0', label: 'Off' },
            { value: '1', label: 'Moderate' },
            { value: '2', label: 'Strict' },
          ]}
          onChange={(v) => updatePref('safeSearch', parseInt(v))}
        />
        <ToggleField
          label="Deep search by default"
          description="Use thorough multi-source search for every query"
          icon={Zap}
          checked={user.deepSearchDefault}
          onChange={(v) => updatePref('deepSearchDefault', v)}
        />
      </Section>

      <Section title="Appearance">
        <SelectField
          label="Language"
          icon={Globe}
          value={user.language}
          options={[
            { value: 'en', label: 'English' },
            { value: 'es', label: 'Spanish' },
            { value: 'fr', label: 'French' },
            { value: 'de', label: 'German' },
            { value: 'ja', label: 'Japanese' },
            { value: 'zh', label: 'Chinese' },
            { value: 'hi', label: 'Hindi' },
          ]}
          onChange={(v) => updatePref('language', v)}
        />
      </Section>

      {saving && (
        <p className="text-xs text-[#888] text-center">Saving...</p>
      )}
    </div>
  );
}

// ── Account tab ─────────────────────────────────────────────────────────────

function AccountTab({ user, onLogout }: { user: AuthUser; onLogout: () => void }) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  const handleDelete = async () => {
    if (!confirmDelete) {
      setConfirmDelete(true);
      return;
    }
    await fetch('/api/user/profile', { method: 'DELETE' });
    onLogout();
  };

  return (
    <div className="space-y-6">
      <Section title="Usage">
        <div className="grid grid-cols-3 gap-4">
          <StatCard label="Searches" value={user.totalSearches} />
          <StatCard label="Sessions" value={user.totalSessions} />
          <StatCard label="Member since" value={user.memberSince ? new Date(user.memberSince).toLocaleDateString('en', { month: 'short', year: 'numeric' }) : '-'} />
        </div>
      </Section>

      <Section title="Connected account">
        <div className="flex items-center justify-between p-4 bg-[#1a1a1a] rounded-xl border border-[#2a2b2d]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-[#232425] flex items-center justify-center">
              <ExternalLink size={18} className="text-[#888]" />
            </div>
            <div>
              <p className="text-white text-sm font-medium">Elixpo Accounts</p>
              <p className="text-[#888] text-xs">{user.email}</p>
            </div>
          </div>
          <span className="text-xs text-sage-main">Connected</span>
        </div>
      </Section>

      <Section title="Danger zone">
        <button
          onClick={onLogout}
          className="w-full flex items-center justify-between px-4 py-3 bg-[#1a1a1a] rounded-xl border border-[#2a2b2d] text-[#ccc] hover:text-white hover:border-[#444] transition-colors"
        >
          <span className="text-sm">Sign out</span>
          <ChevronRight size={16} />
        </button>
        <button
          onClick={handleDelete}
          className={`w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl border text-sm font-medium transition-colors ${
            confirmDelete
              ? 'bg-red-500/10 border-red-500/30 text-red-400 hover:bg-red-500/20'
              : 'bg-[#1a1a1a] border-[#2a2b2d] text-red-400 hover:border-red-500/30'
          }`}
        >
          <Trash2 size={16} />
          {confirmDelete ? 'Click again to permanently delete' : 'Delete account'}
        </button>
      </Section>
    </div>
  );
}

// ── Shared UI components ────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-xs font-bold text-[#888] uppercase tracking-wider mb-3">{title}</h3>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function Field({
  label, value, onChange, placeholder, multiline, maxLength,
}: {
  label: string; value: string; onChange: (v: string) => void; placeholder?: string;
  multiline?: boolean; maxLength?: number;
}) {
  const cls = "w-full bg-[#1a1a1a] border border-[#2a2b2d] rounded-xl px-4 py-3 text-white text-sm placeholder-[#555] focus:outline-none focus:border-[#444] transition-colors";
  return (
    <div>
      <label className="text-[#ccc] text-sm mb-1.5 block">{label}</label>
      {multiline ? (
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          maxLength={maxLength}
          rows={3}
          className={`${cls} resize-none`}
        />
      ) : (
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className={cls}
        />
      )}
    </div>
  );
}

function SelectField({
  label, icon: Icon, value, options, onChange,
}: {
  label: string; icon: typeof Globe; value: string;
  options: { value: string; label: string }[]; onChange: (v: string) => void;
}) {
  return (
    <div className="flex items-center justify-between p-4 bg-[#1a1a1a] rounded-xl border border-[#2a2b2d]">
      <div className="flex items-center gap-3">
        <Icon size={18} className="text-[#888]" />
        <span className="text-white text-sm">{label}</span>
      </div>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-[#232425] border border-[#333] rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none cursor-pointer"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  );
}

function ToggleField({
  label, description, icon: Icon, checked, onChange,
}: {
  label: string; description?: string; icon: typeof Zap;
  checked: boolean; onChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between p-4 bg-[#1a1a1a] rounded-xl border border-[#2a2b2d]">
      <div className="flex items-center gap-3">
        <Icon size={18} className="text-[#888]" />
        <div>
          <span className="text-white text-sm">{label}</span>
          {description && <p className="text-[#555] text-xs mt-0.5">{description}</p>}
        </div>
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`w-11 h-6 rounded-full transition-colors relative ${checked ? 'bg-lime-main' : 'bg-[#333]'}`}
      >
        <div className={`absolute top-0.5 w-5 h-5 rounded-full bg-white shadow transition-transform ${checked ? 'left-[22px]' : 'left-0.5'}`} />
      </button>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="p-4 bg-[#1a1a1a] rounded-xl border border-[#2a2b2d] text-center">
      <p className="text-white text-lg font-semibold font-display">{value}</p>
      <p className="text-[#888] text-xs mt-1">{label}</p>
    </div>
  );
}
