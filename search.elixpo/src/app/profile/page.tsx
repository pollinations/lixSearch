'use client';

import dynamic from 'next/dynamic';

const SettingsContent = dynamic(() => import('@/components/SettingsContent'), { ssr: false });

export default function ProfilePage() {
  return <SettingsContent />;
}
