import dynamic from 'next/dynamic';

export const runtime = 'edge';

const SettingsContent = dynamic(() => import('@/components/SettingsContent'), { ssr: false });

export default function SettingsPage() {
  return <SettingsContent />;
}
