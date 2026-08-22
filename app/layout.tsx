import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import { BeginnerModeProvider } from '@/app/components/BeginnerMode';
import { EvidenceModeProvider } from '@/app/components/EvidenceMode';

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

export const metadata: Metadata = { title: 'Learning Language Models', description: 'A living, hands-on guide to training, evaluating, and understanding small language models on personal hardware.' };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${geistSans.variable} ${geistMono.variable}`}><EvidenceModeProvider><BeginnerModeProvider>{children}</BeginnerModeProvider></EvidenceModeProvider></body></html>;
}
