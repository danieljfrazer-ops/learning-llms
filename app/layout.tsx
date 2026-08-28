import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import { BeginnerModeProvider } from '@/app/components/BeginnerMode';
import { EvidenceModeProvider } from '@/app/components/EvidenceMode';

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

export const metadata: Metadata = {
  title: { default: 'Learning Language Models', template: '%s · Learning Language Models' },
  description: 'A reproducible, beginner-friendly course for training, evaluating, adapting, and understanding small language models on personal hardware.',
  applicationName: 'Learning Language Models',
  keywords: ['language models', 'LLM education', 'MLX', 'machine learning', 'TinyStories', 'Shakespeare', 'LoRA', 'text to SQL'],
  icons: { icon: '/favicon.svg' },
  robots: { index: true, follow: true },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${geistSans.variable} ${geistMono.variable}`}><EvidenceModeProvider><BeginnerModeProvider>{children}</BeginnerModeProvider></EvidenceModeProvider></body></html>;
}
