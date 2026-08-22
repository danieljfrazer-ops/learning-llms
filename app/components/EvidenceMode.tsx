'use client';

import { createContext, useContext, useSyncExternalStore } from 'react';

export type EvidenceMode = 'reference' | 'local';
type EvidenceContextValue = { mode: EvidenceMode; setMode: (mode: EvidenceMode) => void };
const EvidenceContext = createContext<EvidenceContextValue>({ mode: 'reference', setMode: () => undefined });
const storageKey = 'learning-llms-evidence-mode';

export function EvidenceModeProvider({ children }: { children: React.ReactNode }) {
  const mode = useSyncExternalStore(
    (notify) => { window.addEventListener('storage', notify); window.addEventListener('evidence-mode-change', notify); return () => { window.removeEventListener('storage', notify); window.removeEventListener('evidence-mode-change', notify); }; },
    () => window.localStorage.getItem(storageKey) === 'local' ? 'local' : 'reference',
    () => 'reference' as EvidenceMode,
  );
  const setMode = (next: EvidenceMode) => { window.localStorage.setItem(storageKey, next); window.dispatchEvent(new Event('evidence-mode-change')); };
  return <EvidenceContext.Provider value={{ mode, setMode }}><div data-evidence-mode={mode}>{children}</div></EvidenceContext.Provider>;
}

export function useEvidenceMode() { return useContext(EvidenceContext); }
export function useEvidenceUrl(filename: string) { return `/data/${useEvidenceMode().mode}/${filename}`; }

export function ReferenceOnly({ children }: { children: React.ReactNode }) {
  return useEvidenceMode().mode === 'reference' ? <>{children}</> : null;
}

export function LocalOnly({ children }: { children: React.ReactNode }) {
  return useEvidenceMode().mode === 'local' ? <>{children}</> : null;
}

export function LocalEvidencePlaceholder({ title }: { title: string }) {
  return <LocalOnly><div className="local-evidence-placeholder"><strong>Your evidence goes here</strong><p>Run this stage with My Lab selected. The training script writes ignored local results, and the matching dashboard refreshes without altering the published reference evidence. Until then, use the surrounding lesson as the procedure and compare your measurements afterward.</p><small>Waiting for: {title}</small></div></LocalOnly>;
}

export function StageState({ reference }: { reference: string }) {
  const { mode } = useEvidenceMode();
  return <>{mode === 'reference' ? reference : 'guide ready'}</>;
}

export function EvidenceModeToggle() {
  const { mode, setMode } = useEvidenceMode();
  const local = mode === 'local';
  return <label className="evidence-toggle"><span><strong>{local ? 'My lab' : 'Reference results'}</strong><small>{local ? 'Your runs and blank states' : 'Published course evidence'}</small></span><input type="checkbox" checked={local} onChange={event => setMode(event.target.checked ? 'local' : 'reference')} aria-label="Show my local experiment results" /><i aria-hidden="true" /></label>;
}

export function EvidenceMachineCard() {
  const { mode } = useEvidenceMode();
  return mode === 'reference'
    ? <div className="machine-card"><span className="live-dot" /> REFERENCE LAB<strong>MacBook Air · 32 GB</strong><small>Apple M5 · MLX 0.32</small></div>
    : <div className="machine-card local-machine-card"><span className="live-dot" /> MY LAB<strong>Your hardware</strong><small>Run scripts/system_report.py</small></div>;
}
