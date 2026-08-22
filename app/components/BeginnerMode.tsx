'use client';

import { createContext, useContext, useSyncExternalStore } from 'react';

type BeginnerContextValue = { enabled: boolean; setEnabled: (enabled: boolean) => void };
const BeginnerContext = createContext<BeginnerContextValue>({ enabled: false, setEnabled: () => undefined });

export function BeginnerModeProvider({ children }: { children: React.ReactNode }) {
  const enabled = useSyncExternalStore(
    (notify) => { window.addEventListener('storage', notify); window.addEventListener('beginner-mode-change', notify); return () => { window.removeEventListener('storage', notify); window.removeEventListener('beginner-mode-change', notify); }; },
    () => window.localStorage.getItem('learning-llms-beginner-mode') === 'on',
    () => false,
  );
  const setEnabled = (next: boolean) => { window.localStorage.setItem('learning-llms-beginner-mode', next ? 'on' : 'off'); window.dispatchEvent(new Event('beginner-mode-change')); };
  return <BeginnerContext.Provider value={{ enabled, setEnabled }}><div data-beginner-mode={enabled ? 'on' : 'off'}>{children}</div></BeginnerContext.Provider>;
}

export function useBeginnerMode() { return useContext(BeginnerContext); }

export function BeginnerModeToggle() {
  const { enabled, setEnabled } = useBeginnerMode();
  return <label className="beginner-toggle"><span><strong>Beginner mode</strong><small>{enabled ? 'Extra explanations on' : 'Standard lesson view'}</small></span><input type="checkbox" checked={enabled} onChange={event => setEnabled(event.target.checked)} aria-label="Show extra beginner explanations" /><i aria-hidden="true" /></label>;
}

export function BeginnerOnly({ children, className = '' }: { children: React.ReactNode; className?: string }) {
  const { enabled } = useBeginnerMode();
  return enabled ? <div className={className}>{children}</div> : null;
}
