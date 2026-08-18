"use client";

import { useEffect, useRef, useState } from "react";
import { fetchDriveJson, invalidateDriveFileList } from "./drive";

const POLL_MS = 30_000; // export.py's own cycle is ~2min; this just re-checks

export interface DriveJsonState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

export function useDriveJson<T>(filename: string | null): DriveJsonState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const nonce = useRef(0);

  const load = async (bustCache: boolean) => {
    if (!filename) return;
    if (bustCache) invalidateDriveFileList();
    const my = ++nonce.current;
    setLoading(true);
    try {
      const result = await fetchDriveJson<T>(filename);
      if (my !== nonce.current) return;
      setData(result);
      setError(result === null ? `"${filename}" not found in the Drive folder yet.` : null);
    } catch (e) {
      if (my !== nonce.current) return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (my === nonce.current) setLoading(false);
    }
  };

  useEffect(() => {
    // Deliberate: load() flips `loading` true synchronously before its first
    // await, on every filename change / poll tick -- that's the desired
    // behavior (show a loading state while re-fetching), not an accidental
    // render loop, so the set-state-in-effect rule is a false positive here.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load(false);
    const id = setInterval(() => load(false), POLL_MS);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filename]);

  return { data, loading, error, refresh: () => load(true) };
}
