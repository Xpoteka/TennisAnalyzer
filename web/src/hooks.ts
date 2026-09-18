import { useCallback, useEffect, useRef, useState } from "react";

/** Load data, and reload it every `intervalMs` while `poll` holds (pass 0 to not poll). */
export function useData<T>(
  load: () => Promise<T>,
  deps: unknown[],
  intervalMs = 0,
  poll: (data: T | undefined) => boolean = () => true,
) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<string>();
  const loadRef = useRef(load);
  loadRef.current = load;
  const pollRef = useRef(poll);
  pollRef.current = poll;
  const dataRef = useRef<T | undefined>(undefined);

  const reload = useCallback(async () => {
    try {
      const value = await loadRef.current();
      dataRef.current = value;
      setData(value);
      setError(undefined);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      await reload();
      if (!stopped && intervalMs > 0 && pollRef.current(dataRef.current)) {
        timer = setTimeout(tick, intervalMs);
      }
    };
    dataRef.current = undefined;
    setData(undefined);
    tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, error, reload };
}
