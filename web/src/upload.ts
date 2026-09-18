// Resumable chunked uploads (see tennis/api/uploads.py). The server keys a partial upload by
// file name and size, so dropping the same file again continues where it stopped.

import { ApiError, request } from "./api";

const PIECE = 32 * 1024 * 1024;

type Status = { done: boolean; received: number; size: number; path?: string };

export type UploadProgress = { sent: number; size: number; rate: number };

export async function uploadFile(
  file: File,
  onProgress: (p: UploadProgress) => void,
  signal?: AbortSignal,
): Promise<string> {
  const q = (extra = "") =>
    `/api/uploads?name=${encodeURIComponent(file.name)}&size=${file.size}${extra}`;
  let status = await request<Status>("GET", q());
  const started = performance.now();
  const startOffset = status.received;
  while (!status.done) {
    if (signal?.aborted) throw new DOMException("Upload cancelled", "AbortError");
    const offset = status.received;
    const piece = file.slice(offset, Math.min(offset + PIECE, file.size));
    try {
      status = await request<Status>("PUT", q(`&offset=${offset}`), piece, { signal });
    } catch (err) {
      // Another tab or a retry moved the server ahead: continue from where it is.
      if (err instanceof ApiError && err.status === 409 && typeof err.body.received === "number") {
        status = { done: false, received: err.body.received, size: file.size };
        continue;
      }
      throw err;
    }
    const seconds = (performance.now() - started) / 1000;
    onProgress({
      sent: status.received,
      size: file.size,
      rate: seconds > 0 ? (status.received - startOffset) / seconds : 0,
    });
  }
  onProgress({ sent: file.size, size: file.size, rate: 0 });
  if (!status.path) throw new Error("the server did not say where the file went");
  return status.path;
}
