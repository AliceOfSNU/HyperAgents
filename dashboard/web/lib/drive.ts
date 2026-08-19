"use client";

// Fetches go through our own Next.js API routes (app/api/drive/*), which
// proxy to Google Drive server-side. Necessary because Drive's alt=media
// (file content download) endpoint doesn't reliably support CORS for
// cross-origin browser fetches -- files.list (metadata) does support it,
// but alt=media doesn't, confirmed against a live deployment where the
// list call succeeded and the content call was blocked. Routing everything
// through our own server-side routes also means the Drive API key never
// ships to the client bundle at all anymore.

const FILE_LIST_TTL_MS = 20_000;
let fileIdCache: Record<string, string> | null = null;
let fileIdCacheAt = 0;
let inFlightList: Promise<Record<string, string>> | null = null;

async function getFileIdMap(fresh = false): Promise<Record<string, string>> {
  if (!fresh && fileIdCache && Date.now() - fileIdCacheAt < FILE_LIST_TTL_MS) {
    return fileIdCache;
  }
  if (!fresh && inFlightList) return inFlightList;

  inFlightList = fetch("/api/drive/list", { cache: "no-store" })
    .then(async (res) => {
      if (!res.ok) {
        throw new Error(`Drive file list failed: ${res.status} ${await res.text()}`);
      }
      return res.json();
    })
    .then((data) => {
      const map: Record<string, string> = {};
      for (const f of data.files ?? []) map[f.name] = f.id;
      fileIdCache = map;
      fileIdCacheAt = Date.now();
      return map;
    })
    .finally(() => {
      inFlightList = null;
    });

  return inFlightList;
}

export async function fetchDriveJson<T>(filename: string): Promise<T | null> {
  const map = await getFileIdMap();
  const id = map[filename];
  if (!id) return null;
  const res = await fetch(`/api/drive/file/${id}`, { cache: "no-store" });
  if (!res.ok) return null;
  return res.json() as Promise<T>;
}

/** Force a re-list on the next fetch (call after a manual refresh action). */
export function invalidateDriveFileList() {
  fileIdCache = null;
}
