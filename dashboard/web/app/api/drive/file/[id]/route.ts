import { NextResponse } from "next/server";
import { getDriveAccessToken } from "@/lib/driveServerAuth";

// See ../../list/route.ts and lib/driveServerAuth.ts: this proxies Drive's
// alt=media (file content) download server-side using real OAuth
// credentials, since API-key-authenticated content downloads are
// categorically blocked by Google's automated-abuse detection.
export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;

  let accessToken: string;
  try {
    accessToken = await getDriveAccessToken();
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }

  const url = `https://www.googleapis.com/drive/v3/files/${id}?alt=media`;
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${accessToken}` },
    cache: "no-store",
  });
  if (!res.ok) {
    return NextResponse.json({ error: await res.text() }, { status: res.status });
  }
  const data = await res.json();
  return NextResponse.json(data, { headers: { "Cache-Control": "no-store" } });
}
