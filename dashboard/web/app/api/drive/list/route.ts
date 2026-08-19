import { NextResponse } from "next/server";
import { getDriveAccessToken } from "@/lib/driveServerAuth";

// Server-side only. See lib/driveServerAuth.ts for why this uses OAuth
// rather than the public API key we started with.
const FOLDER_ID = process.env.DRIVE_FOLDER_ID;

export async function GET() {
  if (!FOLDER_ID) {
    return NextResponse.json({ error: "DRIVE_FOLDER_ID is not set on the server." }, { status: 500 });
  }

  let accessToken: string;
  try {
    accessToken = await getDriveAccessToken();
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }

  const q = encodeURIComponent(`'${FOLDER_ID}' in parents and trashed=false`);
  const url = `https://www.googleapis.com/drive/v3/files?q=${q}&fields=files(id,name)&pageSize=1000`;

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
