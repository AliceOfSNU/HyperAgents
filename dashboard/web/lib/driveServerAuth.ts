// Server-only OAuth token refresh, shared by the app/api/drive/* routes.
//
// Why OAuth instead of the public API key we started with: files.list
// (metadata) works fine with an API key, but files.get?alt=media (actual
// file content) categorically returns Google's automated-abuse block page
// for API-key-authenticated requests -- confirmed with a brand-new,
// zero-usage key, so it's not a rate limit, it's how Google treats
// unauthenticated content downloads. A real OAuth-authenticated request
// (the account owner's own credentials, same ones dashboard/scripts/drive_auth.py
// uses locally) doesn't hit this at all.
//
// Server-side only: GOOGLE_OAUTH_CLIENT_ID/SECRET/REFRESH_TOKEN must never
// get a NEXT_PUBLIC_ prefix.

let cachedToken: { accessToken: string; expiresAt: number } | null = null;

export async function getDriveAccessToken(): Promise<string> {
  if (cachedToken && Date.now() < cachedToken.expiresAt - 30_000) {
    return cachedToken.accessToken;
  }

  const clientId = process.env.GOOGLE_OAUTH_CLIENT_ID;
  const clientSecret = process.env.GOOGLE_OAUTH_CLIENT_SECRET;
  const refreshToken = process.env.GOOGLE_OAUTH_REFRESH_TOKEN;
  if (!clientId || !clientSecret || !refreshToken) {
    throw new Error(
      "GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET / GOOGLE_OAUTH_REFRESH_TOKEN are not set on the server."
    );
  }

  const res = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      client_id: clientId,
      client_secret: clientSecret,
      refresh_token: refreshToken,
      grant_type: "refresh_token",
    }),
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`OAuth token refresh failed: ${res.status} ${await res.text()}`);
  }
  const data = await res.json();
  cachedToken = { accessToken: data.access_token, expiresAt: Date.now() + data.expires_in * 1000 };
  return cachedToken.accessToken;
}
