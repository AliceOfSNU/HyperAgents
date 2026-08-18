# RSI Dashboard

Tracks/controls `generate_loop.py` research-domain runs. Data flows:

```
outputs/<run_id>/...  --(export.py)-->  local JSON  --(upload.py, OAuth)-->  Google Drive
                                                                                   |
                                                                    (Next.js API routes, same OAuth)
                                                                                   v
                                                                      Next.js dashboard on Vercel
```

Everything -- local upload and the deployed dashboard's reads -- authenticates to
Drive as your own Google account via OAuth (the same refresh token, minted once).
There's no API key anywhere in this design anymore: an earlier "public link + API
key" version worked for listing files but Google's Drive API categorically blocks
API-key-authenticated `alt=media` (file content) downloads with an automated-abuse
page -- confirmed with a brand-new, zero-usage key, so it's not a rate limit, it's
just how that endpoint treats unauthenticated content requests. Real OAuth doesn't
hit this.

## One-time setup (you)

**1. OAuth consent screen** (project `project-3d3d9d1f-31ec-443a-855`):
   - Go to https://console.cloud.google.com/apis/credentials/consent?project=project-3d3d9d1f-31ec-443a-855
   - User type: **External**
   - App name: anything (e.g. "RSI Dashboard"). User support email / developer contact: your email.
   - Scopes screen: skip (no changes needed).
   - Test users: add **khjune29@gmail.com**.
   - Save. Leave it in "Testing" status -- no Google verification needed for personal use with a test-user allowlist.

**2. OAuth Client ID**:
   - Go to https://console.cloud.google.com/apis/credentials?project=project-3d3d9d1f-31ec-443a-855
   - Create Credentials -> OAuth client ID -> Application type: **Desktop app** -> name it anything -> Create.
   - Download the JSON, save it as exactly:
     `HyperAgents/.dashboard_secrets/oauth_client.json`

**3. Run the one-time consent flow:**
   ```
   cd HyperAgents
   venv_nat/bin/python3 dashboard/scripts/drive_auth.py
   ```
   It prints a URL -- open it in a browser, sign in as khjune29@gmail.com, approve. A
   refresh token is cached at `.dashboard_secrets/token.json`; every later script run
   (including the web app's API routes, once you copy the token into Vercel -- see
   below) reuses it silently, no more prompts.

Both files under `.dashboard_secrets/` are gitignored -- never commit them.

## What gets uploaded

`export.py` reads every `outputs/generate_*/` run directory and writes compact JSON
files (run index, per-run detail, per-agent detail) into a local
`dashboard/export/` staging directory. `upload.py` pushes everything in that staging
directory into a Drive folder (created automatically on first run, named
"RSI Dashboard Data"). Raw log files are uploaded/updated as plain text files,
refreshed on a slower cadence (every ~10 min) since they're large and change less
usefully often.

`run_loop.py` runs export+upload on a loop and is the thing you actually leave running.

## Web dashboard

`web/` is a Next.js app. The browser never talks to Google directly -- it calls our
own API routes (`app/api/drive/list`, `app/api/drive/file/[id]`), which run
server-side on Vercel and proxy to Drive using OAuth (`lib/driveServerAuth.ts`
refreshes an access token from a stored refresh token on each request). Required
server-side env vars (no `NEXT_PUBLIC_` prefix -- none of this reaches the client):

- `DRIVE_FOLDER_ID` -- the folder ID `upload.py` prints on its first run
- `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET` -- from `.dashboard_secrets/oauth_client.json`
- `GOOGLE_OAUTH_REFRESH_TOKEN` -- from `.dashboard_secrets/token.json`

## Running it

1. **Start the sync loop** (leave this running -- it's what keeps Drive current):
   ```
   cd HyperAgents
   nohup venv_nat/bin/python3 dashboard/scripts/run_loop.py > dashboard_loop.log 2>&1 &
   ```
   The first cycle prints the Drive folder ID it created.

2. **Local dev**: copy `web/.env.local.example` to `web/.env.local`, fill in the four
   vars listed above, then:
   ```
   cd dashboard/web
   npm run dev
   ```

3. **Deploy to Vercel**:
   ```
   cd dashboard/web
   npx vercel login       # one-time, opens a browser
   npx vercel link        # creates/links a Vercel project
   npx vercel env add DRIVE_FOLDER_ID production --sensitive
   npx vercel env add GOOGLE_OAUTH_CLIENT_ID production --sensitive
   npx vercel env add GOOGLE_OAUTH_CLIENT_SECRET production --sensitive
   npx vercel env add GOOGLE_OAUTH_REFRESH_TOKEN production --sensitive
   npx vercel --prod
   ```
