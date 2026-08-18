"""Shared Google Drive OAuth helper for the dashboard's local scripts.

One-time setup: after creating a Desktop-app OAuth Client ID in Google Cloud
Console (see dashboard/README.md), save the downloaded JSON as
.dashboard_secrets/oauth_client.json, then run this file directly once:

    python3 dashboard/scripts/drive_auth.py

It opens a consent URL (visit it, sign in, grant Drive access) and caches a
refresh token in .dashboard_secrets/token.json. Every other script in this
directory imports get_drive_service() from here, which reuses that cached
token (and auto-refreshes it) with no further prompts.
"""
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SECRETS_DIR = Path(__file__).resolve().parent.parent.parent / ".dashboard_secrets"
CLIENT_SECRET_PATH = SECRETS_DIR / "oauth_client.json"
TOKEN_PATH = SECRETS_DIR / "token.json"

# drive.file: access only to files/folders this app creates -- not full
# read/write access to the user's entire Drive.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _load_or_refresh_credentials():
    creds = None
    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _run_consent_flow():
    if not CLIENT_SECRET_PATH.exists():
        raise FileNotFoundError(
            f"{CLIENT_SECRET_PATH} not found. Download the OAuth Desktop Client "
            "JSON from Google Cloud Console and save it there first (see "
            "dashboard/README.md)."
        )
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_PATH), SCOPES)
    creds = flow.run_local_server(port=0, open_browser=False)
    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    return creds


def get_credentials():
    creds = _load_or_refresh_credentials()
    if not creds or not creds.valid:
        creds = _run_consent_flow()
    return creds


def get_drive_service():
    return build("drive", "v3", credentials=get_credentials())


if __name__ == "__main__":
    creds = get_credentials()
    print("Drive OAuth credentials ready." if creds.valid else "FAILED to obtain valid credentials.", file=sys.stderr)
