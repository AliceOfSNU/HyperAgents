"""Pushes dashboard/export/*.json and referenced log files to a shared
Google Drive folder. Owns all Drive state (file IDs, sharing, upload
cadence) -- export.py never touches Drive directly.

Folder is auto-created on first run ("RSI Dashboard Data"), shared "anyone
with the link can view" so the public, referrer-restricted API key used by
the web dashboard can read it. A local ID cache (dashboard/export/.drive_ids.json)
maps filename -> Drive file ID so re-runs update in place instead of
creating duplicates, and tracks each log file's last-uploaded time so logs
only get pushed every LOG_UPLOAD_INTERVAL_S, not every cycle (they're large
and change less usefully often than the JSON).
"""
import json
import sys
import time
from pathlib import Path

from googleapiclient.http import MediaInMemoryUpload, MediaFileUpload

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drive_auth import get_drive_service  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXPORT_DIR = Path(__file__).resolve().parent.parent / "export"
ID_CACHE_PATH = EXPORT_DIR / ".drive_ids.json"
FOLDER_NAME = "RSI Dashboard Data"
LOG_UPLOAD_INTERVAL_S = 10 * 60


def _load_cache():
    if ID_CACHE_PATH.exists():
        return json.loads(ID_CACHE_PATH.read_text(encoding="utf-8"))
    return {"folder_id": None, "files": {}}  # files: name -> {id, last_uploaded}


def _save_cache(cache):
    ID_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _ensure_public_readable(service, file_id):
    perms = service.permissions().list(fileId=file_id, fields="permissions(type,role)").execute()
    if any(p.get("type") == "anyone" for p in perms.get("permissions", [])):
        return
    service.permissions().create(
        fileId=file_id, body={"type": "anyone", "role": "reader"}, fields="id",
    ).execute()


def _ensure_folder(service, cache):
    if cache.get("folder_id"):
        return cache["folder_id"]
    results = service.files().list(
        q=f"name='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id)",
    ).execute()
    existing = results.get("files", [])
    if existing:
        folder_id = existing[0]["id"]
    else:
        folder = service.files().create(
            body={"name": FOLDER_NAME, "mimeType": "application/vnd.google-apps.folder"},
            fields="id",
        ).execute()
        folder_id = folder["id"]
    _ensure_public_readable(service, folder_id)
    cache["folder_id"] = folder_id
    _save_cache(cache)
    return folder_id


def _upsert_file(service, cache, folder_id, name, media, mime_type):
    file_id = cache["files"].get(name, {}).get("id")
    if file_id:
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        created = service.files().create(
            body={"name": name, "parents": [folder_id], "mimeType": mime_type},
            media_body=media, fields="id",
        ).execute()
        file_id = created["id"]
        _ensure_public_readable(service, file_id)
    cache["files"].setdefault(name, {})["id"] = file_id
    return file_id


def _upload_json_files(service, cache, folder_id):
    # Path.glob("*.json") matches dotfiles too (unlike shell globbing), so
    # this would otherwise upload our own ID_CACHE_PATH as if it were
    # dashboard data.
    for path in sorted(EXPORT_DIR.glob("*.json")):
        if path.name == ID_CACHE_PATH.name:
            continue
        media = MediaInMemoryUpload(path.read_bytes(), mimetype="application/json")
        _upsert_file(service, cache, folder_id, path.name, media, "application/json")


def _due_for_log_upload(cache, log_name):
    last = cache["files"].get(log_name, {}).get("last_uploaded", 0)
    return time.time() - last >= LOG_UPLOAD_INTERVAL_S


def _upload_due_logs(service, cache, folder_id):
    """Upload/update any referenced log file that's due, and return
    {agent_json_filename: {genid_key: drive_file_id}} isn't needed -- we
    inject links directly by rewriting each agent_*.json's log_drive_link
    field in place before the JSON upload pass runs."""
    for agent_path in sorted(EXPORT_DIR.glob("agent_*.json")):
        agent = json.loads(agent_path.read_text(encoding="utf-8"))
        local_rel = agent.get("log_local_path")
        if not local_rel:
            continue
        log_name = f"log_{agent_path.stem}.log"
        source_path = REPO_ROOT / local_rel
        if not source_path.exists():
            continue
        cached = cache["files"].get(log_name, {})
        if cached.get("id") and not _due_for_log_upload(cache, log_name):
            file_id = cached["id"]
        else:
            media = MediaFileUpload(str(source_path), mimetype="text/plain", resumable=False)
            file_id = _upsert_file(service, cache, folder_id, log_name, media, "text/plain")
            cache["files"][log_name]["last_uploaded"] = time.time()
        agent["log_drive_link"] = f"https://drive.google.com/file/d/{file_id}/view"
        agent_path.write_text(json.dumps(agent, indent=2), encoding="utf-8")


def upload_all():
    if not EXPORT_DIR.exists() or not any(EXPORT_DIR.glob("*.json")):
        print("Nothing to upload -- run export.py first.")
        return
    service = get_drive_service()
    cache = _load_cache()
    is_new_folder = not cache.get("folder_id")
    folder_id = _ensure_folder(service, cache)
    _upload_due_logs(service, cache, folder_id)  # mutates agent_*.json in place first
    _upload_json_files(service, cache, folder_id)  # then uploads the (possibly updated) JSON
    _save_cache(cache)
    print(f"Uploaded {len(list(EXPORT_DIR.glob('*.json')))} JSON file(s) to Drive folder {folder_id}")
    if is_new_folder:
        print(
            f"\nFirst run: created Drive folder '{FOLDER_NAME}'.\n"
            f"  NEXT_PUBLIC_DRIVE_FOLDER_ID={folder_id}\n"
            f"  View it: https://drive.google.com/drive/folders/{folder_id}\n"
            "Set that env var (alongside NEXT_PUBLIC_DRIVE_API_KEY) in dashboard/web/.env.local "
            "and in the Vercel project settings.\n"
        )


if __name__ == "__main__":
    upload_all()
