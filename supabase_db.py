"""
Supabase Database Helper Module for UIDAI Biometric System.
Handles storage of enrolled members, 512-d embeddings, verification logs, and face images.
"""

import os
import io
import time
from datetime import datetime
import numpy as np

# Load environment variables if python-dotenv available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

# Default project credentials (can be overridden via env or streamlit secrets)
DEFAULT_URL = os.environ.get("SUPABASE_URL", "https://xhrcpwjfjubuizqiqsid.supabase.co")
DEFAULT_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_wvxdai9xSINwsX7nNyO1dw_himsF_i2")

_client_instance = None


def get_supabase_client(force_refresh=False):
    """Returns a singleton Supabase client instance."""
    global _client_instance, create_client, Client
    if force_refresh:
        _client_instance = None

    if _client_instance is not None:
        return _client_instance

    if create_client is None:
        try:
            from supabase import create_client as _cc, Client as _cl
            create_client = _cc
            Client = _cl
        except ImportError:
            return None

    # Check streamlit secrets first if available
    url = DEFAULT_URL
    key = DEFAULT_KEY
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "SUPABASE_URL" in st.secrets:
                url = st.secrets["SUPABASE_URL"]
            if "SUPABASE_KEY" in st.secrets:
                key = st.secrets["SUPABASE_KEY"]
    except Exception:
        pass

    if not url or not key:
        return None

    try:
        _client_instance = create_client(url, key)
        return _client_instance
    except Exception as e:
        print(f"[Supabase] Connection error: {e}")
        return None


def is_connected():
    """Verify active connection to Supabase."""
    client = get_supabase_client()
    if client is None:
        return False
    try:
        client.table("enrolled_members").select("aadhaar_id").limit(1).execute()
        return True
    except Exception as e:
        print(f"[Supabase] Connection check failed: {e}")
        return False


def fetch_all_enrolled():
    """
    Fetch all enrolled members and their recent verification logs from Supabase.
    Returns: dict matching st.session_state.enrolled_db schema:
      {
         aadhaar_id: {
             "name": str,
             "embedding": np.ndarray,
             "history": list of dicts,
             "image_url": str | None
         }
      }
    """
    client = get_supabase_client()
    if client is None:
        return {}

    try:
        members_res = client.table("enrolled_members").select("*").execute()
        members = members_res.data or []

        # Fetch verification logs ordered by timestamp desc
        logs_res = client.table("verification_logs").select("*").order("timestamp", desc=True).execute()
        all_logs = logs_res.data or []

        logs_by_aadhaar = {}
        for log in all_logs:
            aid = log.get("aadhaar_id")
            if aid not in logs_by_aadhaar:
                logs_by_aadhaar[aid] = []
            
            # Format timestamp nicely
            raw_ts = log.get("timestamp", "")
            clean_ts = raw_ts.replace("T", " ")[:19] if raw_ts else datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            logs_by_aadhaar[aid].append({
                "timestamp": clean_ts,
                "match_rate": log.get("match_rate", "N/A"),
                "status": log.get("status", "Unknown")
            })

        db = {}
        for m in members:
            aid = str(m["aadhaar_id"])
            emb = m.get("embedding", [])
            db[aid] = {
                "name": m.get("name", "Unknown"),
                "embedding": np.array(emb, dtype=np.float32) if emb else np.zeros(512),
                "history": logs_by_aadhaar.get(aid, []),
                "image_url": m.get("image_url")
            }
        return db

    except Exception as e:
        print(f"[Supabase] Error fetching enrolled members: {e}")
        return {}


def save_enrolled_member(aadhaar_id, name, embedding, image_bytes=None):
    """
    Save or update an enrolled member in Supabase.
    Optionally uploads face image to 'face-images' bucket if accessible.
    """
    client = get_supabase_client()
    if client is None:
        return False

    image_url = None
    if image_bytes is not None:
        try:
            filename = f"{aadhaar_id}_{int(time.time())}.jpg"
            client.storage.from_("face-images").upload(
                file=image_bytes,
                path=filename,
                file_options={"content-type": "image/jpeg", "upsert": "true"}
            )
            image_url = client.storage.from_("face-images").get_public_url(filename)
        except Exception:
            # Storage upload optional
            image_url = None

    emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
    payload = {
        "aadhaar_id": str(aadhaar_id),
        "name": str(name),
        "embedding": emb_list,
    }
    if image_url:
        payload["image_url"] = image_url

    try:
        client.table("enrolled_members").upsert(payload).execute()
        return True
    except Exception as e:
        print(f"[Supabase] Error upserting member {aadhaar_id}: {e}")
        return False


def log_verification(aadhaar_id, match_rate, status, similarity_score=0.0):
    """Save an identity verification attempt to verification_logs table."""
    client = get_supabase_client()
    if client is None:
        return False

    try:
        payload = {
            "aadhaar_id": str(aadhaar_id),
            "match_rate": str(match_rate),
            "status": str(status),
            "similarity_score": float(similarity_score)
        }
        client.table("verification_logs").insert(payload).execute()
        return True
    except Exception as e:
        print(f"[Supabase] Error logging verification for {aadhaar_id}: {e}")
        return False


def clear_all_enrolled():
    """Clear all records from Supabase enrolled_members (cascades to logs)."""
    client = get_supabase_client()
    if client is None:
        return False

    try:
        # Delete all records where aadhaar_id is not empty
        client.table("enrolled_members").delete().neq("aadhaar_id", "").execute()
        return True
    except Exception as e:
        print(f"[Supabase] Error clearing database: {e}")
        return False


def sync_local_to_supabase(local_db):
    """
    Push all local records in local_db to Supabase enrolled_members.
    Returns: tuple (success_count, fail_count)
    """
    success = 0
    fail = 0
    if not is_connected():
        return 0, len(local_db)

    for aid, data in local_db.items():
        name = data.get("name", "Unknown")
        emb = data.get("embedding")
        if emb is not None:
            ok = save_enrolled_member(aid, name, emb)
            if ok:
                success += 1
            else:
                fail += 1
    return success, fail

