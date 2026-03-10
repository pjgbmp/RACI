# app.py (RUTA 1 - Supabase + user_profiles) — ALINEADO con el db.py que pegaste
# -----------------------------------------------------------------------------
# ✅ Cambios clave:
# 1) Imports: SOLO funciones que existen en tu db.py.
# 2) Login: guarda st.session_state["sb_session"] y ["sb_user"] (requerido por sb_user()).
# 3) Storage: usa supabase-py directo (get_conn() ya ES el cliente; no existe .client).
# 4) Todo lo que NO está en db.py (RACI, deps, comments, history, approvals, notifications,
#    templates, attachments, outbound) se implementa aquí con supa_table(...).
# -----------------------------------------------------------------------------

import io
import json
import os
import uuid as uuidlib
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from auth_supabase import sign_in, sign_out
from ics_utils import task_to_ics_event, tasks_to_ics_calendar

# ✅ db.py (tu versión completa pegada)
from db import (
    get_conn, now_iso,
    # user_profiles
    upsert_user_profile, get_user_profile, get_user_profile_by_username, fetch_profiles,
    # grupos / permisos
    list_my_groups, create_group, get_group_by_join_code, add_group_member,
    list_group_members, user_role_in_group, has_permission,
    seed_group_permissions, transfer_group_leadership,
    list_group_role_permissions, set_group_role_permission,
    # proyectos
    list_projects, create_project,
    list_project_members, ensure_project_member, remove_project_member,
    # tareas
    list_tasks_by_project, list_tasks_for_board, get_task, create_task,
    update_task_due_date, update_task_status,
)

st.set_page_config(page_title="Workflow Organización de equipo", layout="wide", page_icon="📋")

# ─── CSS Global ───────────────────────────────────────────────────────────────
_ACTIVE_PAGE = None  # se actualiza abajo tras resolver `menu`

_CSS = """
<style>
/* ── Fuentes y base ──────────────────────────────── */
[data-testid="stAppViewContainer"] { background-color: #F7F9FC; }
[data-testid="stSidebar"] { background-color: #1E2435 !important; }
[data-testid="stSidebar"] * { color: #D6DCF0 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #D6DCF0 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,179,237,0.18) !important;
    border-color: #63B3ED !important;
}
[data-testid="stSidebar"] .stSuccess { background: rgba(72,187,120,0.15) !important; }

/* ── Tarjetas de tarea ─────────────────────────── */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background: #FFFFFF;
    border-radius: 10px;
}

/* ── Navegación superior — botón activo ──────────── */
div[data-active-page="true"] > button {
    background: linear-gradient(135deg,#4C8BF5,#3B5BDB) !important;
    color: white !important;
    border-color: #3B5BDB !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(76,139,245,0.35) !important;
}

/* Botón nav normal */
.stButton > button {
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(0,0,0,0.12);
}

/* ── Métricas ─────────────────────────────────── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E8EDF5;
    border-radius: 12px;
    padding: 14px 18px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* ── Dataframes ───────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid #E8EDF5;
}

/* ── Alertas ──────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Divider ──────────────────────────────────── */
hr { border-color: #E8EDF5 !important; }

/* ── Forms ────────────────────────────────────── */
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E8EDF5;
    border-radius: 12px;
    padding: 18px !important;
}

/* ── Expander ─────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E8EDF5 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
}

/* ── Status badges (usados con st.markdown) ──── */
.badge-todo       { background:#E8EDF5; color:#4A5568; padding:3px 9px; border-radius:20px; font-size:0.82em; font-weight:600; }
.badge-doing      { background:#EBF8FF; color:#2B6CB0; padding:3px 9px; border-radius:20px; font-size:0.82em; font-weight:600; }
.badge-blocked    { background:#FFF5F5; color:#C53030; padding:3px 9px; border-radius:20px; font-size:0.82em; font-weight:600; }
.badge-approval   { background:#FFFBEB; color:#B7791F; padding:3px 9px; border-radius:20px; font-size:0.82em; font-weight:600; }
.badge-done       { background:#F0FFF4; color:#276749; padding:3px 9px; border-radius:20px; font-size:0.82em; font-weight:600; }
.badge-urgente    { background:#FFF5F5; color:#C53030; padding:3px 9px; border-radius:20px; font-size:0.82em; font-weight:600; }
.badge-alta       { background:#FFFBEB; color:#B7791F; padding:3px 9px; border-radius:20px; font-size:0.82em; font-weight:600; }
.badge-media      { background:#EBF8FF; color:#2B6CB0; padding:3px 9px; border-radius:20px; font-size:0.82em; font-weight:600; }
.badge-baja       { background:#F0FFF4; color:#276749; padding:3px 9px; border-radius:20px; font-size:0.82em; font-weight:600; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# =============================================================================
# Helpers generales
# =============================================================================
def safe_selectbox_dict(label, dict_options, format_func, key=None, index=0):
    if not dict_options:
        st.warning(f"No hay opciones disponibles para: {label}")
        return None
    return st.selectbox(
        label,
        options=dict_options,
        format_func=format_func,
        key=key,
        index=min(index, len(dict_options) - 1),
    )


def fmt_user(u: dict) -> str:
    uname = u.get("username") or ""
    return f"{u.get('full_name','(sin nombre)')} (@{uname})"


def _to_dt(x):
    return pd.to_datetime(x, errors="coerce", utc=True)



def week_bounds(d: date):
    start = d - timedelta(days=d.weekday())  # lunes
    end = start + timedelta(days=6)          # domingo
    return start, end


def logged_in() -> bool:
    return st.session_state.get("user_id") is not None


def require_login():
    if not logged_in():
        st.stop()


def set_menu(page: str):
    st.session_state["menu"] = page
    st.rerun()


# =============================================================================
# Session defaults
# =============================================================================
for k, v in {
    "user_id": None,
    "username": "",
    "full_name": "",
    "is_global_admin": False,
    "active_group_id": None,
    "selected_task_id": None,
    "menu": "Resumen",
    # ✅ claves que db.py requiere para sb_user()
    "sb_user": None,
    "sb_session": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =============================================================================
# Supabase helpers (low-level)
# =============================================================================
def supa_table(name: str):
    # get_conn() retorna el cliente supabase-py actuando como user (por db.py)
    return get_conn().table(name)


def chunked(lst, n=200):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas problemáticas para que openpyxl pueda serializarlas:
    - Timestamps tz-aware  → string ISO 8601
    - Nullable Int64/Float64 → int/float nativos (NaN para nulos)
    - pd.NA → None
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        # Timestamps con timezone → string
        if pd.api.types.is_datetime64_any_dtype(s):
            try:
                if getattr(s.dt, "tz", None) is not None:
                    df[col] = s.dt.strftime("%Y-%m-%d %H:%M:%S").where(s.notna(), other=None)
                else:
                    df[col] = s.dt.strftime("%Y-%m-%d %H:%M:%S").where(s.notna(), other=None)
            except Exception:
                df[col] = s.astype(str).where(s.notna(), other=None)
        # Nullable Int64 / Float64 → tipos nativos
        elif hasattr(s, "dtype") and str(s.dtype) in ("Int64", "Int32", "Float64", "Float32"):
            df[col] = s.to_numpy(dtype=float, na_value=float("nan"))
        # Reemplazar pd.NA con None para que openpyxl no falle
        else:
            df[col] = s.where(s.notna(), other=None)
    return df


def safe_execute(query, default=None):
    """Wrapper seguro para llamadas a Supabase. Retorna default si falla."""
    try:
        return getattr(query.execute(), "data", default) or default
    except Exception as e:
        st.warning(f"⚠️ Error al consultar la base de datos: {e}")
        return default if default is not None else []


# =============================================================================
# STORAGE (Adjuntos) — ✅ FIX: sin .client, usando supabase-py directo
# =============================================================================
def storage_bucket_name() -> str:
    try:
        return st.secrets["connections"]["supabase"].get("SUPABASE_STORAGE_BUCKET", "attachments")
    except Exception:
        return "attachments"


def storage_upload_bytes(bucket: str, path: str, data: bytes, content_type: str | None = None):
    client = get_conn()
    opts = {"contentType": content_type} if content_type else None
    return client.storage.from_(bucket).upload(path, data, file_options=opts)


def storage_signed_url(bucket: str, path: str, expires_in_seconds: int = 3600) -> str:
    client = get_conn()
    res = client.storage.from_(bucket).create_signed_url(path, expires_in_seconds)
    return (res or {}).get("signedURL") or (res or {}).get("signedUrl") or ""


# =============================================================================
# (NO estaban en db.py) — Implementaciones directas vía tablas
# =============================================================================

# ---------- RACI ----------
def get_task_roles(task_id: int) -> dict[str, list[dict]]:
    rows = getattr(
        supa_table("task_roles")
        .select("user_id,role")
        .eq("task_id", int(task_id))
        .execute(),
        "data",
        [],
    ) or []

    by_role: dict[str, list[str]] = {"A": [], "R": [], "C": [], "I": []}
    for r in rows:
        role = r.get("role")
        uid = r.get("user_id")
        if role in by_role and uid:
            by_role[role].append(uid)

    # perfiles
    all_uids = sorted({u for xs in by_role.values() for u in xs})
    prof_by = fetch_profiles(all_uids)

    def pack(uid: str) -> dict:
        p = prof_by.get(uid, {})
        return {"id": uid, "user_id": uid, "name": p.get("full_name", "(sin nombre)"), "username": p.get("username", "")}

    out = {k: [pack(uid) for uid in v] for k, v in by_role.items()}
    # A único: si hay múltiples, nos quedamos con el primero (por convención)
    if len(out["A"]) > 1:
        out["A"] = [out["A"][0]]
    return out


def get_accountable_id(task_id: int) -> str | None:
    r = getattr(
        supa_table("task_roles")
        .select("user_id")
        .eq("task_id", int(task_id))
        .eq("role", "A")
        .limit(1)
        .execute(),
        "data",
        [],
    ) or []
    return (r[0].get("user_id") if r else None)


def is_responsible(task_id: int, user_id: str) -> bool:
    r = getattr(
        supa_table("task_roles")
        .select("task_id")
        .eq("task_id", int(task_id))
        .eq("role", "R")
        .eq("user_id", user_id)
        .limit(1)
        .execute(),
        "data",
        [],
    ) or []
    return bool(r)


def upsert_task_roles(
    task_id: int,
    user_id: str,
    A_id: str,
    R_ids: list[str],
    C_ids: list[str],
    I_ids: list[str],
) -> None:
    # Validaciones
    if not A_id:
        raise ValueError("Falta Dueño (A).")
    if not R_ids:
        raise ValueError("Debe haber al menos 1 Ejecutor (R).")

    # delete all and insert fresh (simple, consistente)
    supa_table("task_roles").delete().eq("task_id", int(task_id)).execute()

    rows = [{"task_id": int(task_id), "user_id": A_id, "role": "A"}]
    rows += [{"task_id": int(task_id), "user_id": x, "role": "R"} for x in sorted(set(R_ids))]
    rows += [{"task_id": int(task_id), "user_id": x, "role": "C"} for x in sorted(set(C_ids or []))]
    rows += [{"task_id": int(task_id), "user_id": x, "role": "I"} for x in sorted(set(I_ids or []))]

    supa_table("task_roles").insert(rows).execute()


# ---------- Dependencias ----------
def list_task_dependencies_open(task_id: int) -> list[dict]:
    # Espera tabla: task_dependencies(task_id, depends_on_task_id)
    links = getattr(
        supa_table("task_dependencies")
        .select("depends_on_task_id")
        .eq("task_id", int(task_id))
        .execute(),
        "data",
        [],
    ) or []
    dep_ids = [int(x["depends_on_task_id"]) for x in links if x.get("depends_on_task_id") is not None]
    if not dep_ids:
        return []

    deps = getattr(
        supa_table("tasks")
        .select("id,title,status,priority,due_date")
        .in_("id", dep_ids)
        .execute(),
        "data",
        [],
    ) or []

    return [d for d in deps if d.get("status") != "done"]


def set_task_dependencies(task_id: int, depends_on_task_ids: list[int]) -> None:
    supa_table("task_dependencies").delete().eq("task_id", int(task_id)).execute()
    if depends_on_task_ids:
        rows = [{"task_id": int(task_id), "depends_on_task_id": int(x)} for x in sorted(set(depends_on_task_ids))]
        supa_table("task_dependencies").insert(rows).execute()


# ---------- Comments ----------
def list_task_comments(task_id: int, limit: int = 50) -> list[dict]:
    rows = getattr(
        supa_table("task_comments")
        .select("id,task_id,user_id,comment,progress_pct,next_step,created_at")
        .eq("task_id", int(task_id))
        .order("id", desc=True)
        .limit(limit)
        .execute(),
        "data",
        [],
    ) or []
    uids = sorted({r["user_id"] for r in rows if r.get("user_id")})
    prof_by = fetch_profiles(uids)
    for r in rows:
        p = prof_by.get(r.get("user_id", ""), {})
        r["full_name"] = p.get("full_name", "(sin nombre)")
    return rows


def add_task_comment(task_id: int, user_id: str, comment: str, progress_pct: int | None, next_step: str | None) -> None:
    supa_table("task_comments").insert(
        {
            "task_id": int(task_id),
            "user_id": user_id,
            "comment": comment,
            "progress_pct": int(progress_pct) if progress_pct is not None else None,
            "next_step": next_step,
            "created_at": now_iso(),
        }
    ).execute()


# ---------- History ----------
def list_task_history(task_id: int, limit: int = 200) -> list[dict]:
    rows = getattr(
        supa_table("task_history")
        .select("id,task_id,user_id,field,old_value,new_value,created_at")
        .eq("task_id", int(task_id))
        .order("id", desc=True)
        .limit(limit)
        .execute(),
        "data",
        [],
    ) or []
    uids = sorted({r["user_id"] for r in rows if r.get("user_id")})
    prof_by = fetch_profiles(uids)
    for r in rows:
        p = prof_by.get(r.get("user_id", ""), {})
        r["user_name"] = p.get("full_name", "(sin nombre)")
    return rows


def log_history(task_id: int, user_id: str, field: str, old_value, new_value) -> None:
    supa_table("task_history").insert(
        {
            "task_id": int(task_id),
            "user_id": user_id,
            "field": field,
            "old_value": None if old_value is None else str(old_value),
            "new_value": None if new_value is None else str(new_value),
            "created_at": now_iso(),
        }
    ).execute()


# ---------- Approvals ----------
def get_pending_approval(task_id: int) -> dict | None:
    rows = getattr(
        supa_table("approval_requests")
        .select("*")
        .eq("task_id", int(task_id))
        .eq("status", "pending")
        .order("id", desc=True)
        .limit(1)
        .execute(),
        "data",
        [],
    ) or []
    return rows[0] if rows else None


def create_approval_request(task_id: int, requested_by: str, accountable_user_id: str, request_note: str | None = None):
    supa_table("approval_requests").insert(
        {
            "task_id": int(task_id),
            "requested_by": requested_by,
            "accountable_user_id": accountable_user_id,
            "status": "pending",
            "request_note": request_note,
            "requested_at": now_iso(),
            "decided_at": None,
            "decision_note": None,
        }
    ).execute()


def decide_approval(approval_id: int, decision: str, decision_note: str | None = None):
    assert decision in ("approved", "rejected")
    supa_table("approval_requests").update(
        {
            "status": decision,
            "decided_at": now_iso(),
            "decision_note": decision_note,
        }
    ).eq("id", int(approval_id)).execute()


def list_approvals_by_group(group_id: int, project_id: int | None = None) -> list[dict]:
    # Trae tasks del grupo (por proyectos) y luego approvals por task_id
    projs = list_projects(group_id)
    if not projs:
        return []
    pids = [int(project_id)] if project_id else [int(p["id"]) for p in projs]

    task_ids: list[int] = []
    for pid in pids:
        rows = list_tasks_by_project(pid, limit=1200)
        task_ids.extend([int(t["id"]) for t in rows])
    task_ids = sorted(set(task_ids))
    if not task_ids:
        return []

    appr: list[dict] = []
    for part in chunked(task_ids, 200):
        rows = getattr(
            supa_table("approval_requests")
            .select("id,task_id,requested_by,accountable_user_id,status,requested_at,decided_at,decision_note,request_note")
            .in_("task_id", part)
            .execute(),
            "data",
            [],
        ) or []
        appr.extend(rows)

    # enriquecer accountable full_name
    uids = sorted({a["accountable_user_id"] for a in appr if a.get("accountable_user_id")})
    prof_by = fetch_profiles(uids)
    for a in appr:
        p = prof_by.get(a.get("accountable_user_id", ""), {})
        a["full_name"] = p.get("full_name", "(sin nombre)")
    return appr


# ---------- Notifications ----------
def add_notification(target_user_id: str, kind: str, message: str, task_id: int | None = None) -> None:
    supa_table("notifications").insert(
        {
            "user_id": target_user_id,
            "kind": kind,
            "message": message,
            "task_id": int(task_id) if task_id is not None else None,
            "created_at": now_iso(),
            "read_at": None,
        }
    ).execute()



def unread_notifications_count(user_id: str) -> int:
    rows = getattr(
        supa_table("notifications")
        .select("id")
        .eq("user_id", user_id)
        .is_("read_at", "null")
        .execute(),
        "data",
        [],
    ) or []
    return len(rows)


def list_notifications(user_id: str, limit: int = 20) -> list[dict]:
    return getattr(
        supa_table("notifications")
        .select("id,kind,message,task_id,created_at,read_at")
        .eq("user_id", user_id)
        .order("id", desc=True)
        .limit(limit)
        .execute(),
        "data",
        [],
    ) or []


def mark_notifications_read(user_id: str) -> None:
    supa_table("notifications").update({"read_at": now_iso()}).eq("user_id", user_id).is_("read_at", "null").execute()


# ---------- Templates ----------
def list_templates(group_id: int) -> list[dict]:
    return getattr(
        supa_table("templates")
        .select("*")
        .eq("group_id", int(group_id))
        .order("id", desc=True)
        .execute(),
        "data",
        [],
    ) or []


def create_template(group_id: int, name: str, description: str | None, created_by: str) -> int:
    row = getattr(
        supa_table("templates")
        .insert(
            {
                "group_id": int(group_id),
                "name": name,
                "description": description,
                "created_by": created_by,
                "created_at": now_iso(),
            }
        )
        .execute(),
        "data",
        None,
    )
    if isinstance(row, list) and row:
        return int(row[0]["id"])
    if isinstance(row, dict) and row.get("id"):
        return int(row["id"])
    raise RuntimeError("No pude crear template (respuesta sin id).")


def list_template_tasks(template_id: int) -> list[dict]:
    return getattr(
        supa_table("template_tasks")
        .select("*")
        .eq("template_id", int(template_id))
        .order("id", desc=False)
        .execute(),
        "data",
        [],
    ) or []


def add_template_task(
    template_id: int,
    title: str,
    description: str | None,
    dod: str,
    priority: str,
    requires_approval: bool,
    days_from_start: int,
    tags_csv: str | None,
    component_name: str | None,
) -> None:
    supa_table("template_tasks").insert(
        {
            "template_id": int(template_id),
            "title": title,
            "description": description,
            "dod": dod,
            "priority": priority,
            "requires_approval": bool(requires_approval),
            "days_from_start": int(days_from_start),
            "tags_csv": tags_csv,
            "component_name": component_name,
        }
    ).execute()


# ---------- Attachments metadata ----------
def add_task_attachment(task_id: int, user_id: str, filename: str, storage_path: str, storage_bucket: str) -> None:
    supa_table("task_attachments").insert(
        {
            "task_id": int(task_id),
            "user_id": user_id,
            "filename": filename,
            "storage_bucket": storage_bucket,
            "storage_path": storage_path,
            "created_at": now_iso(),
        }
    ).execute()


def list_task_attachments(task_id: int, limit: int = 50) -> list[dict]:
    rows = getattr(
        supa_table("task_attachments")
        .select("id,task_id,user_id,filename,storage_bucket,storage_path,created_at")
        .eq("task_id", int(task_id))
        .order("id", desc=True)
        .limit(limit)
        .execute(),
        "data",
        [],
    ) or []
    uids = sorted({r["user_id"] for r in rows if r.get("user_id")})
    prof_by = fetch_profiles(uids)
    for r in rows:
        p = prof_by.get(r.get("user_id", ""), {})
        r["full_name"] = p.get("full_name", "(sin nombre)")
    return rows


# ---------- Outbound queue ----------
def enqueue_outbound(channel: str, payload: str, to_address: str | None = None) -> None:
    supa_table("outbound_queue").insert(
        {
            "channel": channel,
            "payload": payload,
            "to_address": to_address,
            "status": "queued",
            "created_at": now_iso(),
        }
    ).execute()


# =============================================================================
# Notify wrapper (in-app + optional email queue)
# =============================================================================
def notify(
    target_user_id: str,
    kind: str,
    message: str,
    task_id: int | None = None,
    send_email: bool = False,
    email_subject: str | None = None,
):
    # 1) In-app
    add_notification(target_user_id, kind, message, int(task_id) if task_id is not None else None)

    # 2) Email queue (optional)
    if send_email:
        prof = get_user_profile(target_user_id)
        to_email = (prof or {}).get("email")
        if not to_email:
            st.warning("No se pudo encolar email: el usuario destino no tiene email en user_profiles.")
            return

        payload_obj = {
            "to": to_email,
            "subject": email_subject or "Notificación del Workflow Team Manager",
            "message": message,
            "task_id": task_id,
            "kind": kind,
            "user_id": target_user_id,
            "created_at": now_iso(),
        }
        enqueue_outbound("email", payload=json.dumps(payload_obj, ensure_ascii=False), to_address=to_email)


# =============================================================================
# KPIs / Resumen
# =============================================================================
@st.cache_data(ttl=60)
def _project_map(group_id: int) -> dict[int, dict]:
    """Cachea el mapa de proyectos por 60 s para evitar llamadas repetidas."""
    projs = list_projects(group_id)
    return {int(p["id"]): p for p in projs}


@st.cache_data(ttl=60)
def _cached_group_members(group_id: int) -> list[dict]:
    """Versión cacheada de list_group_members."""
    return list_group_members(group_id)


@st.cache_data(ttl=60)
def _cached_fetch_profiles(uids_tuple: tuple) -> dict:
    """Versión cacheada de fetch_profiles (recibe tuple para ser hasheable)."""
    return fetch_profiles(list(uids_tuple))


def _fetch_group_task_ids(group_id: int, project_id: int | None = None, limit_per_project: int = 600) -> list[int]:
    pmap = _project_map(group_id)
    if not pmap:
        return []
    pids = [int(project_id)] if project_id is not None else list(pmap.keys())

    task_ids: list[int] = []
    for pid in pids:
        tasks = list_tasks_by_project(pid, limit=limit_per_project)
        task_ids.extend([int(t["id"]) for t in tasks])
    return sorted(set(task_ids))


def _bulk_task_roles(task_ids: list[int]) -> list[dict]:
    if not task_ids:
        return []
    out: list[dict] = []
    for part in chunked(task_ids, 200):
        res = supa_table("task_roles").select("task_id,user_id,role").in_("task_id", part).execute()
        out.extend(getattr(res, "data", []) or [])
    return out


def _bulk_tasks(task_ids: list[int]) -> list[dict]:
    if not task_ids:
        return []
    out: list[dict] = []
    for part in chunked(task_ids, 200):
        res = supa_table("tasks").select("*").in_("id", part).execute()
        out.extend(getattr(res, "data", []) or [])
    return out


def compute_kpis_per_person(group_id: int, project_id: int | None = None) -> pd.DataFrame:
    members = list_group_members(group_id)
    if not members:
        return pd.DataFrame()

    task_ids = _fetch_group_task_ids(group_id, project_id=project_id)
    if not task_ids:
        return pd.DataFrame([{
            "user_id": m["user_id"],
            "full_name": m["full_name"],
            "todo": 0, "doing": 0, "blocked": 0, "awaiting_approval": 0, "done": 0,
            "wip_doing": 0, "overdue_open": 0,
            "avg_aging_days_open": None,
            "avg_lead_time_days_done_60d": None,
            "on_time_rate_done": None
        } for m in members])

    roles = _bulk_task_roles(task_ids)
    roles_r = [r for r in roles if r.get("role") == "R"]
    tasks = _bulk_tasks(task_ids)
    t_by_id = {int(t["id"]): t for t in tasks}

    rows = []
    for r in roles_r:
        tid = int(r["task_id"])
        t = t_by_id.get(tid)
        if not t:
            continue
        rows.append({
            "user_id": r["user_id"],
            "task_id": tid,
            "status": t.get("status"),
            "priority": t.get("priority"),
            "created_at": t.get("created_at"),
            "completed_at": t.get("completed_at"),
            "due_date": t.get("due_date"),
            "target_date": t.get("target_date"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    prof_by = fetch_profiles(df["user_id"].unique().tolist())
    df["full_name"] = df["user_id"].map(lambda x: prof_by.get(x, {}).get("full_name", "(sin nombre)"))

    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["completed_at_dt"] = pd.to_datetime(df["completed_at"], errors="coerce", utc=True)
    df["due_dt"] = pd.to_datetime(df["due_date"], errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")

    status_counts = (df.groupby(["user_id", "full_name", "status"]).size().unstack(fill_value=0).reset_index())
    for col in ["todo", "doing", "blocked", "awaiting_approval", "done"]:
        if col not in status_counts.columns:
            status_counts[col] = 0
    status_counts["wip_doing"] = status_counts["doing"]

    open_mask = df["status"].isin(["todo", "doing", "blocked", "awaiting_approval"])
    overdue_mask = open_mask & df["due_dt"].notna() & (df["due_dt"].dt.date < now.date())
    overdue = (df[overdue_mask].groupby(["user_id", "full_name"]).size().reset_index(name="overdue_open"))

    open_df = df[open_mask & df["created_at_dt"].notna()].copy()
    open_df["aging_days"] = (now - open_df["created_at_dt"]).dt.days
    aging = (open_df.groupby(["user_id", "full_name"])["aging_days"].mean().reset_index(name="avg_aging_days_open"))

    done_df = df[(df["status"] == "done") & df["created_at_dt"].notna() & df["completed_at_dt"].notna()].copy()
    done_df["lead_days"] = (done_df["completed_at_dt"] - done_df["created_at_dt"]).dt.days
    cutoff = now - pd.Timedelta(days=60)
    done_60 = done_df[done_df["completed_at_dt"] >= cutoff]
    lead = (done_60.groupby(["user_id", "full_name"])["lead_days"].mean().reset_index(name="avg_lead_time_days_done_60d"))

    done_due = done_df[done_df["due_dt"].notna()].copy()
    if not done_due.empty:
        done_due["on_time"] = done_due["completed_at_dt"].dt.date <= done_due["due_dt"].dt.date
        ontime = (done_due.groupby(["user_id", "full_name"])["on_time"].mean().reset_index(name="on_time_rate_done"))
    else:
        ontime = pd.DataFrame(columns=["user_id", "full_name", "on_time_rate_done"])

    base_users = pd.DataFrame([{"user_id": m["user_id"], "full_name": m["full_name"]} for m in members])
    out = base_users.merge(status_counts, on=["user_id", "full_name"], how="left")
    out = out.merge(overdue, on=["user_id", "full_name"], how="left")
    out = out.merge(aging, on=["user_id", "full_name"], how="left")
    out = out.merge(lead, on=["user_id", "full_name"], how="left")
    out = out.merge(ontime, on=["user_id", "full_name"], how="left")

    for col in ["todo", "doing", "blocked", "awaiting_approval", "done", "wip_doing", "overdue_open"]:
        out[col] = out[col].fillna(0).astype(int)

    if "avg_aging_days_open" in out.columns:
        out["avg_aging_days_open"] = out["avg_aging_days_open"].round(1)
    if "avg_lead_time_days_done_60d" in out.columns:
        out["avg_lead_time_days_done_60d"] = out["avg_lead_time_days_done_60d"].round(1)
    if "on_time_rate_done" in out.columns:
        out["on_time_rate_done"] = pd.to_numeric(out["on_time_rate_done"], errors="coerce")
        out["on_time_rate_done"] = (out["on_time_rate_done"] * 100).round(0).astype("Int64")

    # Renombrar columnas internas a español para la UI
    rename_map = {
        "full_name": "nombre",
        "todo": "por_hacer",
        "doing": "en_curso",
        "blocked": "bloqueadas",
        "awaiting_approval": "en_aprobacion",
        "done": "completadas",
        "wip_doing": "wip_en_curso",
        "overdue_open": "vencidas_abiertas",
        "avg_aging_days_open": "antiguedad_prom_dias",
        "avg_lead_time_days_done_60d": "tiempo_cierre_prom_60d",
        "on_time_rate_done": "tasa_cumplimiento_pct",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    return out


def kpi_badges_row(df_kpi: pd.DataFrame):
    if df_kpi.empty:
        return
    total_overdue = int(df_kpi.get("vencidas_abiertas", pd.Series([0])).sum())
    total_wip = int(df_kpi.get("wip_en_curso", pd.Series([0])).sum())
    avg_aging = df_kpi.get("antiguedad_prom_dias", pd.Series(dtype=float)).dropna().mean()
    ontime = df_kpi.get("tasa_cumplimiento_pct", pd.Series(dtype=float)).dropna().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Tareas vencidas abiertas", total_overdue)
    c2.metric("🔵 En curso (WIP)", total_wip)
    c3.metric("📅 Antigüedad promedio (días)", round(float(avg_aging), 1) if pd.notna(avg_aging) else "—")
    c4.metric("✅ Cumplimiento a tiempo (%)", int(ontime) if pd.notna(ontime) else "—")


def approval_cycle_hours(task_id: int) -> float | None:
    r = getattr(
        supa_table("approval_requests")
        .select("requested_at,decided_at,status")
        .eq("task_id", int(task_id))
        .order("id", desc=True)
        .limit(1)
        .execute(),
        "data",
        None
    )
    if not r:
        return None
    row = r[0]
    req = _to_dt(row.get("requested_at"))
    if pd.isna(req):
        return None
    if row.get("status") == "pending":
        end = pd.Timestamp.now()
    else:
        end = _to_dt(row.get("decided_at"))
        if pd.isna(end):
            end = pd.Timestamp.now()
    return round(float((end - req).total_seconds() / 3600), 2)


def approvals_kpi_by_accountable(group_id: int, project_id: int | None = None) -> pd.DataFrame:
    appr = list_approvals_by_group(group_id, project_id=project_id)
    df = pd.DataFrame(appr)
    if df.empty:
        return pd.DataFrame(columns=["a_user_id", "full_name", "pending_count", "avg_cycle_hours_pending", "avg_cycle_hours_approved_60d"])

    df = df.rename(columns={"accountable_user_id": "a_user_id"})
    # utc=True garantiza tz-aware en TODAS las filas, evitando TypeError en restas
    df["requested_at_dt"] = pd.to_datetime(df["requested_at"], errors="coerce", utc=True)
    df["decided_at_dt"]   = pd.to_datetime(df["decided_at"],   errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC")

    pending = df["status"] == "pending"
    decided = ~pending

    # Pendientes: now - requested_at
    req_pend = df.loc[pending, "requested_at_dt"]
    valid_pend = req_pend.notna()
    df.loc[pending & df.index.isin(req_pend[valid_pend].index), "cycle_hours"] = (
        (now - req_pend[valid_pend]).dt.total_seconds() / 3600
    )

    # Decididas: decided_at - requested_at (solo filas donde ambas son válidas)
    req_dec = df.loc[decided, "requested_at_dt"]
    dec_dec = df.loc[decided, "decided_at_dt"]
    both_valid = decided & df["requested_at_dt"].notna() & df["decided_at_dt"].notna()
    if both_valid.any():
        df.loc[both_valid, "cycle_hours"] = (
            (df.loc[both_valid, "decided_at_dt"] - df.loc[both_valid, "requested_at_dt"])
            .dt.total_seconds() / 3600
        )

    pend = (df[pending].groupby(["a_user_id", "full_name"])
            .agg(pendientes_count=("task_id", "count"),
                 horas_ciclo_pendientes=("cycle_hours", "mean"))
            .reset_index())

    cutoff = now - pd.Timedelta(days=60)
    dec60 = df[decided & (df["decided_at_dt"] >= cutoff)]
    dec = (dec60.groupby(["a_user_id", "full_name"])
           .agg(horas_ciclo_aprobadas_60d=("cycle_hours", "mean"))
           .reset_index())

    out = pd.merge(pend, dec, on=["a_user_id", "full_name"], how="outer")
    out = out.rename(columns={"full_name": "nombre"})
    out["pendientes_count"] = out["pendientes_count"].fillna(0).astype(int)
    for col in ["horas_ciclo_pendientes", "horas_ciclo_aprobadas_60d"]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce").round(2).astype("Float64")
    return out


# =============================================================================
# Auth (SOLO LOGIN) — ✅ FIX: guarda sb_session y sb_user
# =============================================================================
st.sidebar.title("Acceso")

if not logged_in():
    st.sidebar.markdown(
        '<div style="text-align:center;padding:18px 0 10px;">'
        '<div style="font-size:2rem">📋</div>'
        '<div style="font-size:1.1rem;font-weight:700;color:#D6DCF0">Workflow Manager</div>'
        '<div style="font-size:0.78rem;color:#8A9BC0;margin-top:4px">Inicia sesión para continuar</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    email = st.sidebar.text_input("📧 Email")
    p = st.sidebar.text_input("🔐 Contraseña", type="password")

    if st.sidebar.button("Entrar →", use_container_width=True, type="primary"):
        try:
            res = sign_in(email.strip(), p)

            user = getattr(res, "user", None)
            sess = getattr(res, "session", None)

            if not user or not sess:
                st.sidebar.error("No se pudo iniciar sesión (respuesta sin user/session).")
                st.stop()

            # ✅ requerido por db.py (sb_user())
            st.session_state["sb_user"] = user
            st.session_state["sb_session"] = sess

            prof = get_user_profile(user.id)

            if not prof:
                upsert_user_profile(
                    user.id,
                    full_name=email.strip(),
                    username=None,
                    email=email.strip(),
                    is_active=True,
                )
                prof = get_user_profile(user.id) or {}

            st.session_state["user_id"] = user.id
            st.session_state["username"] = prof.get("username") or ""
            st.session_state["full_name"] = prof.get("full_name") or ""
            st.session_state["is_global_admin"] = bool(prof.get("is_global_admin", False))

            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Login falló: {e}")

    st.stop()

# Logged in — avatar + nombre
_fn = st.session_state.get("full_name") or "Usuario"
_un = st.session_state.get("username") or ""
_initials = "".join(w[0].upper() for w in _fn.split()[:2]) or "U"
st.sidebar.markdown(
    f'<div style="display:flex;align-items:center;gap:10px;padding:10px 0 14px;">'
    f'<div style="width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,#4C8BF5,#3B5BDB);'
    f'display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1rem">{_initials}</div>'
    f'<div><div style="font-weight:600;font-size:0.95rem">{_fn}</div>'
    f'<div style="font-size:0.78rem;opacity:0.65">@{_un}</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)
if st.sidebar.button("🚪 Cerrar sesión", use_container_width=True):
    sign_out()
    for k in [
        "user_id", "username", "full_name", "active_group_id", "selected_task_id", "is_global_admin",
        "sb_user", "sb_session"
    ]:
        if k in ["user_id", "active_group_id", "selected_task_id", "sb_user", "sb_session"]:
            st.session_state[k] = None
        elif k == "is_global_admin":
            st.session_state[k] = False
        else:
            st.session_state[k] = ""
    st.session_state["menu"] = "Resumen"
    st.rerun()


# =============================================================================
# Grupo: selección / creación / join
# =============================================================================
my_groups = list_my_groups(st.session_state["user_id"])

st.sidebar.subheader("Equipo / Grupo")

if my_groups:
    names = [f"[{g['id']}] {g['name']} ({g.get('my_role')})" for g in my_groups]
    idx = 0
    if st.session_state["active_group_id"]:
        for i, g in enumerate(my_groups):
            if int(g["id"]) == int(st.session_state["active_group_id"]):
                idx = i
                break
    pick = st.sidebar.selectbox("Grupo activo", options=names, index=idx)
    active = my_groups[names.index(pick)]
    st.session_state["active_group_id"] = int(active["id"])
else:
    st.sidebar.info("No perteneces a ningún grupo aún.")

with st.sidebar.expander("➕ Crear grupo"):
    gname = st.text_input("Nombre del grupo", key="gname")
    if st.button("Crear grupo"):
        if not gname.strip():
            st.error("Pon un nombre.")
        else:
            join_code = uuidlib.uuid4().hex[:8].upper()
            gid = create_group(gname.strip(), st.session_state["user_id"], join_code)
            add_group_member(gid, st.session_state["user_id"], role="leader")
            seed_group_permissions(gid)
            st.success(f"Grupo creado. Código de acceso: {join_code}")
            st.session_state["active_group_id"] = int(gid)
            st.rerun()

with st.sidebar.expander("🔑 Unirme a un grupo"):
    code = st.text_input("Código de acceso (join code)", key="joincode")
    if st.button("Unirme"):
        g = get_group_by_join_code(code.strip().upper())
        if not g:
            st.error("Código inválido.")
        else:
            add_group_member(int(g["id"]), st.session_state["user_id"], role="member")
            seed_group_permissions(int(g["id"]))
            st.success("Listo. Ya perteneces al grupo.")
            st.session_state["active_group_id"] = int(g["id"])
            st.rerun()

if not st.session_state["active_group_id"]:
    st.title("Workflow Team Manager")
    st.info("Crea o únete a un grupo para empezar.")
    st.stop()

GROUP_ID = int(st.session_state["active_group_id"])


# =============================================================================
# Header + navegación
# =============================================================================
unread = unread_notifications_count(st.session_state["user_id"])
_notif_badge = f'<span style="background:#E53935;color:white;border-radius:20px;padding:2px 8px;font-size:0.78em;font-weight:700;margin-left:6px">{unread}</span>' if unread > 0 else ""
st.markdown(
    f'<div style="display:flex;align-items:center;justify-content:space-between;'
    f'background:linear-gradient(135deg,#1E2435 0%,#2D3A5F 100%);'
    f'padding:16px 24px;border-radius:14px;margin-bottom:12px;">'
    f'<span style="color:white;font-size:1.6rem;font-weight:700">📋 Workflow · Organización de Equipo</span>'
    f'<span style="color:#A0B0D0;font-size:0.9rem">🔔 Notificaciones{_notif_badge}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

PAGES = ["Resumen", "Proyectos", "Tablero", "Tarea (detalle)", "Plantillas", "Export", "Gobernanza", "Reportes", "Ayuda"]
if st.session_state.get("is_global_admin"):
    PAGES = ["Admin"] + PAGES

if st.session_state.get("menu") == "Admin" and not st.session_state.get("is_global_admin"):
    st.session_state["menu"] = "Resumen"

# Inyectar CSS para el botón activo (el botón cuyo label == menu actual)
_active = st.session_state.get("menu", "Resumen")
_active_css = "".join([
    f"""<style>
    button[kind="secondary"][data-testid$="topnav_{pg.replace(' ','_').replace('(','').replace(')','')}"]:not([disabled]),
    div.stButton > button[aria-label="{pg}"]
    {{ background: linear-gradient(135deg,#4C8BF5,#3B5BDB) !important;
       color: white !important; font-weight: 700 !important;
       box-shadow: 0 2px 8px rgba(76,139,245,0.4) !important; }}
    </style>"""
    for pg in PAGES if pg == _active
])
st.markdown(_active_css, unsafe_allow_html=True)

ICONS = {
    "Resumen": "📊", "Proyectos": "📁", "Tablero": "📌",
    "Tarea (detalle)": "🔍", "Plantillas": "📋", "Export": "📤",
    "Gobernanza": "⚖️", "Reportes": "📈", "Ayuda": "❓", "Admin": "🛡️",
}
nav_cols = st.columns(len(PAGES))
for i, page in enumerate(PAGES):
    with nav_cols[i]:
        icon = ICONS.get(page, "")
        label = f"{icon} {page}" if icon else page
        is_active = (page == _active)
        btn_type = "primary" if is_active else "secondary"
        if st.button(label, key=f"topnav_{page}", use_container_width=True, type=btn_type):
            set_menu(page)

sidebar_idx = PAGES.index(st.session_state["menu"]) if st.session_state["menu"] in PAGES else 0
menu = st.sidebar.radio("Menú", PAGES, index=sidebar_idx)
if menu != st.session_state["menu"]:
    st.session_state["menu"] = menu


# =============================================================================
# Funciones soporte (permisos + WIP)
# =============================================================================
def can_edit_due_date(group_id: int, task_id: int, user_id: str) -> bool:
    if not has_permission(group_id, user_id, "task_change_due_date"):
        return False
    role = user_role_in_group(group_id, user_id)
    if role == "leader":
        return True
    return get_accountable_id(task_id) == user_id


def can_change_status(group_id: int, task_id: int, user_id: str) -> bool:
    if not has_permission(group_id, user_id, "task_change_status"):
        return False
    role = user_role_in_group(group_id, user_id)
    if role == "leader":
        return True
    return is_responsible(task_id, user_id)


def wip_limit_ok(project_id: int, user_id: str) -> tuple[bool, str]:
    proj = supa_table("projects").select("wip_limit_doing").eq("id", int(project_id)).limit(1).execute()
    row = getattr(proj, "data", None)
    limit = int(row[0]["wip_limit_doing"]) if row else 3

    doing = supa_table("tasks").select("id").eq("project_id", int(project_id)).eq("status", "doing").execute()
    doing_ids = [int(x["id"]) for x in (getattr(doing, "data", []) or [])]
    if not doing_ids:
        return True, ""

    cnt = 0
    for part in chunked(doing_ids, 200):
        rr = supa_table("task_roles").select("task_id").in_("task_id", part).eq("role", "R").eq("user_id", user_id).execute()
        cnt += len(getattr(rr, "data", []) or [])

    if cnt >= limit:
        return False, f"WIP excedido: ya tienes {cnt} tareas en DOING (límite {limit})."
    return True, ""


def task_r_names(task_id: int) -> str:
    raw = supa_table("task_roles").select("user_id,role").eq("task_id", int(task_id)).eq("role", "R").execute()
    ids = [x["user_id"] for x in (getattr(raw, "data", []) or [])]
    if not ids:
        return "—"
    prof_by = fetch_profiles(ids)
    names = [prof_by.get(uid, {}).get("full_name", "(sin nombre)") for uid in ids]
    return ", ".join(sorted(names))


def task_a_name(task_id: int) -> str:
    a = get_accountable_id(task_id)
    if not a:
        return "—"
    p = get_user_profile(a)
    return (p.get("full_name") if p else "—") or "—"


# ── Versiones BULK para build_weekly_report ───────────────────────────────────

def _bulk_raci_names(task_ids: list[int]) -> tuple[dict[int, str], dict[int, str]]:
    """
    Retorna (r_names, a_names) como dicts {task_id: str} en UNA sola consulta.
    Reemplaza los bucles de task_r_names / task_a_name por tarea.
    """
    if not task_ids:
        return {}, {}

    all_roles: list[dict] = []
    for part in chunked(task_ids, 200):
        rows = safe_execute(
            supa_table("task_roles").select("task_id,user_id,role").in_("task_id", part),
            default=[]
        )
        all_roles.extend(rows)

    all_uids = list({r["user_id"] for r in all_roles if r.get("user_id")})
    prof_by = fetch_profiles(all_uids) if all_uids else {}

    r_names: dict[int, list[str]] = {}
    a_names: dict[int, str] = {}

    for row in all_roles:
        tid = int(row["task_id"])
        uid = row.get("user_id", "")
        name = prof_by.get(uid, {}).get("full_name", "(sin nombre)")
        if row["role"] == "R":
            r_names.setdefault(tid, []).append(name)
        elif row["role"] == "A":
            if tid not in a_names:
                a_names[tid] = name

    return (
        {tid: ", ".join(sorted(ns)) for tid, ns in r_names.items()},
        a_names,
    )


def _bulk_approval_cycle(task_ids: list[int]) -> dict[int, float | None]:
    """
    Retorna {task_id: cycle_hours} en UNA sola consulta de approval_requests.
    Reemplaza el bucle de approval_cycle_hours por tarea.
    """
    if not task_ids:
        return {}

    all_appr: list[dict] = []
    for part in chunked(task_ids, 200):
        rows = safe_execute(
            supa_table("approval_requests")
            .select("task_id,requested_at,decided_at,status")
            .in_("task_id", part)
            .order("id", desc=True),
            default=[]
        )
        all_appr.extend(rows)

    # Solo la aprobación más reciente por tarea
    seen: set[int] = set()
    result: dict[int, float | None] = {}
    now = pd.Timestamp.now(tz="UTC")

    for row in all_appr:
        tid = int(row["task_id"])
        if tid in seen:
            continue
        seen.add(tid)
        req = _to_dt(row.get("requested_at"))
        if pd.isna(req):
            result[tid] = None
            continue
        if row.get("status") == "pending":
            end = now
        else:
            end = _to_dt(row.get("decided_at"))
            if pd.isna(end):
                end = now
        result[tid] = round(float((end - req).total_seconds() / 3600), 2)

    return result


def build_weekly_report(group_id: int, week_start_d: date, week_end_d: date, project_id: int | None = None):
    pmap = _project_map(group_id)
    if not pmap:
        return pd.DataFrame(), pd.DataFrame()

    pids = [int(project_id)] if project_id is not None else list(pmap.keys())

    rows_all = []
    for pid in pids:
        tasks = list_tasks_by_project(pid, limit=800)
        for t in tasks:
            t = dict(t)
            t["project_name"] = pmap[pid]["name"]
            rows_all.append(t)

    df = pd.DataFrame(rows_all)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["completed_at_dt"] = pd.to_datetime(df["completed_at"], errors="coerce", utc=True)
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["due_dt"] = pd.to_datetime(df["due_date"], errors="coerce", utc=True)

    ws_ts = pd.Timestamp(week_start_d, tz="UTC")
    we_ts = pd.Timestamp(week_end_d, tz="UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)

    done = df[
        (df["status"] == "done")
        & df["completed_at_dt"].notna()
        & (df["completed_at_dt"] >= ws_ts)
        & (df["completed_at_dt"] <= we_ts)
    ].copy()

    pending = df[df["status"].isin(["todo", "doing", "blocked", "awaiting_approval"])].copy()

    # Pre-carga bulk de RACI y aprobaciones (evita N llamadas por tarea)
    all_ids_report = list(df["id"].astype(int).unique())
    bulk_r, bulk_a = _bulk_raci_names(all_ids_report)
    bulk_appr = _bulk_approval_cycle(all_ids_report)

    def enrich(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in.empty:
            return df_in
        df_in = df_in.copy()
        df_in["lead_days"] = None
        ok = df_in["created_at_dt"].notna() & df_in["completed_at_dt"].notna()
        df_in.loc[ok, "lead_days"] = (df_in.loc[ok, "completed_at_dt"] - df_in.loc[ok, "created_at_dt"]).dt.days
        # Bulk lookups — sin llamadas individuales por fila
        df_in["horas_aprobacion"] = df_in["id"].astype(int).map(lambda x: bulk_appr.get(x))
        df_in["ejecutores"] = df_in["id"].astype(int).map(lambda x: bulk_r.get(x, "—"))
        df_in["dueno"] = df_in["id"].astype(int).map(lambda x: bulk_a.get(x, "—"))
        return df_in

    return enrich(done), enrich(pending)


# =============================================================================
# ADMIN GLOBAL
# =============================================================================
if menu == "Admin":
    if not st.session_state.get("is_global_admin"):
        st.error("Acceso denegado.")
        st.stop()

    st.subheader("Panel Admin Global")

    groups = getattr(supa_table("groups").select("*").order("id", desc=True).execute(), "data", []) or []
    st.markdown("### Grupos")
    st.dataframe(pd.DataFrame(groups) if groups else pd.DataFrame(), use_container_width=True)

    if not groups:
        st.info("No hay grupos.")
        st.stop()

    gsel = st.selectbox("Selecciona un grupo", options=groups, format_func=lambda g: f"[{g['id']}] {g.get('name','(sin nombre)')}")
    gid = int(gsel["id"])

    st.markdown("### Miembros del grupo")
    gm = getattr(supa_table("group_members").select("user_id,role,joined_at").eq("group_id", gid).execute(), "data", []) or []
    if gm:
        prof_by = fetch_profiles([x["user_id"] for x in gm])
        rows = []
        for m in gm:
            p = prof_by.get(m["user_id"], {})
            rows.append({
                "user_id": m["user_id"],
                "full_name": p.get("full_name"),
                "username": p.get("username"),
                "email": p.get("email"),
                "role": m.get("role"),
                "joined_at": m.get("joined_at"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Este grupo no tiene miembros.")

    st.markdown("### Proyectos")
    projs = getattr(supa_table("projects").select("*").eq("group_id", gid).order("id", desc=True).execute(), "data", []) or []
    st.dataframe(pd.DataFrame(projs) if projs else pd.DataFrame(), use_container_width=True)

    if projs:
        psel = st.selectbox("Ver tareas del proyecto", options=projs, format_func=lambda p: f"[{p['id']}] {p['name']}")
        pid = int(psel["id"])
        tasks = getattr(supa_table("tasks").select("*").eq("project_id", pid).order("id", desc=True).limit(500).execute(), "data", []) or []
        st.markdown("### Tareas (últimas 500)")
        st.dataframe(pd.DataFrame(tasks) if tasks else pd.DataFrame(), use_container_width=True)

    st.stop()


# Helper para encabezados de página
def _page_header(icon: str, title: str, subtitle: str = "", color: str = "#4C8BF5"):
    st.markdown(
        f'<div style="border-left:4px solid {color};padding:6px 0 6px 14px;margin-bottom:18px;">'
        f'<div style="font-size:1.45rem;font-weight:700">{icon} {title}</div>'
        + (f'<div style="color:#718096;font-size:0.88rem;margin-top:2px">{subtitle}</div>' if subtitle else "")
        + f'</div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# RESUMEN
# =============================================================================
if menu == "Resumen":
    uid = st.session_state["user_id"]
    _page_header("📊", "Resumen", "Vista general de trabajo, KPIs y riesgos", "#4C8BF5")

    st.markdown("### Vista general de tareas (filtros por RACI y fechas)")

    pmap = _project_map(GROUP_ID)
    if not pmap:
        st.info("No hay proyectos todavía.")
        st.stop()

    projs = [{"id": None, "name": "Todos los proyectos"}] + [{"id": pid, "name": p["name"]} for pid, p in pmap.items()]
    proj_pick = safe_selectbox_dict("Proyecto", projs, format_func=lambda x: x["name"], key="sum_proj")
    proj_filter = proj_pick["id"]

    members = list_group_members(GROUP_ID)
    member_opts = [{"user_id": None, "full_name": "(cualquiera)", "username": ""}] + members
    person_pick = safe_selectbox_dict(
        "Persona (para filtrar RACI)",
        member_opts,
        format_func=lambda x: fmt_user(x) if x["user_id"] else "(cualquiera)",
        key="sum_person"
    )

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        role_pick = st.selectbox("Rol RACI", ["Todos", "A", "R", "C", "I"], index=0, key="sum_role")
    with r2:
        status_pick = st.multiselect(
            "Estado",
            ["todo", "doing", "blocked", "awaiting_approval", "done"],
            default=["todo", "doing", "blocked", "awaiting_approval"],
            key="sum_status"
        )
    with r3:
        window = st.selectbox("Fechas próximas", ["(sin filtro)", "Próximos 7 días", "Próximos 14 días", "Próximos 30 días", "Rango"], key="sum_window")
    with r4:
        date_field = st.selectbox("Campo fecha", ["due_date", "target_date", "start_date"], index=0, key="sum_date_field")

    date_from = None
    date_to = None
    if window == "Rango":
        cA, cB = st.columns(2)
        with cA:
            date_from = st.date_input("Desde", value=date.today(), key="sum_from")
        with cB:
            date_to = st.date_input("Hasta", value=date.today() + timedelta(days=14), key="sum_to")
    elif window.startswith("Próximos"):
        n = int(window.split()[1])
        date_from = date.today()
        date_to = date.today() + timedelta(days=n)

    task_ids = _fetch_group_task_ids(GROUP_ID, project_id=int(proj_filter) if proj_filter else None, limit_per_project=700)
    tasks = _bulk_tasks(task_ids)
    roles_raw = _bulk_task_roles(task_ids)

    roles_by_task: dict[int, dict[str, list[str]]] = {}
    all_user_ids = set()
    for rr in roles_raw:
        tid = int(rr["task_id"])
        roles_by_task.setdefault(tid, {"A": [], "R": [], "C": [], "I": []})
        roles_by_task[tid][rr["role"]].append(rr["user_id"])
        all_user_ids.add(rr["user_id"])

    prof_by = fetch_profiles(list(all_user_ids))

    def a_name_from_ids(ids: list[str]) -> str:
        if not ids:
            return "—"
        return prof_by.get(ids[0], {}).get("full_name", "(sin nombre)")

    def names_from_ids(ids: list[str]) -> str:
        if not ids:
            return "—"
        nms = [prof_by.get(x, {}).get("full_name", "(sin nombre)") for x in ids]
        return ", ".join(sorted(nms))

    rows = []
    for t in tasks:
        tid = int(t["id"])
        rset = roles_by_task.get(tid, {"A": [], "R": [], "C": [], "I": []})

        if status_pick and t.get("status") not in status_pick:
            continue

        if date_from and date_to:
            raw = t.get(date_field)
            if not raw:
                continue
            try:
                d = date.fromisoformat(raw)
            except Exception:
                continue
            if not (date_from <= d <= date_to):
                continue

        if person_pick and person_pick["user_id"]:
            puid = person_pick["user_id"]
            if role_pick == "Todos":
                if not (puid in rset["A"] or puid in rset["R"] or puid in rset["C"] or puid in rset["I"]):
                    continue
            else:
                if puid not in rset.get(role_pick, []):
                    continue

        rows.append({
            "project_id": int(t["project_id"]),
            "project": pmap.get(int(t["project_id"]), {}).get("name", f"#{t['project_id']}"),
            "id": tid,
            "title": t.get("title"),
            "status": t.get("status"),
            "priority": t.get("priority"),
            "start_date": t.get("start_date"),
            "target_date": t.get("target_date"),
            "due_date": t.get("due_date"),
            "A": a_name_from_ids(rset["A"]),
            "R": names_from_ids(rset["R"]),
            "C": names_from_ids(rset["C"]),
            "I": names_from_ids(rset["I"]),
            "created_at": t.get("created_at"),
            "updated_at": t.get("updated_at"),
        })

    df_all = pd.DataFrame(rows)
    if df_all.empty:
        st.info("No hay tareas que coincidan con los filtros.")
    else:
        st.dataframe(
            df_all.sort_values(["priority", "due_date"], ascending=[True, True]),
            use_container_width=True,
            height=320
        )

    st.divider()
    st.subheader("Resumen por RACI (personal)")

    def fetch_by_role(role: str):
        res = supa_table("task_roles").select("task_id").eq("user_id", uid).eq("role", role).execute()
        tids = [int(x["task_id"]) for x in (getattr(res, "data", []) or [])]
        if not tids:
            return []

        tasks_ = _bulk_tasks(tids)
        allowed_pids = set(pmap.keys())
        out = []
        for t in tasks_:
            if int(t["project_id"]) not in allowed_pids:
                continue
            out.append({
                "id": int(t["id"]),
                "title": t.get("title"),
                "status": t.get("status"),
                "priority": t.get("priority"),
                "due_date": t.get("due_date"),
                "project": pmap[int(t["project_id"])]["name"],
            })

        pr_rank = {"urgente": 1, "alta": 2, "media": 3, "baja": 4}
        out.sort(key=lambda x: (pr_rank.get(x["priority"], 99), x["due_date"] or "9999-12-31"))
        return out

    R = fetch_by_role("R")
    A = fetch_by_role("A")
    C = fetch_by_role("C")
    I = fetch_by_role("I")

    cols = st.columns(4)
    with cols[0]:
        st.markdown("### Ejecuto")
        st.dataframe(pd.DataFrame(R) if R else pd.DataFrame(), use_container_width=True, height=220)
    with cols[1]:
        st.markdown("### Debo aprobar")
        st.dataframe(pd.DataFrame(A) if A else pd.DataFrame(), use_container_width=True, height=220)
    with cols[2]:
        st.markdown("### Me consultan")
        st.dataframe(pd.DataFrame(C) if C else pd.DataFrame(), use_container_width=True, height=220)
    with cols[3]:
        st.markdown("### Solo informado")
        st.dataframe(pd.DataFrame(I) if I else pd.DataFrame(), use_container_width=True, height=220)

    st.divider()
    st.subheader("Riesgos y control")

    all_tasks = _bulk_tasks(_fetch_group_task_ids(GROUP_ID, project_id=int(proj_filter) if proj_filter else None))
    today = date.today()

    overdue_rows = []
    blocked_rows = []
    for t in all_tasks:
        pid = int(t["project_id"])
        pname = pmap.get(pid, {}).get("name", f"#{pid}")
        if t.get("status") == "blocked":
            blocked_rows.append({
                "id": int(t["id"]),
                "title": t.get("title"),
                "blocked_reason": t.get("blocked_reason"),
                "project": pname,
            })
        if t.get("status") != "done" and t.get("due_date"):
            try:
                dd = date.fromisoformat(t["due_date"])
                if dd < today:
                    overdue_rows.append({
                        "id": int(t["id"]),
                        "title": t.get("title"),
                        "due_date": t.get("due_date"),
                        "status": t.get("status"),
                        "project": pname,
                    })
            except Exception:
                pass

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Vencidas")
        st.dataframe(pd.DataFrame(overdue_rows) if overdue_rows else pd.DataFrame(), use_container_width=True, height=260)
    with c2:
        st.markdown("### Bloqueadas")
        st.dataframe(pd.DataFrame(blocked_rows) if blocked_rows else pd.DataFrame(), use_container_width=True, height=260)

    st.divider()
    st.subheader("KPIs por persona")

    proj_ids_for_kpi = [{"id": None, "name": "Todos"}] + [{"id": pid, "name": p["name"]} for pid, p in pmap.items()]
    proj_kpi = safe_selectbox_dict("Filtrar KPIs por proyecto (opcional)", proj_ids_for_kpi, format_func=lambda x: x["name"], key="kpi_proj_filter")
    kpi_df = compute_kpis_per_person(GROUP_ID, project_id=int(proj_kpi["id"]) if proj_kpi and proj_kpi["id"] else None)

    if kpi_df.empty:
        st.info("No hay datos todavía.")
    else:
        kpi_badges_row(kpi_df)

        show_cols = [
            "nombre",
            "por_hacer", "en_curso", "bloqueadas", "en_aprobacion", "completadas",
            "wip_en_curso", "vencidas_abiertas",
            "antiguedad_prom_dias",
            "tiempo_cierre_prom_60d",
            "tasa_cumplimiento_pct",
        ]
        show_cols = [c for c in show_cols if c in kpi_df.columns]
        st.dataframe(kpi_df[show_cols], use_container_width=True, height=420)

        # Alerta WIP: usuarios que superan el límite por defecto (3)
        wip_col = "wip_en_curso"
        if wip_col in kpi_df.columns:
            over_wip = kpi_df[kpi_df[wip_col] >= 3][["nombre", wip_col]]
            if not over_wip.empty:
                nombres = ", ".join(over_wip["nombre"].tolist())
                st.warning(f"⚠️ **Límite WIP alcanzado o superado** por: {nombres}. Considera redistribuir tareas.")

        chart_df = kpi_df[["nombre", "wip_en_curso", "vencidas_abiertas"]].copy() if "nombre" in kpi_df.columns else pd.DataFrame()
        if not chart_df.empty:
            fig = px.bar(
                chart_df, x="nombre",
                y=["wip_en_curso", "vencidas_abiertas"],
                barmode="group",
                labels={"value": "Cantidad", "nombre": "Persona", "variable": "Métrica",
                        "wip_en_curso": "En curso (WIP)", "vencidas_abiertas": "Vencidas abiertas"},
                color_discrete_map={"wip_en_curso": "#4C8BF5", "vencidas_abiertas": "#E53935"},
            )
            fig.update_layout(height=360, xaxis_title="", yaxis_title="", legend_title="")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Ciclo de aprobación (en aprobación)")

    proj_ap = safe_selectbox_dict("Filtrar aprobaciones por proyecto (opcional)", proj_ids_for_kpi, format_func=lambda x: x["name"], key="appr_proj_filter")
    appr_df = approvals_kpi_by_accountable(GROUP_ID, project_id=int(proj_ap["id"]) if proj_ap and proj_ap["id"] else None)
    if appr_df.empty:
        st.info("No hay aprobaciones registradas todavía.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("⏳ Aprobaciones pendientes", int(appr_df["pendientes_count"].sum()))
        c2.metric("🕐 Horas promedio pendientes", round(float(appr_df["horas_ciclo_pendientes"].dropna().mean()), 1) if appr_df["horas_ciclo_pendientes"].notna().any() else "—")
        c3.metric("✅ Horas promedio aprobadas (60d)", round(float(appr_df["horas_ciclo_aprobadas_60d"].dropna().mean()), 1) if appr_df["horas_ciclo_aprobadas_60d"].notna().any() else "—")
        st.dataframe(appr_df.sort_values(["pendientes_count", "horas_ciclo_pendientes"], ascending=False),
                     use_container_width=True, height=320)


# =============================================================================
# PROYECTOS
# =============================================================================
elif menu == "Proyectos":
    uid = st.session_state["user_id"]
    is_leader = (user_role_in_group(GROUP_ID, uid) == "leader")
    _page_header("📁", "Proyectos", "Gestión de proyectos y miembros", "#38B2AC")
    projs = list_projects(GROUP_ID)
    st.dataframe(pd.DataFrame(projs) if projs else pd.DataFrame(), use_container_width=True)

    st.divider()
    st.subheader("Miembros por proyecto (UI)")

    if not projs:
        st.info("No hay proyectos.")
    else:
        proj = safe_selectbox_dict("Proyecto a administrar", projs, format_func=lambda p: f"[{p['id']}] {p['name']}", key="pm_proj")
        pid = int(proj["id"])

        # ✅ En db.py NO existe project_manage_members, usamos project_edit (sí existe en ACTIONS)
        can_manage = (user_role_in_group(GROUP_ID, uid) == "leader") or has_permission(GROUP_ID, uid, "project_edit")

        current = list_project_members(pid)
        st.markdown("#### Miembros actuales")
        st.dataframe(pd.DataFrame(current) if current else pd.DataFrame(), use_container_width=True)

        group_members = list_group_members(GROUP_ID)
        current_ids = set([m["user_id"] for m in current]) if current else set()
        candidates = [m for m in group_members if m["user_id"] not in current_ids]

        if not can_manage:
            st.caption("No tienes permisos para modificar miembros del proyecto.")
        else:
            st.markdown("#### Agregar miembros")
            if candidates:
                opt_map = {fmt_user(m): m["user_id"] for m in candidates}
                add_sel = st.multiselect("Selecciona miembros a agregar", options=list(opt_map.keys()))
                if st.button("Agregar al proyecto"):
                    if not add_sel:
                        st.warning("Selecciona al menos 1.")
                    else:
                        for k in add_sel:
                            ensure_project_member(pid, opt_map[k])
                        st.success("Miembros agregados.")
                        st.rerun()
            else:
                st.caption("No hay miembros disponibles para agregar (ya están todos en el proyecto).")

            st.markdown("#### Remover miembros")
            if current:
                cur_map = {fmt_user(m): m["user_id"] for m in current}
                rem = st.multiselect("Selecciona miembros a remover", options=list(cur_map.keys()))
                if st.button("Remover del proyecto"):
                    if not rem:
                        st.warning("Selecciona al menos 1.")
                    else:
                        if len(current) - len(rem) <= 0:
                            st.error("No puedes dejar el proyecto sin miembros.")
                        else:
                            for k in rem:
                                remove_project_member(pid, cur_map[k])
                            st.success("Miembros removidos.")
                            st.rerun()

    st.divider()
    st.subheader("Crear proyecto")

    if not has_permission(GROUP_ID, uid, "project_create"):
        st.info("No tienes permiso para crear proyectos.")
    else:
        with st.form("new_project"):
            name = st.text_input("Nombre")
            desc = st.text_area("Descripción", height=70)
            wip = st.number_input("WIP límite DOING por ejecutor", min_value=1, max_value=10, value=3, step=1)
            submit = st.form_submit_button("Crear")

        if submit:
            if not name.strip():
                st.error("Falta nombre.")
            else:
                pid = create_project(GROUP_ID, name.strip(), desc.strip() or None, int(wip), uid)
                ensure_project_member(pid, uid)
                st.success("Proyecto creado.")
                st.rerun()

    st.divider()
    st.subheader("Miembros del grupo y liderazgo")

    members = list_group_members(GROUP_ID)
    st.dataframe(pd.DataFrame(members), use_container_width=True)

    if is_leader and has_permission(GROUP_ID, uid, "group_transfer_lead"):
        st.markdown("#### Transferir liderazgo")
        choices = [m for m in members if m["role"] != "leader"]
        if choices:
            opt = safe_selectbox_dict("Elegir nuevo líder", choices, format_func=lambda x: fmt_user(x))
            if opt and st.button("Hacer líder"):
                transfer_group_leadership(GROUP_ID, uid, opt["user_id"])
                st.success("Liderazgo transferido.")
                st.rerun()

    if is_leader and has_permission(GROUP_ID, uid, "group_manage_members"):
        st.markdown("#### Código del grupo")
        g = supa_table("groups").select("join_code").eq("id", GROUP_ID).limit(1).execute()
        join_code = (getattr(g, "data", [{}])[0] or {}).get("join_code")
        st.info(f"Código del grupo: **{join_code}** (compártelo con tu equipo)")


# =============================================================================
# TABLERO
# =============================================================================
elif menu == "Tablero":
    uid = st.session_state["user_id"]
    projs = list_projects(GROUP_ID)
    if not projs:
        st.info("Crea un proyecto primero.")
        st.stop()

    proj = safe_selectbox_dict("Proyecto", projs, format_func=lambda p: f"[{p['id']}] {p['name']}")
    if not proj:
        st.stop()
    pid = int(proj["id"])
    _page_header("📌", "Tablero Kanban", f"Proyecto: {proj['name']}", "#805AD5")

    # tags
    tags = getattr(supa_table("tags").select("*").eq("group_id", GROUP_ID).execute(), "data", []) or []
    tag_opts = {f"{t['kind']}:{t['name']}": int(t["id"]) for t in tags}

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        q = st.text_input("Búsqueda global (título/desc/DoD)")
    with f2:
        f_status = st.multiselect("Estado", ["todo", "doing", "blocked", "awaiting_approval", "done"],
                                  default=["todo", "doing", "blocked", "awaiting_approval"])
    with f3:
        f_priority = st.multiselect("Prioridad", ["urgente", "alta", "media", "baja"],
                                    default=["urgente", "alta", "media", "baja"])
    with f4:
        only_mine = st.checkbox("Solo donde soy R/A", value=False)

    tag_filter = st.multiselect("Filtrar por tags/componentes", options=list(tag_opts.keys()))
    only_overdue = st.checkbox("Solo vencidas", value=False)
    only_blocked = st.checkbox("Solo bloqueadas", value=False)

    tasks = list_tasks_for_board(
        group_id=GROUP_ID,
        project_id=pid,
        statuses=f_status,
        priorities=f_priority,
        q=q.strip() if q else None,
        only_overdue=only_overdue,
        only_blocked=only_blocked,
        only_mine_user_id=uid if only_mine else None,
        tag_ids_filter=[tag_opts[k] for k in tag_filter] if tag_filter else None
    )

    by_status = {s: [] for s in ["todo", "doing", "blocked", "awaiting_approval", "done"]}
    for t in tasks:
        by_status[t["status"]].append(t)

    _STATUS_BADGE = {
        "todo": "badge-todo", "doing": "badge-doing",
        "blocked": "badge-blocked", "awaiting_approval": "badge-approval", "done": "badge-done",
    }
    _PRIO_BADGE = {
        "urgente": "badge-urgente", "alta": "badge-alta",
        "media": "badge-media", "baja": "badge-baja",
    }
    _STATUS_LABEL = {
        "todo": "Por hacer", "doing": "En curso", "blocked": "Bloqueada",
        "awaiting_approval": "En aprobación", "done": "Completada",
    }

    def render_task_card(t):
        roles = get_task_roles(int(t["id"]))
        A = roles["A"][0]["name"] if roles["A"] else "—"
        Rs = ", ".join([x["name"] for x in roles["R"]]) if roles["R"] else "—"
        due = t.get("due_date") or "—"
        status = t.get("status", "todo")
        prio = t.get("priority", "media")

        deps = list_task_dependencies_open(int(t["id"]))
        aging_days = None
        try:
            created = datetime.fromisoformat(t["created_at"])
            aging_days = (datetime.now() - created).days
        except Exception:
            pass

        # Color de borde según prioridad
        _border_colors = {"urgente": "#FC8181", "alta": "#F6AD55", "media": "#63B3ED", "baja": "#68D391"}
        border_c = _border_colors.get(prio, "#CBD5E0")

        with st.container(border=True):
            sbadge = _STATUS_BADGE.get(status, "badge-todo")
            slabel = _STATUS_LABEL.get(status, status)
            pbadge = _PRIO_BADGE.get(prio, "badge-media")
            st.markdown(
                f'<span class="{pbadge}">{prio.upper()}</span>&nbsp;&nbsp;'
                f'<span style="color:#718096;font-size:0.82em">#{t["id"]} · vence {due}'
                + (f' · {aging_days}d' if aging_days is not None else "")
                + f'</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{t['title']}**")
            st.caption(f"👤 Dueño: {A}  •  🔧 Ejecutores: {Rs}")
            if deps:
                st.markdown(f'<span class="badge-blocked">⛔ {len(deps)} dependencia(s) pendiente(s)</span>', unsafe_allow_html=True)

            if st.button("Abrir →", key=f"open_{t['id']}", use_container_width=True):
                st.session_state["selected_task_id"] = int(t["id"])
                st.session_state["menu"] = "Tarea (detalle)"
                st.rerun()

    cols = st.columns(5)
    # Conteo WIP del usuario activo en este proyecto
    doing_ids_board = [int(t["id"]) for t in by_status.get("doing", [])]
    wip_current_user = sum(
        1 for tid in doing_ids_board
        if any(
            rr.get("user_id") == uid and rr.get("role") == "R"
            for rr in (safe_execute(
                supa_table("task_roles").select("user_id,role")
                .eq("task_id", tid).eq("user_id", uid).eq("role", "R"),
                default=[]
            ) or [])
        )
    ) if doing_ids_board else 0

    proj_row = safe_execute(
        supa_table("projects").select("wip_limit_doing").eq("id", pid).limit(1),
        default=[{"wip_limit_doing": 3}]
    )
    wip_limit_board = int((proj_row[0] if proj_row else {}).get("wip_limit_doing") or 3)

    if wip_current_user >= wip_limit_board:
        st.warning(f"⚠️ Has alcanzado tu límite WIP: **{wip_current_user}/{wip_limit_board}** tareas en curso. Completa una antes de tomar más trabajo.")

    _COL_COLORS = {
        "todo": "#F7F9FC", "doing": "#EBF8FF",
        "blocked": "#FFF5F5", "awaiting_approval": "#FFFBEB", "done": "#F0FFF4",
    }
    _COL_BORDER = {
        "todo": "#CBD5E0", "doing": "#63B3ED",
        "blocked": "#FC8181", "awaiting_approval": "#F6AD55", "done": "#68D391",
    }
    labels = {
        "todo": "📋 Por hacer",
        "doing": f"🔵 En curso ({len(by_status['doing'])})",
        "blocked": f"🔴 Bloqueadas ({len(by_status['blocked'])})",
        "awaiting_approval": f"🟡 En aprobación ({len(by_status['awaiting_approval'])})",
        "done": f"✅ Completadas ({len(by_status['done'])})",
    }
    for s, col in zip(labels.keys(), cols):
        with col:
            bg = _COL_COLORS.get(s, "#F7F9FC")
            brd = _COL_BORDER.get(s, "#CBD5E0")
            st.markdown(
                f'<div style="background:{bg};border-top:3px solid {brd};'
                f'border-radius:10px 10px 0 0;padding:8px 12px;margin-bottom:6px;">'
                f'<strong>{labels[s]}</strong></div>',
                unsafe_allow_html=True,
            )
            for t in by_status[s]:
                render_task_card(t)

    st.divider()
    st.subheader("Crear tarea")

    if not has_permission(GROUP_ID, uid, "task_create"):
        st.info("No tienes permiso para crear tareas.")
    else:
        members = list_project_members(pid)
        if not members:
            st.warning("No hay miembros en el proyecto. Agrega miembros en la pestaña Proyectos.")
        m_opts = {fmt_user(m): m["user_id"] for m in members}

        t_tags = [t for t in tags if t["kind"] == "tag"]
        t_comps = [t for t in tags if t["kind"] == "component"]
        tag_names = [x["name"] for x in t_tags]
        comp_names = [x["name"] for x in t_comps]

        with st.form("new_task"):
            title = st.text_input("Título")
            desc = st.text_area("Descripción", height=80)
            dod = st.text_area("Criterio de aceptación (DoD)", height=80)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                priority = st.selectbox("Prioridad", ["urgente", "alta", "media", "baja"], index=2)

            # ✅ date_input no soporta None directo: usamos checkbox
            with c2:
                use_start = st.checkbox("Usar start_date", value=False)
                start_d = st.date_input("Start date", value=date.today()) if use_start else None
            with c3:
                target_d = st.date_input("Target (SLA interno)", value=date.today() + timedelta(days=7))
            with c4:
                due_d = st.date_input("Due date (deadline)", value=date.today() + timedelta(days=10))

            requires_approval = st.checkbox("Requiere aprobación del dueño para cerrar", value=True)

            st.markdown("Responsabilidad y comunicación (RACI)")
            if m_opts:
                A = st.selectbox("Dueño (A) - único", options=list(m_opts.keys()))
                R = st.multiselect("Ejecutores (R) - mínimo 1", options=list(m_opts.keys()))
                C = st.multiselect("Consultados (C)", options=list(m_opts.keys()))
                I = st.multiselect("Informados (I)", options=list(m_opts.keys()))
            else:
                A = None; R = []; C = []; I = []

            st.markdown("Etiquetas y componente")
            selected_tags = st.multiselect("Tags", options=tag_names)
            selected_component = st.selectbox("Componente", options=["(ninguno)"] + comp_names)

            existing = list_tasks_by_project(pid, limit=300)
            dep_map = {f"#{x['id']} {x['title']}": int(x["id"]) for x in existing}
            deps_sel = st.multiselect("Depende de (opcional)", options=list(dep_map.keys()))
            send_email_now = st.checkbox("Enviar correo (solo notificaciones importantes)", value=False)

            submit = st.form_submit_button("Crear tarea")

        if submit:
            if not title.strip():
                st.error("Falta título.")
            elif not dod.strip():
                st.error("Toda tarea debe tener DoD.")
            elif not R:
                st.error("Debe haber al menos 1 ejecutor (R).")
            else:
                task_id = create_task(
                    project_id=pid,
                    title=title.strip(),
                    description=desc.strip() or None,
                    dod=dod.strip(),
                    priority=priority,
                    status="todo",
                    start_date=start_d,
                    target_date=target_d,
                    due_date=due_d,
                    requires_approval=requires_approval,
                    created_by=uid
                )

                upsert_task_roles(
                    task_id=int(task_id),
                    user_id=uid,
                    A_id=m_opts[A],
                    R_ids=[m_opts[x] for x in R],
                    C_ids=[m_opts[x] for x in C],
                    I_ids=[m_opts[x] for x in I]
                )

                set_task_dependencies(int(task_id), [dep_map[d] for d in deps_sel] if deps_sel else [])

                # tags: crea si no existe
                tag_ids = []
                for tn in selected_tags:
                    # get_or_create_tag no está en db.py, lo hacemos inline
                    r = getattr(supa_table("tags").select("id").eq("group_id", GROUP_ID).eq("name", tn).eq("kind", "tag").limit(1).execute(), "data", []) or []
                    tid = int(r[0]["id"]) if r else int(getattr(supa_table("tags").insert({"group_id": GROUP_ID, "name": tn, "kind": "tag"}).execute(), "data", [{}])[0]["id"])
                    tag_ids.append(tid)

                if selected_component != "(ninguno)":
                    r = getattr(supa_table("tags").select("id").eq("group_id", GROUP_ID).eq("name", selected_component).eq("kind", "component").limit(1).execute(), "data", []) or []
                    tid = int(r[0]["id"]) if r else int(getattr(supa_table("tags").insert({"group_id": GROUP_ID, "name": selected_component, "kind": "component"}).execute(), "data", [{}])[0]["id"])
                    tag_ids.append(tid)

                # set_task_tags no está en db.py, lo hacemos inline
                supa_table("task_tags").delete().eq("task_id", int(task_id)).execute()
                if tag_ids:
                    supa_table("task_tags").insert([{"task_id": int(task_id), "tag_id": int(tid)} for tid in sorted(set(tag_ids))]).execute()

                notify(m_opts[A], "info", f"Nueva tarea #{task_id}: eres dueño.", int(task_id),
                       send_email=send_email_now, email_subject=f"Nueva tarea #{task_id} (Dueño)")

                for x in R:
                    notify(m_opts[x], "info", f"Nueva tarea #{task_id}: eres ejecutor.", int(task_id),
                           send_email=send_email_now, email_subject=f"Nueva tarea #{task_id} (Ejecutor)")

                for x in C:
                    notify(m_opts[x], "input_request", f"Te consultan en tarea #{task_id}.", int(task_id), send_email=False)
                for x in I:
                    notify(m_opts[x], "info", f"FYI: tarea #{task_id} creada.", int(task_id), send_email=False)

                st.success("Tarea creada.")
                st.rerun()


# =============================================================================
# TAREA (DETALLE)
# =============================================================================
elif menu == "Tarea (detalle)":
    uid = st.session_state["user_id"]
    _page_header("🔍", "Detalle de tarea", "Responsabilidades, avances y aprobaciones", "#DD6B20")

    pmap = _project_map(GROUP_ID)
    projs = [{"id": pid, "name": p["name"]} for pid, p in pmap.items()]
    if not projs:
        st.info("No hay proyectos.")
        st.stop()

    selected = st.session_state.get("selected_task_id")

    proj = safe_selectbox_dict("Proyecto", projs, format_func=lambda x: f"[{x['id']}] {x['name']}")
    if not proj:
        st.stop()
    pid = int(proj["id"])

    tasks = list_tasks_by_project(pid, limit=300)
    if not tasks:
        st.info("No hay tareas.")
        st.stop()

    ids = [int(t["id"]) for t in tasks]
    default_idx = ids.index(int(selected)) if (selected is not None and int(selected) in ids) else 0

    task_pick = st.selectbox("Tarea", options=tasks, index=default_idx, format_func=lambda x: f"#{x['id']} {x['title']}")
    task_id = int(task_pick["id"])

    t = get_task(task_id)
    if not t:
        st.error("Tarea no encontrada.")
        st.stop()

    roles = get_task_roles(task_id)
    A_id = get_accountable_id(task_id)
    deps = list_task_dependencies_open(task_id)

    st.markdown(f"### #{t['id']} — {t['title']}")
    st.caption(f"Proyecto: {pmap[int(t['project_id'])]['name']} • Estado: {t['status']} • Prioridad: {t['priority']}")

    st.markdown("#### Qué se pide exactamente (DoD / criterio de aceptación)")
    st.write(t["dod"])

    st.markdown("#### Descripción")
    st.write(t.get("description") or "")

    if deps:
        st.warning("Esta tarea tiene dependencias pendientes. Debe completarse primero:")
        st.dataframe(pd.DataFrame(deps), use_container_width=True)

    st.markdown("#### SLA / tiempos")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.write(f"**Inicio:** {t.get('start_date') or '—'}")
    with c2:
        st.write(f"**Objetivo (SLA):** {t.get('target_date') or '—'}")
    with c3:
        st.write(f"**Vencimiento:** {t.get('due_date') or '—'}")
    with c4:
        try:
            aging = (datetime.now() - datetime.fromisoformat(t["created_at"])).days
            st.write(f"**Antigüedad:** {aging} días")
        except Exception:
            st.write("**Antigüedad:** —")

    st.markdown("#### Responsabilidad y comunicación")
    st.write(f"Dueño: {roles['A'][0]['name'] if roles['A'] else '—'}")
    st.write("Ejecutores: " + (", ".join([x["name"] for x in roles["R"]]) or "—"))
    st.write("Consultados: " + (", ".join([x["name"] for x in roles["C"]]) or "—"))
    st.write("Informados: " + (", ".join([x["name"] for x in roles["I"]]) or "—"))

    st.divider()
    st.subheader("Acciones")

    # due_date
    if can_edit_due_date(GROUP_ID, task_id, uid):
        new_due = st.date_input("Cambiar due date", value=date.fromisoformat(t["due_date"]) if t.get("due_date") else date.today())
        if st.button("Guardar due date"):
            old = t.get("due_date")
            update_task_due_date(task_id, new_due)
            log_history(task_id, uid, "due_date", old, new_due.isoformat())
            if A_id:
                add_notification(A_id, "info", f"Se cambió due date de la tarea #{task_id}.", task_id)
            st.success("Due date actualizado.")
            st.rerun()
    else:
        st.caption("No puedes cambiar due date (solo dueño o líder, según gobernanza).")

    # status
    if can_change_status(GROUP_ID, task_id, uid):
        new_status = st.selectbox(
            "Cambiar estado",
            ["todo", "doing", "blocked", "awaiting_approval", "done"],
            index=["todo", "doing", "blocked", "awaiting_approval", "done"].index(t["status"])
        )

        blocked_reason = None
        unblock_owner = None

        if new_status == "blocked":
            blocked_reason = st.text_input("Motivo de bloqueo")
            members = list_group_members(GROUP_ID)
            opts = {fmt_user(m): m["user_id"] for m in members}
            who = st.selectbox("Quién debe destrabar", options=list(opts.keys()))
            unblock_owner = opts[who]

        send_email_approval = False
        send_email_blocked = False

        if new_status == "done" and int(t.get("requires_approval") or 0) == 1:
            send_email_approval = st.checkbox(
                "Enviar correo al dueño (A) por solicitud de aprobación",
                value=True,
                key=f"email_appr_{task_id}"
            )

        if new_status == "blocked":
            send_email_blocked = st.checkbox(
                "Enviar correo al destrabador (bloqueo)",
                value=True,
                key=f"email_block_{task_id}"
            )

        if st.button("Aplicar estado"):
            if new_status in ("doing", "awaiting_approval", "done") and deps:
                update_task_status(task_id, "blocked", "Dependencias pendientes", None, None)
                log_history(task_id, uid, "status", t["status"], "blocked")
                st.error("No se puede avanzar: dependencias pendientes. Se marcó como BLOCKED.")
                st.rerun()

            if new_status == "doing":
                ok, msg = wip_limit_ok(int(t["project_id"]), uid)
                if not ok:
                    st.error(msg)
                    st.stop()

            if new_status == "done" and int(t.get("requires_approval") or 0) == 1:
                if not A_id:
                    st.error("No hay dueño (A).")
                    st.stop()
                update_task_status(task_id, "awaiting_approval", None, None, None)
                create_approval_request(task_id, uid, A_id, request_note="Solicitud de cierre")
                log_history(task_id, uid, "status", t["status"], "awaiting_approval")
                notify(
                    A_id,
                    "approval_request",
                    f"Se solicita tu aprobación para cerrar tarea #{task_id}.",
                    task_id,
                    send_email=send_email_approval,
                    email_subject=f"Aprobación requerida: tarea #{task_id}"
                )
                st.success("Enviado a aprobación.")
                st.rerun()

            completed_at = now_iso() if (new_status == "done" and int(t.get("requires_approval") or 0) == 0) else None
            update_task_status(task_id, new_status, blocked_reason, unblock_owner, completed_at)
            log_history(task_id, uid, "status", t["status"], new_status)

            if A_id:
                add_notification(A_id, "status_change", f"Estado de tarea #{task_id} cambió a {new_status}.", task_id)

            if new_status == "blocked" and unblock_owner:
                notify(
                    unblock_owner,
                    "blocked",
                    f"Tarea #{task_id} bloqueada: {blocked_reason or '(sin motivo)'}",
                    task_id,
                    send_email=send_email_blocked,
                    email_subject=f"Tarea bloqueada #{task_id}"
                )

            if new_status in ("blocked", "done"):
                res = supa_table("task_roles").select("user_id").eq("task_id", task_id).eq("role", "I").execute()
                for rr in (getattr(res, "data", []) or []):
                    add_notification(rr["user_id"], "info", f"FYI: tarea #{task_id} ahora está {new_status}.", task_id)

            st.success("Estado actualizado.")
            st.rerun()
    else:
        st.caption("No puedes cambiar estado (solo ejecutor o líder).")

    # Aprobaciones
    st.divider()
    st.subheader("Aprobaciones")

    pend = get_pending_approval(task_id)
    if pend:
        st.info("Hay una aprobación pendiente.")
        if A_id == uid:
            decision_note = st.text_input("Nota de decisión")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Aprobar"):
                    decide_approval(int(pend["id"]), "approved", decision_note or None)
                    update_task_status(task_id, "done", None, None, now_iso())
                    log_history(task_id, uid, "approval", "pending", "approved")
                    log_history(task_id, uid, "status", "awaiting_approval", "done")

                    res = supa_table("task_roles").select("user_id").eq("task_id", task_id).eq("role", "R").execute()
                    for r in (getattr(res, "data", []) or []):
                        add_notification(r["user_id"], "info", f"Tarea #{task_id} aprobada y cerrada.", task_id)

                    st.success("Aprobado.")
                    st.rerun()
            with b2:
                if st.button("Rechazar"):
                    decide_approval(int(pend["id"]), "rejected", decision_note or None)
                    update_task_status(task_id, "doing", None, None, None)
                    log_history(task_id, uid, "approval", "pending", "rejected")
                    log_history(task_id, uid, "status", "awaiting_approval", "doing")
                    st.warning("Rechazado. Devuelto a DOING.")
                    st.rerun()
        else:
            st.caption("Solo el dueño (A) puede decidir.")
    else:
        st.caption("No hay aprobaciones pendientes.")

    # Comentarios
    st.divider()
    st.subheader("Avances / notas")

    comm = list_task_comments(task_id, limit=50)
    for c in comm[:15]:
        st.caption(f"{c.get('full_name','(sin nombre)')} • {c['created_at']} • progreso {c['progress_pct'] if c.get('progress_pct') is not None else '—'}%")
        st.write(c["comment"])
        if c.get("next_step"):
            st.caption(f"Próximo paso: {c['next_step']}")

    new_comment = st.text_area("Nuevo comentario", height=80)
    progress = st.slider("Progreso (%)", 0, 100, 0)
    next_step = st.text_input("Próximo paso (opcional)")
    if st.button("Guardar avance"):
        add_task_comment(task_id, uid, new_comment.strip() or "(sin texto)", int(progress), next_step.strip() or None)
        log_history(task_id, uid, "comment", None, "added")
        if A_id and A_id != uid:
            add_notification(A_id, "info", f"Nuevo avance en tarea #{task_id}.", task_id)
        st.success("Guardado.")
        st.rerun()

    # Archivos (Storage)
    st.divider()
    st.subheader("Archivos")

    if has_permission(GROUP_ID, uid, "task_upload_files"):
        up = st.file_uploader("Subir archivo")
        if st.button("Guardar archivo"):
            if up is None:
                st.error("Sube un archivo.")
            else:
                bucket = storage_bucket_name()
                unique = f"task_{task_id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuidlib.uuid4().hex}_{up.name}"
                try:
                    storage_upload_bytes(bucket=bucket, path=unique, data=up.getvalue(), content_type=up.type)
                    add_task_attachment(task_id, uid, up.name, storage_path=unique, storage_bucket=bucket)
                    log_history(task_id, uid, "attachment", None, up.name)
                    if A_id:
                        add_notification(A_id, "info", f"Archivo subido en tarea #{task_id}: {up.name}", task_id)
                    st.success("Archivo guardado.")
                    st.rerun()
                except Exception as e:
                    st.error(f"No se pudo subir a Storage: {e}")
    else:
        st.caption("No tienes permiso para subir archivos.")

    atts = list_task_attachments(task_id, limit=50)
    for a in atts[:20]:
        st.caption(f"{a.get('full_name','(sin nombre)')} • {a['created_at']} • {a['filename']}")
        try:
            url = storage_signed_url(a["storage_bucket"], a["storage_path"], expires_in_seconds=3600)
            if url:
                st.link_button("Descargar", url, use_container_width=False)
        except Exception:
            st.caption("(No disponible descarga directa: falta signed URL)")

    # Editar RACI
    st.divider()
    st.subheader("Editar responsabilidades")

    if has_permission(GROUP_ID, uid, "task_change_raci"):
        members = list_group_members(GROUP_ID)
        opts = {fmt_user(m): m["user_id"] for m in members}

        current = get_task_roles(task_id)

        def default_keys(role):
            keys = []
            for x in current[role]:
                for k, v in opts.items():
                    if v == x["id"]:
                        keys.append(k)
                        break
            return keys

        current_A_id = current["A"][0]["id"] if current["A"] else None
        A_pick_default = None
        if current_A_id:
            for k, v in opts.items():
                if v == current_A_id:
                    A_pick_default = k
                    break

        all_keys = list(opts.keys())
        A_pick = st.selectbox("Dueño (A)", options=all_keys,
                              index=(all_keys.index(A_pick_default) if A_pick_default in all_keys else 0))
        R_pick = st.multiselect("Ejecutores (R)", options=all_keys, default=default_keys("R"))
        C_pick = st.multiselect("Consultados (C)", options=all_keys, default=default_keys("C"))
        I_pick = st.multiselect("Informados (I)", options=all_keys, default=default_keys("I"))

        if st.button("Guardar responsabilidades"):
            try:
                upsert_task_roles(
                    task_id=task_id,
                    user_id=uid,
                    A_id=opts[A_pick],
                    R_ids=[opts[x] for x in R_pick],
                    C_ids=[opts[x] for x in C_pick],
                    I_ids=[opts[x] for x in I_pick]
                )

                add_notification(opts[A_pick], "info", f"Actualización: eres dueño de tarea #{task_id}.", task_id)
                for x in R_pick:
                    add_notification(opts[x], "info", f"Actualización: eres ejecutor de tarea #{task_id}.", task_id)
                for x in C_pick:
                    add_notification(opts[x], "input_request", f"Actualización: te consultan en tarea #{task_id}.", task_id)
                for x in I_pick:
                    add_notification(opts[x], "info", f"Actualización FYI en tarea #{task_id}.", task_id)

                st.success("Actualizado.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    else:
        st.caption("No tienes permiso para editar responsabilidades.")


# =============================================================================
# PLANTILLAS
# =============================================================================
elif menu == "Plantillas":
    uid = st.session_state["user_id"]
    _page_header("📋", "Plantillas", "Reutiliza estructuras de proyecto", "#2F855A")

    if not has_permission(GROUP_ID, uid, "template_manage"):
        st.info("No tienes permiso para gestionar plantillas.")
        st.stop()

    templates = list_templates(GROUP_ID)
    st.dataframe(pd.DataFrame(templates) if templates else pd.DataFrame(), use_container_width=True)

    with st.expander("➕ Crear plantilla"):
        with st.form("tpl"):
            name = st.text_input("Nombre plantilla")
            desc = st.text_area("Descripción", height=70)
            submit = st.form_submit_button("Crear")
        if submit:
            if not name.strip():
                st.error("Falta nombre.")
            else:
                create_template(GROUP_ID, name.strip(), desc.strip() or None, uid)
                st.success("Plantilla creada.")
                st.rerun()

    if templates:
        tpl = safe_selectbox_dict("Plantilla", templates, format_func=lambda x: f"[{x['id']}] {x['name']}")
        tpl_id = int(tpl["id"])

        st.markdown("#### Tareas dentro de plantilla")
        tpls = list_template_tasks(tpl_id)
        st.dataframe(pd.DataFrame(tpls) if tpls else pd.DataFrame(), use_container_width=True)

        st.markdown("#### Agregar tarea a plantilla")
        with st.form("tpl_task"):
            tt = st.text_input("Título")
            td = st.text_area("Descripción", height=60)
            dod = st.text_area("DoD", height=60)
            priority = st.selectbox("Prioridad", ["urgente", "alta", "media", "baja"], index=2)
            req = st.checkbox("Requiere aprobación", value=True)
            days_from = st.number_input("Días desde inicio del proyecto (para target/due)", min_value=0, max_value=365, value=0, step=1)
            tags_csv = st.text_input("Tags (separados por coma)", value="finanzas,operaciones")
            component = st.text_input("Componente (opcional)", value="")
            submit = st.form_submit_button("Agregar a plantilla")
        if submit:
            if not (tt.strip() and dod.strip()):
                st.error("Título y DoD son obligatorios.")
            else:
                add_template_task(
                    template_id=tpl_id,
                    title=tt.strip(),
                    description=td.strip() or None,
                    dod=dod.strip(),
                    priority=priority,
                    requires_approval=req,
                    days_from_start=int(days_from),
                    tags_csv=tags_csv.strip() or None,
                    component_name=component.strip() or None,
                )
                st.success("Agregado.")
                st.rerun()

        st.divider()
        st.subheader("Aplicar plantilla a un proyecto")

        projs = list_projects(GROUP_ID)
        if not projs:
            st.info("Crea un proyecto primero.")
            st.stop()

        proj = safe_selectbox_dict("Proyecto destino", projs, format_func=lambda x: f"[{x['id']}] {x['name']}")
        start = st.date_input("Fecha inicio base para plan (start_date)", value=date.today())

        if st.button("Aplicar"):
            tasks_tpl = list_template_tasks(tpl_id)
            for tt in tasks_tpl:
                target = start + timedelta(days=int(tt["days_from_start"]))
                due = start + timedelta(days=int(tt["days_from_start"]) + 2)

                task_id = create_task(
                    project_id=int(proj["id"]),
                    title=tt["title"],
                    description=tt.get("description"),
                    dod=tt["dod"],
                    priority=tt["priority"],
                    status="todo",
                    start_date=start,
                    target_date=target,
                    due_date=due,
                    requires_approval=bool(tt["requires_approval"]),
                    created_by=uid
                )

                # tags/component por CSV
                tag_ids = []
                if tt.get("tags_csv"):
                    for name in [x.strip() for x in tt["tags_csv"].split(",") if x.strip()]:
                        r = getattr(supa_table("tags").select("id").eq("group_id", GROUP_ID).eq("name", name).eq("kind", "tag").limit(1).execute(), "data", []) or []
                        tid = int(r[0]["id"]) if r else int(getattr(supa_table("tags").insert({"group_id": GROUP_ID, "name": name, "kind": "tag"}).execute(), "data", [{}])[0]["id"])
                        tag_ids.append(tid)

                if tt.get("component_name"):
                    cn = tt["component_name"]
                    r = getattr(supa_table("tags").select("id").eq("group_id", GROUP_ID).eq("name", cn).eq("kind", "component").limit(1).execute(), "data", []) or []
                    tid = int(r[0]["id"]) if r else int(getattr(supa_table("tags").insert({"group_id": GROUP_ID, "name": cn, "kind": "component"}).execute(), "data", [{}])[0]["id"])
                    tag_ids.append(tid)

                supa_table("task_tags").delete().eq("task_id", int(task_id)).execute()
                if tag_ids:
                    supa_table("task_tags").insert([{"task_id": int(task_id), "tag_id": int(tid)} for tid in sorted(set(tag_ids))]).execute()

            st.success("Plantilla aplicada. Ahora asigna responsabilidades a cada tarea.")
            st.rerun()


# =============================================================================
# EXPORT
# =============================================================================
elif menu == "Export":
    uid = st.session_state["user_id"]
    _page_header("📤", "Exportar datos", "CSV, Excel e ICS para calendario", "#2B6CB0")

    if not has_permission(GROUP_ID, uid, "export_data"):
        st.info("No tienes permiso de export.")
        st.stop()

    projs = list_projects(GROUP_ID)
    if not projs:
        st.info("No hay proyectos.")
        st.stop()

    proj = safe_selectbox_dict("Proyecto", projs, format_func=lambda x: f"[{x['id']}] {x['name']}")
    pid = int(proj["id"])

    rows = list_tasks_by_project(pid, limit=800)
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    st.dataframe(df, use_container_width=True)

    st.download_button("Descargar CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=f"tasks_project_{pid}.csv")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _excel_safe(df).to_excel(writer, index=False, sheet_name="Tareas")
    st.download_button("Descargar Excel", data=buf.getvalue(), file_name=f"tareas_proyecto_{pid}.xlsx")

    events = []
    for r in rows:
        events.append(task_to_ics_event(r["id"], r["title"], r.get("due_date"), r.get("dod")))
    ics = tasks_to_ics_calendar(events, cal_name=f"Proyecto {proj['name']}")
    st.download_button("Descargar ICS", data=ics.encode("utf-8"), file_name=f"tasks_project_{pid}.ics")


# =============================================================================
# GOBERNANZA
# =============================================================================
elif menu == "Gobernanza":
    uid = st.session_state["user_id"]
    _page_header("⚖️", "Gobernanza", "Permisos y reglas del grupo", "#744210")

    role = user_role_in_group(GROUP_ID, uid)
    if role != "leader":
        st.info("Solo líderes del grupo pueden gestionar gobernanza.")
        st.stop()

    st.markdown("### Permisos finos por rol (acciones)")
    perms = list_group_role_permissions(GROUP_ID)
    dfp = pd.DataFrame(perms)
    st.dataframe(dfp, use_container_width=True)

    st.markdown("#### Editar permisos (cuidado: esto afecta la gobernanza)")
    action = st.selectbox("Acción", options=sorted(dfp["action"].unique().tolist()) if not dfp.empty else [])
    role_pick = st.selectbox("Rol", options=["leader", "member"])
    allowed = st.checkbox("Permitido", value=True)
    if st.button("Guardar permiso"):
        set_group_role_permission(GROUP_ID, role_pick, action, allowed)
        st.success("Actualizado.")
        st.rerun()

    st.divider()
    st.markdown("### Reglas recomendadas")
    st.write("- Toda tarea debe tener DoD (criterio de aceptación).")
    st.write("- Si requiere aprobación: no se cierra sin decisión del Dueño.")
    st.write("- Blocked requiere motivo y dueño de desbloqueo.")
    st.write("- Limitar WIP para evitar sobrecarga y multitarea.")


# =============================================================================
# REPORTES
# =============================================================================
elif menu == "Reportes":
    _page_header("📈", "Reportes", "Reporte semanal de tareas completadas y pendientes", "#1A365D")

    pmap = _project_map(GROUP_ID)
    projs = [{"id": None, "name": "Todos"}] + [{"id": pid, "name": p["name"]} for pid, p in pmap.items()]

    proj_filter = safe_selectbox_dict("Proyecto (opcional)", projs, format_func=lambda x: x["name"], key="rep_proj")
    pid = int(proj_filter["id"]) if proj_filter and proj_filter["id"] else None

    ref = st.date_input("Semana de referencia", value=date.today())
    ws, we = week_bounds(ref)
    st.caption(f"Semana: {ws.isoformat()} → {we.isoformat()}")

    df_done, df_pending = build_weekly_report(GROUP_ID, ws, we, project_id=pid)

    st.markdown("### ✅ Completadas")
    st.dataframe(df_done, use_container_width=True, height=280)

    st.markdown("### ⏳ Pendientes")
    st.dataframe(df_pending, use_container_width=True, height=280)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _excel_safe(df_done).to_excel(writer, index=False, sheet_name="Completadas")
        _excel_safe(df_pending).to_excel(writer, index=False, sheet_name="Pendientes")
    st.download_button("Descargar Excel del reporte", data=buf.getvalue(), file_name=f"reporte_semanal_{ws.isoformat()}.xlsx")


# =============================================================================
# AYUDA
# =============================================================================
elif menu == "Ayuda":
    _page_header("❓", "Ayuda", "Guía rápida de uso del sistema", "#553C9A")
    st.markdown("""
### Objetivo
Esta herramienta organiza proyectos y tareas por grupo, con control de responsabilidades, trazabilidad y gobernanza.

---

## Glosario

RACI:
- R: ejecuta
- A: dueño final / aprueba
- C: consultado
- I: informado

DoD: Definition of Done. Criterio de aceptación.  
WIP: Límite de tareas en DOING por persona.  
SLA: objetivo de tiempo (target_date).  
Aging: días desde creación de tarea.  
ICS: export a calendario.

---

## Reglas
- 1 solo Dueño (A) por tarea.
- mínimo 1 Ejecutor (R).
- Si hay dependencias pendientes, no se deja cerrar/avanzar.
- Si requiere aprobación, no se cierra sin decisión del Dueño.
""")


# =============================================================================
# Sidebar: notificaciones quick view
# =============================================================================
if logged_in():
    uid_notif = st.session_state["user_id"]
    unread = unread_notifications_count(uid_notif)
    label = f"🔔 Notificaciones ({unread} sin leer)" if unread > 0 else "🔔 Notificaciones"

    if "show_notifs" not in st.session_state:
        st.session_state["show_notifs"] = False

    if st.sidebar.button(label):
        st.session_state["show_notifs"] = not st.session_state["show_notifs"]

    if st.session_state.get("show_notifs"):
        rows = list_notifications(uid_notif, limit=20)
        st.sidebar.markdown("### Últimas notificaciones")
        if not rows:
            st.sidebar.info("Sin notificaciones.")
        for r in rows:
            icon = "📭" if r.get("read_at") else "📬"
            st.sidebar.caption(f"{icon} {r['created_at']} • {r['kind']}")
            st.sidebar.write(r["message"])
        if st.sidebar.button("Marcar todas como leídas", key="btn_mark_read"):
            mark_notifications_read(uid_notif)
            st.session_state["show_notifs"] = False
            st.rerun()
