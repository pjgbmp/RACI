# db.py (RUTA 1 - Supabase, con user_profiles)
from __future__ import annotations
import os
from datetime import datetime, date, timezone
from typing import Any

import streamlit as st
from st_supabase_connection import SupabaseConnection


# ----------------------------
# Config / Constantes
# ----------------------------
ACTIONS = [
    "group_manage_members",
    "group_transfer_lead",
    "project_create",
    "project_edit",
    "task_create",
    "task_edit",
    "task_change_status",
    "task_change_due_date",
    "task_change_raci",          # UI lo llama "Responsabilidades"
    "task_upload_files",
    "template_manage",
    "export_data",
]

DEFAULT_ROLE_PERMS = {
    "leader": set(ACTIONS),
    "member": {
        "task_change_status",
        "task_upload_files",
        "export_data",
    },
}


def now_iso() -> str:
    # ISO UTC con Z (ideal para timestamptz)
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@st.cache_resource
def get_conn() -> SupabaseConnection:
    cfg = st.secrets.get("connections", {}).get("supabase", {})
    url = cfg.get("url") or os.getenv("SUPABASE_URL")
    key = cfg.get("key") or os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("Faltan Supabase url/key en secrets.toml o variables de entorno.")

    return st.connection("supabase", type=SupabaseConnection, url=url, key=key)


def _single(res) -> dict | None:
    data = getattr(res, "data", None)
    if not data:
        return None
    return data[0]


def _many(res) -> list[dict]:
    return getattr(res, "data", []) or []


def _count(res) -> int:
    c = getattr(res, "count", None)
    return int(c or 0)


# ----------------------------
# user_profiles
# ----------------------------
def upsert_user_profile(
    user_id: str,
    full_name: str,
    username: str | None = None,
    email: str | None = None,
    is_active: bool = True,
) -> None:
    conn = get_conn()
    # No mandes created_at en upsert: si ya existe, no queremos pisarlo
    conn.table("user_profiles").upsert(
        {
            "user_id": user_id,
            "full_name": full_name.strip(),
            "username": (username.strip() if username else None),
            "email": (email.strip() if email else None),
            "is_active": bool(is_active),
        },
        on_conflict="user_id",
    ).execute()


def get_user_profile(user_id: str) -> dict | None:
    conn = get_conn()
    return _single(conn.table("user_profiles").select("*").eq("user_id", user_id).limit(1).execute())


def get_user_profile_by_username(username: str) -> dict | None:
    conn = get_conn()
    return _single(conn.table("user_profiles").select("*").eq("username", username).limit(1).execute())


def fetch_profiles(user_ids: list[str]) -> dict[str, dict]:
    """
    Devuelve dict: {user_id: profile_dict}
    """
    if not user_ids:
        return {}
    conn = get_conn()
    rows = _many(
        conn.table("user_profiles")
        .select("user_id,full_name,username,email,is_active")
        .in_("user_id", list(set(user_ids)))
        .execute()
    )
    return {r["user_id"]: r for r in rows}


# ----------------------------
# Grupos / Miembros / Permisos
# ----------------------------
def list_my_groups(user_id: str) -> list[dict]:
    conn = get_conn()

    memberships = _many(
        conn.table("group_members")
        .select("group_id,role,joined_at")
        .eq("user_id", user_id)
        .execute()
    )
    if not memberships:
        return []

    group_ids = [m["group_id"] for m in memberships]
    role_by_gid = {m["group_id"]: m["role"] for m in memberships}

    groups = _many(conn.table("groups").select("*").in_("id", group_ids).order("id", desc=True).execute())
    for g in groups:
        g["my_role"] = role_by_gid.get(g["id"])
    return groups


def get_group(group_id: int) -> dict | None:
    conn = get_conn()
    return _single(conn.table("groups").select("*").eq("id", group_id).limit(1).execute())


def get_group_by_join_code(join_code: str) -> dict | None:
    conn = get_conn()
    return _single(conn.table("groups").select("*").eq("join_code", join_code).limit(1).execute())


def create_group(name: str, created_by: str, join_code: str) -> int:
    conn = get_conn()
    row = _single(
        conn.table("groups")
        .insert(
            {
                "name": name.strip(),
                "join_code": join_code.strip(),
                "created_by": created_by,
                "created_at": now_iso(),
            }
        )
        .execute()
    )
    return int(row["id"])


def add_group_member(group_id: int, user_id: str, role: str = "member") -> None:
    conn = get_conn()
    conn.table("group_members").upsert(
        {
            "group_id": int(group_id),
            "user_id": user_id,
            "role": role,
            "joined_at": now_iso(),
        },
        on_conflict="group_id,user_id",
    ).execute()


def remove_group_member(group_id: int, user_id: str) -> None:
    conn = get_conn()
    conn.table("group_members").delete().eq("group_id", int(group_id)).eq("user_id", user_id).execute()


def list_group_members(group_id: int) -> list[dict]:
    """
    Devuelve lista de miembros con perfil:
      - id/user_id (uuid)
      - full_name, username, email
      - role, joined_at
    """
    conn = get_conn()
    gm = _many(conn.table("group_members").select("user_id,role,joined_at").eq("group_id", int(group_id)).execute())
    if not gm:
        return []

    uids = [x["user_id"] for x in gm]
    prof_by = fetch_profiles(uids)

    out: list[dict] = []
    for m in gm:
        p = prof_by.get(m["user_id"], {})
        out.append(
            {
                "id": m["user_id"],
                "user_id": m["user_id"],
                "full_name": p.get("full_name", "(sin nombre)"),
                "username": p.get("username", "") or "",
                "email": p.get("email"),
                "is_active": p.get("is_active", True),
                "role": m["role"],
                "joined_at": m["joined_at"],
            }
        )

    out.sort(key=lambda x: (0 if x["role"] == "leader" else 1, (x["full_name"] or "").lower()))
    return out


def seed_group_permissions(group_id: int) -> None:
    """
    Inserta defaults.
    """
    conn = get_conn()
    rows = []
    for role, actions in DEFAULT_ROLE_PERMS.items():
        for action in ACTIONS:
            allowed = True if action in actions else False
            rows.append(
                {
                    "group_id": int(group_id),
                    "role": role,
                    "action": action,
                    "allowed": allowed,
                }
            )
    conn.table("group_role_permissions").upsert(rows, on_conflict="group_id,role,action").execute()


def user_role_in_group(group_id: int, user_id: str) -> str | None:
    conn = get_conn()
    r = _single(
        conn.table("group_members")
        .select("role")
        .eq("group_id", int(group_id))
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    return r["role"] if r else None


def has_permission(group_id: int, user_id: str, action: str) -> bool:
    role = user_role_in_group(group_id, user_id)
    if not role:
        return False
    conn = get_conn()
    r = _single(
        conn.table("group_role_permissions")
        .select("allowed")
        .eq("group_id", int(group_id))
        .eq("role", role)
        .eq("action", action)
        .limit(1)
        .execute()
    )
    return bool(r["allowed"]) if r else False


def set_group_role_permission(group_id: int, role: str, action: str, allowed: bool) -> None:
    conn = get_conn()
    conn.table("group_role_permissions").upsert(
        {
            "group_id": int(group_id),
            "role": role,
            "action": action,
            "allowed": bool(allowed),
        },
        on_conflict="group_id,role,action",
    ).execute()


def list_group_role_permissions(group_id: int) -> list[dict]:
    conn = get_conn()
    return _many(
        conn.table("group_role_permissions")
        .select("group_id,role,action,allowed")
        .eq("group_id", int(group_id))
        .order("role", desc=False)
        .order("action", desc=False)
        .execute()
    )


def transfer_group_leadership(group_id: int, from_user_id: str, to_user_id: str) -> None:
    conn = get_conn()
    conn.table("group_members").update({"role": "member"}).eq("group_id", int(group_id)).eq("user_id", from_user_id).execute()
    conn.table("group_members").update({"role": "leader"}).eq("group_id", int(group_id)).eq("user_id", to_user_id).execute()


# ----------------------------
# Proyectos
# ----------------------------
def list_projects(group_id: int) -> list[dict]:
    conn = get_conn()
    return _many(conn.table("projects").select("*").eq("group_id", int(group_id)).order("id", desc=True).execute())


def get_project(project_id: int) -> dict | None:
    conn = get_conn()
    return _single(conn.table("projects").select("*").eq("id", int(project_id)).limit(1).execute())


def create_project(group_id: int, name: str, description: str | None, wip_limit_doing: int, created_by: str) -> int:
    conn = get_conn()
    row = _single(
        conn.table("projects")
        .insert(
            {
                "group_id": int(group_id),
                "name": name.strip(),
                "description": (description.strip() if description else None),
                "wip_limit_doing": int(wip_limit_doing),
                "created_by": created_by,
                "created_at": now_iso(),
            }
        )
        .execute()
    )
    return int(row["id"])


def ensure_project_member(project_id: int, user_id: str) -> None:
    conn = get_conn()
    conn.table("project_members").upsert({"project_id": int(project_id), "user_id": user_id}, on_conflict="project_id,user_id").execute()


def remove_project_member(project_id: int, user_id: str) -> None:
    conn = get_conn()
    conn.table("project_members").delete().eq("project_id", int(project_id)).eq("user_id", user_id).execute()


def list_project_members(project_id: int) -> list[dict]:
    conn = get_conn()
    pm = _many(conn.table("project_members").select("user_id").eq("project_id", int(project_id)).execute())
    if not pm:
        return []
    uids = [x["user_id"] for x in pm]
    prof_by = fetch_profiles(uids)
    out = [
        {
            "id": uid,
            "user_id": uid,
            "full_name": prof_by.get(uid, {}).get("full_name", "(sin nombre)"),
            "username": prof_by.get(uid, {}).get("username", "") or "",
        }
        for uid in uids
    ]
    out.sort(key=lambda x: (x["full_name"] or "").lower())
    return out


# ----------------------------
# Tags / Componentes
# ----------------------------
def list_tags(group_id: int, kind: str | None = None) -> list[dict]:
    conn = get_conn()
    q = conn.table("tags").select("*").eq("group_id", int(group_id))
    if kind:
        q = q.eq("kind", kind)
    return _many(q.order("kind", desc=False).order("name", desc=False).execute())


def get_or_create_tag(group_id: int, name: str, kind: str) -> int:
    conn = get_conn()
    r = _single(conn.table("tags").select("id").eq("group_id", int(group_id)).eq("name", name).eq("kind", kind).limit(1).execute())
    if r:
        return int(r["id"])
    row = _single(conn.table("tags").insert({"group_id": int(group_id), "name": name, "kind": kind}).execute())
    return int(row["id"])


def set_task_tags(task_id: int, tag_ids: list[int]) -> None:
    conn = get_conn()
    conn.table("task_tags").delete().eq("task_id", int(task_id)).execute()
    if tag_ids:
        rows = [{"task_id": int(task_id), "tag_id": int(tid)} for tid in sorted(set(tag_ids))]
        conn.table("task_tags").upsert(rows, on_conflict="task_id,tag_id").execute()


def list_task_tags(task_id: int) -> list[dict]:
    conn = get_conn()
    links = _many(conn.table("task_tags").select("tag_id").eq("task_id", int(task_id)).execute())
    if not links:
        return []
    tag_ids = [x["tag_id"] for x in links]
    tags = _many(conn.table("tags").select("*").in_("id", tag_ids).execute())
    tags.sort(key=lambda x: (x.get("kind", ""), x.get("name", "")))
    return tags


# ----------------------------
# Tareas
# ----------------------------
def create_task(
    project_id: int,
    title: str,
    description: str | None,
    dod: str,
    priority: str,
    status: str,
    start_date: date | None,
    target_date: date | None,
    due_date: date | None,
    requires_approval: bool,
    created_by: str,
) -> int:
    conn = get_conn()
    row = _single(
        conn.table("tasks")
        .insert(
            {
                "project_id": int(project_id),
                "title": title.strip(),
                "description": (description.strip() if description else None),
                "dod": dod.strip(),
                "priority": priority,
                "status": status,
                "start_date": start_date.isoformat() if start_date else None,
                "target_date": target_date.isoformat() if target_date else None,
                "due_date": due_date.isoformat() if due_date else None,
                "requires_approval": bool(requires_approval),
                "created_by": created_by,
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "completed_at": None,
                "blocked_reason": None,
                "unblock_owner_user_id": None,
            }
        )
        .execute()
    )
    return int(row["id"])


def get_task(task_id: int) -> dict | None:
    conn = get_conn()
    return _single(conn.table("tasks").select("*").eq("id", int(task_id)).limit(1).execute())


def list_tasks_by_project(project_id: int, limit: int = 300) -> list[dict]:
    conn = get_conn()
    return _many(
        conn.table("tasks")
        .select("id,title,priority,status,due_date,created_at,updated_at,completed_at,requires_approval,project_id,blocked_reason,unblock_owner_user_id,target_date,start_date")
        .eq("project_id", int(project_id))
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )


def list_tasks_for_board(
    group_id: int,
    project_id: int,
    statuses: list[str],
    priorities: list[str],
    q: str | None,
    only_overdue: bool,
    only_blocked: bool,
    only_mine_user_id: str | None,
    tag_ids_filter: list[int] | None,
) -> list[dict]:
    conn = get_conn()
    base = (
        conn.table("tasks")
        .select("*")
        .eq("project_id", int(project_id))
        .in_("status", statuses)
        .in_("priority", priorities)
    )

    if only_blocked:
        base = base.eq("status", "blocked")

    rows = _many(base.order("id", desc=False).execute())

    if q and q.strip():
        s = q.strip().lower()

        def hit(t: dict) -> bool:
            return (
                s in (t.get("title") or "").lower()
                or s in (t.get("description") or "").lower()
                or s in (t.get("dod") or "").lower()
            )

        rows = [t for t in rows if hit(t)]

    if only_mine_user_id:
        mine = _many(
            conn.table("task_roles")
            .select("task_id,role")
            .eq("user_id", only_mine_user_id)
            .in_("role", ["R", "A"])
            .execute()
        )
        mine_ids = {x["task_id"] for x in mine}
        rows = [t for t in rows if t["id"] in mine_ids]

    if tag_ids_filter:
        links = _many(
            conn.table("task_tags")
            .select("task_id,tag_id")
            .in_("tag_id", [int(x) for x in tag_ids_filter])
            .execute()
        )
        allowed_task_ids = {x["task_id"] for x in links}
        rows = [t for t in rows if t["id"] in allowed_task_ids]

    if only_overdue:
        today = date.today()

        def is_overdue(t: dict) -> bool:
            if t.get("status") == "done":
                return False
            d = t.get("due_date")
            if not d:
                return False
            try:
                return date.fromisoformat(d) < today
            except Exception:
                return False

        rows = [t for t in rows if is_overdue(t)]

    pr_rank = {"urgente": 1, "alta": 2, "media": 3, "baja": 4}

    def keyf(t: dict):
        due = t.get("due_date") or "9999-12-31"
        return (pr_rank.get(t.get("priority"), 99), due)

    rows.sort(key=keyf)
    return rows


def update_task_due_date(task_id: int, due_date_new: date | None) -> None:
    conn = get_conn()
    conn.table("tasks").update({"due_date": due_date_new.isoformat() if due_date_new else None, "updated_at": now_iso()}).eq("id", int(task_id)).execute()


def update_task_status(
    task_id: int,
    new_status: str,
    blocked_reason: str | None,
    unblock_owner_user_id: str | None,
    completed_at_iso: str | None,
) -> None:
    conn = get_conn()
    payload: dict[str, Any] = {
        "status": new_status,
        "updated_at": now_iso(),
        "blocked_reason": (blocked_reason if new_status == "blocked" else None),
        "unblock_owner_user_id": (unblock_owner_user_id if new_status == "blocked" else None),
    }
    if completed_at_iso:
        payload["completed_at"] = completed_at_iso
    conn.table("tasks").update(payload).eq("id", int(task_id)).execute()


# ----------------------------
# RACI (task_roles)
# ----------------------------
def get_task_roles(task_id: int) -> dict[str, list[dict]]:
    conn = get_conn()
    rows = _many(conn.table("task_roles").select("role,user_id").eq("task_id", int(task_id)).execute())
    out = {"A": [], "R": [], "C": [], "I": []}
    if not rows:
        return out
    uids = [r["user_id"] for r in rows]
    prof_by = fetch_profiles(uids)

    rows.sort(key=lambda r: (r["role"], (prof_by.get(r["user_id"], {}).get("full_name", "") or "").lower()))
    for r in rows:
        p = prof_by.get(r["user_id"], {})
        out[r["role"]].append({"id": r["user_id"], "name": p.get("full_name", "(sin nombre)")})
    return out


def get_accountable_id(task_id: int) -> str | None:
    conn = get_conn()
    r = _single(conn.table("task_roles").select("user_id").eq("task_id", int(task_id)).eq("role", "A").limit(1).execute())
    return r["user_id"] if r else None


def is_responsible(task_id: int, user_id: str) -> bool:
    conn = get_conn()
    r = _single(conn.table("task_roles").select("task_id").eq("task_id", int(task_id)).eq("role", "R").eq("user_id", user_id).limit(1).execute())
    return r is not None


def upsert_task_roles(task_id: int, actor_id: str, A_id: str, R_ids: list[str], C_ids: list[str], I_ids: list[str]) -> None:
    if not A_id:
        raise ValueError("Debe existir 1 dueño (A) por tarea.")
    if not R_ids:
        raise ValueError("Debe existir al menos 1 ejecutor (R).")

    conn = get_conn()

    before = get_task_roles(task_id)

    conn.table("task_roles").delete().eq("task_id", int(task_id)).execute()

    ts = now_iso()
    rows = []
    rows.append({"task_id": int(task_id), "user_id": A_id, "role": "A", "assigned_at": ts})
    for uid in sorted(set(R_ids)):
        rows.append({"task_id": int(task_id), "user_id": uid, "role": "R", "assigned_at": ts})
    for uid in sorted(set(C_ids)):
        rows.append({"task_id": int(task_id), "user_id": uid, "role": "C", "assigned_at": ts})
    for uid in sorted(set(I_ids)):
        rows.append({"task_id": int(task_id), "user_id": uid, "role": "I", "assigned_at": ts})

    conn.table("task_roles").upsert(rows, on_conflict="task_id,user_id,role").execute()

    after = get_task_roles(task_id)
    log_history(task_id, actor_id, "responsibilities", str(before), str(after))


# ----------------------------
# Dependencias
# ----------------------------
def list_task_dependencies_open(task_id: int) -> list[dict]:
    conn = get_conn()
    deps = _many(conn.table("task_dependencies").select("depends_on_task_id").eq("task_id", int(task_id)).execute())
    if not deps:
        return []
    dep_ids = [d["depends_on_task_id"] for d in deps]

    tasks = _many(conn.table("tasks").select("id,title,status").in_("id", dep_ids).execute())
    out = [t for t in tasks if t.get("status") != "done"]
    out.sort(key=lambda x: int(x["id"]))
    return out


def set_task_dependencies(task_id: int, depends_on_task_ids: list[int]) -> None:
    conn = get_conn()
    conn.table("task_dependencies").delete().eq("task_id", int(task_id)).execute()
    rows = [{"task_id": int(task_id), "depends_on_task_id": int(d)} for d in sorted(set(depends_on_task_ids)) if int(d) != int(task_id)]
    if rows:
        conn.table("task_dependencies").upsert(rows, on_conflict="task_id,depends_on_task_id").execute()


# ----------------------------
# Comentarios
# ----------------------------
def list_task_comments(task_id: int, limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = _many(
        conn.table("task_comments")
        .select("*")
        .eq("task_id", int(task_id))
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    if not rows:
        return []
    uids = [r["user_id"] for r in rows]
    prof_by = fetch_profiles(uids)
    for r in rows:
        r["full_name"] = prof_by.get(r["user_id"], {}).get("full_name", "(sin nombre)")
    return rows


def add_task_comment(task_id: int, user_id: str, comment: str, progress_pct: int | None, next_step: str | None) -> int:
    conn = get_conn()
    row = _single(
        conn.table("task_comments")
        .insert(
            {
                "task_id": int(task_id),
                "user_id": user_id,
                "comment": comment,
                "progress_pct": progress_pct,
                "next_step": next_step,
                "created_at": now_iso(),
            }
        )
        .execute()
    )
    return int(row["id"])


# ----------------------------
# Adjuntos (metadata) - archivo real va a Storage
# ----------------------------
def add_task_attachment(task_id: int, user_id: str, filename: str, storage_path: str, storage_bucket: str | None = None) -> int:
    conn = get_conn()
    bucket = storage_bucket or st.secrets["connections"]["supabase"].get("SUPABASE_STORAGE_BUCKET", "attachments")
    row = _single(
        conn.table("task_attachments")
        .insert(
            {
                "task_id": int(task_id),
                "user_id": user_id,
                "filename": filename,
                "storage_bucket": bucket,
                "storage_path": storage_path,
                "created_at": now_iso(),
            }
        )
        .execute()
    )
    return int(row["id"])


def list_task_attachments(task_id: int, limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = _many(
        conn.table("task_attachments")
        .select("*")
        .eq("task_id", int(task_id))
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    if not rows:
        return []
    uids = [r["user_id"] for r in rows]
    prof_by = fetch_profiles(uids)
    for r in rows:
        r["full_name"] = prof_by.get(r["user_id"], {}).get("full_name", "(sin nombre)")
    return rows


# ----------------------------
# Auditoría / History
# ----------------------------
def log_history(task_id: int, user_id: str, field: str, old_v: str | None, new_v: str | None) -> None:
    conn = get_conn()
    conn.table("task_history").insert(
        {
            "task_id": int(task_id),
            "user_id": user_id,
            "field": field,
            "old_value": old_v,
            "new_value": new_v,
            "created_at": now_iso(),
        }
    ).execute()


def list_task_history(task_id: int, limit: int = 100) -> list[dict]:
    conn = get_conn()
    rows = _many(
        conn.table("task_history")
        .select("*")
        .eq("task_id", int(task_id))
        .order("id", desc=True)
        .limit(limit)
        .execute()
    )
    if not rows:
        return []
    uids = [r["user_id"] for r in rows]
    prof_by = fetch_profiles(uids)
    for r in rows:
        r["full_name"] = prof_by.get(r["user_id"], {}).get("full_name", "(sin nombre)")
    return rows


# ----------------------------
# Aprobaciones
# ----------------------------
def create_approval_request(task_id: int, requested_by: str, accountable_user_id: str, request_note: str | None = None) -> int:
    conn = get_conn()
    row = _single(
        conn.table("approval_requests")
        .insert(
            {
                "task_id": int(task_id),
                "requested_by": requested_by,
                "accountable_user_id": accountable_user_id,
                "status": "pending",
                "request_note": request_note,
                "decision_note": None,
                "requested_at": now_iso(),
                "decided_at": None,
            }
        )
        .execute()
    )
    return int(row["id"])


def get_latest_approval(task_id: int) -> dict | None:
    conn = get_conn()
    return _single(conn.table("approval_requests").select("*").eq("task_id", int(task_id)).order("id", desc=True).limit(1).execute())


def get_pending_approval(task_id: int) -> dict | None:
    conn = get_conn()
    return _single(
        conn.table("approval_requests")
        .select("*")
        .eq("task_id", int(task_id))
        .eq("status", "pending")
        .order("id", desc=True)
        .limit(1)
        .execute()
    )


def decide_approval(approval_id: int, decision: str, decision_note: str | None) -> None:
    if decision not in ("approved", "rejected"):
        raise ValueError("decision debe ser 'approved' o 'rejected'")
    conn = get_conn()
    conn.table("approval_requests").update({"status": decision, "decision_note": decision_note, "decided_at": now_iso()}).eq("id", int(approval_id)).execute()


def list_approvals_by_group(group_id: int, project_id: int | None = None) -> list[dict]:
    conn = get_conn()
    projects = list_projects(group_id)
    if not projects:
        return []
    proj_ids = [int(project_id)] if project_id is not None else [p["id"] for p in projects]

    tasks = _many(conn.table("tasks").select("id,project_id").in_("project_id", proj_ids).execute())
    if not tasks:
        return []
    task_ids = [t["id"] for t in tasks]

    appr = _many(conn.table("approval_requests").select("*").in_("task_id", task_ids).execute())
    if not appr:
        return []

    uids = list({a["accountable_user_id"] for a in appr if a.get("accountable_user_id")})
    prof_by = fetch_profiles(uids)
    for a in appr:
        a["full_name"] = prof_by.get(a.get("accountable_user_id", ""), {}).get("full_name", "(sin nombre)")
    return appr


# ----------------------------
# Notificaciones
# ----------------------------
def _notify_via_rpc(actor_id: str, target_user_id: str, kind: str, message: str, task_id: int | None) -> int:
    """
    Usa una función RPC (notify_user) para insertar notificaciones a terceros con RLS activo.
    Requiere que crees el RPC en Supabase (te dejo el SQL abajo).
    """
    conn = get_conn()
    res = conn.rpc(
        "notify_user",
        {
            "p_actor_id": actor_id,
            "p_target_user_id": target_user_id,
            "p_kind": kind,
            "p_message": message,
            "p_task_id": int(task_id) if task_id is not None else None,
        },
    ).execute()
    # La función devuelve bigint id
    data = getattr(res, "data", None)
    if isinstance(data, list) and data:
        return int(data[0])
    if isinstance(data, int):
        return int(data)
    # fallback raro
    return int(data or 0)


def add_notification(user_id: str, kind: str, message: str, task_id: int | None, actor_id: str | None = None) -> int:
    """
    - Si estás usando SERVICE ROLE KEY, el insert directo funciona para cualquiera.
    - Si estás usando anon key con RLS, el insert a terceros puede fallar:
        -> en ese caso intentamos RPC notify_user().
    """
    conn = get_conn()

    try:
        row = _single(
            conn.table("notifications")
            .insert(
                {
                    "user_id": user_id,
                    "kind": kind,
                    "message": message,
                    "task_id": int(task_id) if task_id is not None else None,
                    "created_at": now_iso(),
                    "read_at": None,
                }
            )
            .execute()
        )
        return int(row["id"])
    except Exception:
        # Si falla por RLS, y nos dieron actor_id, probamos RPC
        if not actor_id:
            raise
        return _notify_via_rpc(actor_id, user_id, kind, message, task_id)


def unread_notifications_count(user_id: str) -> int:
    conn = get_conn()
    res = conn.table("notifications").select("id", count="exact").eq("user_id", user_id).is_("read_at", None).execute()
    return _count(res)


def list_notifications(user_id: str, limit: int = 30) -> list[dict]:
    conn = get_conn()
    return _many(conn.table("notifications").select("*").eq("user_id", user_id).order("id", desc=True).limit(limit).execute())


def mark_notifications_read(user_id: str, notification_ids: list[int]) -> None:
    if not notification_ids:
        return
    conn = get_conn()
    conn.table("notifications").update({"read_at": now_iso()}).eq("user_id", user_id).in_("id", [int(x) for x in notification_ids]).execute()


# ----------------------------
# Outbound queue
# ----------------------------
def enqueue_outbound(channel: str, payload: str, to_address: str | None = None) -> int:
    if channel not in ("email", "teams"):
        raise ValueError("channel debe ser 'email' o 'teams'")
    conn = get_conn()
    row = _single(
        conn.table("outbound_queue")
        .insert(
            {
                "channel": channel,
                "to_address": to_address,
                "payload": payload,
                "status": "pending",
                "created_at": now_iso(),
                "last_error": None,
            }
        )
        .execute()
    )
    return int(row["id"])


# ----------------------------
# Plantillas
# ----------------------------
def list_templates(group_id: int) -> list[dict]:
    conn = get_conn()
    return _many(conn.table("templates").select("*").eq("group_id", int(group_id)).order("id", desc=True).execute())


def create_template(group_id: int, name: str, description: str | None, created_by: str) -> int:
    conn = get_conn()
    row = _single(
        conn.table("templates")
        .insert(
            {
                "group_id": int(group_id),
                "name": name.strip(),
                "description": (description.strip() if description else None),
                "created_by": created_by,
                "created_at": now_iso(),
            }
        )
        .execute()
    )
    return int(row["id"])


def list_template_tasks(template_id: int) -> list[dict]:
    conn = get_conn()
    return _many(conn.table("template_tasks").select("*").eq("template_id", int(template_id)).order("id", desc=False).execute())


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
) -> int:
    conn = get_conn()
    row = _single(
        conn.table("template_tasks")
        .insert(
            {
                "template_id": int(template_id),
                "title": title.strip(),
                "description": (description.strip() if description else None),
                "dod": dod.strip(),
                "priority": priority,
                "requires_approval": bool(requires_approval),
                "days_from_start": int(days_from_start),
                "tags_csv": (tags_csv.strip() if tags_csv else None),
                "component_name": (component_name.strip() if component_name else None),
            }
        )
        .execute()
    )
    return int(row["id"])
