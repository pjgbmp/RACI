# db.py (RUTA 1 - Supabase, con user_profiles, RLS + admin global boolean)
from __future__ import annotations

import os
from datetime import datetime, date, timezone
from typing import Any, Optional

import streamlit as st
from supabase import create_client


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
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ----------------------------
# Secrets helpers
# ----------------------------
def _sb_cfg() -> dict:
    return st.secrets.get("connections", {}).get("supabase", {})


def _require(v: str | None, msg: str) -> str:
    if not v or not str(v).strip():
        raise RuntimeError(msg)
    return str(v).strip()


def _supabase_url_key() -> tuple[str, str]:
    cfg = _sb_cfg()
    url = cfg.get("url") or os.getenv("SUPABASE_URL")
    anon = (
        cfg.get("anon_key")
        or cfg.get("key")
        or os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_KEY")
    )
    url = _require(url, "Falta SUPABASE_URL / connections.supabase.url")
    anon = _require(anon, "Falta SUPABASE_ANON_KEY / connections.supabase.anon_key")
    return url, anon


# ----------------------------
# Supabase client (anon + user session)
# ----------------------------
@st.cache_resource
def sb_anon():
    """
    Cliente base (anon). No tiene sesión de usuario.
    Útil para llamadas que no requieren auth (pocas).
    """
    url, anon = _supabase_url_key()
    return create_client(url, anon)


def sb_user():
    """
    Cliente que actúa como el usuario logueado (RLS se aplica).
    Requiere que en session_state exista:
      - st.session_state["sb_session"].access_token
      - st.session_state["sb_session"].refresh_token
    """
    url, anon = _supabase_url_key()
    client = create_client(url, anon)

    sess = st.session_state.get("sb_session")
    if not sess:
        raise RuntimeError("No hay sesión activa. Falta st.session_state['sb_session'].")

    # set_session(access, refresh)
    client.auth.set_session(sess.access_token, sess.refresh_token)
    return client


# ----------------------------
# Helpers de respuesta
# ----------------------------
def _data(res):
    return getattr(res, "data", None)


def _single(res) -> dict | None:
    d = _data(res)
    if not d:
        return None
    # supabase-py puede devolver dict o list[dict]
    if isinstance(d, dict):
        return d
    if isinstance(d, list):
        return d[0] if d else None
    return None


def _many(res) -> list[dict]:
    d = _data(res)
    if not d:
        return []
    if isinstance(d, list):
        return d
    if isinstance(d, dict):
        return [d]
    return []


# ----------------------------
# Sesión / perfil
# ----------------------------
def current_user_id() -> str:
    u = st.session_state.get("sb_user")
    if not u:
        raise RuntimeError("No hay usuario en sesión. Falta st.session_state['sb_user'].")
    return u.id


def get_my_profile() -> dict | None:
    uid = current_user_id()
    return _single(sb_user().table("user_profiles").select("*").eq("user_id", uid).limit(1).execute())


def is_global_admin() -> bool:
    p = get_my_profile() or {}
    return bool(p.get("is_global_admin"))


def upsert_user_profile(
    user_id: str,
    full_name: str,
    username: str | None = None,
    email: str | None = None,
    is_active: bool = True,
) -> None:
    """
    IMPORTANTE:
    - Esto corre como el usuario logueado (RLS).
    - Normalmente se usa al registrarse / primer login.
    - Para permitirlo, tu RLS debe dejar: INSERT/UPDATE donde user_id = auth.uid()
    """
    sb_user().table("user_profiles").upsert(
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
    return _single(sb_user().table("user_profiles").select("*").eq("user_id", user_id).limit(1).execute())


def get_user_profile_by_username(username: str) -> dict | None:
    return _single(sb_user().table("user_profiles").select("*").eq("username", username).limit(1).execute())


def fetch_profiles(user_ids: list[str]) -> dict[str, dict]:
    if not user_ids:
        return {}
    rows = _many(
        sb_user()
        .table("user_profiles")
        .select("user_id,full_name,username,email,is_active,is_global_admin")
        .in_("user_id", list(set(user_ids)))
        .execute()
    )
    return {r["user_id"]: r for r in rows}


# ----------------------------
# Grupos / Miembros / Permisos
# ----------------------------
def list_my_groups(user_id: str) -> list[dict]:
    memberships = _many(
        sb_user()
        .table("group_members")
        .select("group_id,role,joined_at")
        .eq("user_id", user_id)
        .execute()
    )
    if not memberships:
        return []

    group_ids = [m["group_id"] for m in memberships]
    role_by_gid = {m["group_id"]: m["role"] for m in memberships}

    groups = _many(sb_user().table("groups").select("*").in_("id", group_ids).order("id", desc=True).execute())
    for g in groups:
        g["my_role"] = role_by_gid.get(g["id"])
    return groups


def get_group(group_id: int) -> dict | None:
    return _single(sb_user().table("groups").select("*").eq("id", int(group_id)).limit(1).execute())


def get_group_by_join_code(join_code: str) -> dict | None:
    return _single(sb_user().table("groups").select("*").eq("join_code", join_code).limit(1).execute())


def create_group(name: str, created_by: str, join_code: str) -> int:
    row = _single(
        sb_user()
        .table("groups")
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
    sb_user().table("group_members").upsert(
        {
            "group_id": int(group_id),
            "user_id": user_id,
            "role": role,
            "joined_at": now_iso(),
        },
        on_conflict="group_id,user_id",
    ).execute()


def remove_group_member(group_id: int, user_id: str) -> None:
    sb_user().table("group_members").delete().eq("group_id", int(group_id)).eq("user_id", user_id).execute()


def list_group_members(group_id: int) -> list[dict]:
    gm = _many(sb_user().table("group_members").select("user_id,role,joined_at").eq("group_id", int(group_id)).execute())
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
    rows = []
    for role, actions in DEFAULT_ROLE_PERMS.items():
        for action in ACTIONS:
            rows.append(
                {
                    "group_id": int(group_id),
                    "role": role,
                    "action": action,
                    "allowed": True if action in actions else False,
                }
            )
    sb_user().table("group_role_permissions").upsert(rows, on_conflict="group_id,role,action").execute()


def user_role_in_group(group_id: int, user_id: str) -> str | None:
    r = _single(
        sb_user()
        .table("group_members")
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
    r = _single(
        sb_user()
        .table("group_role_permissions")
        .select("allowed")
        .eq("group_id", int(group_id))
        .eq("role", role)
        .eq("action", action)
        .limit(1)
        .execute()
    )
    return bool(r["allowed"]) if r else False


def set_group_role_permission(group_id: int, role: str, action: str, allowed: bool) -> None:
    sb_user().table("group_role_permissions").upsert(
        {
            "group_id": int(group_id),
            "role": role,
            "action": action,
            "allowed": bool(allowed),
        },
        on_conflict="group_id,role,action",
    ).execute()


def list_group_role_permissions(group_id: int) -> list[dict]:
    return _many(
        sb_user()
        .table("group_role_permissions")
        .select("group_id,role,action,allowed")
        .eq("group_id", int(group_id))
        .order("role", desc=False)
        .order("action", desc=False)
        .execute()
    )


def transfer_group_leadership(group_id: int, from_user_id: str, to_user_id: str) -> None:
    sb_user().table("group_members").update({"role": "member"}).eq("group_id", int(group_id)).eq("user_id", from_user_id).execute()
    sb_user().table("group_members").update({"role": "leader"}).eq("group_id", int(group_id)).eq("user_id", to_user_id).execute()


# ----------------------------
# Proyectos
# ----------------------------
def list_projects(group_id: int) -> list[dict]:
    return _many(sb_user().table("projects").select("*").eq("group_id", int(group_id)).order("id", desc=True).execute())


def get_project(project_id: int) -> dict | None:
    return _single(sb_user().table("projects").select("*").eq("id", int(project_id)).limit(1).execute())


def create_project(group_id: int, name: str, description: str | None, wip_limit_doing: int, created_by: str) -> int:
    row = _single(
        sb_user()
        .table("projects")
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
    sb_user().table("project_members").upsert(
        {"project_id": int(project_id), "user_id": user_id},
        on_conflict="project_id,user_id",
    ).execute()


def remove_project_member(project_id: int, user_id: str) -> None:
    sb_user().table("project_members").delete().eq("project_id", int(project_id)).eq("user_id", user_id).execute()


def list_project_members(project_id: int) -> list[dict]:
    pm = _many(sb_user().table("project_members").select("user_id").eq("project_id", int(project_id)).execute())
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
# Tags
# ----------------------------
def list_tags(group_id: int, kind: str | None = None) -> list[dict]:
    q = sb_user().table("tags").select("*").eq("group_id", int(group_id))
    if kind:
        q = q.eq("kind", kind)
    return _many(q.order("kind", desc=False).order("name", desc=False).execute())


def get_or_create_tag(group_id: int, name: str, kind: str) -> int:
    r = _single(
        sb_user().table("tags").select("id").eq("group_id", int(group_id)).eq("name", name).eq("kind", kind).limit(1).execute()
    )
    if r:
        return int(r["id"])
    row = _single(sb_user().table("tags").insert({"group_id": int(group_id), "name": name, "kind": kind}).execute())
    return int(row["id"])


def set_task_tags(task_id: int, tag_ids: list[int]) -> None:
    sb_user().table("task_tags").delete().eq("task_id", int(task_id)).execute()
    if tag_ids:
        rows = [{"task_id": int(task_id), "tag_id": int(tid)} for tid in sorted(set(tag_ids))]
        sb_user().table("task_tags").upsert(rows, on_conflict="task_id,tag_id").execute()


def list_task_tags(task_id: int) -> list[dict]:
    links = _many(sb_user().table("task_tags").select("tag_id").eq("task_id", int(task_id)).execute())
    if not links:
        return []
    tag_ids = [x["tag_id"] for x in links]
    tags = _many(sb_user().table("tags").select("*").in_("id", tag_ids).execute())
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
    row = _single(
        sb_user()
        .table("tasks")
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
    return _single(sb_user().table("tasks").select("*").eq("id", int(task_id)).limit(1).execute())


def list_tasks_by_project(project_id: int, limit: int = 300) -> list[dict]:
    return _many(
        sb_user()
        .table("tasks")
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
    base = (
        sb_user()
        .table("tasks")
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
            sb_user()
            .table("task_roles")
            .select("task_id,role")
            .eq("user_id", only_mine_user_id)
            .in_("role", ["R", "A"])
            .execute()
        )
        mine_ids = {x["task_id"] for x in mine}
        rows = [t for t in rows if t["id"] in mine_ids]

    if tag_ids_filter:
        links = _many(
            sb_user()
            .table("task_tags")
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
    sb_user().table("tasks").update(
        {"due_date": due_date_new.isoformat() if due_date_new else None, "updated_at": now_iso()}
    ).eq("id", int(task_id)).execute()


def update_task_status(
    task_id: int,
    new_status: str,
    blocked_reason: str | None,
    unblock_owner_user_id: str | None,
    completed_at_iso: str | None,
) -> None:
    payload: dict[str, Any] = {
        "status": new_status,
        "updated_at": now_iso(),
        "blocked_reason": (blocked_reason if new_status == "blocked" else None),
        "unblock_owner_user_id": (unblock_owner_user_id if new_status == "blocked" else None),
    }
    if completed_at_iso:
        payload["completed_at"] = completed_at_iso
    sb_user().table("tasks").update(payload).eq("id", int(task_id)).execute()


# ----------------------------
# Admin Dashboard helpers (depende de RLS + is_global_admin)
# ----------------------------
def admin_list_all_groups() -> list[dict]:
    if not is_global_admin():
        return []
    return _many(sb_user().table("groups").select("*").order("id", desc=True).execute())


def admin_list_all_group_members() -> list[dict]:
    if not is_global_admin():
        return []
    return _many(sb_user().table("group_members").select("*").order("group_id", desc=False).execute())


def admin_list_all_projects() -> list[dict]:
    if not is_global_admin():
        return []
    return _many(sb_user().table("projects").select("*").order("id", desc=True).execute())


def admin_list_all_tasks(limit: int = 5000) -> list[dict]:
    if not is_global_admin():
        return []
    return _many(sb_user().table("tasks").select("*").order("id", desc=True).limit(limit).execute())
