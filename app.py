# app.py
import os, uuid
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import date, datetime, timedelta

from db import (
    init_db, connect, now_iso, UPLOAD_DIR,
    seed_group_permissions, has_permission, user_role_in_group,
    add_notification, log_history
)
from auth import hash_password, verify_password
from ics_utils import task_to_ics_event, tasks_to_ics_calendar

st.set_page_config(page_title="Workflow Organización de equipo", layout="wide")
init_db()

# -------------------- Helpers: pickle-safe for Streamlit widgets --------------------
def compute_kpis_per_person(con, group_id: int, project_id: int | None = None) -> pd.DataFrame:
    """
    KPIs por persona (basado en rol R = ejecutor).
    - WIP (doing)
    - ToDo / Doing / Blocked / Awaiting / Done
    - Overdue (vencidas)
    - Avg Aging (días) en tareas abiertas
    - Avg Lead Time (días) en tareas cerradas (últimos 60 días)
    - On-time % en tareas cerradas (con due_date)
    """
    # 1) Usuarios del grupo
    users = rows_to_dicts(con.execute("""
        SELECT u.id as user_id, u.full_name
        FROM group_members gm
        JOIN users u ON u.id = gm.user_id
        WHERE gm.group_id=?
        ORDER BY u.full_name
    """, (group_id,)).fetchall())

    if not users:
        return pd.DataFrame()

    # 2) Tareas donde la persona es R (ejecutor)
    base_sql = """
      SELECT
        tr.user_id,
        u.full_name,
        t.id as task_id,
        t.status,
        t.priority,
        t.created_at,
        t.updated_at,
        t.completed_at,
        t.due_date,
        t.target_date
      FROM task_roles tr
      JOIN tasks t ON t.id = tr.task_id
      JOIN projects p ON p.id = t.project_id
      JOIN users u ON u.id = tr.user_id
      WHERE p.group_id = ?
        AND tr.role = 'R'
    """
    params = [group_id]
    if project_id is not None:
        base_sql += " AND t.project_id = ? "
        params.append(project_id)

    rows = rows_to_dicts(con.execute(base_sql, tuple(params)).fetchall())
    df = pd.DataFrame(rows)

    # Si no hay tareas asignadas, devolvemos tabla con ceros
    if df.empty:
        return pd.DataFrame([{
            "user_id": u["user_id"],
            "full_name": u["full_name"],
            "todo": 0, "doing": 0, "blocked": 0, "awaiting_approval": 0, "done": 0,
            "wip_doing": 0, "overdue_open": 0,
            "avg_aging_days_open": None,
            "avg_lead_time_days_done_60d": None,
            "on_time_rate_done": None
        } for u in users])

    # 3) Parse de fechas
    def to_dt(s):
        try:
            return pd.to_datetime(s)
        except Exception:
            return pd.NaT

    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["completed_at_dt"] = pd.to_datetime(df["completed_at"], errors="coerce")
    df["due_dt"] = pd.to_datetime(df["due_date"], errors="coerce")

    now = pd.Timestamp.now()

    # 4) Métricas por status
    status_counts = (df
        .groupby(["user_id", "full_name", "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # asegurar columnas
    for col in ["todo", "doing", "blocked", "awaiting_approval", "done"]:
        if col not in status_counts.columns:
            status_counts[col] = 0

    status_counts["wip_doing"] = status_counts["doing"]

    # 5) Overdue (tareas abiertas con due_date < hoy)
    open_mask = df["status"].isin(["todo", "doing", "blocked", "awaiting_approval"])
    overdue_mask = open_mask & df["due_dt"].notna() & (df["due_dt"].dt.date < now.date())
    overdue = (df[overdue_mask]
        .groupby(["user_id", "full_name"])
        .size()
        .reset_index(name="overdue_open")
    )

    # 6) Aging (días desde creación en tareas abiertas)
    open_df = df[open_mask & df["created_at_dt"].notna()].copy()
    open_df["aging_days"] = (now - open_df["created_at_dt"]).dt.days
    aging = (open_df.groupby(["user_id", "full_name"])["aging_days"]
             .mean()
             .reset_index(name="avg_aging_days_open"))

    # 7) Lead time (días desde created_at a completed_at) en done últimos 60 días
    done_df = df[
    (df["status"] == "done") &
    df["created_at_dt"].notna() &
    df["completed_at_dt"].notna()
    ].copy()
    done_df["lead_days"] = (done_df["completed_at_dt"] - done_df["created_at_dt"]).dt.days
    cutoff = now - pd.Timedelta(days=60)
    done_60 = done_df[done_df["completed_at_dt"] >= cutoff]
    lead = (done_60.groupby(["user_id", "full_name"])["lead_days"]
            .mean()
            .reset_index(name="avg_lead_time_days_done_60d"))

    # 8) On-time rate (done con due_date): completed_at <= due_date
    done_due = done_df[done_df["due_dt"].notna()].copy()
    if not done_due.empty:
        done_due["on_time"] = done_due["completed_at_dt"].dt.date <= done_due["due_dt"].dt.date
        ontime = (done_due.groupby(["user_id", "full_name"])["on_time"]
                  .mean()
                  .reset_index(name="on_time_rate_done"))
    else:
        ontime = pd.DataFrame(columns=["user_id", "full_name", "on_time_rate_done"])

    # 9) Merge todo en una tabla final, incluyendo usuarios sin tareas (left join)
    base_users = pd.DataFrame(users).rename(columns={"user_id": "user_id", "full_name": "full_name"})
    out = base_users.merge(status_counts, on=["user_id", "full_name"], how="left")
    out = out.merge(overdue, on=["user_id", "full_name"], how="left")
    out = out.merge(aging, on=["user_id", "full_name"], how="left")
    out = out.merge(lead, on=["user_id", "full_name"], how="left")
    out = out.merge(ontime, on=["user_id", "full_name"], how="left")

    # nulls -> 0 donde corresponde
    for col in ["todo", "doing", "blocked", "awaiting_approval", "done", "wip_doing", "overdue_open"]:
        out[col] = out[col].fillna(0).astype(int)

    # redondeos amigables
    if "avg_aging_days_open" in out.columns:
        out["avg_aging_days_open"] = out["avg_aging_days_open"].round(1)
    if "avg_lead_time_days_done_60d" in out.columns:
        out["avg_lead_time_days_done_60d"] = out["avg_lead_time_days_done_60d"].round(1)
    if "on_time_rate_done" in out.columns:
        out["on_time_rate_done"] = pd.to_numeric(out["on_time_rate_done"], errors="coerce")
        out["on_time_rate_done"] = (out["on_time_rate_done"] * 100).round(0)
        # opcional: mostrar como entero con NA
        out["on_time_rate_done"] = out["on_time_rate_done"].astype("Int64")  # permite <NA>

    return out


def kpi_badges_row(df_kpi: pd.DataFrame):
    """Mini KPIs globales (para ver salud del grupo rápido)."""
    if df_kpi.empty:
        return
    total_overdue = int(df_kpi["overdue_open"].sum())
    total_wip = int(df_kpi["wip_doing"].sum())
    avg_aging = df_kpi["avg_aging_days_open"].dropna().mean()
    ontime = df_kpi["on_time_rate_done"].dropna().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overdue (abiertas)", total_overdue)
    c2.metric("WIP (doing)", total_wip)
    c3.metric("Aging promedio (días)", round(float(avg_aging), 1) if pd.notna(avg_aging) else "—")
    c4.metric("On-time promedio (%)", int(ontime) if pd.notna(ontime) else "—")

def rows_to_dicts(rows):
    """Convierte sqlite3.Row -> dict (evita error 'cannot pickle sqlite3.Row')"""
    return [dict(r) for r in rows] if rows else []

def safe_selectbox_dict(label, dict_options, format_func, key=None, index=0):
    """Selectbox seguro cuando options son dicts."""
    if not dict_options:
        st.warning(f"No hay opciones disponibles para: {label}")
        return None
    return st.selectbox(label, options=dict_options, format_func=format_func, key=key, index=min(index, len(dict_options)-1))

def _to_dt(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.NaT

def approval_cycle_hours(con, task_id: int) -> float | None:
    """
    Devuelve horas en aprobación para la última solicitud:
    - si está pending: horas desde requested_at hasta ahora
    - si está decidida: horas desde requested_at hasta decided_at
    """
    r = con.execute("""
      SELECT requested_at, decided_at, status
      FROM approval_requests
      WHERE task_id=?
      ORDER BY id DESC
      LIMIT 1
    """, (task_id,)).fetchone()
    if not r or not r["requested_at"]:
        return None

    req = _to_dt(r["requested_at"])
    if pd.isna(req):
        return None

    if r["status"] == "pending":
        end = pd.Timestamp.now()
    else:
        end = _to_dt(r["decided_at"])
        if pd.isna(end):
            end = pd.Timestamp.now()

    return round(float((end - req).total_seconds() / 3600), 2)


def approvals_kpi_by_accountable(con, group_id: int, project_id: int | None = None) -> pd.DataFrame:
    """
    KPIs de aprobación por Dueño (A):
    - pending_count: cuántas aprobaciones pendientes tiene
    - avg_cycle_hours_approved_60d: promedio horas de aprobación de aprobadas/rechazadas en últimos 60 días
    - avg_cycle_hours_pending: promedio horas de pendientes (hasta ahora)
    """
    base = """
      SELECT
        ar.task_id,
        ar.accountable_user_id as a_user_id,
        u.full_name,
        ar.status,
        ar.requested_at,
        ar.decided_at,
        t.project_id
      FROM approval_requests ar
      JOIN tasks t ON t.id=ar.task_id
      JOIN projects p ON p.id=t.project_id
      JOIN users u ON u.id=ar.accountable_user_id
      WHERE p.group_id=?
    """
    params = [group_id]
    if project_id is not None:
        base += " AND t.project_id=? "
        params.append(project_id)

    rows = [dict(x) for x in con.execute(base, tuple(params)).fetchall()]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["a_user_id","full_name","pending_count","avg_cycle_hours_pending","avg_cycle_hours_approved_60d"])

    df["requested_at_dt"] = df["requested_at"].apply(_to_dt)
    df["decided_at_dt"] = df["decided_at"].apply(_to_dt)
    now = pd.Timestamp.now()

    # cycle hours
    df["cycle_hours"] = None
    pending = df["status"] == "pending"
    df.loc[pending, "cycle_hours"] = (now - df.loc[pending, "requested_at_dt"]).dt.total_seconds() / 3600
    decided = ~pending
    df.loc[decided, "cycle_hours"] = (df.loc[decided, "decided_at_dt"] - df.loc[decided, "requested_at_dt"]).dt.total_seconds() / 3600

    # Pending count + avg pending
    pend = (df[pending]
            .groupby(["a_user_id","full_name"])
            .agg(pending_count=("task_id","count"),
                 avg_cycle_hours_pending=("cycle_hours","mean"))
            .reset_index())

    # Approved/rejected last 60d
    cutoff = now - pd.Timedelta(days=60)
    dec60 = df[decided & (df["decided_at_dt"] >= cutoff)]
    dec = (dec60.groupby(["a_user_id","full_name"])
           .agg(avg_cycle_hours_approved_60d=("cycle_hours","mean"))
           .reset_index())

    out = pd.merge(pend, dec, on=["a_user_id","full_name"], how="outer")
    out["pending_count"] = out["pending_count"].fillna(0).astype(int)

    # Asegurar numérico antes de round
    for col in ["avg_cycle_hours_pending", "avg_cycle_hours_approved_60d"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
        # opcional: mostrar como número con NA (no rompe)
    for col in ["avg_cycle_hours_pending", "avg_cycle_hours_approved_60d"]:
        if col in out.columns:
            out[col] = out[col].astype("Float64")  # permite <NA>

    return out

def week_bounds(d: date):
    start = d - timedelta(days=d.weekday())  # lunes
    end = start + timedelta(days=6)          # domingo
    return start, end

def task_r_names(con, task_id: int) -> str:
    rs = con.execute("""
      SELECT u.full_name
      FROM task_roles tr JOIN users u ON u.id=tr.user_id
      WHERE tr.task_id=? AND tr.role='R'
      ORDER BY u.full_name
    """, (task_id,)).fetchall()
    return ", ".join([r["full_name"] for r in rs]) if rs else "—"

def task_a_name(con, task_id: int) -> str:
    a = con.execute("""
      SELECT u.full_name
      FROM task_roles tr JOIN users u ON u.id=tr.user_id
      WHERE tr.task_id=? AND tr.role='A'
      LIMIT 1
    """, (task_id,)).fetchone()
    return a["full_name"] if a else "—"

def build_weekly_report(con, group_id: int, week_start_d: date, week_end_d: date, project_id: int | None = None):
    params = [group_id]
    proj_filter = ""
    if project_id is not None:
        proj_filter = " AND t.project_id=? "
        params.append(project_id)

    # Done in week (by completed_at)
    done = rows_to_dicts(con.execute(f"""
      SELECT t.id, t.title, t.description, t.dod, t.priority, t.status,
             t.created_at, t.completed_at, t.due_date,
             p.name as project_name
      FROM tasks t
      JOIN projects p ON p.id=t.project_id
      WHERE p.group_id=?
        {proj_filter}
        AND t.status='done'
        AND t.completed_at IS NOT NULL
        AND date(t.completed_at) BETWEEN date(?) AND date(?)
      ORDER BY date(t.completed_at) ASC
    """, tuple(params + [week_start_d.isoformat(), week_end_d.isoformat()])).fetchall())

    # Pending (open) as of end of week
    pending = rows_to_dicts(con.execute(f"""
      SELECT t.id, t.title, t.description, t.dod, t.priority, t.status,
             t.created_at, t.due_date,
             p.name as project_name
      FROM tasks t
      JOIN projects p ON p.id=t.project_id
      WHERE p.group_id=?
        {proj_filter}
        AND t.status IN ('todo','doing','blocked','awaiting_approval')
      ORDER BY
        CASE t.priority WHEN 'urgente' THEN 1 WHEN 'alta' THEN 2 WHEN 'media' THEN 3 ELSE 4 END,
        t.due_date
    """, tuple(params)).fetchall())

    def enrich(rows):
        for r in rows:
            # lead time
            try:
                if r.get("completed_at"):
                    created = pd.to_datetime(r["created_at"], errors="coerce")
                    comp = pd.to_datetime(r["completed_at"], errors="coerce")
                    r["lead_days"] = int((comp - created).days) if pd.notna(created) and pd.notna(comp) else None
                else:
                    r["lead_days"] = None
            except Exception:
                r["lead_days"] = None

            # approval cycle
            r["approval_hours"] = approval_cycle_hours(con, r["id"])

            # R / A names
            r["R"] = task_r_names(con, r["id"])
            r["A"] = task_a_name(con, r["id"])
        return rows

    done = enrich(done)
    pending = enrich(pending)

    df_done = pd.DataFrame(done)
    df_pending = pd.DataFrame(pending)

    return df_done, df_pending

def report_to_html(df_done: pd.DataFrame, df_pending: pd.DataFrame, title: str):
    # Selección de columnas (descriptivo)
    cols_done = ["project_name","id","title","priority","due_date","lead_days","approval_hours","R","A","description","dod","completed_at"]
    cols_pend = ["project_name","id","title","status","priority","due_date","approval_hours","R","A","description","dod","created_at"]

    done_html = df_done[cols_done].to_html(index=False, escape=False) if not df_done.empty else "<p>(Sin tareas completadas en el período)</p>"
    pend_html = df_pending[cols_pend].to_html(index=False, escape=False) if not df_pending.empty else "<p>(Sin tareas pendientes)</p>"

    return f"""
    <h2>{title}</h2>
    <h3>✅ Tareas completadas</h3>
    {done_html}
    <h3>⏳ Tareas pendientes (abiertas)</h3>
    {pend_html}
    """

# -------------------- Session defaults --------------------
for k, v in {
    "user_id": None,
    "username": "",
    "full_name": "",
    "active_group_id": None,
    "selected_task_id": None,  # para abrir detalle desde tablero
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

def logged_in():
    return st.session_state["user_id"] is not None

def require_login():
    if not logged_in():
        st.stop()

def fmt_user(u):
    # u puede ser dict o Row; pero aquí lo convertimos a dict por seguridad
    if not isinstance(u, dict):
        u = dict(u)
    return f"{u['full_name']} (@{u['username']})"

# -------------------- DB helpers --------------------
def get_user_by_username(con, username: str):
    return con.execute("SELECT * FROM users WHERE username=? AND is_active=1", (username,)).fetchone()

def get_user_by_id(con, uid: int):
    return con.execute("SELECT * FROM users WHERE id=? AND is_active=1", (uid,)).fetchone()

def list_users(con):
    rows = con.execute("SELECT id, full_name, username, email FROM users WHERE is_active=1 ORDER BY full_name").fetchall()
    return rows_to_dicts(rows)

def list_my_groups(con, user_id: int):
    rows = con.execute("""
      SELECT g.*, gm.role as my_role
      FROM groups g
      JOIN group_members gm ON gm.group_id=g.id
      WHERE gm.user_id=?
      ORDER BY g.id DESC
    """, (user_id,)).fetchall()
    return rows_to_dicts(rows)

def list_group_members(con, group_id: int):
    rows = con.execute("""
      SELECT u.id, u.full_name, u.username, u.email, gm.role, gm.joined_at
      FROM group_members gm
      JOIN users u ON u.id=gm.user_id
      WHERE gm.group_id=?
      ORDER BY gm.role DESC, u.full_name
    """, (group_id,)).fetchall()
    return rows_to_dicts(rows)

def list_projects(con, group_id: int):
    rows = con.execute("""
      SELECT id, group_id, name, description, wip_limit_doing, created_by, created_at
      FROM projects WHERE group_id=? ORDER BY id DESC
    """, (group_id,)).fetchall()
    return rows_to_dicts(rows)

def list_project_members(con, project_id: int):
    rows = con.execute("""
      SELECT u.id, u.full_name, u.username
      FROM project_members pm JOIN users u ON u.id=pm.user_id
      WHERE pm.project_id=?
      ORDER BY u.full_name
    """, (project_id,)).fetchall()
    return rows_to_dicts(rows)

def ensure_project_member(con, project_id: int, user_id: int):
    con.execute("INSERT OR IGNORE INTO project_members(project_id, user_id) VALUES (?,?)", (project_id, user_id))

def get_task_roles(con, task_id: int):
    rows = con.execute("""
      SELECT tr.role, u.id as user_id, u.full_name
      FROM task_roles tr JOIN users u ON u.id=tr.user_id
      WHERE tr.task_id=?
      ORDER BY tr.role, u.full_name
    """, (task_id,)).fetchall()
    out = {"A": [], "R": [], "C": [], "I": []}
    for r in rows:
        rr = dict(r)
        out[rr["role"]].append({"id": rr["user_id"], "name": rr["full_name"]})
    return out

def get_accountable_id(con, task_id: int):
    r = con.execute("SELECT user_id FROM task_roles WHERE task_id=? AND role='A'", (task_id,)).fetchone()
    return r["user_id"] if r else None

def is_responsible(con, task_id: int, user_id: int):
    r = con.execute("SELECT 1 FROM task_roles WHERE task_id=? AND role='R' AND user_id=?", (task_id, user_id)).fetchone()
    return r is not None

def can_edit_due_date(con, group_id: int, task_id: int, user_id: int) -> bool:
    # Permiso fino + decision rights: solo A (o líder del grupo)
    if not has_permission(con, group_id, user_id, "task_change_due_date"):
        return False
    role = user_role_in_group(con, group_id, user_id)
    if role == "leader":
        return True
    return get_accountable_id(con, task_id) == user_id

def can_change_status(con, group_id: int, task_id: int, user_id: int) -> bool:
    if not has_permission(con, group_id, user_id, "task_change_status"):
        return False
    role = user_role_in_group(con, group_id, user_id)
    if role == "leader":
        return True
    return is_responsible(con, task_id, user_id)

def dependencies_open(con, task_id: int) -> list[dict]:
    rows = con.execute("""
      SELECT d.depends_on_task_id as id, t.title, t.status
      FROM task_dependencies d
      JOIN tasks t ON t.id=d.depends_on_task_id
      WHERE d.task_id=? AND t.status!='done'
    """, (task_id,)).fetchall()
    return rows_to_dicts(rows)

def wip_limit_ok(con, project_id: int, user_id: int) -> tuple[bool, str]:
    proj = con.execute("SELECT wip_limit_doing FROM projects WHERE id=?", (project_id,)).fetchone()
    limit = int(proj["wip_limit_doing"]) if proj else 3
    cnt = con.execute("""
      SELECT COUNT(*) as c
      FROM tasks t
      JOIN task_roles tr ON tr.task_id=t.id AND tr.role='R'
      WHERE t.project_id=? AND t.status='doing' AND tr.user_id=?
    """, (project_id, user_id)).fetchone()["c"]
    if cnt >= limit:
        return False, f"WIP excedido: ya tienes {cnt} tareas en DOING (límite {limit})."
    return True, ""

def upsert_task_roles(con, task_id: int, actor_id: int, A_id: int, R_ids: list[int], C_ids: list[int], I_ids: list[int]):
    if A_id is None:
        raise ValueError("Debe existir 1 dueño (A) por tarea.")
    if not R_ids:
        raise ValueError("Debe existir al menos 1 ejecutor (R).")
    before = get_task_roles(con, task_id)
    con.execute("DELETE FROM task_roles WHERE task_id=?", (task_id,))
    ts = now_iso()
    con.execute("INSERT INTO task_roles(task_id, user_id, role, assigned_at) VALUES (?,?, 'A', ?)", (task_id, A_id, ts))
    for uid in sorted(set(R_ids)):
        con.execute("INSERT INTO task_roles(task_id, user_id, role, assigned_at) VALUES (?,?, 'R', ?)", (task_id, uid, ts))
    for uid in sorted(set(C_ids)):
        con.execute("INSERT INTO task_roles(task_id, user_id, role, assigned_at) VALUES (?,?, 'C', ?)", (task_id, uid, ts))
    for uid in sorted(set(I_ids)):
        con.execute("INSERT INTO task_roles(task_id, user_id, role, assigned_at) VALUES (?,?, 'I', ?)", (task_id, uid, ts))
    after = get_task_roles(con, task_id)
    log_history(con, task_id, actor_id, "responsibilities", str(before), str(after))

# -------------------- Sidebar Auth --------------------
st.sidebar.title("Acceso")

if not logged_in():
    tabs = st.sidebar.tabs(["Login", "Crear usuario"])
    with tabs[0]:
        u = st.text_input("Usuario")
        p = st.text_input("Contraseña", type="password")
        if st.button("Entrar"):
            with connect() as con:
                row = get_user_by_username(con, u.strip())
                if not row or not verify_password(p, row["password_hash"]):
                    st.error("Credenciales inválidas.")
                else:
                    st.session_state["user_id"] = row["id"]
                    st.session_state["username"] = row["username"]
                    st.session_state["full_name"] = row["full_name"]
                    st.rerun()

    with tabs[1]:
        nu = st.text_input("Nuevo usuario", key="nu")
        npw = st.text_input("Nueva contraseña", type="password", key="npw")
        fn = st.text_input("Nombre completo", key="fn")
        email = st.text_input("Email (opcional)")
        if st.button("Crear"):
            if not (nu.strip() and npw and fn.strip()):
                st.error("Completa usuario, contraseña y nombre.")
            else:
                with connect() as con:
                    try:
                        con.execute("""
                          INSERT INTO users(username, password_hash, full_name, email, created_at)
                          VALUES (?, ?, ?, ?, ?)
                        """, (nu.strip(), hash_password(npw), fn.strip(), (email.strip() or None), now_iso()))
                        con.commit()
                        st.success("Usuario creado. Inicia sesión.")
                    except Exception as e:
                        st.error(f"No se pudo crear: {e}")
    st.stop()

# Logged in
st.sidebar.success(st.session_state.get("full_name", "Usuario"))
if st.sidebar.button("Cerrar sesión"):
    st.session_state["user_id"] = None
    st.session_state["username"] = ""
    st.session_state["full_name"] = ""
    st.session_state["active_group_id"] = None
    st.session_state["selected_task_id"] = None
    st.rerun()

# -------------------- Group selection / creation / join --------------------
with connect() as con:
    my_groups = list_my_groups(con, st.session_state["user_id"])

st.sidebar.subheader("Equipo / Grupo")

if my_groups:
    names = [f"[{g['id']}] {g['name']} ({g['my_role']})" for g in my_groups]
    idx = 0
    if st.session_state["active_group_id"]:
        for i, g in enumerate(my_groups):
            if g["id"] == st.session_state["active_group_id"]:
                idx = i
                break
    pick = st.sidebar.selectbox("Grupo activo", options=names, index=idx)
    active = my_groups[names.index(pick)]
    st.session_state["active_group_id"] = active["id"]
else:
    st.sidebar.info("No perteneces a ningún grupo aún.")

with st.sidebar.expander("➕ Crear grupo"):
    gname = st.text_input("Nombre del grupo", key="gname")
    if st.button("Crear grupo"):
        if not gname.strip():
            st.error("Pon un nombre.")
        else:
            with connect() as con:
                join_code = uuid.uuid4().hex[:8].upper()
                con.execute("""
                  INSERT INTO groups(name, join_code, created_by, created_at)
                  VALUES (?, ?, ?, ?)
                """, (gname.strip(), join_code, st.session_state["user_id"], now_iso()))
                gid = con.execute("SELECT last_insert_rowid() as id").fetchone()["id"]
                con.execute("""
                  INSERT INTO group_members(group_id, user_id, role, joined_at)
                  VALUES (?, ?, 'leader', ?)
                """, (gid, st.session_state["user_id"], now_iso()))
                seed_group_permissions(con, gid)
                con.commit()
                st.success(f"Grupo creado. Código de acceso: {join_code}")
                st.session_state["active_group_id"] = gid
                st.rerun()

with st.sidebar.expander("🔑 Unirme a un grupo"):
    code = st.text_input("Código de acceso (join code)", key="joincode")
    if st.button("Unirme"):
        with connect() as con:
            g = con.execute("SELECT * FROM groups WHERE join_code=?", (code.strip().upper(),)).fetchone()
            if not g:
                st.error("Código inválido.")
            else:
                con.execute("""
                  INSERT OR IGNORE INTO group_members(group_id, user_id, role, joined_at)
                  VALUES (?, ?, 'member', ?)
                """, (g["id"], st.session_state["user_id"], now_iso()))
                seed_group_permissions(con, g["id"])
                con.commit()
                st.success("Listo. Ya perteneces al grupo.")
                st.session_state["active_group_id"] = g["id"]
                st.rerun()

# Require group
if not st.session_state["active_group_id"]:
    st.title("Workflow Team Manager")
    st.info("Crea o únete a un grupo para empezar.")
    st.stop()

GROUP_ID = st.session_state["active_group_id"]

# -------------------- Header & Notifications --------------------
st.title("Workflow Organización de Equipo")

with connect() as con:
    unread = con.execute("""
      SELECT COUNT(*) as c FROM notifications
      WHERE user_id=? AND read_at IS NULL
    """, (st.session_state["user_id"],)).fetchone()["c"]
st.caption(f"🔔 Notificaciones sin leer: {unread}")

# -------------------- Menu --------------------
menu = st.sidebar.radio(
    "Menú",
    ["Resumen", "Proyectos", "Tablero", "Tarea (detalle)", "Plantillas", "Export", "Gobernanza", "Reportes", "Ayuda"]
)

# -------------------- Resumen --------------------
if menu == "Resumen":
    with connect() as con:
        uid = st.session_state["user_id"]

        st.subheader("Resumen de trabajo (qué hay por hacer)")

        def fetch_by_role(role: str):
            rows = con.execute("""
              SELECT t.id, t.title, t.status, t.priority, t.due_date, p.name as project
              FROM tasks t
              JOIN projects p ON p.id=t.project_id
              JOIN task_roles tr ON tr.task_id=t.id
              WHERE p.group_id=? AND tr.user_id=? AND tr.role=?
              ORDER BY
                CASE t.priority WHEN 'urgente' THEN 1 WHEN 'alta' THEN 2 WHEN 'media' THEN 3 ELSE 4 END,
                t.due_date
            """, (GROUP_ID, uid, role)).fetchall()
            return rows_to_dicts(rows)

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

        overdue = rows_to_dicts(con.execute("""
          SELECT t.id, t.title, t.due_date, t.status, p.name as project
          FROM tasks t JOIN projects p ON p.id=t.project_id
          WHERE p.group_id=? AND t.due_date IS NOT NULL
            AND date(t.due_date) < date('now') AND t.status!='done'
          ORDER BY date(t.due_date) ASC
        """, (GROUP_ID,)).fetchall())

        blocked = rows_to_dicts(con.execute("""
          SELECT t.id, t.title, t.blocked_reason, p.name as project
          FROM tasks t JOIN projects p ON p.id=t.project_id
          WHERE p.group_id=? AND t.status='blocked'
          ORDER BY t.updated_at DESC
        """, (GROUP_ID,)).fetchall())

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Vencidas")
            st.dataframe(pd.DataFrame(overdue) if overdue else pd.DataFrame(),
                         use_container_width=True, height=260)
        with c2:
            st.markdown("### Bloqueadas")
            st.dataframe(pd.DataFrame(blocked) if blocked else pd.DataFrame(),
                         use_container_width=True, height=260)
        st.divider()
        st.subheader("KPIs por persona")

        with connect() as con:
            # filtro por proyecto (opcional)
            projs = rows_to_dicts(con.execute(
                "SELECT id, name FROM projects WHERE group_id=? ORDER BY id DESC",
                (GROUP_ID,)
            ).fetchall())

            proj_filter = st.selectbox(
                "Filtrar KPIs por proyecto (opcional)",
                options=[None] + [p["id"] for p in projs],
                format_func=lambda x: "Todos los proyectos" if x is None else f"[{x}] " + next(p["name"] for p in projs if p["id"] == x),
                key="kpi_proj_filter"
            )

            kpi_df = compute_kpis_per_person(con, GROUP_ID, project_id=proj_filter)
            if kpi_df.empty:
                st.info("No hay datos todavía.")
            else:
                kpi_badges_row(kpi_df)

                # Tabla principal
                show_cols = [
                    "full_name",
                    "todo", "doing", "blocked", "awaiting_approval", "done",
                    "wip_doing", "overdue_open",
                    "avg_aging_days_open",
                    "avg_lead_time_days_done_60d",
                    "on_time_rate_done"
                ]
                st.dataframe(kpi_df[show_cols], use_container_width=True, height=420)

                # Gráfico: WIP + Overdue por persona
                chart_df = kpi_df[["full_name", "wip_doing", "overdue_open"]].copy()
                fig = px.bar(chart_df, x="full_name", y=["wip_doing", "overdue_open"], barmode="group",
                            labels={"value":"Cantidad", "full_name":"Persona", "variable":"Métrica"})
                fig.update_layout(height=360, xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

                # Gráfico: estado por persona (stack)
                long = kpi_df.melt(
                    id_vars=["full_name"],
                    value_vars=["todo","doing","blocked","awaiting_approval","done"],
                    var_name="status",
                    value_name="count"
                )
                fig2 = px.bar(long, x="full_name", y="count", color="status",
                            labels={"count":"Tareas", "full_name":"Persona", "status":"Estado"})
                fig2.update_layout(height=380, xaxis_title="", yaxis_title="")
                st.plotly_chart(fig2, use_container_width=True)

                # Ranking (rápido) para gestión
                st.markdown("#### Señales rápidas (para liderazgo)")
                top_overdue = kpi_df.sort_values("overdue_open", ascending=False).head(5)[["full_name","overdue_open"]]
                top_wip = kpi_df.sort_values("wip_doing", ascending=False).head(5)[["full_name","wip_doing"]]
                c1, c2 = st.columns(2)
                with c1:
                    st.caption("Top 5 con más vencidas")
                    st.dataframe(top_overdue, use_container_width=True)
                with c2:
                    st.caption("Top 5 con más WIP")
                    st.dataframe(top_wip, use_container_width=True)
        st.divider()
        st.subheader("Ciclo de aprobación (awaiting_approval)")

        with connect() as con:
            projs = rows_to_dicts(con.execute(
                "SELECT id, name FROM projects WHERE group_id=? ORDER BY id DESC",
                (GROUP_ID,)
            ).fetchall())

            proj_filter2 = st.selectbox(
                "Filtrar aprobaciones por proyecto (opcional)",
                options=[None] + [p["id"] for p in projs],
                format_func=lambda x: "Todos los proyectos" if x is None else f"[{x}] " + next(p["name"] for p in projs if p["id"] == x),
                key="appr_proj_filter"
            )

            appr_df = approvals_kpi_by_accountable(con, GROUP_ID, project_id=proj_filter2)

            if appr_df.empty:
                st.info("No hay aprobaciones registradas todavía.")
            else:
                # KPIs globales rápidos
                c1, c2, c3 = st.columns(3)
                c1.metric("Pendientes totales", int(appr_df["pending_count"].sum()))
                c2.metric("Promedio horas pendientes", float(appr_df["avg_cycle_hours_pending"].dropna().mean()) if appr_df["avg_cycle_hours_pending"].notna().any() else "—")
                c3.metric("Promedio horas aprobadas (60d)", float(appr_df["avg_cycle_hours_approved_60d"].dropna().mean()) if appr_df["avg_cycle_hours_approved_60d"].notna().any() else "—")

                st.dataframe(appr_df.sort_values(["pending_count","avg_cycle_hours_pending"], ascending=False),
                            use_container_width=True, height=320)


# -------------------- Proyectos --------------------
elif menu == "Proyectos":
    with connect() as con:
        uid = st.session_state["user_id"]
        is_leader = user_role_in_group(con, GROUP_ID, uid) == "leader"

        st.subheader("Proyectos del grupo")
        projs = list_projects(con, GROUP_ID)
        st.dataframe(pd.DataFrame(projs) if projs else pd.DataFrame(), use_container_width=True)
        st.divider()
        st.subheader("Miembros por proyecto (UI)")

        # seleccionar proyecto
        projs = list_projects(con, GROUP_ID)
        if not projs:
            st.info("No hay proyectos.")
        else:
            proj = safe_selectbox_dict("Proyecto a administrar", projs, format_func=lambda p: f"[{p['id']}] {p['name']}", key="pm_proj")
            pid = proj["id"]

            # permiso: líder o action dedicada
            can_manage = (user_role_in_group(con, GROUP_ID, uid) == "leader") or has_permission(con, GROUP_ID, uid, "project_manage_members")

            current = list_project_members(con, pid)
            st.markdown("#### Miembros actuales")
            st.dataframe(pd.DataFrame(current) if current else pd.DataFrame(), use_container_width=True)

            group_members = rows_to_dicts(con.execute("""
                SELECT u.id, u.full_name, u.username
                FROM group_members gm JOIN users u ON u.id=gm.user_id
                WHERE gm.group_id=?
                ORDER BY u.full_name
            """, (GROUP_ID,)).fetchall())

            current_ids = set([m["id"] for m in current])
            candidates = [m for m in group_members if m["id"] not in current_ids]

            if not can_manage:
                st.caption("No tienes permisos para modificar miembros del proyecto.")
            else:
                st.markdown("#### Agregar miembros")
                if candidates:
                    opt_map = {fmt_user(m): m["id"] for m in candidates}
                    add_sel = st.multiselect("Selecciona miembros a agregar", options=list(opt_map.keys()))
                    if st.button("Agregar al proyecto"):
                        if not add_sel:
                            st.warning("Selecciona al menos 1.")
                        else:
                            for k in add_sel:
                                ensure_project_member(con, pid, opt_map[k])
                            con.commit()
                            st.success("Miembros agregados.")
                            st.rerun()
                else:
                    st.caption("No hay miembros del grupo disponibles para agregar (ya están todos en el proyecto).")

                st.markdown("#### Remover miembros")
                if current:
                    cur_map = {fmt_user(m): m["id"] for m in current}
                    rem = st.multiselect("Selecciona miembros a remover", options=list(cur_map.keys()))
                    if st.button("Remover del proyecto"):
                        if not rem:
                            st.warning("Selecciona al menos 1.")
                        else:
                            # evitar dejar el proyecto vacío
                            if len(current) - len(rem) <= 0:
                                st.error("No puedes dejar el proyecto sin miembros.")
                            else:
                                for k in rem:
                                    con.execute("DELETE FROM project_members WHERE project_id=? AND user_id=?", (pid, cur_map[k]))
                                con.commit()
                                st.success("Miembros removidos.")
                                st.rerun()
                else:
                    st.caption("No hay miembros en el proyecto.")

        st.divider()
        st.subheader("Crear proyecto")
        if not has_permission(con, GROUP_ID, uid, "project_create"):
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
                    con.execute("""
                      INSERT INTO projects(group_id, name, description, wip_limit_doing, created_by, created_at)
                      VALUES (?, ?, ?, ?, ?, ?)
                    """, (GROUP_ID, name.strip(), desc.strip(), int(wip), uid, now_iso()))
                    pid = con.execute("SELECT last_insert_rowid() as id").fetchone()["id"]
                    ensure_project_member(con, pid, uid)
                    con.commit()
                    st.success("Proyecto creado.")
                    st.rerun()

        st.divider()
        st.subheader("Miembros del grupo y liderazgo")
        members = list_group_members(con, GROUP_ID)
        st.dataframe(pd.DataFrame(members), use_container_width=True)

        if is_leader and has_permission(con, GROUP_ID, uid, "group_transfer_lead"):
            st.markdown("#### Transferir liderazgo")
            choices = [m for m in members if m["role"] != "leader"]
            if choices:
                opt = safe_selectbox_dict("Elegir nuevo líder", choices, format_func=lambda x: fmt_user(x))
                if opt and st.button("Hacer líder"):
                    con.execute("UPDATE group_members SET role='member' WHERE group_id=? AND user_id=?", (GROUP_ID, uid))
                    con.execute("UPDATE group_members SET role='leader' WHERE group_id=? AND user_id=?", (GROUP_ID, opt["id"]))
                    con.commit()
                    st.success("Liderazgo transferido.")
                    st.rerun()
            else:
                st.caption("No hay otro miembro para transferir el liderazgo.")

        if is_leader and has_permission(con, GROUP_ID, uid, "group_manage_members"):
            st.markdown("#### Agregar miembro por código")
            g = con.execute("SELECT join_code FROM groups WHERE id=?", (GROUP_ID,)).fetchone()
            st.info(f"Código del grupo: **{g['join_code']}** (compártelo con tu equipo)")

# -------------------- Tablero (FIXED) --------------------
elif menu == "Tablero":
    with connect() as con:
        uid = st.session_state["user_id"]

        projs = rows_to_dicts(con.execute(
            "SELECT id, name, description, wip_limit_doing FROM projects WHERE group_id=? ORDER BY id DESC",
            (GROUP_ID,)
        ).fetchall())
        if not projs:
            st.info("Crea un proyecto primero.")
            st.stop()

        proj = safe_selectbox_dict("Proyecto", projs, format_func=lambda p: f"[{p['id']}] {p['name']}")
        if not proj:
            st.stop()
        pid = proj["id"]

        st.subheader("Tablero (Kanban) + filtros avanzados")

        tags = rows_to_dicts(con.execute(
            "SELECT * FROM tags WHERE group_id=? ORDER BY kind, name",
            (GROUP_ID,)
        ).fetchall())
        tag_opts = {f"{t['kind']}:{t['name']}": t["id"] for t in tags}

        # filtros
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            q = st.text_input("Búsqueda global (título/desc/DoD)")
        with f2:
            f_status = st.multiselect("Estado", ["todo","doing","blocked","awaiting_approval","done"],
                                      default=["todo","doing","blocked","awaiting_approval"])
        with f3:
            f_priority = st.multiselect("Prioridad", ["urgente","alta","media","baja"],
                                        default=["urgente","alta","media","baja"])
        with f4:
            only_mine = st.checkbox("Solo donde soy R/A", value=False)

        tag_filter = st.multiselect("Filtrar por tags/componentes", options=list(tag_opts.keys()))
        only_overdue = st.checkbox("Solo vencidas", value=False)
        only_blocked = st.checkbox("Solo bloqueadas", value=False)

        base = """
          SELECT t.*, p.name as project_name
          FROM tasks t
          JOIN projects p ON p.id=t.project_id
          WHERE p.group_id=? AND t.project_id=?
        """
        params = [GROUP_ID, pid]

        base += " AND t.status IN ({}) ".format(",".join(["?"]*len(f_status)))
        params += f_status
        base += " AND t.priority IN ({}) ".format(",".join(["?"]*len(f_priority)))
        params += f_priority

        if only_overdue:
            base += " AND t.due_date IS NOT NULL AND date(t.due_date) < date('now') AND t.status!='done' "
        if only_blocked:
            base += " AND t.status='blocked' "

        if q.strip():
            base += " AND (t.title LIKE ? OR t.description LIKE ? OR t.dod LIKE ?) "
            s = f"%{q.strip()}%"
            params += [s, s, s]

        if only_mine:
            base += """
              AND EXISTS(
                SELECT 1 FROM task_roles tr
                WHERE tr.task_id=t.id AND tr.user_id=? AND tr.role IN ('R','A')
              )
            """
            params.append(uid)

        if tag_filter:
            tag_ids = [tag_opts[k] for k in tag_filter]
            base += f"""
              AND EXISTS(
                SELECT 1 FROM task_tags tt
                WHERE tt.task_id=t.id AND tt.tag_id IN ({",".join(["?"]*len(tag_ids))})
              )
            """
            params += tag_ids

        base += " ORDER BY CASE t.priority WHEN 'urgente' THEN 1 WHEN 'alta' THEN 2 WHEN 'media' THEN 3 ELSE 4 END, t.due_date "

        tasks = rows_to_dicts(con.execute(base, tuple(params)).fetchall())

        by_status = {s: [] for s in ["todo","doing","blocked","awaiting_approval","done"]}
        for t in tasks:
            by_status[t["status"]].append(t)

        def render_task_card(t):
            roles = get_task_roles(con, t["id"])
            A = roles["A"][0]["name"] if roles["A"] else "—"
            Rs = ", ".join([x["name"] for x in roles["R"]]) if roles["R"] else "—"
            due = t["due_date"] or "—"

            deps = dependencies_open(con, t["id"])
            aging_days = None
            try:
                created = datetime.fromisoformat(t["created_at"])
                aging_days = (datetime.now() - created).days
            except Exception:
                pass

            with st.container(border=True):
                st.caption(f"#{t['id']} • {t['priority']} • due {due} • aging {aging_days}d" if aging_days is not None else f"#{t['id']} • {t['priority']} • due {due}")
                st.write(t["title"])
                st.caption(f"Dueño: {A}")
                st.caption(f"Ejecutores: {Rs}")
                if deps:
                    st.warning(f"Dependencias pendientes: {len(deps)}")

                if st.button("Abrir detalle", key=f"open_{t['id']}"):
                    st.session_state["selected_task_id"] = t["id"]
                    st.info("Ve a 'Tarea (detalle)' para ver/editar.")
                    # no rerun for now; Streamlit will rerender anyway

        cols = st.columns(5)
        labels = {
            "todo": "Por hacer",
            "doing": "En curso",
            "blocked": "Bloqueadas",
            "awaiting_approval": "En aprobación",
            "done": "Hechas"
        }
        for s, col in zip(labels.keys(), cols):
            with col:
                st.markdown(f"### {labels[s]}")
                for t in by_status[s]:
                    render_task_card(t)

        # Crear tarea
        st.divider()
        st.subheader("Crear tarea")
        if not has_permission(con, GROUP_ID, uid, "task_create"):
            st.info("No tienes permiso para crear tareas.")
        else:
            members = list_project_members(con, pid)
            if not members:
                st.warning("No hay miembros en el proyecto. Agrega al menos al creador en project_members.")
            m_opts = {fmt_user(m): m["id"] for m in members}

            t_tags = rows_to_dicts(con.execute("SELECT * FROM tags WHERE group_id=? AND kind='tag' ORDER BY name", (GROUP_ID,)).fetchall())
            t_comps = rows_to_dicts(con.execute("SELECT * FROM tags WHERE group_id=? AND kind='component' ORDER BY name", (GROUP_ID,)).fetchall())
            tag_names = [x["name"] for x in t_tags]
            comp_names = [x["name"] for x in t_comps]

            with st.form("new_task"):
                title = st.text_input("Título")
                desc = st.text_area("Descripción", height=80)
                dod = st.text_area("Criterio de aceptación (DoD)", height=80)

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    priority = st.selectbox("Prioridad", ["urgente","alta","media","baja"], index=2)
                with c2:
                    start_d = st.date_input("Start date (opcional)", value=None)
                with c3:
                    target_d = st.date_input("Target (SLA interno)", value=date.today()+timedelta(days=7))
                with c4:
                    due_d = st.date_input("Due date (deadline)", value=date.today()+timedelta(days=10))

                requires_approval = st.checkbox("Requiere aprobación del dueño para cerrar", value=True)

                st.markdown("Responsabilidad y comunicación (RACI por debajo)")
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

                existing = rows_to_dicts(con.execute("SELECT id, title FROM tasks WHERE project_id=? ORDER BY id DESC LIMIT 200", (pid,)).fetchall())
                dep_map = {f"#{x['id']} {x['title']}": x["id"] for x in existing}
                deps_sel = st.multiselect("Depende de (opcional)", options=list(dep_map.keys()))

                submit = st.form_submit_button("Crear tarea")

            if submit:
                if not title.strip():
                    st.error("Falta título.")
                elif not dod.strip():
                    st.error("Toda tarea debe tener DoD.")
                elif not R:
                    st.error("Debe haber al menos 1 ejecutor (R).")
                else:
                    con.execute("""
                      INSERT INTO tasks(project_id, title, description, dod, priority, status,
                                        start_date, target_date, due_date, requires_approval,
                                        created_by, created_at, updated_at)
                      VALUES (?, ?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pid, title.strip(), desc.strip(), dod.strip(), priority,
                        start_d.isoformat() if start_d else None,
                        target_d.isoformat() if target_d else None,
                        due_d.isoformat() if due_d else None,
                        int(requires_approval),
                        uid, now_iso(), now_iso()
                    ))
                    task_id = con.execute("SELECT last_insert_rowid() as id").fetchone()["id"]

                    try:
                        upsert_task_roles(
                            con, task_id, uid,
                            m_opts[A],
                            [m_opts[x] for x in R],
                            [m_opts[x] for x in C],
                            [m_opts[x] for x in I]
                        )
                    except Exception as e:
                        con.rollback()
                        st.error(str(e))
                        st.stop()

                    for d in deps_sel:
                        con.execute("INSERT OR IGNORE INTO task_dependencies(task_id, depends_on_task_id) VALUES (?,?)",
                                    (task_id, dep_map[d]))

                    def get_or_create_tag(name, kind):
                        r = con.execute("SELECT id FROM tags WHERE group_id=? AND name=? AND kind=?", (GROUP_ID, name, kind)).fetchone()
                        if r: return r["id"]
                        con.execute("INSERT INTO tags(group_id, name, kind) VALUES (?,?,?)", (GROUP_ID, name, kind))
                        return con.execute("SELECT last_insert_rowid() as id").fetchone()["id"]

                    for tn in selected_tags:
                        tid = get_or_create_tag(tn, "tag")
                        con.execute("INSERT OR IGNORE INTO task_tags(task_id, tag_id) VALUES (?,?)", (task_id, tid))
                    if selected_component != "(ninguno)":
                        cid = get_or_create_tag(selected_component, "component")
                        con.execute("INSERT OR IGNORE INTO task_tags(task_id, tag_id) VALUES (?,?)", (task_id, cid))

                    add_notification(con, m_opts[A], "info", f"Nueva tarea #{task_id}: eres dueño.", task_id)
                    for x in R:
                        add_notification(con, m_opts[x], "info", f"Nueva tarea #{task_id}: eres ejecutor.", task_id)
                    for x in C:
                        add_notification(con, m_opts[x], "input_request", f"Te consultan en tarea #{task_id}.", task_id)
                    for x in I:
                        add_notification(con, m_opts[x], "info", f"FYI: tarea #{task_id} creada.", task_id)

                    con.commit()
                    st.success("Tarea creada.")
                    st.rerun()

# -------------------- Tarea (detalle) (FIXED) --------------------
elif menu == "Tarea (detalle)":
    with connect() as con:
        uid = st.session_state["user_id"]

        st.subheader("Detalle de tarea")

        projs = rows_to_dicts(con.execute(
            "SELECT id, name FROM projects WHERE group_id=? ORDER BY id DESC",
            (GROUP_ID,)
        ).fetchall())
        if not projs:
            st.info("No hay proyectos.")
            st.stop()

        # si venimos desde tablero
        selected = st.session_state.get("selected_task_id")

        proj = safe_selectbox_dict("Proyecto", projs, format_func=lambda x: f"[{x['id']}] {x['name']}")
        if not proj:
            st.stop()
        pid = proj["id"]

        tasks = rows_to_dicts(con.execute(
            "SELECT id, title FROM tasks WHERE project_id=? ORDER BY id DESC LIMIT 300",
            (pid,)
        ).fetchall())
        if not tasks:
            st.info("No hay tareas.")
            st.stop()

        ids = [t["id"] for t in tasks]
        default_idx = ids.index(selected) if (selected in ids) else 0

        task_pick = st.selectbox(
            "Tarea",
            options=tasks,
            index=default_idx,
            format_func=lambda x: f"#{x['id']} {x['title']}"
        )
        task_id = task_pick["id"]

        t = con.execute("""
          SELECT t.*, p.group_id, p.name as project_name
          FROM tasks t JOIN projects p ON p.id=t.project_id
          WHERE t.id=?
        """, (task_id,)).fetchone()
        if not t:
            st.error("Tarea no encontrada.")
            st.stop()

        group_id = t["group_id"]

        roles = get_task_roles(con, task_id)
        A_id = get_accountable_id(con, task_id)
        deps = dependencies_open(con, task_id)

        st.markdown(f"### #{t['id']} — {t['title']}")
        st.caption(f"Proyecto: {t['project_name']} • Estado: {t['status']} • Prioridad: {t['priority']}")

        st.markdown("#### Qué se pide exactamente (DoD / criterio de aceptación)")
        st.write(t["dod"])

        st.markdown("#### Descripción")
        st.write(t["description"] or "")

        if deps:
            st.warning("Esta tarea tiene dependencias pendientes. Debe completarse primero:")
            st.dataframe(pd.DataFrame(deps), use_container_width=True)

        st.markdown("#### SLA / tiempos")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.write(f"Start: {t['start_date'] or '—'}")
        with c2:
            st.write(f"Target: {t['target_date'] or '—'}")
        with c3:
            st.write(f"Due: {t['due_date'] or '—'}")
        with c4:
            try:
                aging = (datetime.now() - datetime.fromisoformat(t["created_at"])).days
                st.write(f"Aging: {aging} días")
            except Exception:
                st.write("Aging: —")

        st.markdown("#### Responsabilidad y comunicación")
        st.write(f"Dueño: {roles['A'][0]['name'] if roles['A'] else '—'}")
        st.write("Ejecutores: " + (", ".join([x["name"] for x in roles["R"]]) or "—"))
        st.write("Consultados: " + (", ".join([x["name"] for x in roles["C"]]) or "—"))
        st.write("Informados: " + (", ".join([x["name"] for x in roles["I"]]) or "—"))

        st.divider()
        st.subheader("Acciones")

        # due_date (decision rights)
        if can_edit_due_date(con, group_id, task_id, uid):
            new_due = st.date_input("Cambiar due date", value=date.fromisoformat(t["due_date"]) if t["due_date"] else date.today())
            if st.button("Guardar due date"):
                old = t["due_date"]
                con.execute("UPDATE tasks SET due_date=?, updated_at=? WHERE id=?", (new_due.isoformat(), now_iso(), task_id))
                log_history(con, task_id, uid, "due_date", old, new_due.isoformat())
                if A_id:
                    add_notification(con, A_id, "info", f"Se cambió due date de la tarea #{task_id}.", task_id)
                con.commit()
                st.success("Due date actualizado.")
                st.rerun()
        else:
            st.caption("No puedes cambiar due date (solo dueño o líder, según gobernanza).")

        # status
        if can_change_status(con, group_id, task_id, uid):
            new_status = st.selectbox("Cambiar estado", ["todo","doing","blocked","awaiting_approval","done"],
                                      index=["todo","doing","blocked","awaiting_approval","done"].index(t["status"]))
            blocked_reason = None
            unblock_owner = None

            if new_status == "blocked":
                blocked_reason = st.text_input("Motivo de bloqueo")
                members = rows_to_dicts(con.execute("""
                  SELECT u.id, u.full_name, u.username
                  FROM group_members gm JOIN users u ON u.id=gm.user_id
                  WHERE gm.group_id=?
                  ORDER BY u.full_name
                """, (group_id,)).fetchall())
                opts = {fmt_user(m): m["id"] for m in members}
                who = st.selectbox("Quién debe destrabar", options=list(opts.keys()))
                unblock_owner = opts[who]

            if st.button("Aplicar estado"):
                if new_status in ("doing","awaiting_approval","done") and deps:
                    con.execute("UPDATE tasks SET status='blocked', blocked_reason=?, updated_at=? WHERE id=?",
                                ("Dependencias pendientes", now_iso(), task_id))
                    log_history(con, task_id, uid, "status", t["status"], "blocked")
                    con.commit()
                    st.error("No se puede avanzar: dependencias pendientes. Se marcó como BLOCKED.")
                    st.rerun()

                if new_status == "doing":
                    ok, msg = wip_limit_ok(con, t["project_id"], uid)
                    if not ok:
                        st.error(msg)
                        st.stop()

                if new_status == "done" and int(t["requires_approval"]) == 1:
                    if not A_id:
                        st.error("No hay dueño (A).")
                        st.stop()
                    con.execute("UPDATE tasks SET status='awaiting_approval', updated_at=? WHERE id=?",
                                (now_iso(), task_id))
                    con.execute("""
                      INSERT INTO approval_requests(task_id, requested_by, accountable_user_id, status, request_note, requested_at)
                      VALUES (?, ?, ?, 'pending', ?, ?)
                    """, (task_id, uid, A_id, "Solicitud de cierre", now_iso()))
                    log_history(con, task_id, uid, "status", t["status"], "awaiting_approval")
                    add_notification(con, A_id, "approval_request", f"Se solicita tu aprobación para cerrar tarea #{task_id}.", task_id)
                    con.commit()
                    st.success("Enviado a aprobación.")
                    st.rerun()

                completed_at = now_iso() if (new_status == "done" and int(t["requires_approval"]) == 0) else None

                con.execute("""
                  UPDATE tasks
                  SET status=?, blocked_reason=?, unblock_owner_user_id=?, updated_at=?, completed_at=COALESCE(?, completed_at)
                  WHERE id=?
                """, (
                    new_status,
                    blocked_reason if new_status == "blocked" else None,
                    unblock_owner if new_status == "blocked" else None,
                    now_iso(),
                    completed_at,
                    task_id
                ))
                log_history(con, task_id, uid, "status", t["status"], new_status)

                if A_id:
                    add_notification(con, A_id, "status_change", f"Estado de tarea #{task_id} cambió a {new_status}.", task_id)
                if new_status in ("blocked","done"):
                    informed = con.execute("SELECT user_id FROM task_roles WHERE task_id=? AND role='I'", (task_id,)).fetchall()
                    for rr in informed:
                        add_notification(con, rr["user_id"], "info", f"FYI: tarea #{task_id} ahora está {new_status}.", task_id)

                con.commit()
                st.success("Estado actualizado.")
                st.rerun()
        else:
            st.caption("No puedes cambiar estado (solo ejecutor o líder).")

        # Aprobaciones
        st.divider()
        st.subheader("Aprobaciones")
        pend = con.execute("""
          SELECT * FROM approval_requests WHERE task_id=? AND status='pending' ORDER BY id DESC LIMIT 1
        """, (task_id,)).fetchone()

        if pend:
            st.info("Hay una aprobación pendiente.")
            if A_id == uid:
                decision_note = st.text_input("Nota de decisión")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Aprobar"):
                        con.execute("""
                          UPDATE approval_requests
                          SET status='approved', decision_note=?, decided_at=?
                          WHERE id=?
                        """, (decision_note, now_iso(), pend["id"]))
                        con.execute("UPDATE tasks SET status='done', updated_at=?, completed_at=? WHERE id=?",
                                    (now_iso(), now_iso(), task_id))
                        log_history(con, task_id, uid, "approval", "pending", "approved")
                        log_history(con, task_id, uid, "status", "awaiting_approval", "done")
                        rs = con.execute("SELECT user_id FROM task_roles WHERE task_id=? AND role='R'", (task_id,)).fetchall()
                        for r in rs:
                            add_notification(con, r["user_id"], "info", f"Tarea #{task_id} aprobada y cerrada.", task_id)
                        con.commit()
                        st.success("Aprobado.")
                        st.rerun()
                with b2:
                    if st.button("Rechazar"):
                        con.execute("""
                          UPDATE approval_requests
                          SET status='rejected', decision_note=?, decided_at=?
                          WHERE id=?
                        """, (decision_note, now_iso(), pend["id"]))
                        con.execute("UPDATE tasks SET status='doing', updated_at=? WHERE id=?",
                                    (now_iso(), task_id))
                        log_history(con, task_id, uid, "approval", "pending", "rejected")
                        log_history(con, task_id, uid, "status", "awaiting_approval", "doing")
                        con.commit()
                        st.warning("Rechazado. Devuelto a DOING.")
                        st.rerun()
            else:
                st.caption("Solo el dueño (A) puede decidir.")
        else:
            st.caption("No hay aprobaciones pendientes.")

        # Comentarios
        st.divider()
        st.subheader("Avances / notas")
        comm = rows_to_dicts(con.execute("""
          SELECT c.*, u.full_name
          FROM task_comments c JOIN users u ON u.id=c.user_id
          WHERE c.task_id=? ORDER BY c.id DESC
        """, (task_id,)).fetchall())

        for c in comm[:15]:
            st.caption(f"{c['full_name']} • {c['created_at']} • progreso {c['progress_pct'] if c['progress_pct'] is not None else '—'}%")
            st.write(c["comment"])
            if c.get("next_step"):
                st.caption(f"Próximo paso: {c['next_step']}")

        new_comment = st.text_area("Nuevo comentario", height=80)
        progress = st.slider("Progreso (%)", 0, 100, 0)
        next_step = st.text_input("Próximo paso (opcional)")
        if st.button("Guardar avance"):
            con.execute("""
              INSERT INTO task_comments(task_id, user_id, comment, progress_pct, next_step, created_at)
              VALUES (?, ?, ?, ?, ?, ?)
            """, (task_id, uid, new_comment.strip() or "(sin texto)", int(progress), next_step.strip() or None, now_iso()))
            log_history(con, task_id, uid, "comment", None, "added")
            if A_id and A_id != uid:
                add_notification(con, A_id, "info", f"Nuevo avance en tarea #{task_id}.", task_id)
            con.commit()
            st.success("Guardado.")
            st.rerun()

        # Archivos
        st.divider()
        st.subheader("Archivos")
        if has_permission(con, group_id, uid, "task_upload_files"):
            up = st.file_uploader("Subir archivo")
            if st.button("Guardar archivo"):
                if up is None:
                    st.error("Sube un archivo.")
                else:
                    task_dir = os.path.join(UPLOAD_DIR, f"task_{task_id}")
                    os.makedirs(task_dir, exist_ok=True)
                    unique = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}_{up.name}"
                    path = os.path.join(task_dir, unique)
                    with open(path, "wb") as f:
                        f.write(up.getbuffer())
                    con.execute("""
                      INSERT INTO task_attachments(task_id, user_id, filename, stored_path, created_at)
                      VALUES (?, ?, ?, ?, ?)
                    """, (task_id, uid, up.name, path, now_iso()))
                    log_history(con, task_id, uid, "attachment", None, up.name)
                    if A_id:
                        add_notification(con, A_id, "info", f"Archivo subido en tarea #{task_id}: {up.name}", task_id)
                    con.commit()
                    st.success("Archivo guardado.")
                    st.rerun()
        else:
            st.caption("No tienes permiso para subir archivos.")

        atts = rows_to_dicts(con.execute("""
          SELECT a.*, u.full_name
          FROM task_attachments a JOIN users u ON u.id=a.user_id
          WHERE a.task_id=? ORDER BY a.id DESC
        """, (task_id,)).fetchall())

        for a in atts[:20]:
            st.caption(f"{a['full_name']} • {a['created_at']} • {a['filename']}")
            if os.path.exists(a["stored_path"]):
                with open(a["stored_path"], "rb") as f:
                    st.download_button("Descargar", data=f, file_name=a["filename"], key=f"dl_{a['id']}")

        # Editar responsabilidades
        st.divider()
        st.subheader("Editar responsabilidades")
        if has_permission(con, group_id, uid, "task_change_raci"):
            members = rows_to_dicts(con.execute("""
              SELECT u.id, u.full_name, u.username
              FROM group_members gm JOIN users u ON u.id=gm.user_id
              WHERE gm.group_id=? ORDER BY u.full_name
            """, (group_id,)).fetchall())
            opts = {fmt_user(m): m["id"] for m in members}

            current = get_task_roles(con, task_id)

            def default_keys(role):
                keys = []
                for x in current[role]:
                    for k in opts.keys():
                        if x["name"] in k:
                            keys.append(k)
                            break
                return keys

            # Dueño actual
            current_A_name = current["A"][0]["name"] if current["A"] else None
            A_pick_default = None
            if current_A_name:
                for k in opts.keys():
                    if current_A_name in k:
                        A_pick_default = k
                        break
            A_pick = st.selectbox("Dueño (A)", options=list(opts.keys()),
                                  index=(list(opts.keys()).index(A_pick_default) if A_pick_default in opts else 0))
            R_pick = st.multiselect("Ejecutores (R)", options=list(opts.keys()), default=default_keys("R"))
            C_pick = st.multiselect("Consultados (C)", options=list(opts.keys()), default=default_keys("C"))
            I_pick = st.multiselect("Informados (I)", options=list(opts.keys()), default=default_keys("I"))

            if st.button("Guardar responsabilidades"):
                try:
                    upsert_task_roles(
                        con, task_id, uid,
                        opts[A_pick],
                        [opts[x] for x in R_pick],
                        [opts[x] for x in C_pick],
                        [opts[x] for x in I_pick]
                    )
                    add_notification(con, opts[A_pick], "info", f"Actualización: eres dueño de tarea #{task_id}.", task_id)
                    for x in R_pick:
                        add_notification(con, opts[x], "info", f"Actualización: eres ejecutor de tarea #{task_id}.", task_id)
                    for x in C_pick:
                        add_notification(con, opts[x], "input_request", f"Actualización: te consultan en tarea #{task_id}.", task_id)
                    for x in I_pick:
                        add_notification(con, opts[x], "info", f"Actualización FYI en tarea #{task_id}.", task_id)
                    con.commit()
                    st.success("Actualizado.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        else:
            st.caption("No tienes permiso para editar responsabilidades.")

# -------------------- Plantillas --------------------
elif menu == "Plantillas":
    with connect() as con:
        uid = st.session_state["user_id"]
        st.subheader("Plantillas de proyecto")
        if not has_permission(con, GROUP_ID, uid, "template_manage"):
            st.info("No tienes permiso para gestionar plantillas.")
            st.stop()

        templates = rows_to_dicts(con.execute("SELECT * FROM templates WHERE group_id=? ORDER BY id DESC", (GROUP_ID,)).fetchall())
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
                    con.execute("""
                      INSERT INTO templates(group_id, name, description, created_by, created_at)
                      VALUES (?, ?, ?, ?, ?)
                    """, (GROUP_ID, name.strip(), desc.strip(), uid, now_iso()))
                    con.commit()
                    st.success("Plantilla creada.")
                    st.rerun()

        if templates:
            tpl = safe_selectbox_dict("Plantilla", templates, format_func=lambda x: f"[{x['id']}] {x['name']}")
            tpl_id = tpl["id"]

            st.markdown("#### Tareas dentro de plantilla")
            tpls = rows_to_dicts(con.execute("SELECT * FROM template_tasks WHERE template_id=? ORDER BY id", (tpl_id,)).fetchall())
            st.dataframe(pd.DataFrame(tpls) if tpls else pd.DataFrame(), use_container_width=True)

            st.markdown("#### Agregar tarea a plantilla")
            with st.form("tpl_task"):
                tt = st.text_input("Título")
                td = st.text_area("Descripción", height=60)
                dod = st.text_area("DoD", height=60)
                priority = st.selectbox("Prioridad", ["urgente","alta","media","baja"], index=2)
                req = st.checkbox("Requiere aprobación", value=True)
                days_from = st.number_input("Días desde inicio del proyecto (para target/due)", min_value=0, max_value=365, value=0, step=1)
                tags_csv = st.text_input("Tags (separados por coma)", value="finanzas,operaciones")
                component = st.text_input("Componente (opcional)", value="")
                submit = st.form_submit_button("Agregar a plantilla")
            if submit:
                if not (tt.strip() and dod.strip()):
                    st.error("Título y DoD son obligatorios.")
                else:
                    con.execute("""
                      INSERT INTO template_tasks(template_id, title, description, dod, priority, requires_approval, days_from_start, tags_csv, component_name)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (tpl_id, tt.strip(), td.strip(), dod.strip(), priority, int(req), int(days_from), tags_csv.strip(), component.strip() or None))
                    con.commit()
                    st.success("Agregado.")
                    st.rerun()

            st.divider()
            st.subheader("Aplicar plantilla a un proyecto")
            projs = rows_to_dicts(con.execute("SELECT id, name FROM projects WHERE group_id=? ORDER BY id DESC", (GROUP_ID,)).fetchall())
            if not projs:
                st.info("Crea un proyecto primero.")
                st.stop()
            proj = safe_selectbox_dict("Proyecto destino", projs, format_func=lambda x: f"[{x['id']}] {x['name']}")
            start = st.date_input("Fecha inicio base para plan (start_date)", value=date.today())
            if st.button("Aplicar"):
                tasks_tpl = rows_to_dicts(con.execute("SELECT * FROM template_tasks WHERE template_id=? ORDER BY id", (tpl_id,)).fetchall())
                for tt in tasks_tpl:
                    target = (start + timedelta(days=int(tt["days_from_start"]))).isoformat()
                    due = (start + timedelta(days=int(tt["days_from_start"]) + 2)).isoformat()
                    con.execute("""
                      INSERT INTO tasks(project_id, title, description, dod, priority, status,
                                        start_date, target_date, due_date, requires_approval,
                                        created_by, created_at, updated_at)
                      VALUES (?, ?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?, ?, ?)
                    """, (proj["id"], tt["title"], tt["description"], tt["dod"], tt["priority"],
                          start.isoformat(), target, due, int(tt["requires_approval"]),
                          uid, now_iso(), now_iso()))
                    task_id = con.execute("SELECT last_insert_rowid() as id").fetchone()["id"]

                    def get_or_create(name, kind):
                        r = con.execute("SELECT id FROM tags WHERE group_id=? AND name=? AND kind=?", (GROUP_ID, name, kind)).fetchone()
                        if r: return r["id"]
                        con.execute("INSERT INTO tags(group_id, name, kind) VALUES (?,?,?)", (GROUP_ID, name, kind))
                        return con.execute("SELECT last_insert_rowid() as id").fetchone()["id"]

                    if tt.get("tags_csv"):
                        for name in [x.strip() for x in tt["tags_csv"].split(",") if x.strip()]:
                            tid = get_or_create(name, "tag")
                            con.execute("INSERT OR IGNORE INTO task_tags(task_id, tag_id) VALUES (?,?)", (task_id, tid))
                    if tt.get("component_name"):
                        cid = get_or_create(tt["component_name"], "component")
                        con.execute("INSERT OR IGNORE INTO task_tags(task_id, tag_id) VALUES (?,?)", (task_id, cid))

                con.commit()
                st.success("Plantilla aplicada. Ahora asigna responsabilidades a cada tarea.")
                st.rerun()

# -------------------- Export (FIXED) --------------------
elif menu == "Export":
    with connect() as con:
        uid = st.session_state["user_id"]
        st.subheader("Export")

        if not has_permission(con, GROUP_ID, uid, "export_data"):
            st.info("No tienes permiso de export.")
            st.stop()

        projs = rows_to_dicts(con.execute(
            "SELECT id, name FROM projects WHERE group_id=? ORDER BY id DESC",
            (GROUP_ID,)
        ).fetchall())
        if not projs:
            st.info("No hay proyectos.")
            st.stop()

        proj = safe_selectbox_dict("Proyecto", projs, format_func=lambda x: f"[{x['id']}] {x['name']}")
        pid = proj["id"]

        rows = con.execute("""
          SELECT t.*
          FROM tasks t
          WHERE t.project_id=?
          ORDER BY t.id
        """, (pid,)).fetchall()
        df = pd.DataFrame(rows_to_dicts(rows)) if rows else pd.DataFrame()
        st.dataframe(df, use_container_width=True)

        st.download_button("Descargar CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"tasks_project_{pid}.csv")

        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="tasks")
        st.download_button("Descargar Excel", data=buf.getvalue(),
                           file_name=f"tasks_project_{pid}.xlsx")

        events = []
        for r in rows_to_dicts(rows):
            events.append(task_to_ics_event(r["id"], r["title"], r["due_date"], r["dod"]))
        ics = tasks_to_ics_calendar(events, cal_name=f"Proyecto {proj['name']}")
        st.download_button("Descargar ICS", data=ics.encode("utf-8"),
                           file_name=f"tasks_project_{pid}.ics")

# -------------------- Gobernanza --------------------
elif menu == "Gobernanza":
    with connect() as con:
        uid = st.session_state["user_id"]
        st.subheader("Gobernanza del grupo")

        role = user_role_in_group(con, GROUP_ID, uid)
        if role != "leader":
            st.info("Solo líderes del grupo pueden gestionar gobernanza.")
            st.stop()

        st.markdown("### Permisos finos por rol (acciones)")
        perms = rows_to_dicts(con.execute("""
          SELECT role, action, allowed FROM group_role_permissions
          WHERE group_id=? ORDER BY role, action
        """, (GROUP_ID,)).fetchall())

        dfp = pd.DataFrame(perms)
        st.dataframe(dfp, use_container_width=True)

        st.markdown("#### Editar permisos (cuidado: esto afecta la gobernanza)")
        action = st.selectbox("Acción", options=sorted(dfp["action"].unique().tolist()))
        role_pick = st.selectbox("Rol", options=["leader","member"])
        allowed = st.checkbox("Permitido", value=True)
        if st.button("Guardar permiso"):
            con.execute("""
              INSERT OR REPLACE INTO group_role_permissions(group_id, role, action, allowed)
              VALUES (?, ?, ?, ?)
            """, (GROUP_ID, role_pick, action, int(allowed)))
            con.commit()
            st.success("Actualizado.")
            st.rerun()

        st.divider()
        st.markdown("### Reglas recomendadas")
        st.write("- Toda tarea debe tener DoD (criterio de aceptación).")
        st.write("- Si requiere aprobación: no se cierra sin decisión del Dueño.")
        st.write("- Blocked requiere motivo y dueño de desbloqueo.")
        st.write("- Limitar WIP para evitar sobrecarga y multitarea.")
# -------------------- Reportes --------------------
elif menu == "Reportes":
    with connect() as con:
        st.subheader("Reporte semanal")

        projs = rows_to_dicts(con.execute(
            "SELECT id, name FROM projects WHERE group_id=? ORDER BY id DESC",
            (GROUP_ID,)
        ).fetchall())

        proj_filter = st.selectbox(
            "Proyecto (opcional)",
            options=[None] + [p["id"] for p in projs],
            format_func=lambda x: "Todos" if x is None else f"[{x}] " + next(p["name"] for p in projs if p["id"] == x)
        )

        ref = st.date_input("Semana de referencia", value=date.today())
        ws, we = week_bounds(ref)
        st.caption(f"Semana: {ws.isoformat()} → {we.isoformat()}")

        df_done, df_pending = build_weekly_report(con, GROUP_ID, ws, we, project_id=proj_filter)

        st.markdown("### ✅ Completadas")
        st.dataframe(df_done, use_container_width=True, height=280)

        st.markdown("### ⏳ Pendientes")
        st.dataframe(df_pending, use_container_width=True, height=280)

        # Export Excel
        import io
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_done.to_excel(writer, index=False, sheet_name="done")
            df_pending.to_excel(writer, index=False, sheet_name="pending")
        st.download_button("Descargar Excel del reporte", data=buf.getvalue(), file_name=f"weekly_report_{ws.isoformat()}.xlsx")

        # Email (si configurado)
        #st.divider()
        #st.subheader("Enviar por email")
        #to_list = st.text_input("Destinatarios (separados por coma)", value="")
        #if st.button("Enviar reporte"):
            #emails = [x.strip() for x in to_list.split(",") if x.strip()]
            #if not emails:
                #st.error("Pon al menos 1 email.")
            #else:
                #html = report_to_html(df_done, df_pending, title=f"Reporte semanal {ws.isoformat()} → {we.isoformat()}")
                #try:
                    #send_email_smtp(
                        #subject=f"Reporte semanal - {ws.isoformat()} a {we.isoformat()}",
                        #to_emails=emails,
                        #html_body=html
                    #)
                    #st.success("Email enviado.")
                #except Exception as e:
                    #st.error(f"No se pudo enviar el email: {e}")


# -------------------- Ayuda (glosario de acrónimos) --------------------
elif menu == "Ayuda":
    st.subheader("Guía rápida: cómo usar el sistema")

    st.markdown("""
### Objetivo
Esta herramienta organiza proyectos y tareas por grupo, con control de responsabilidades, trazabilidad y gobernanza.

---

## Glosario (acrónimos y términos)

**RACI**: *Responsibility Assignment Matrix*. Modelo para definir roles:
- **R (Responsible)**: ejecuta
- **A (Accountable)**: dueño final / aprueba
- **C (Consulted)**: consultado (ida-vuelta)
- **I (Informed)**: informado (solo ida)

**DoD (Definition of Done)**: “Definición de hecho”. Criterio de aceptación para cerrar una tarea sin discusión.

**WIP (Work In Progress)**: trabajo en curso. Límite de tareas en DOING por persona para evitar multitarea excesiva.

**SLA (Service Level Agreement)**: en esta app se usa como “objetivo de tiempo” (target_date). No es contrato legal aquí, es control operativo.

**Aging**: días desde que se creó la tarea (sirve para detectar estancamiento).

**FYI**: *For Your Information*. Notificación informativa, no requiere acción.

**ICS**: formato de calendario iCalendar. Te permite exportar fechas para Google Calendar/Outlook.

**CSV**: archivo de texto tabular (separado por comas). Export simple para Excel/BI.

**KPI (Key Performance Indicator)**: métrica clave (ej: vencidas, lead time, % completadas).

**ETA (Estimated Time of Arrival / Estimated Time to Complete)**: estimación de cuándo estará listo (no implementado aún como campo, pero puede añadirse).

**Lead time**: tiempo desde “todo” hasta “done”. (Base de datos permite calcularlo usando timestamps.)

**Blocked**: bloqueada (requiere motivo + responsable de desbloqueo).

**Approval workflow**: flujo de aprobación. Si una tarea requiere aprobación, pasa a “awaiting_approval” antes de “done”.
                
**Kanban**: Principios y prácticas de control y de manejo para mejorar el flujo de trabajo y sus riesgos, cambios progresivos y WIP.

---

## Reglas que el sistema fuerza (buenas prácticas)
- 1 solo **Dueño (A)** por tarea.
- mínimo 1 **Ejecutor (R)**.
- Tarea sin DoD = tarea ambigua (evítalo).
- Si hay dependencias pendientes, no se deja cerrar/avanzar.
- Si requiere aprobación, no se cierra sin decisión del Dueño.

---

## Uso recomendado
- Revisión semanal obligatoria (panel Resumen).
- WIP bajo (2–3) por persona, puede variar.
- Blocked siempre con motivo y responsable.
- DoD siempre (aunque sea corto).
""")

# -------------------- Sidebar notifications quick view --------------------
with connect() as con:
    if st.sidebar.button("Ver notificaciones"):
        rows = con.execute("""
          SELECT kind, message, created_at FROM notifications
          WHERE user_id=? ORDER BY id DESC LIMIT 20
        """, (st.session_state["user_id"],)).fetchall()
        st.sidebar.markdown("### Últimas notificaciones")
        for r in rows_to_dicts(rows):
            st.sidebar.caption(f"{r['created_at']} • {r['kind']}")
            st.sidebar.write(r["message"])
