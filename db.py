# db.py
import os
import sqlite3
from datetime import datetime

DB_PATH = "workflow.db"
UPLOAD_DIR = "uploads"

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
        "export_data",  # opcional: puedes quitar si prefieres
    },
}

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def connect():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON;")
    return con

def init_db():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with connect() as con:
        cur = con.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        )
        """)

        # Grupos/Equipos
        cur.execute("""
        CREATE TABLE IF NOT EXISTS groups(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            join_code TEXT UNIQUE NOT NULL,
            created_by INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(created_by) REFERENCES users(id)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS group_members(
            group_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('leader','member')) DEFAULT 'member',
            joined_at TEXT NOT NULL,
            PRIMARY KEY(group_id, user_id),
            FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)

        # Permisos finos por grupo y rol
        cur.execute("""
        CREATE TABLE IF NOT EXISTS group_role_permissions(
            group_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('leader','member')),
            action TEXT NOT NULL,
            allowed INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY(group_id, role, action),
            FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
        )
        """)

        # Proyectos dentro de grupo
        cur.execute("""
        CREATE TABLE IF NOT EXISTS projects(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            wip_limit_doing INTEGER NOT NULL DEFAULT 3,
            created_by INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE,
            FOREIGN KEY(created_by) REFERENCES users(id)
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS project_members(
            project_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            PRIMARY KEY(project_id, user_id),
            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)

        # Tareas
        cur.execute("""
        CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            dod TEXT NOT NULL, -- criterio de aceptación
            priority TEXT NOT NULL CHECK(priority IN ('urgente','alta','media','baja')) DEFAULT 'media',
            status TEXT NOT NULL CHECK(status IN ('todo','doing','blocked','awaiting_approval','done')) DEFAULT 'todo',

            start_date TEXT,     -- YYYY-MM-DD
            target_date TEXT,    -- YYYY-MM-DD (SLA interno / objetivo)
            due_date TEXT,       -- YYYY-MM-DD (deadline)

            requires_approval INTEGER NOT NULL DEFAULT 0,

            blocked_reason TEXT,
            unblock_owner_user_id INTEGER, -- quien debe destrabar

            created_by INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,

            FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE,
            FOREIGN KEY(unblock_owner_user_id) REFERENCES users(id),
            FOREIGN KEY(created_by) REFERENCES users(id)
        )
        """)

        # Responsabilidades (RACI debajo, pero UI amigable)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS task_roles(
            task_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('A','R','C','I')),
            assigned_at TEXT NOT NULL,
            PRIMARY KEY(task_id, user_id, role),
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        # 1 A por tarea (enforce)
        cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_one_A_per_task
        ON task_roles(task_id)
        WHERE role='A'
        """)

        # Dependencias
        cur.execute("""
        CREATE TABLE IF NOT EXISTS task_dependencies(
            task_id INTEGER NOT NULL,
            depends_on_task_id INTEGER NOT NULL,
            PRIMARY KEY(task_id, depends_on_task_id),
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(depends_on_task_id) REFERENCES tasks(id) ON DELETE CASCADE
        )
        """)

        # Tags y componentes (component puede ser tag tipo)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS tags(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('tag','component')) DEFAULT 'tag',
            UNIQUE(group_id, name, kind),
            FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS task_tags(
            task_id INTEGER NOT NULL,
            tag_id INTEGER NOT NULL,
            PRIMARY KEY(task_id, tag_id),
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
        )
        """)

        # Comentarios / avances
        cur.execute("""
        CREATE TABLE IF NOT EXISTS task_comments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            comment TEXT NOT NULL,
            progress_pct INTEGER,
            next_step TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)

        # Adjuntos
        cur.execute("""
        CREATE TABLE IF NOT EXISTS task_attachments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)

        # Auditoría
        cur.execute("""
        CREATE TABLE IF NOT EXISTS task_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            field TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """)

        # Aprobaciones
        cur.execute("""
        CREATE TABLE IF NOT EXISTS approval_requests(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            requested_by INTEGER NOT NULL,
            accountable_user_id INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected')) DEFAULT 'pending',
            request_note TEXT,
            decision_note TEXT,
            requested_at TEXT NOT NULL,
            decided_at TEXT,
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE,
            FOREIGN KEY(requested_by) REFERENCES users(id),
            FOREIGN KEY(accountable_user_id) REFERENCES users(id)
        )
        """)

        # Notificaciones in-app (+ base para email/teams)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS notifications(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            message TEXT NOT NULL,
            task_id INTEGER,
            created_at TEXT NOT NULL,
            read_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
        )
        """)

        # Outbox para futuras integraciones (email/teams)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS outbound_queue(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel TEXT NOT NULL CHECK(channel IN ('email','teams')),
            to_address TEXT,
            payload TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('pending','sent','failed')) DEFAULT 'pending',
            created_at TEXT NOT NULL,
            last_error TEXT
        )
        """)

        # Plantillas
        cur.execute("""
        CREATE TABLE IF NOT EXISTS templates(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            created_by INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(group_id, name),
            FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE,
            FOREIGN KEY(created_by) REFERENCES users(id)
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS template_tasks(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            dod TEXT NOT NULL,
            priority TEXT NOT NULL CHECK(priority IN ('urgente','alta','media','baja')) DEFAULT 'media',
            requires_approval INTEGER NOT NULL DEFAULT 0,
            days_from_start INTEGER NOT NULL DEFAULT 0, -- para target/due relativo
            tags_csv TEXT,
            component_name TEXT,
            FOREIGN KEY(template_id) REFERENCES templates(id) ON DELETE CASCADE
        )
        """)

        con.commit()

def seed_group_permissions(con, group_id: int):
    # Inserta la matriz default si no existe
    for role, actions in DEFAULT_ROLE_PERMS.items():
        for action in ACTIONS:
            allowed = 1 if action in actions else 0
            con.execute("""
              INSERT OR IGNORE INTO group_role_permissions(group_id, role, action, allowed)
              VALUES (?, ?, ?, ?)
            """, (group_id, role, action, allowed))

def user_role_in_group(con, group_id: int, user_id: int) -> str | None:
    r = con.execute("""
      SELECT role FROM group_members WHERE group_id=? AND user_id=?
    """, (group_id, user_id)).fetchone()
    return r["role"] if r else None

def has_permission(con, group_id: int, user_id: int, action: str) -> bool:
    role = user_role_in_group(con, group_id, user_id)
    if not role:
        return False
    r = con.execute("""
      SELECT allowed FROM group_role_permissions
      WHERE group_id=? AND role=? AND action=?
    """, (group_id, role, action)).fetchone()
    return bool(r["allowed"]) if r else False

def add_notification(con, user_id: int, kind: str, message: str, task_id: int | None):
    con.execute("""
      INSERT INTO notifications(user_id, kind, message, task_id, created_at)
      VALUES (?, ?, ?, ?, ?)
    """, (user_id, kind, message, task_id, now_iso()))

def log_history(con, task_id: int, user_id: int, field: str, old_v: str | None, new_v: str | None):
    con.execute("""
      INSERT INTO task_history(task_id, user_id, field, old_value, new_value, created_at)
      VALUES (?, ?, ?, ?, ?, ?)
    """, (task_id, user_id, field, old_v, new_v, now_iso()))
