# ics_utils.py
from datetime import datetime

def _dtstamp():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

def task_to_ics_event(task_id: int, title: str, due_date_iso: str | None, description: str = "") -> str:
    # Evento "all-day" si hay due_date
    if not due_date_iso:
        return ""
    dt = due_date_iso.replace("-", "")
    uid = f"task-{task_id}@workflow"
    # Calcular DTEND = día siguiente (requerido por Google Calendar / RFC 5545)
    from datetime import datetime as _dt
    dt_obj = _dt.strptime(dt, "%Y%m%d")
    from datetime import timedelta as _td
    dtend = (dt_obj + _td(days=1)).strftime("%Y%m%d")

    lines = [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{_dtstamp()}",
        f"DTSTART;VALUE=DATE:{dt}",
        f"DTEND;VALUE=DATE:{dtend}",
        f"SUMMARY:{title}",
    ]
    if description:
        safe_desc = description.replace("\n", "\\n")
        lines.append(f"DESCRIPTION:{safe_desc}")
    lines.append("END:VEVENT")
    return "\n".join(lines)

def tasks_to_ics_calendar(events: list[str], cal_name: str = "Workflow Tasks") -> str:
    body = "\n".join([e for e in events if e.strip()])
    return "\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//WorkflowApp//EN",
        f"X-WR-CALNAME:{cal_name}",
        body,
        "END:VCALENDAR"
    ])
