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
    lines = [
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{_dtstamp()}",
        f"DTSTART;VALUE=DATE:{dt}",
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
