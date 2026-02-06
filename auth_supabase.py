import streamlit as st
from supabase import create_client


def _cfg() -> dict:
    return st.secrets.get("connections", {}).get("supabase", {})


def _require(v: str | None, msg: str) -> str:
    if not v or not str(v).strip():
        raise RuntimeError(msg)
    return str(v).strip()


@st.cache_resource
def sb_base():
    """
    Cliente ANON base (sin sesión). Se usa para sign_in y sign_out.
    """
    cfg = _cfg()
    url = _require(cfg.get("url"), "Falta connections.supabase.url en secrets.toml")
    anon = _require(
        cfg.get("anon_key") or cfg.get("key"),
        "Falta connections.supabase.anon_key (o key) en secrets.toml",
    )
    return create_client(url, anon)


def sb_user():
    """
    Cliente que actúa como el usuario logueado (RLS aplica y auth.uid() funciona).
    Requiere st.session_state['sb_session'].
    """
    cfg = _cfg()
    url = _require(cfg.get("url"), "Falta connections.supabase.url en secrets.toml")
    anon = _require(
        cfg.get("anon_key") or cfg.get("key"),
        "Falta connections.supabase.anon_key (o key) en secrets.toml",
    )

    sess = st.session_state.get("sb_session")
    if not sess:
        raise RuntimeError("No hay sesión activa (sb_session). Inicia sesión primero.")

    client = create_client(url, anon)
    client.auth.set_session(sess.access_token, sess.refresh_token)
    return client


def sign_in(email: str, password: str):
    """
    Inicia sesión y guarda user/session en st.session_state:
      - sb_session
      - sb_user
    """
    res = sb_base().auth.sign_in_with_password({"email": email, "password": password})

    # Guardar sesión para que db.py pueda usarla
    st.session_state["sb_session"] = res.session
    st.session_state["sb_user"] = res.user

    return res


def sign_out():
    """
    Cierra sesión y limpia session_state.
    """
    try:
        sb_base().auth.sign_out()
    except Exception:
        pass

    for k in ["sb_session", "sb_user"]:
        if k in st.session_state:
            del st.session_state[k]


def ensure_session_from_refresh_token():
    """
    Opcional: si tú guardas refresh_token en algún lado (cookie/kv),
    puedes rehidratar la sesión aquí.
    En Streamlit puro, normalmente no persistes cookies, así que esto
    se usa solo si tú implementas persistencia.
    """
    return
