import streamlit as st
from supabase import create_client

@st.cache_resource
def sb_anon():
    cfg = st.secrets["connections"]["supabase"]
    # soporta tanto "anon_key" como "key" por si aún no cambiaste secrets
    anon_key = cfg.get("anon_key") or cfg.get("key")
    if not anon_key:
        raise RuntimeError("Falta anon_key (o key) en [connections.supabase] de secrets.toml")
    return create_client(cfg["url"], anon_key)

@st.cache_resource
def sb_admin():
    cfg = st.secrets["connections"]["supabase"]
    # soporta tanto "service_role_key" como "key" por si aún usas service role como key
    srv = cfg.get("service_role_key") or cfg.get("key")
    if not srv:
        raise RuntimeError("Falta service_role_key (o key) en [connections.supabase] de secrets.toml")
    return create_client(cfg["url"], srv)

def sign_in(email: str, password: str):
    return sb_anon().auth.sign_in_with_password({"email": email, "password": password})

def sign_out():
    try:
        sb_anon().auth.sign_out()
    except Exception:
        pass

def admin_create_user(email: str, password: str, email_confirm: bool = True):
    return sb_admin().auth.admin.create_user({
        "email": email,
        "password": password,
        "email_confirm": bool(email_confirm),
    })
