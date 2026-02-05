# auth_supabase.py
import streamlit as st
from supabase import create_client

@st.cache_resource
def sb():
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]  # anon
    return create_client(url, key)

def sign_up(email: str, password: str):
    return sb().auth.sign_up({"email": email, "password": password})

def sign_in(email: str, password: str):
    return sb().auth.sign_in_with_password({"email": email, "password": password})

def sign_out():
    try:
        sb().auth.sign_out()
    except Exception:
        pass
