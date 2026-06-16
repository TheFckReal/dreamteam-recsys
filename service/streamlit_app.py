"""Streamlit frontend for the DreamTeam RecSys API."""

from __future__ import annotations

from datetime import datetime
import time
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DreamTeam RecSys",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = "http://localhost:8000"
MODELS = ["dummy", "svd_v1", "ials_v1", "vae_v1"]

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ---- Global palette ---- */
    :root {
        --accent: #7C3AED;
        --accent-light: #A78BFA;
        --bg-card: #1E1E2E;
        --bg-card2: #252535;
        --success: #10B981;
        --error: #EF4444;
        --warning: #F59E0B;
        --text-muted: #94A3B8;
    }

    /* ---- Main background ---- */
    .stApp {
        background: linear-gradient(135deg, #0F0F1A 0%, #1A1A2E 50%, #16213E 100%);
        color: #E2E8F0;
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E1E2E 0%, #252535 100%);
        border-right: 1px solid #2D2D44;
    }
    [data-testid="stSidebar"] * { color: #CBD5E1; }

    /* ---- Metric cards ---- */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid #2D2D44;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.1);
        transition: transform .2s;
    }
    [data-testid="stMetric"]:hover { transform: translateY(-2px); }
    [data-testid="stMetricValue"] { color: var(--accent-light) !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED, #6D28D9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: .5rem 1.5rem;
        font-weight: 600;
        letter-spacing: .03em;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.35);
        transition: all .2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #8B5CF6, #7C3AED);
        box-shadow: 0 6px 18px rgba(124, 58, 237, 0.5);
        transform: translateY(-1px);
    }

    /* ---- Input fields ---- */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: #252535;
        border: 1px solid #3D3D5C;
        border-radius: 8px;
        color: #E2E8F0;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 2px rgba(124,58,237,.25);
    }

    /* ---- Dataframe / Table ---- */
    [data-testid="stDataFrame"] {
        border: 1px solid #2D2D44;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---- Custom cards ---- */
    .recsys-card {
        background: var(--bg-card);
        border: 1px solid #2D2D44;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,.3);
    }
    .recsys-card h3 { color: var(--accent-light); margin-top: 0; }

    /* ---- Result boxes ---- */
    .result-success {
        background: linear-gradient(135deg, rgba(16,185,129,.15), rgba(16,185,129,.05));
        border: 1px solid rgba(16,185,129,.4);
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
    }
    .result-error {
        background: linear-gradient(135deg, rgba(239,68,68,.15), rgba(239,68,68,.05));
        border: 1px solid rgba(239,68,68,.4);
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
    }

    /* ---- Badge ---- */
    .badge {
        display: inline-block;
        padding: .2rem .7rem;
        border-radius: 9999px;
        font-size: .75rem;
        font-weight: 600;
        letter-spacing: .05em;
        text-transform: uppercase;
    }
    .badge-success { background: rgba(16,185,129,.2); color: #34D399; border: 1px solid rgba(16,185,129,.4); }
    .badge-error   { background: rgba(239,68,68,.2);  color: #F87171; border: 1px solid rgba(239,68,68,.4); }
    .badge-model   { background: rgba(124,58,237,.2); color: #A78BFA; border: 1px solid rgba(124,58,237,.4); }

    /* ---- Section header ---- */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #A78BFA, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.2rem;
    }

    /* ---- Plotly charts background ---- */
    .js-plotly-plot .plotly .main-svg {
        background: transparent !important;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: 10px;
        padding: .3rem;
        gap: .3rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 7px;
        color: var(--text-muted);
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
    }

    /* ---- Hide Streamlit branding ---- */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "token": None,
        "username": None,
        "is_admin": False,
        "api_url": DEFAULT_API_URL,
        "last_prediction": None,
        "prediction_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if st.session_state.token:
        h["Authorization"] = f"Bearer {st.session_state.token}"
    return h


def _base() -> str:
    return st.session_state.api_url.rstrip("/")


def api_login(username: str, password: str) -> tuple[bool, str]:
    try:
        r = requests.post(
            f"{_base()}/token",
            data={"username": username, "password": password},
            timeout=5,
        )
        if r.status_code == 200:
            data = r.json()
            st.session_state.token = data["access_token"]
            st.session_state.username = username
            return True, "OK"
        return False, r.json().get("detail", "Неверные данные")
    except requests.exceptions.ConnectionError:
        return False, "Нет соединения с API"
    except Exception as e:
        return False, str(e)


def api_check_admin() -> bool:
    try:
        r = requests.get(f"{_base()}/admin/info", headers=_headers(), timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def api_predict(
    user_id: int,
    item_id: str,
    model_key: str,
    model_params: dict | None = None,
) -> dict:
    # FastAPI-эндпоинт /forward объявляет два body-параметра (model_input + model_params),
    # поэтому тело должно быть "embedded": {"model_input": {...}, "model_params": ...}.
    try:
        payload = {
            "model_input": {
                "user_id": user_id,
                "item_id": item_id,
                "model_key": model_key,
            },
            "model_params": model_params,
        }
        r = requests.post(f"{_base()}/forward", json=payload, headers=_headers(), timeout=15)
        return {"ok": r.status_code == 200, "status_code": r.status_code, "data": r.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "status_code": 0, "data": {"detail": "Нет соединения с API"}}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "status_code": 0, "data": {"detail": str(e)}}


def api_recommend(user_id: int, model_key: str, top_k: int) -> dict:
    # /recommend объявлен с одним Pydantic body-параметром → плоское тело.
    try:
        payload = {"user_id": user_id, "model_key": model_key, "top_k": top_k}
        r = requests.post(f"{_base()}/recommend", json=payload, headers=_headers(), timeout=30)
        return {"ok": r.status_code == 200, "status_code": r.status_code, "data": r.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "status_code": 0, "data": {"detail": "Нет соединения с API"}}
    except requests.exceptions.RequestException as e:
        return {"ok": False, "status_code": 0, "data": {"detail": str(e)}}


def api_history() -> dict:
    try:
        r = requests.get(f"{_base()}/history", headers=_headers(), timeout=5)
        return {"ok": r.status_code == 200, "data": r.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "data": {"detail": "Нет соединения с API"}}
    except Exception as e:
        return {"ok": False, "data": {"detail": str(e)}}


def api_stats() -> dict:
    try:
        r = requests.get(f"{_base()}/stats", timeout=5)
        return {"ok": r.status_code == 200, "data": r.json()}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "data": {"detail": "Нет соединения с API"}}
    except Exception as e:
        return {"ok": False, "data": {"detail": str(e)}}


def api_admin_info() -> dict:
    try:
        r = requests.get(f"{_base()}/admin/info", headers=_headers(), timeout=5)
        return {"ok": r.status_code == 200, "data": r.json()}
    except Exception as e:
        return {"ok": False, "data": {"detail": str(e)}}


# ---------------------------------------------------------------------------
# Plotly chart helpers
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT: dict[str, Any] = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(30,30,46,0.6)",
    "font": {"color": "#CBD5E1", "family": "Inter, sans-serif"},
    "margin": {"l": 20, "r": 20, "t": 40, "b": 20},
    "xaxis": {"gridcolor": "#2D2D44", "zerolinecolor": "#2D2D44"},
    "yaxis": {"gridcolor": "#2D2D44", "zerolinecolor": "#2D2D44"},
}

PURPLE_SEQ = ["#7C3AED", "#8B5CF6", "#A78BFA", "#C4B5FD", "#DDD6FE"]


def _gauge(value: float, title: str, color: str = "#10B981") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * 100,
            number={"suffix": "%", "font": {"color": color, "size": 42}},
            title={"text": title, "font": {"color": "#94A3B8", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#3D3D5C", "tickfont": {"color": "#94A3B8"}},
                "bar": {"color": color},
                "bgcolor": "#252535",
                "bordercolor": "#3D3D5C",
                "steps": [
                    {"range": [0, 50], "color": "rgba(239,68,68,.1)"},
                    {"range": [50, 80], "color": "rgba(245,158,11,.1)"},
                    {"range": [80, 100], "color": "rgba(16,185,129,.1)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.75,
                    "value": value * 100,
                },
            },
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=20, r=20, t=30, b=10))
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> str:
    with st.sidebar:
        # Logo / Title
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0 1.5rem;">
                <div style="font-size:3rem;">🎯</div>
                <div style="font-size:1.3rem; font-weight:700;
                            background:linear-gradient(90deg,#A78BFA,#60A5FA);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    DreamTeam RecSys
                </div>
                <div style="font-size:.75rem; color:#64748B; margin-top:.3rem;">
                    Recommendation Engine
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # API URL
        with st.expander("⚙️ Настройки API", expanded=False):
            new_url = st.text_input(
                "Base URL", value=st.session_state.api_url, label_visibility="collapsed"
            )
            if new_url != st.session_state.api_url:
                st.session_state.api_url = new_url
                st.rerun()

        st.markdown("<hr style='border-color:#2D2D44;margin:.5rem 0 1rem;'>", unsafe_allow_html=True)

        # Auth block
        if not st.session_state.token:
            st.markdown("**🔐 Войти**")
            username = st.text_input("Логин", placeholder="dreamer")
            password = st.text_input("Пароль", type="password", placeholder="••••••••")
            if st.button("Войти", width="stretch"):
                with st.spinner("Авторизация..."):
                    ok, msg = api_login(username, password)
                if ok:
                    st.session_state.is_admin = api_check_admin()
                    st.success("Вход выполнен!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)

            st.markdown(
                "<div style='color:#64748B;font-size:.75rem;margin-top:.5rem;'>"
                "Тестовые данные: <br>"
                "<code>dreamer / secret</code><br>"
                "<code>admin / admin</code>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            role_badge = (
                '<span class="badge badge-model">Admin</span>'
                if st.session_state.is_admin
                else '<span class="badge badge-success">User</span>'
            )
            st.markdown(
                f"""
                <div style="background:#252535;border-radius:10px;padding:.8rem 1rem;margin-bottom:.8rem;">
                    <div style="font-size:.8rem;color:#64748B;">Вы вошли как</div>
                    <div style="font-weight:700;font-size:1rem;color:#E2E8F0;margin:.2rem 0;">
                        👤 {st.session_state.username}
                    </div>
                    {role_badge}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Выйти", width="stretch"):
                st.session_state.token = None
                st.session_state.username = None
                st.session_state.is_admin = False
                st.rerun()

        st.markdown("<hr style='border-color:#2D2D44;margin:1rem 0;'>", unsafe_allow_html=True)

        # Navigation
        st.markdown(
            "<div style='font-size:.7rem;color:#64748B;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.5rem;'>Навигация</div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Навигация",
            options=[
                "🏠 Обзор",
                "🔮 Предсказание",
                "🎁 Рекомендации",
                "📜 История",
                "📊 Статистика",
                "🔑 Админ",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border-color:#2D2D44;margin:1rem 0;'>", unsafe_allow_html=True)

        # API status indicator
        try:
            r = requests.get(f"{_base()}/stats", timeout=2)
            status_ok = r.status_code == 200
        except Exception:
            status_ok = False

        color = "#10B981" if status_ok else "#EF4444"
        label = "Online" if status_ok else "Offline"
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:.5rem;font-size:.8rem;color:#94A3B8;">
                <div style="width:8px;height:8px;border-radius:50%;background:{color};
                            box-shadow:0 0 6px {color};"></div>
                API {label} — {_base()}
            </div>
            """,
            unsafe_allow_html=True,
        )

    return page


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview() -> None:
    st.markdown('<div class="section-header">🏠 Обзор системы</div>', unsafe_allow_html=True)

    result = api_stats()
    if not result["ok"]:
        st.markdown(
            f'<div class="recsys-card result-error">❌ Нет доступа к API: {result["data"].get("detail","")}</div>',
            unsafe_allow_html=True,
        )
        return

    stats = result["data"]
    total = stats["total_requests"]
    success_rate = stats["success_rate"]
    avg_dur = stats["avg_duration_ms"]
    p95 = stats.get("duration_quantiles", {}).get("p95", 0)

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📥 Всего запросов", f"{total:,}")
    c2.metric("✅ Успешность", f"{success_rate * 100:.1f}%")
    c3.metric("⚡ Среднее время", f"{avg_dur:.1f} мс")
    c4.metric("📈 p95 задержка", f"{p95} мс")

    st.markdown("<br>", unsafe_allow_html=True)

    col_gauge, col_models = st.columns([1, 2])

    with col_gauge:
        st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
        st.markdown("<h3>📉 Success Rate</h3>", unsafe_allow_html=True)
        fig_gauge = _gauge(success_rate, "Success Rate")
        st.plotly_chart(fig_gauge, use_container_width=True, key="gauge_overview")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_models:
        by_model = stats.get("by_model", [])
        if by_model:
            st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
            st.markdown("<h3>🤖 Запросы по моделям</h3>", unsafe_allow_html=True)
            df_m = pd.DataFrame(by_model)
            fig_bar = px.bar(
                df_m,
                x="model_key",
                y="request_count",
                color="model_key",
                color_discrete_sequence=PURPLE_SEQ,
                text="request_count",
            )
            fig_bar.update_traces(textposition="outside", marker_line_width=0)
            fig_bar.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="")
            st.plotly_chart(fig_bar, use_container_width=True, key="bar_models_overview")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="recsys-card"><h3>🤖 Запросы по моделям</h3>'
                '<p style="color:#64748B;">Данных пока нет. Сделайте первое предсказание!</p></div>',
                unsafe_allow_html=True,
            )

    # Quantiles chart
    quantiles = stats.get("duration_quantiles", {})
    if quantiles:
        st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
        st.markdown("<h3>⏱️ Перцентили задержки</h3>", unsafe_allow_html=True)
        q_labels = list(quantiles.keys())
        q_vals = list(quantiles.values())
        fig_q = go.Figure(
            go.Bar(
                x=q_labels,
                y=q_vals,
                marker=dict(
                    color=q_vals,
                    colorscale=[[0, "#10B981"], [0.5, "#F59E0B"], [1, "#EF4444"]],
                    showscale=False,
                ),
                text=[f"{v} мс" for v in q_vals],
                textposition="outside",
            )
        )
        fig_q.update_layout(**PLOTLY_LAYOUT, title="", yaxis_title="мс")
        st.plotly_chart(fig_q, use_container_width=True, key="bar_quantiles_overview")
        st.markdown("</div>", unsafe_allow_html=True)


def page_predict() -> None:
    st.markdown('<div class="section-header">🔮 Получить предсказание</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
        st.markdown("<h3>Параметры запроса</h3>", unsafe_allow_html=True)

        user_id = int(st.number_input("User ID", min_value=0, value=42, step=1))
        item_id = st.text_input("Item ID", value="nfmcg_100", placeholder="nfmcg_6702130")

        model_key = st.selectbox("Модель", MODELS, index=0)

        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🚀 Получить предсказание", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if run:
            with st.spinner("Запрос к модели..."):
                t0 = time.time()
                res = api_predict(user_id, item_id, model_key)
                elapsed = (time.time() - t0) * 1000

            if res["ok"]:
                data = res["data"]
                st.session_state.last_prediction = {**data, "_elapsed_ms": elapsed}
                st.session_state.prediction_history.append(
                    {
                        "ts": datetime.now().strftime("%H:%M:%S"),
                        "user_id": data["user_id"],
                        "item_id": data["item_id"],
                        "model": data["model_key"],
                        "score": data["score"],
                        "latency_ms": round(elapsed, 1),
                    }
                )

            st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
            st.markdown("<h3>Результат</h3>", unsafe_allow_html=True)

            if res["ok"]:
                data = res["data"]
                score_color = "#10B981" if data["score"] >= 0.5 else "#F59E0B"
                st.markdown(
                    f"""
                    <div class="result-success">
                        <div style="font-size:.85rem;color:#6EE7B7;margin-bottom:.8rem;">
                            ✅ Успешно · {elapsed:.1f} мс
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;">
                            <div>
                                <div style="font-size:.75rem;color:#94A3B8;">User ID</div>
                                <div style="font-size:1.4rem;font-weight:700;color:#E2E8F0;">{data['user_id']}</div>
                            </div>
                            <div>
                                <div style="font-size:.75rem;color:#94A3B8;">Item ID</div>
                                <div style="font-size:1.4rem;font-weight:700;color:#E2E8F0;">{data['item_id']}</div>
                            </div>
                            <div>
                                <div style="font-size:.75rem;color:#94A3B8;">Модель</div>
                                <span class="badge badge-model">{data['model_key']}</span>
                            </div>
                            <div>
                                <div style="font-size:.75rem;color:#94A3B8;">Score</div>
                                <div style="font-size:2rem;font-weight:800;color:{score_color};">
                                    {data['score']:.4f}
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                detail = res["data"].get("detail", "Неизвестная ошибка")
                code = res["status_code"]
                st.markdown(
                    f"""
                    <div class="result-error">
                        <div style="font-size:.85rem;color:#FCA5A5;margin-bottom:.5rem;">
                            ❌ Ошибка {code if code else "соединения"}
                        </div>
                        <div style="color:#E2E8F0;">{detail}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        elif st.session_state.last_prediction:
            data = st.session_state.last_prediction
            st.markdown(
                '<div class="recsys-card"><h3>Последнее предсказание</h3>',
                unsafe_allow_html=True,
            )
            elapsed = data.get("_elapsed_ms", 0)
            score_color = "#10B981" if data["score"] >= 0.5 else "#F59E0B"
            st.markdown(
                f"""
                <div class="result-success">
                    <div style="font-size:.85rem;color:#6EE7B7;margin-bottom:.8rem;">
                        ✅ {elapsed:.1f} мс
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:.8rem;">
                        <div>
                            <div style="font-size:.75rem;color:#94A3B8;">User ID</div>
                            <div style="font-size:1.4rem;font-weight:700;color:#E2E8F0;">{data['user_id']}</div>
                        </div>
                        <div>
                            <div style="font-size:.75rem;color:#94A3B8;">Item ID</div>
                            <div style="font-size:1.4rem;font-weight:700;color:#E2E8F0;">{data['item_id']}</div>
                        </div>
                        <div>
                            <div style="font-size:.75rem;color:#94A3B8;">Модель</div>
                            <span class="badge badge-model">{data['model_key']}</span>
                        </div>
                        <div>
                            <div style="font-size:.75rem;color:#94A3B8;">Score</div>
                            <div style="font-size:2rem;font-weight:800;color:{score_color};">{data['score']:.4f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="recsys-card" style="display:flex;flex-direction:column;align-items:center;
                     justify-content:center;min-height:200px;text-align:center;">
                    <div style="font-size:3rem;margin-bottom:1rem;">🎯</div>
                    <div style="color:#94A3B8;">Заполните форму и нажмите<br><strong>«Получить предсказание»</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Session history table
    if st.session_state.prediction_history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
        st.markdown("<h3>📋 История сессии</h3>", unsafe_allow_html=True)
        df_hist = pd.DataFrame(st.session_state.prediction_history[::-1])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if len(df_hist) > 1:
            fig_score = px.line(
                df_hist[::-1],
                x="ts",
                y="score",
                color="model",
                markers=True,
                color_discrete_sequence=PURPLE_SEQ,
                labels={"ts": "Время", "score": "Score"},
            )
            fig_score.update_layout(**PLOTLY_LAYOUT, title="")
            st.plotly_chart(fig_score, use_container_width=True, key="line_session_scores")
        st.markdown("</div>", unsafe_allow_html=True)


KNOWN_USERS = {
    "dummy": "любой целый ID (42, 123, …)",
    "svd_v1": "388, 1267, 1762, 2267, 2874, 3395, 3489, 4016 (1.6M пользователей)",
    "vae_v1": "388, 1267, 1762, 2267, 2874, 3395, 3489, 4016 (200k пользователей)",
    "ials_v1": "зависит от артефактов iALS",
}


def page_recommend() -> None:
    st.markdown(
        '<div class="section-header">🎁 Топ-N рекомендаций</div>', unsafe_allow_html=True
    )

    st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
    st.markdown("<h3>Параметры</h3>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        user_id = int(st.number_input("User ID", min_value=0, value=388, step=1, key="rec_user"))
    with c2:
        model_key = st.selectbox("Модель", MODELS, index=0, key="rec_model")
    with c3:
        top_k = int(st.slider("top-N", min_value=5, max_value=50, value=15, step=1))

    hint = KNOWN_USERS.get(model_key, "")

    run = st.button("🎯 Построить рекомендации", width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    if not run:
        st.markdown(
            """
            <div class="recsys-card" style="display:flex;flex-direction:column;align-items:center;
                 justify-content:center;min-height:160px;text-align:center;">
                <div style="font-size:3rem;margin-bottom:1rem;">🎁</div>
                <div style="color:#94A3B8;">Выберите пользователя и модель,<br>
                    затем нажмите <strong>«Построить рекомендации»</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.spinner(f"Запрашиваю top-{top_k} рекомендаций..."):
        t0 = time.time()
        res = api_recommend(user_id, model_key, top_k)
        elapsed = (time.time() - t0) * 1000

    if not res["ok"]:
        detail = res["data"].get("detail", "Неизвестная ошибка")
        code = res["status_code"]
        # Подсказка при ошибке 404 (user not found)
        hint_html = ""
        if code == 404:
            hint_html = (
                f"<div style='margin-top:.5rem;font-size:.8rem;color:#94A3B8;'>"
                f"Попробуйте один из валидных user_id: {hint}</div>"
            )
        st.markdown(
            f"""
            <div class="recsys-card result-error">
                <div style="font-size:.85rem;color:#FCA5A5;margin-bottom:.5rem;">
                    ❌ Ошибка {code if code else "соединения"}
                </div>
                <div style="color:#E2E8F0;">{detail}</div>
                {hint_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    data = res["data"]
    recs = data.get("recommendations", [])

    if not recs:
        st.markdown(
            '<div class="recsys-card"><p style="color:#64748B;text-align:center;">'
            "Модель не вернула рекомендаций для этого пользователя.</p></div>",
            unsafe_allow_html=True,
        )
        return

    # Summary
    df = pd.DataFrame(recs)
    c1, c2, c3 = st.columns(3)
    c1.metric("👤 User ID", data["user_id"])
    c2.metric("🤖 Модель", data["model_key"])
    c3.metric("⚡ Время", f"{elapsed:.1f} мс")

    st.markdown("<br>", unsafe_allow_html=True)

    col_table, col_chart = st.columns([1, 1], gap="large")

    with col_table:
        st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
        st.markdown(f"<h3>📋 Топ-{len(recs)} айтемов</h3>", unsafe_allow_html=True)
        st.dataframe(
            df[["rank", "item_id", "score"]],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_chart:
        st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
        st.markdown("<h3>📊 Score по айтемам</h3>", unsafe_allow_html=True)
        df_chart = df.copy()
        df_chart["item_id"] = df_chart["item_id"].astype(str)
        fig = px.bar(
            df_chart,
            x="score",
            y="item_id",
            orientation="h",
            color="score",
            color_continuous_scale=["#C4B5FD", "#7C3AED"],
            labels={"score": "Score", "item_id": "Item ID"},
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True, key="bar_recommendations")
        st.markdown("</div>", unsafe_allow_html=True)


def page_history() -> None:
    st.markdown('<div class="section-header">📜 История запросов</div>', unsafe_allow_html=True)

    if not st.session_state.token:
        st.markdown(
            """
            <div class="recsys-card result-error">
                🔐 Для просмотра истории необходимо войти в систему.
                <br><span style="color:#94A3B8;font-size:.875rem;">
                    Используйте боковую панель для авторизации.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        st.button("🔄 Обновить", width="stretch")

    result = api_history()

    if not result["ok"]:
        st.markdown(
            f'<div class="recsys-card result-error">❌ {result["data"].get("detail","Ошибка")}</div>',
            unsafe_allow_html=True,
        )
        return

    rows = result["data"]
    if not rows:
        st.markdown(
            '<div class="recsys-card"><p style="color:#64748B;text-align:center;">История пуста. Сделайте первое предсказание!</p></div>',
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Summary metrics
    total = len(df)
    ok_cnt = int((df["status"] == "ok").sum())
    avg_dur = float(df["duration_ms"].mean())
    unique_models = int(df["model_key"].nunique())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📥 Всего записей", total)
    c2.metric("✅ Успешных", ok_cnt)
    c3.metric("⚡ Среднее время", f"{avg_dur:.1f} мс")
    c4.metric("🤖 Моделей", unique_models)

    tab_table, tab_charts = st.tabs(["📋 Таблица", "📊 Графики"])

    with tab_table:
        # Color status column
        display_cols = ["id", "user_id", "item_id", "model_key", "status", "duration_ms", "created_at"]
        existing_cols = [c for c in display_cols if c in df.columns]

        def _style_status(val: str) -> str:
            if val == "ok":
                return "color: #34D399; font-weight: 600"
            return "color: #F87171; font-weight: 600"

        styled = df[existing_cols].style.map(_style_status, subset=["status"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with tab_charts:
        c_left, c_right = st.columns(2)

        with c_left:
            st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
            st.markdown("<h3>Статусы запросов</h3>", unsafe_allow_html=True)
            status_counts = df["status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            fig_pie = px.pie(
                status_counts,
                names="status",
                values="count",
                color="status",
                color_discrete_map={"ok": "#10B981", "error": "#EF4444"},
                hole=0.5,
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1"),
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(font=dict(color="#CBD5E1")),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="pie_status")
            st.markdown("</div>", unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
            st.markdown("<h3>Распределение задержки</h3>", unsafe_allow_html=True)
            fig_hist = px.histogram(
                df,
                x="duration_ms",
                nbins=20,
                color_discrete_sequence=["#7C3AED"],
                labels={"duration_ms": "Задержка (мс)"},
            )
            fig_hist.update_traces(marker_line_width=0)
            fig_hist.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_hist, use_container_width=True, key="hist_duration")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
        st.markdown("<h3>Задержка по времени</h3>", unsafe_allow_html=True)
        fig_line = px.scatter(
            df,
            x="created_at",
            y="duration_ms",
            color="model_key",
            color_discrete_sequence=PURPLE_SEQ,
            labels={"created_at": "Время", "duration_ms": "Задержка (мс)"},
        )
        fig_line.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_line, use_container_width=True, key="scatter_latency_history")
        st.markdown("</div>", unsafe_allow_html=True)


def page_statistics() -> None:
    st.markdown('<div class="section-header">📊 Аналитика и статистика</div>', unsafe_allow_html=True)

    result = api_stats()
    if not result["ok"]:
        st.markdown(
            f'<div class="recsys-card result-error">❌ {result["data"].get("detail","Ошибка")}</div>',
            unsafe_allow_html=True,
        )
        return

    stats = result["data"]
    total = stats["total_requests"]
    rc = stats.get("request_characteristics", {})
    by_model = stats.get("by_model", [])

    # Key metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📥 Всего запросов", f"{total:,}")
    c2.metric("⚡ Avg задержка", f"{stats['avg_duration_ms']:.1f} мс")
    c3.metric("🔥 Max задержка", f"{stats['max_duration_ms']} мс")
    c4.metric("🤖 Активных моделей", rc.get("distinct_models", 0))

    st.markdown("<br>", unsafe_allow_html=True)

    # Gauge + quantiles
    col_gauge, col_quant = st.columns([1, 2])

    with col_gauge:
        st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
        st.markdown("<h3>✅ Success Rate</h3>", unsafe_allow_html=True)
        fig_g = _gauge(stats["success_rate"], "")
        st.plotly_chart(fig_g, use_container_width=True, key="gauge_stats")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_quant:
        quantiles = stats.get("duration_quantiles", {})
        if quantiles:
            st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
            st.markdown("<h3>⏱️ Перцентили задержки</h3>", unsafe_allow_html=True)
            q_df = pd.DataFrame(
                [{"Перцентиль": k.upper(), "мс": v} for k, v in quantiles.items()]
            )
            fig_q = px.bar(
                q_df,
                x="Перцентиль",
                y="мс",
                color="мс",
                color_continuous_scale=[[0, "#10B981"], [0.5, "#F59E0B"], [1, "#EF4444"]],
                text="мс",
            )
            fig_q.update_traces(texttemplate="%{text} мс", textposition="outside", marker_line_width=0)
            fig_q.update_coloraxes(showscale=False)
            fig_q.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_q, use_container_width=True, key="bar_quantiles_stats")
            st.markdown("</div>", unsafe_allow_html=True)

    # Model breakdown
    if by_model:
        st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
        st.markdown("<h3>🤖 Детализация по моделям</h3>", unsafe_allow_html=True)

        df_m = pd.DataFrame(by_model)
        tab1, tab2, tab3 = st.tabs(["Запросы", "Средняя задержка", "Таблица"])

        with tab1:
            fig_rc = px.bar(
                df_m,
                x="model_key",
                y="request_count",
                color="model_key",
                color_discrete_sequence=PURPLE_SEQ,
                text="request_count",
                labels={"model_key": "Модель", "request_count": "Запросов"},
            )
            fig_rc.update_traces(textposition="outside", marker_line_width=0)
            fig_rc.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig_rc, use_container_width=True, key="bar_model_requests")

        with tab2:
            fig_dur = px.bar(
                df_m,
                x="model_key",
                y="avg_duration",
                color="model_key",
                color_discrete_sequence=PURPLE_SEQ,
                text=df_m["avg_duration"].round(1),
                labels={"model_key": "Модель", "avg_duration": "Avg задержка (мс)"},
            )
            fig_dur.update_traces(texttemplate="%{text} мс", textposition="outside", marker_line_width=0)
            fig_dur.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig_dur, use_container_width=True, key="bar_model_duration")

        with tab3:
            display_cols = [c for c in ["model_key", "request_count", "avg_duration", "avg_request_size"] if c in df_m.columns]
            st.dataframe(df_m[display_cols], use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Request characteristics
    if rc:
        st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
        st.markdown("<h3>📦 Характеристики запросов</h3>", unsafe_allow_html=True)
        rc_c1, rc_c2 = st.columns(2)
        rc_c1.metric("Avg размер запроса", f"{rc.get('avg_request_size_bytes', 0):.1f} байт")
        rc_c2.metric("Avg токенов", f"{rc.get('avg_token_count', 0):.1f}")
        st.markdown("</div>", unsafe_allow_html=True)


def page_admin() -> None:
    st.markdown('<div class="section-header">🔑 Панель администратора</div>', unsafe_allow_html=True)

    if not st.session_state.token:
        st.markdown(
            '<div class="recsys-card result-error">🔐 Требуется авторизация.</div>',
            unsafe_allow_html=True,
        )
        return

    result = api_admin_info()

    if not result["ok"]:
        code_hint = ""
        detail = result["data"].get("detail", "Нет доступа")
        if "403" in str(result) or not st.session_state.is_admin:
            code_hint = "У вашего аккаунта нет прав администратора."
        st.markdown(
            f"""
            <div class="recsys-card result-error">
                🚫 Доступ запрещён<br>
                <span style="color:#94A3B8;font-size:.875rem;">{detail or code_hint}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    data = result["data"]
    st.markdown(
        f"""
        <div class="recsys-card result-success">
            <div style="font-size:1.1rem;font-weight:700;color:#6EE7B7;margin-bottom:.8rem;">
                ✅ Административный доступ подтверждён
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
                <div>
                    <div style="font-size:.75rem;color:#94A3B8;">Статус</div>
                    <div style="font-size:1rem;font-weight:600;color:#E2E8F0;">{data.get('status','')}</div>
                </div>
                <div>
                    <div style="font-size:.75rem;color:#94A3B8;">Администратор</div>
                    <div style="font-size:1rem;font-weight:600;color:#E2E8F0;">👤 {data.get('user','')}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
    st.markdown("<h3>📡 Информация о системе</h3>", unsafe_allow_html=True)

    df_info = pd.DataFrame(
        {
            "Параметр": ["API Base URL", "Аутентифицирован", "Роль", "Token"],
            "Значение": [
                _base(),
                f"✅ {st.session_state.username}",
                "👑 Администратор",
                f"…{st.session_state.token[-20:]}",
            ],
        }
    )
    st.dataframe(df_info, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="recsys-card">', unsafe_allow_html=True)
    st.markdown("<h3>🧪 Быстрый тест эндпоинтов</h3>", unsafe_allow_html=True)
    if st.button("Запустить проверку", width="content"):
        endpoints = [
            ("GET", "/stats", lambda: requests.get(f"{_base()}/stats", timeout=3)),
            ("GET", "/history", lambda: requests.get(f"{_base()}/history", headers=_headers(), timeout=3)),
            ("GET", "/admin/info", lambda: requests.get(f"{_base()}/admin/info", headers=_headers(), timeout=3)),
        ]
        rows_ep = []
        for method, path, fn in endpoints:
            try:
                t0 = time.time()
                r = fn()
                dur = (time.time() - t0) * 1000
                rows_ep.append({"Метод": method, "Путь": path, "Статус": r.status_code, "Время": f"{dur:.1f} мс"})
            except Exception as exc:
                rows_ep.append({"Метод": method, "Путь": path, "Статус": "err", "Время": str(exc)})

        def _color_status(val: str | int) -> str:
            if str(val) == "200":
                return "color:#34D399;font-weight:600"
            return "color:#F87171;font-weight:600"

        df_ep = pd.DataFrame(rows_ep)
        styled_ep = df_ep.style.map(_color_status, subset=["Статус"])
        st.dataframe(styled_ep, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    page = render_sidebar()

    if page == "🏠 Обзор":
        page_overview()
    elif page == "🔮 Предсказание":
        page_predict()
    elif page == "🎁 Рекомендации":
        page_recommend()
    elif page == "📜 История":
        page_history()
    elif page == "📊 Статистика":
        page_statistics()
    elif page == "🔑 Админ":
        page_admin()


if __name__ == "__main__":
    main()
