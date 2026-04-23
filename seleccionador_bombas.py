import os
import re
import base64
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, curve_fit


# ==============================
# Configuración general
# ==============================
NPSH_MARGIN_M = 0.5
VALID_USERNAME = "Diego"
VALID_PASSWORD = "Vog1234"
APP_SUBTITLE = "Series N-NP-N(V)"
APP_TITLE = "Seleccionador Bombas Normalizadas"

IEC_MOTORS_KW = [
    0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5, 7.5,
    11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110, 132, 160, 200, 250,
    315, 355, 400, 500, 630,
]

st.set_page_config(
    layout="wide",
    page_title=APP_TITLE,
)


# ==============================
# Utilidades visuales / formato
# ==============================
def file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def svg_to_data_uri(path: str) -> str:
    return f"data:image/svg+xml;base64,{file_to_base64(path)}"


def png_to_data_uri(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{file_to_base64(path)}"


def image_to_data_uri(path: str) -> str:
    if path.lower().endswith(".svg"):
        return svg_to_data_uri(path)
    return png_to_data_uri(path)


def find_logo_path() -> Optional[str]:
    possible_paths = [
        "Logo Vogt.svg",
        "Logo_Vogt.svg",
        "logo_vogt.svg",
        "Logo Vogt.png",
        "logo_vogt.png",
        os.path.join(os.getcwd(), "Logo Vogt.svg"),
        os.path.join(os.getcwd(), "Logo_Vogt.svg"),
        os.path.join(os.getcwd(), "logo_vogt.svg"),
        os.path.join(os.getcwd(), "Logo Vogt.png"),
        os.path.join(os.getcwd(), "logo_vogt.png"),
        "/mnt/data/Logo Vogt.svg",
        "/mnt/data/Logo_Vogt.svg",
        "/mnt/data/logo_vogt.svg",
        "/mnt/data/Logo Vogt.png",
        "/mnt/data/logo_vogt.png",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .main-block {padding-top: 0.35rem;}

            .app-title {
                font-size: 2.55rem;
                font-weight: 800;
                line-height: 1.02;
                margin: 0;
                color: #1f2937;
            }

            .app-subtitle {
                color: #4b5563;
                font-weight: 700;
                margin-top: 0.24rem;
                font-size: 1.22rem;
                line-height: 1.1;
            }

            .menu-card {
                border: 1px solid #d7dee8;
                border-radius: 16px;
                padding: 1.15rem 1.2rem;
                background: #ffffff;
                min-height: 150px;
            }

            .menu-card h4 {
                margin: 0 0 0.55rem 0;
                color: #0f172a;
                font-size: 1.10rem;
            }

            .small-note {
                font-size: 0.95rem;
                color: #4b5563;
            }

            .filter-list-title {
                font-size: 1.95rem;
                font-weight: 700;
                color: #0b4f96;
                margin-bottom: 0.3rem;
            }

            .results-count {
                font-size: 2.00rem;
                font-weight: 800;
                color: #0b4f96;
                margin: 0 0 0.7rem 0;
            }

            .manual-panel {
                border-right: 1px solid #d7dee8;
                padding-right: 1rem;
            }

            .section-divider {
                border-top: 1px solid #d7dee8;
                margin-top: 0.75rem;
                margin-bottom: 0.75rem;
            }

            .stDataFrame, .stTable {
                font-size: 0.95rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


def safe_float(value, default=np.nan) -> float:
    try:
        return float(str(value).replace(",", ".").strip())
    except Exception:
        return default


def fmt2(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{float(value):.2f}"


def fmt1(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{float(value):.1f}"


def fmt0(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    return f"{float(value):.0f}"


def style_numeric_df(
    df: pd.DataFrame,
    zero_dec_cols: Optional[List[str]] = None,
    one_dec_cols: Optional[List[str]] = None,
):
    zero_dec_cols = set(zero_dec_cols or [])
    one_dec_cols = set(one_dec_cols or [])
    formatters = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if col in zero_dec_cols:
                formatters[col] = lambda x: "" if pd.isna(x) else f"{x:.0f}"
            elif col in one_dec_cols:
                formatters[col] = lambda x: "" if pd.isna(x) else f"{x:.1f}"
            else:
                formatters[col] = lambda x: "" if pd.isna(x) else f"{x:.2f}"

    return df.style.format(formatters)


def style_results_df(df: pd.DataFrame):
    styler = style_numeric_df(
        df,
        zero_dec_cols=["RPM", "Polos", "D_Impulsor (mm)"],
        one_dec_cols=["Q Op. (m3/h)", "H Op. (m)"],
    )
    if "Status NPSH" in df.columns:
        styler = styler.applymap(
            lambda x: "background-color: #fde2e1" if x is False else "",
            subset=["Status NPSH"]
        )
    return styler


def keyify(prefix: str, value) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", f"{prefix}_{value}")


# ==============================
# Modelo hidráulico
# ==============================
def get_service_factor(power_kw: float) -> float:
    if power_kw < 22:
        return 1.15
    if power_kw < 55:
        return 1.10
    return 1.05


def select_motor(max_power_kw: float) -> float:
    req_power = max_power_kw * get_service_factor(max_power_kw)
    for motor_kw in IEC_MOTORS_KW:
        if motor_kw >= req_power:
            return motor_kw
    return IEC_MOTORS_KW[-1]


def poly2(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * (x ** 2) + b * x + c


def infer_poles_from_rpm(rpm: Optional[float]) -> Optional[int]:
    if rpm is None or pd.isna(rpm):
        return None
    rpm = float(rpm)
    if rpm >= 2500:
        return 2
    if rpm >= 1200:
        return 4
    return 6


def discharge_from_model(modelo: str) -> Optional[int]:
    if not modelo:
        return None
    m = re.match(r"\s*(\d+)\s*-", str(modelo))
    return int(m.group(1)) if m else None


class PumpCurveBase:
    def __init__(self, diam: float, puntos: List[Dict[str, float]]) -> None:
        self.diam = float(diam)
        self.puntos = sorted(puntos, key=lambda p: p["Q"])

        q_raw = np.array([p["Q"] for p in self.puntos], dtype=float)
        h_raw = np.array([p["H"] for p in self.puntos], dtype=float)
        eta_raw = np.array([p["eta"] for p in self.puntos], dtype=float)
        npsh_raw = np.array([p["NPSH"] for p in self.puntos], dtype=float)

        self.popt_h, _ = curve_fit(poly2, q_raw, h_raw)
        self.a, self.b, self.c = self.popt_h

        if self.a < 0 and self.b > 0:
            self.stable_q_min = max(0.0, -self.b / (2 * self.a))
        else:
            self.stable_q_min = 0.0

        unique_q, unique_idx = np.unique(q_raw, return_index=True)
        self.unique_q = unique_q
        self.eta_values = eta_raw[unique_idx]
        self.npsh_values = npsh_raw[unique_idx]

        self.interp_eta = PchipInterpolator(self.unique_q, self.eta_values)
        self.interp_npsh = PchipInterpolator(self.unique_q, self.npsh_values)

        self.q_min = float(np.min(self.unique_q))
        self.q_max = float(np.max(self.unique_q))

        qq = np.linspace(max(0.1, self.q_min), self.q_max, 240)
        ee = self.interp_eta(qq)
        idx_bep = int(np.argmax(ee))
        self.q_bep = float(qq[idx_bep])
        self.eta_bep = float(ee[idx_bep])

    def get_h(self, q: float) -> float:
        return float(poly2(np.array([q]), *self.popt_h)[0])

    def get_eta(self, q: float) -> float:
        q_eval = float(np.clip(q, self.unique_q[0], self.unique_q[-1]))
        return float(self.interp_eta(q_eval))

    def get_npshr(self, q: float) -> float:
        q_eval = float(np.clip(q, self.unique_q[0], self.unique_q[-1]))
        return float(self.interp_npsh(q_eval))

    def get_power(self, q: float, density: float = 1000.0, viscosity_cf: float = 1.0) -> float:
        h = self.get_h(q)
        eta = max(0.01, self.get_eta(q) * viscosity_cf)
        power_w = (q / 3600.0) * h * density * 9.81 / (eta / 100.0)
        return float(power_w / 1000.0)


class TrimmedCurve:
    def __init__(self, base_curve: PumpCurveBase, trim_diam: float) -> None:
        self.base = base_curve
        self.diam = float(trim_diam)
        self.ratio = self.diam / self.base.diam

    def get_h(self, q: float) -> float:
        q_base = q / self.ratio
        return (self.ratio ** 2) * self.base.get_h(q_base)

    def get_eta(self, q: float) -> float:
        q_base = q / self.ratio
        base_eta = self.base.get_eta(q_base)
        penalty = max(0.0, 1.0 - self.ratio) * 10.0
        return max(0.0, base_eta - penalty)

    def get_npshr(self, q: float) -> float:
        q_base = q / self.ratio
        return self.base.get_npshr(q_base) * (self.ratio ** 2)

    def get_power(self, q: float, density: float = 1000.0, viscosity_cf: float = 1.0) -> float:
        h = self.get_h(q)
        eta = max(0.01, self.get_eta(q) * viscosity_cf)
        power_w = (q / 3600.0) * h * density * 9.81 / (eta / 100.0)
        return float(power_w / 1000.0)

    def get_max_power(self, end_q: float, density: float = 1000.0, viscosity_cf: float = 1.0) -> float:
        qq = np.linspace(max(0.1, 0.02 * end_q), max(end_q, 0.1), 120)
        powers = [self.get_power(qi, density=density, viscosity_cf=viscosity_cf) for qi in qq]
        return float(max(powers)) if powers else 0.0


class SystemCurve:
    def __init__(self, h_stat: float, k: float) -> None:
        self.h_stat = float(h_stat)
        self.k = float(k)

    def get_h(self, q: float) -> float:
        return self.h_stat + self.k * (q ** 2)


class PumpDatabase:
    def __init__(self) -> None:
        self.families: List[Dict] = []

    def load_from_csv(self, file_obj) -> None:
        df = pd.read_csv(file_obj, sep=";", decimal=".")
        df.columns = [str(col).strip() for col in df.columns]

        vital_cols = ["Marca", "modelo", "diametro_mm", "Q_m3/h", "H_m"]
        for col in vital_cols:
            if col not in df.columns:
                raise ValueError(f"Falta columna vital requerida: {col}")

        series_col = "Serie" if "Serie" in df.columns else None
        has_rpm = "RPM" in df.columns

        groupby_cols = []
        if series_col:
            groupby_cols.append(series_col)
        groupby_cols += ["Marca", "modelo"]
        if has_rpm:
            groupby_cols.append("RPM")

        self.families = []
        for name, group in df.groupby(groupby_cols, dropna=False):
            name = name if isinstance(name, tuple) else (name,)
            idx = 0
            serie = None
            if series_col:
                serie = str(name[idx]) if pd.notna(name[idx]) else "-"
                idx += 1

            marca = str(name[idx])
            idx += 1
            modelo = str(name[idx])
            idx += 1

            rpm_val = safe_float(name[idx]) if has_rpm else np.nan
            rpm_val = None if np.isnan(rpm_val) else float(rpm_val)

            diametros = sorted(group["diametro_mm"].dropna().astype(float).unique().tolist())
            if not diametros:
                continue

            curvas: List[PumpCurveBase] = []
            for d in diametros:
                group_d = group[group["diametro_mm"].astype(float) == d]
                puntos: List[Dict[str, float]] = []

                for _, row in group_d.iterrows():
                    q = safe_float(row["Q_m3/h"])
                    h = safe_float(row["H_m"])
                    eta = safe_float(row.get("h_%", row.get("eta", 0.0)), 0.0)
                    npsh = safe_float(row.get("NPSH_m", row.get("NPSH", 0.0)), 0.0)
                    if not np.isnan(q) and not np.isnan(h):
                        puntos.append({"Q": q, "H": h, "eta": eta, "NPSH": npsh})

                if len(puntos) >= 3:
                    try:
                        curvas.append(PumpCurveBase(d, puntos))
                    except Exception:
                        continue

            if not curvas:
                continue

            curvas = sorted(curvas, key=lambda c: c.diam)

            self.families.append(
                {
                    "serie": serie if serie else marca,
                    "marca": marca,
                    "modelo": modelo,
                    "rpm": rpm_val,
                    "polos": infer_poles_from_rpm(rpm_val),
                    "descarga_dn": discharge_from_model(modelo),
                    "D_min": float(min(diametros)),
                    "D_max": float(max(diametros)),
                    "diametros_disponibles": [float(d) for d in diametros],
                    "curvas": curvas,
                }
            )

    def get_families(self) -> List[Dict]:
        return self.families


# ==============================
# Cálculo hidráulico
# ==============================
def find_trim_diameter(
    q_req: float,
    h_req: float,
    base_curve: PumpCurveBase,
    d_min: float,
    d_max: float,
) -> Optional[float]:
    if d_min > d_max or d_min <= 0:
        return None

    min_ratio = max(0.01, d_min / base_curve.diam)
    max_ratio = min(1.0, d_max / base_curve.diam)
    if min_ratio > max_ratio:
        return None

    def objective(ratio: float) -> float:
        q_base = q_req / ratio
        return (ratio ** 2) * base_curve.get_h(q_base) - h_req

    try:
        f_min = objective(min_ratio)
        f_max = objective(max_ratio)

        if f_min == 0:
            return base_curve.diam * min_ratio
        if f_max == 0:
            return base_curve.diam * max_ratio
        if f_min * f_max > 0:
            return None

        ratio_opt = brentq(objective, min_ratio, max_ratio, xtol=1e-5)
        d_req = base_curve.diam * ratio_opt

        if d_min <= d_req <= d_max:
            return float(d_req)
        return None
    except Exception:
        return None


def find_operating_point(
    trim_curve: TrimmedCurve,
    sys_curve: Optional[SystemCurve],
    q_max: Optional[float] = None,
) -> Optional[float]:
    if sys_curve is None:
        return None

    q_upper = q_max if q_max is not None else trim_curve.base.q_max * trim_curve.ratio * 1.35
    q_upper = max(q_upper, 0.5)

    def objective(q: float) -> float:
        return trim_curve.get_h(q) - sys_curve.get_h(q)

    try:
        q0 = 0.001
        if objective(q0) < 0:
            return None
        return float(brentq(objective, q0, q_upper))
    except Exception:
        return None


def viscosity_correction(viscosidad_cst: float) -> float:
    if viscosidad_cst <= 1.0:
        return 1.0
    if viscosidad_cst <= 5.0:
        return 0.97
    if viscosidad_cst <= 20.0:
        return 0.93
    return 0.88


# ==============================
# Session state
# ==============================
def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "username": "",
        "page": "login",
        "loaded_families": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ==============================
# Encabezado / navegación
# ==============================
def render_top_header() -> None:
    logo_path = find_logo_path()

    col_logo, col_title = st.columns([2.2, 4.8], vertical_alignment="center")

    with col_logo:
        if logo_path:
            logo_uri = image_to_data_uri(logo_path)
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:flex-start; height:165px;">
                    <img src="{logo_uri}" style="max-height:155px; width:auto;">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:4.4rem; font-weight:900; color:#0059aa;'>VOGT</div>",
                unsafe_allow_html=True,
            )

    with col_title:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; justify-content:flex-start; min-height:165px; margin-left:-10px;">
                <div>
                    <div class="app-title">{APP_TITLE}</div>
                    <div class="app-subtitle">{APP_SUBTITLE}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def go_to(page_name: str) -> None:
    st.session_state.page = page_name
    st.rerun()


def logout() -> None:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.page = "login"
    st.rerun()


# ==============================
# Login
# ==============================
def login_view() -> None:
    render_top_header()
    st.markdown("<div class='main-block'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2, 1.3, 1.2])
    with c2:
        st.markdown("### Acceso")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submit = st.form_submit_button("Ingresar", use_container_width=True)

        if submit:
            if username == VALID_USERNAME and password == VALID_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.page = "menu"
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")


# ==============================
# Menú principal
# ==============================
def main_menu_view() -> None:
    render_top_header()

    top_left, top_right = st.columns([6, 1])
    with top_right:
        st.button("Cerrar sesión", use_container_width=True, on_click=logout)

    st.markdown("## Menu principal")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class='menu-card'>
                <h4>Selección de Bombas por punto hidraulico</h4>
                <div class='small-note'>
                    Evalúa el punto Q-H requerido, calcula el diámetro necesario y revisa las curvas características.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Abrir selección por punto",
            key="btn_hydraulic",
            use_container_width=True,
            on_click=go_to,
            args=("hydraulic",),
        )

    with c2:
        st.markdown(
            """
            <div class='menu-card'>
                <h4>Seleccion de Bombas Manual</h4>
                <div class='small-note'>
                    Explora la lista completa de modelos, filtra por serie, polos o diámetro de descarga y revisa sus curvas.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Abrir selección manual",
            key="btn_manual",
            use_container_width=True,
            on_click=go_to,
            args=("manual",),
        )


# ==============================
# Carga de base
# ==============================
def load_database_widget() -> Optional[List[Dict]]:
    uploaded_file = st.sidebar.file_uploader("Cargar Base Oficial (CSV)", type=["csv"])

    if uploaded_file is not None:
        db = PumpDatabase()
        try:
            db.load_from_csv(uploaded_file)
            st.session_state.loaded_families = db.get_families()
        except Exception as exc:
            st.sidebar.error(f"Error de integridad en CSV: {exc}")
            st.session_state.loaded_families = None

    families = st.session_state.loaded_families

    if families is not None:
        st.sidebar.success(f"Base cargada: {len(families)} familias activas")
        if st.sidebar.button("Quitar base cargada", use_container_width=True):
            st.session_state.loaded_families = None
            st.rerun()
        return families

    st.info("Carga la base CSV para habilitar la selección.")
    return None


def build_catalog_df(families: List[Dict]) -> pd.DataFrame:
    rows = []
    for fam in families:
        rows.append(
            {
                "Serie (Marca)": fam["serie"],
                "Marca": fam["marca"],
                "Modelo": fam["modelo"],
                "Polos": float(fam["polos"]) if fam["polos"] is not None else np.nan,
                "RPM": float(fam["rpm"]) if fam["rpm"] is not None else np.nan,
                "Descarga DN": float(fam["descarga_dn"]) if fam["descarga_dn"] is not None else np.nan,
                "D mín. (mm)": float(fam["D_min"]),
                "D máx. (mm)": float(fam["D_max"]),
                "Diámetros disponibles": ", ".join(
                    str(int(d)) if float(d).is_integer() else f"{d:.2f}"
                    for d in fam["diametros_disponibles"]
                ),
                "_fam": fam,
            }
        )
    return pd.DataFrame(rows)


# ==============================
# Gráficos
# ==============================
def metric_label(metric: str) -> str:
    return {
        "H": "Altura H (m)",
        "ETA": "Eficiencia (%)",
        "P": "Potencia (kW)",
        "NPSH": "NPSHr (m)",
    }[metric]


def curve_values(
    curve_obj,
    metric: str,
    q_values: np.ndarray,
    density: float = 1000.0,
    visco_cf: float = 1.0,
) -> List[float]:
    values: List[float] = []
    for q in q_values:
        if metric == "H":
            values.append(curve_obj.get_h(float(q)))
        elif metric == "ETA":
            eta = curve_obj.get_eta(float(q)) if hasattr(curve_obj, "get_eta") else 0.0
            values.append(float(eta * visco_cf))
        elif metric == "P":
            values.append(float(curve_obj.get_power(float(q), density=density, viscosity_cf=visco_cf)))
        elif metric == "NPSH":
            values.append(float(curve_obj.get_npshr(float(q))))
    return values


def apply_graph_box(fig: go.Figure) -> None:
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1.2,
        linecolor="black",
        mirror=True,
        ticks="outside",
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1.2,
        linecolor="black",
        mirror=True,
        ticks="outside",
    )


def plot_family_metric(
    fam: Dict,
    metric: str,
    title: str,
    show_all_diameters: bool,
    selected_trim: Optional[TrimmedCurve] = None,
    selected_real_diam: Optional[float] = None,
    op_q: Optional[float] = None,
    q_req: Optional[float] = None,
    h_req: Optional[float] = None,
    sys_curve: Optional[SystemCurve] = None,
    density: float = 1000.0,
    visco_cf: float = 1.0,
) -> go.Figure:
    fig = go.Figure()

    if show_all_diameters:
        for curve in fam["curvas"]:
            qq = np.linspace(curve.q_min, curve.q_max, 140)
            fig.add_trace(
                go.Scatter(
                    x=qq,
                    y=curve_values(curve, metric, qq, density=density, visco_cf=visco_cf),
                    mode="lines",
                    name=f"D={curve.diam:.0f} mm",
                    line=dict(width=1.5, color="rgba(130,130,130,0.75)"),
                    hovertemplate="Q: %{x:.2f} m³/h<br>Valor: %{y:.2f}<extra></extra>",
                )
            )

    if selected_trim is not None:
        q_max_plot = selected_trim.base.q_max * selected_trim.ratio
        qq_sel = np.linspace(
            max(0.05, selected_trim.base.q_min * selected_trim.ratio),
            q_max_plot,
            180,
        )
        fig.add_trace(
            go.Scatter(
                x=qq_sel,
                y=curve_values(selected_trim, metric, qq_sel, density=density, visco_cf=visco_cf),
                mode="lines",
                name=(
                    f"Diámetro escogido = {selected_trim.diam:.0f} mm"
                    if selected_real_diam is None
                    else f"D = {selected_real_diam:.0f} mm"
                ),
                line=dict(width=3, color="#0059aa"),
                hovertemplate="Q: %{x:.2f} m³/h<br>Valor: %{y:.2f}<extra></extra>",
            )
        )
    elif not show_all_diameters and fam["curvas"]:
        curve = min(
            fam["curvas"],
            key=lambda c: abs(c.diam - (selected_real_diam or fam["curvas"][-1].diam)),
        )
        qq = np.linspace(curve.q_min, curve.q_max, 140)
        fig.add_trace(
            go.Scatter(
                x=qq,
                y=curve_values(curve, metric, qq, density=density, visco_cf=visco_cf),
                mode="lines",
                name=f"D={curve.diam:.0f} mm",
                line=dict(width=3, color="#0059aa"),
                hovertemplate="Q: %{x:.2f} m³/h<br>Valor: %{y:.2f}<extra></extra>",
            )
        )

    if metric == "H" and sys_curve is not None:
        if op_q is not None:
            q_end_sys = op_q
        elif q_req is not None:
            q_end_sys = q_req
        else:
            q_end_sys = max([c.q_max for c in fam["curvas"]])

        qq_sys = np.linspace(0.0, max(q_end_sys, 0.5), 160)
        fig.add_trace(
            go.Scatter(
                x=qq_sys,
                y=[sys_curve.get_h(float(q)) for q in qq_sys],
                mode="lines",
                name="Curva del sistema",
                line=dict(width=2, color="#f59e0b", dash="dash"),
                hovertemplate="Q: %{x:.2f} m³/h<br>H: %{y:.2f} m<extra></extra>",
            )
        )

    if metric == "H" and q_req is not None and h_req is not None:
        fig.add_trace(
            go.Scatter(
                x=[q_req],
                y=[h_req],
                mode="markers",
                name="Punto requerido",
                marker=dict(size=10, color="#ef4444", symbol="x"),
                hovertemplate="Q: %{x:.1f} m³/h<br>H: %{y:.1f} m<extra></extra>",
            )
        )

    if op_q is not None and selected_trim is not None:
        if metric == "H":
            op_y = selected_trim.get_h(op_q)
        elif metric == "ETA":
            op_y = selected_trim.get_eta(op_q) * visco_cf
        elif metric == "P":
            op_y = selected_trim.get_power(op_q, density=density, viscosity_cf=visco_cf)
        else:
            op_y = selected_trim.get_npshr(op_q)

        fig.add_trace(
            go.Scatter(
                x=[op_q],
                y=[op_y],
                mode="markers",
                name="Punto operativo",
                marker=dict(size=11, color="#dc2626"),
                hovertemplate="Q: %{x:.1f}<br>Valor: %{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=380,
        margin=dict(l=15, r=15, t=55, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(
        title_text="Caudal Q (m³/h)",
        showgrid=True,
        gridcolor="rgba(200,200,200,0.35)",
        zeroline=False,
        tickformat=".2f",
    )
    fig.update_yaxes(
        title_text=metric_label(metric),
        showgrid=True,
        gridcolor="rgba(200,200,200,0.35)",
        zeroline=False,
        tickformat=".2f",
    )
    apply_graph_box(fig)
    return fig


def render_characteristic_curves(
    fam: Dict,
    show_all_diameters: bool,
    selected_trim: Optional[TrimmedCurve] = None,
    selected_real_diam: Optional[float] = None,
    op_q: Optional[float] = None,
    q_req: Optional[float] = None,
    h_req: Optional[float] = None,
    sys_curve: Optional[SystemCurve] = None,
    density: float = 1000.0,
    visco_cf: float = 1.0,
) -> None:
    st.subheader("Curvas Caracteristicas")
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    with c1:
        st.plotly_chart(
            plot_family_metric(
                fam=fam,
                metric="H",
                title="Altura",
                show_all_diameters=show_all_diameters,
                selected_trim=selected_trim,
                selected_real_diam=selected_real_diam,
                op_q=op_q,
                q_req=q_req,
                h_req=h_req,
                sys_curve=sys_curve,
                density=density,
                visco_cf=visco_cf,
            ),
            use_container_width=True,
        )

    with c2:
        st.plotly_chart(
            plot_family_metric(
                fam=fam,
                metric="ETA",
                title="Eficiencia",
                show_all_diameters=show_all_diameters,
                selected_trim=selected_trim,
                selected_real_diam=selected_real_diam,
                op_q=op_q,
                density=density,
                visco_cf=visco_cf,
            ),
            use_container_width=True,
        )

    with c3:
        st.plotly_chart(
            plot_family_metric(
                fam=fam,
                metric="P",
                title="Potencia",
                show_all_diameters=show_all_diameters,
                selected_trim=selected_trim,
                selected_real_diam=selected_real_diam,
                op_q=op_q,
                density=density,
                visco_cf=visco_cf,
            ),
            use_container_width=True,
        )

    with c4:
        st.plotly_chart(
            plot_family_metric(
                fam=fam,
                metric="NPSH",
                title="NPSH",
                show_all_diameters=show_all_diameters,
                selected_trim=selected_trim,
                selected_real_diam=selected_real_diam,
                op_q=op_q,
                density=density,
                visco_cf=visco_cf,
            ),
            use_container_width=True,
        )


def family_curve_summary_table(fam: Dict) -> pd.DataFrame:
    rows = []
    for curve in fam["curvas"]:
        rows.append(
            {
                "Diámetro (mm)": round(curve.diam),
                "Q mín. (m3/h)": round(curve.q_min, 2),
                "Q máx. (m3/h)": round(curve.q_max, 2),
                "H @ Q mín. (m)": round(curve.get_h(curve.q_min), 2),
                "H @ Q máx. (m)": round(curve.get_h(curve.q_max), 2),
                "Q BEP (m3/h)": round(curve.q_bep, 2),
                "η BEP (%)": round(curve.eta_bep, 2),
            }
        )
    return pd.DataFrame(rows)


# ==============================
# Vista selección por punto
# ==============================
def evaluate_families(
    families: List[Dict],
    q_req: float,
    h_req: float,
    npsha: float,
    densidad: float,
    viscosidad: float,
    usar_sistema: bool,
    h_est: float,
    k_sys: float,
) -> List[Dict]:
    sys_curve = SystemCurve(h_stat=h_est, k=k_sys) if usar_sistema else None
    visco_cf = viscosity_correction(viscosidad)
    results: List[Dict] = []

    for fam in families:
        valid_curves = [c for c in fam["curvas"] if c.get_h(q_req) >= h_req]
        if valid_curves:
            base_curve = valid_curves[0]
        else:
            base_curve = fam["curvas"][-1]

        d_req = find_trim_diameter(q_req, h_req, base_curve, fam["D_min"], fam["D_max"])
        if d_req is None:
            continue

        if not (fam["D_min"] <= d_req <= fam["D_max"]):
            continue

        if base_curve.stable_q_min > q_req:
            continue

        trim_curve = TrimmedCurve(base_curve, d_req)
        op_q = find_operating_point(
            trim_curve,
            sys_curve,
            q_max=base_curve.q_max * trim_curve.ratio,
        )
        if op_q is None:
            op_q = q_req

        h_op = trim_curve.get_h(op_q)
        eta_op = trim_curve.get_eta(op_q) * visco_cf
        p_kw = trim_curve.get_power(op_q, density=densidad, viscosity_cf=visco_cf)
        npshr = trim_curve.get_npshr(op_q)
        p_max = trim_curve.get_max_power(base_curve.q_max * trim_curve.ratio, density=densidad, viscosity_cf=visco_cf)
        motor_kw = select_motor(p_max)
        npsh_status = npsha >= (npshr + NPSH_MARGIN_M)

        results.append(
            {
                "Serie (Marca)": fam["serie"],
                "Marca": fam["marca"],
                "Modelo": fam["modelo"],
                "Polos": float(fam["polos"]) if fam["polos"] is not None else np.nan,
                "RPM": float(fam["rpm"]) if fam["rpm"] is not None else np.nan,
                "D_Impulsor (mm)": int(round(d_req)),
                "Q Op. (m3/h)": round(op_q, 1),
                "H Op. (m)": round(h_op, 1),
                "Eficiencia (%)": round(eta_op, 2),
                "Potencia (kW)": round(p_kw, 2),
                "NPSHr (m)": round(npshr, 2),
                "Status NPSH": npsh_status,
                "Motor IEC (kW)": round(motor_kw, 2),
                "_trim": trim_curve,
                "_fam": fam,
                "_sys_curve": sys_curve,
                "_q_req": q_req,
                "_h_req": h_req,
                "_visco_cf": visco_cf,
                "_densidad": densidad,
            }
        )

    results.sort(key=lambda x: (-x["Status NPSH"], -x["Eficiencia (%)"], x["Potencia (kW)"]))
    return results


def hydraulic_selection_view(families: List[Dict]) -> None:
    st.sidebar.header("1. Parámetros del Fluido")
    fluid_name = st.sidebar.text_input("Fluido", "Agua limpia")
    densidad = st.sidebar.number_input("Densidad (kg/m³)", value=1000.0, step=10.0, format="%.2f")
    viscosidad = st.sidebar.number_input("Viscosidad cinemática (cSt)", value=1.0, step=0.5, format="%.2f")

    st.sidebar.header("2. Punto de Operación")
    q_req = st.sidebar.number_input("Caudal solicitado Q (m³/h)", value=50.0, step=1.0, format="%.1f")
    h_req = st.sidebar.number_input("Altura solicitada H (m)", value=30.0, step=1.0, format="%.1f")
    npsha = st.sidebar.number_input("NPSH disponible (m)", value=10.0, step=0.5, format="%.2f")

    st.sidebar.header("3. Curva del Sistema")
    usar_sistema = st.sidebar.checkbox("Considerar curva del sistema", value=True)
    h_est = st.sidebar.number_input("Carga estática (m)", value=0.0, disabled=not usar_sistema, format="%.2f")
    k_calc = (h_req - h_est) / (q_req ** 2) if q_req > 0 else 0.0
    k_sys = st.sidebar.number_input(
        "Coef. fricción sistema (k)",
        value=float(k_calc),
        format="%.6f",
        disabled=not usar_sistema,
    )

    st.sidebar.header("4. Visualización")
    show_all_diameters = st.sidebar.radio(
        "Diámetros a mostrar",
        ["Solo el escogido para el punto", "Todos los diámetros de la bomba seleccionada"],
        index=0,
    ) == "Todos los diámetros de la bomba seleccionada"

    st.markdown("### Selección de Bombas por punto hidraulico")
    st.caption(f"Fluido: {fluid_name} · Densidad: {fmt2(densidad)} kg/m³ · Viscosidad: {fmt2(viscosidad)} cSt")
    st.caption(f"Punto requerido: Q = {fmt1(q_req)} m³/h · H = {fmt1(h_req)} m")

    results = evaluate_families(
        families=families,
        q_req=q_req,
        h_req=h_req,
        npsha=npsha,
        densidad=densidad,
        viscosidad=viscosidad,
        usar_sistema=usar_sistema,
        h_est=h_est,
        k_sys=k_sys,
    )

    if not results:
        st.warning("No se encontraron bombas que cumplan con el punto solicitado dentro del rango real de diámetros del modelo.")
        return

    st.subheader("Bombas que cumplen con los requisitos")
    df_res = pd.DataFrame(results).drop(
        columns=["_trim", "_fam", "_sys_curve", "_q_req", "_h_req", "_visco_cf", "_densidad", "Motor IEC (kW)"]
    )

    ordered_cols = [
        "Serie (Marca)",
        "Marca",
        "Modelo",
        "Polos",
        "RPM",
        "D_Impulsor (mm)",
        "Q Op. (m3/h)",
        "H Op. (m)",
        "Eficiencia (%)",
        "Potencia (kW)",
        "NPSHr (m)",
        "Status NPSH",
    ]
    df_res = df_res[ordered_cols]

    st.dataframe(
        style_results_df(df_res),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    labels = [
        f"{r['Serie (Marca)']} | {r['Modelo']} | {fmt0(r['Polos'])} polos | η={fmt2(r['Eficiencia (%)'])}% | D={fmt0(r['D_Impulsor (mm)'])} mm"
        for r in results
    ]
    selected_idx = st.selectbox(
        "Selecciona una bomba para revisar curvas",
        range(len(results)),
        format_func=lambda i: labels[i],
    )

    selected = results[selected_idx]
    fam = selected["_fam"]
    trim_curve = selected["_trim"]

    st.markdown(f"#### {selected['Modelo']} · {fmt0(selected['Polos'])} polos")

    detail_cols = st.columns(5)
    detail_cols[0].metric("Serie", selected["Serie (Marca)"])
    detail_cols[1].metric("Modelo", selected["Modelo"])
    detail_cols[2].metric("Diámetro aproximado", f"{fmt0(selected['D_Impulsor (mm)'])} mm")
    detail_cols[3].metric(
        "Punto operativo",
        f"Q={fmt1(selected['Q Op. (m3/h)'])} | H={fmt1(selected['H Op. (m)'])}"
    )
    detail_cols[4].metric("NPSH", "OK" if selected["Status NPSH"] else "Revisar")

    render_characteristic_curves(
        fam=fam,
        show_all_diameters=show_all_diameters,
        selected_trim=trim_curve,
        selected_real_diam=selected["D_Impulsor (mm)"],
        op_q=selected["Q Op. (m3/h)"],
        q_req=q_req,
        h_req=h_req,
        sys_curve=selected["_sys_curve"],
        density=densidad,
        visco_cf=selected["_visco_cf"],
    )

    st.markdown("#### Datos de la bomba seleccionada")
    st.dataframe(
        style_numeric_df(
            family_curve_summary_table(fam),
            zero_dec_cols=["Diámetro (mm)"],
        ),
        use_container_width=True,
        hide_index=True,
    )


# ==============================
# Selección manual - formato ajustado
# ==============================
def reset_manual_filters():
    keys_to_delete = [k for k in st.session_state.keys() if k.startswith("manualflt_")]
    for key in keys_to_delete:
        del st.session_state[key]


def render_checkbox_filter_group(
    options: List,
    prefix: str,
    limit: int = 5,
    formatter=lambda x: str(x),
):
    show_key = f"manualflt_showall_{prefix}"
    if show_key not in st.session_state:
        st.session_state[show_key] = False

    selected = []
    visible_options = options if st.session_state[show_key] else options[:limit]

    for opt in visible_options:
        ckey = keyify(f"manualflt_{prefix}", opt)
        checked = st.checkbox(formatter(opt), key=ckey)
        if checked:
            selected.append(opt)

    if len(options) > limit:
        label = "Show all" if not st.session_state[show_key] else "Show less"
        if st.button(label, key=f"toggle_{prefix}"):
            st.session_state[show_key] = not st.session_state[show_key]
            st.rerun()

    return selected


def manual_selection_view(families: List[Dict]) -> None:
    st.markdown("### Seleccion de Bombas Manual")
    catalog_df = build_catalog_df(families)

    if catalog_df.empty:
        st.warning("La base cargada no contiene familias utilizables.")
        return

    catalog_df = catalog_df.sort_values(
        by=["Descarga DN", "Modelo", "Polos"],
        ascending=[True, True, True],
        na_position="last"
    ).reset_index(drop=True)

    left, right = st.columns([1.05, 4.0], gap="large")

    with left:
        st.markdown("<div class='manual-panel'>", unsafe_allow_html=True)
        header_cols = st.columns([2, 1])
        with header_cols[0]:
            st.markdown("<div class='filter-list-title'>Filter list</div>", unsafe_allow_html=True)
        with header_cols[1]:
            if st.button("Reset", key="manual_reset"):
                reset_manual_filters()
                st.rerun()

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        series_options = sorted([x for x in catalog_df["Serie (Marca)"].dropna().unique().tolist()])
        model_options = sorted([x for x in catalog_df["Modelo"].dropna().unique().tolist()])
        poles_options = sorted([int(x) for x in catalog_df["Polos"].dropna().unique().tolist()])
        dn_options = sorted([int(x) for x in catalog_df["Descarga DN"].dropna().unique().tolist()])

        with st.expander("Serie (Marca)", expanded=True):
            selected_series = render_checkbox_filter_group(
                series_options, "series", limit=5
            )

        with st.expander("Modelo", expanded=True):
            selected_models = render_checkbox_filter_group(
                model_options, "models", limit=5
            )

        with st.expander("Polos", expanded=True):
            selected_poles = render_checkbox_filter_group(
                poles_options, "poles", limit=6, formatter=lambda x: f"{x}"
            )

        with st.expander("Diámetro de descarga", expanded=True):
            selected_dn = render_checkbox_filter_group(
                dn_options, "dn", limit=6, formatter=lambda x: f"{x}"
            )

        model_search = st.text_input("Buscar modelo", key="manualflt_search_model")
        st.markdown("</div>", unsafe_allow_html=True)

    filtered = catalog_df.copy()

    if selected_series:
        filtered = filtered[filtered["Serie (Marca)"].isin(selected_series)]
    if selected_models:
        filtered = filtered[filtered["Modelo"].isin(selected_models)]
    if selected_poles:
        filtered = filtered[filtered["Polos"].isin([float(x) for x in selected_poles])]
    if selected_dn:
        filtered = filtered[filtered["Descarga DN"].isin([float(x) for x in selected_dn])]
    if model_search:
        filtered = filtered[filtered["Modelo"].astype(str).str.contains(model_search, case=False, na=False)]

    filtered = filtered.reset_index(drop=True)

    with right:
        st.markdown(f"<div class='results-count'>{len(filtered)} Results</div>", unsafe_allow_html=True)

        if filtered.empty:
            st.warning("No hay modelos que coincidan con los filtros aplicados.")
            return

        manual_table = pd.DataFrame(
            {
                "Curva": ["Ver"] * len(filtered),
                "Serie (Marca)": filtered["Serie (Marca)"],
                "Modelo": filtered["Modelo"],
                "Polos": filtered["Polos"],
                "Impulsor actual [mm]": filtered["D máx. (mm)"].round().astype(int),
                "Descarga DN": filtered["Descarga DN"],
                "RPM": filtered["RPM"],
            }
        )

        st.dataframe(
            style_numeric_df(
                manual_table,
                zero_dec_cols=["Polos", "Descarga DN", "RPM", "Impulsor actual [mm]"],
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")

        show_all_diameters = st.radio(
            "Diámetros a mostrar",
            ["Todos los diámetros disponibles", "Solo un diámetro"],
            index=0,
            horizontal=True,
        ) == "Todos los diámetros disponibles"

        selection_labels = [
            f"{row['Serie (Marca)']} | {row['Modelo']} | {fmt0(row['Polos'])} polos | D máx = {fmt0(row['D máx. (mm)'])} mm"
            for _, row in filtered.iterrows()
        ]

        selection_index = st.selectbox(
            "Selecciona una bomba",
            range(len(filtered)),
            format_func=lambda i: selection_labels[i],
        )

        selected_row = filtered.iloc[selection_index]
        fam = selected_row["_fam"]

        selected_real_diam = None
        if not show_all_diameters:
            selected_real_diam = st.selectbox(
                "Diámetro a mostrar",
                fam["diametros_disponibles"],
                format_func=lambda x: f"{round(x):.0f} mm",
            )

        st.markdown(f"#### {selected_row['Modelo']} · {fmt0(selected_row['Polos'])} polos")

        kpi_cols = st.columns(6)
        kpi_cols[0].metric("Serie", str(selected_row["Serie (Marca)"]))
        kpi_cols[1].metric("Modelo", str(selected_row["Modelo"]))
        kpi_cols[2].metric("Polos", fmt0(selected_row["Polos"]))
        kpi_cols[3].metric("RPM", fmt0(selected_row["RPM"]))
        kpi_cols[4].metric("D mín. (mm)", fmt0(selected_row["D mín. (mm)"]))
        kpi_cols[5].metric("D máx. (mm)", fmt0(selected_row["D máx. (mm)"]))

        render_characteristic_curves(
            fam=fam,
            show_all_diameters=show_all_diameters,
            selected_real_diam=selected_real_diam,
            density=1000.0,
            visco_cf=1.0,
        )

        st.markdown("#### Diámetros presentes en la base de datos")
        st.dataframe(
            style_numeric_df(
                family_curve_summary_table(fam),
                zero_dec_cols=["Diámetro (mm)"],
            ),
            use_container_width=True,
            hide_index=True,
        )


# ==============================
# Shell de páginas de trabajo
# ==============================
def render_work_page_header() -> None:
    render_top_header()

    with st.sidebar:
        st.success(f"Usuario conectado: {st.session_state.username}")
        if st.button("Volver al menú principal", use_container_width=True):
            go_to("menu")
        if st.button("Cerrar sesión", use_container_width=True):
            logout()
        st.markdown("---")


# ==============================
# App principal
# ==============================
def app() -> None:
    if not st.session_state.authenticated:
        login_view()
        return

    if st.session_state.page == "menu":
        main_menu_view()
        return

    render_work_page_header()

    families = load_database_widget()
    if families is None:
        return

    if st.session_state.page == "hydraulic":
        hydraulic_selection_view(families)
    elif st.session_state.page == "manual":
        manual_selection_view(families)
    else:
        go_to("menu")


if __name__ == "__main__":
    app()
