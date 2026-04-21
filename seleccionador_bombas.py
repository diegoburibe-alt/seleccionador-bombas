import os
import re
import base64
from typing import Dict, List, Optional, Tuple

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


# ==============================
# Utilidades visuales
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


def find_v_path() -> Optional[str]:
    possible_paths = [
        "V.svg",
        "v.svg",
        "V.png",
        "v.png",
        os.path.join(os.getcwd(), "V.svg"),
        os.path.join(os.getcwd(), "v.svg"),
        os.path.join(os.getcwd(), "V.png"),
        os.path.join(os.getcwd(), "v.png"),
        "/mnt/data/V.svg",
        "/mnt/data/v.svg",
        "/mnt/data/V.png",
        "/mnt/data/v.png",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


PAGE_ICON = find_v_path()
st.set_page_config(
    layout="wide",
    page_title=APP_TITLE,
    page_icon=PAGE_ICON if PAGE_ICON else None,
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .main-block {padding-top: 0.6rem;}

            .app-title {
                font-size: 2.15rem;
                font-weight: 800;
                line-height: 1.08;
                margin: 0;
                color: #1f2937;
            }

            .app-subtitle {
                color: #4b5563;
                font-weight: 700;
                margin-top: 0.28rem;
                font-size: 1.04rem;
            }

            .menu-card {
                border: 1px solid #d7dee8;
                border-radius: 16px;
                padding: 1rem 1.2rem;
                background: #ffffff;
                min-height: 110px;
            }

            .menu-card h4 {
                margin: 0 0 0.4rem 0;
                color: #0f172a;
            }

            .small-note {
                font-size: 0.92rem;
                color: #4b5563;
            }

            .stDataFrame, .stTable {font-size: 0.95rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


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


def safe_float(value, default=np.nan) -> float:
    try:
        return float(str(value).replace(",", ".").strip())
    except Exception:
        return default


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
# Construcción de vistas
# ==============================
def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "username": "",
        "screen": "Menú principal",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def render_top_header() -> None:
    logo_path = find_logo_path()
    v_path = find_v_path()

    col_logo, col_title = st.columns([2.2, 4.2], vertical_alignment="center")

    with col_logo:
        if logo_path:
            logo_uri = image_to_data_uri(logo_path)
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; height:140px;">
                    <img src="{logo_uri}" style="max-height:120px; width:auto;">
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='font-size:3rem; font-weight:900; color:#0059aa;'>VOGT</div>",
                unsafe_allow_html=True,
            )

    with col_title:
        v_html = ""
        if v_path:
            v_uri = image_to_data_uri(v_path)
            v_html = f'<img src="{v_uri}" style="height:74px; width:auto;">'
        else:
            v_html = """
            <div style="
                width:74px; height:74px; border-radius:18px;
                background:#0059aa; color:white; display:flex;
                align-items:center; justify-content:center;
                font-size:42px; font-weight:900;
            ">V</div>
            """

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:1rem; min-height:140px;">
                {v_html}
                <div>
                    <div class="app-title">{APP_TITLE}</div>
                    <div class="app-subtitle">{APP_SUBTITLE}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")


# ==============================
# Carga y resumen de base
# ==============================
def build_catalog_df(families: List[Dict]) -> pd.DataFrame:
    rows = []
    for fam in families:
        rows.append(
            {
                "Serie (Marca)": fam["serie"],
                "Marca": fam["marca"],
                "Modelo": fam["modelo"],
                "Polos": fam["polos"],
                "RPM": fam["rpm"],
                "Descarga DN": fam["descarga_dn"],
                "D mín. (mm)": fam["D_min"],
                "D máx. (mm)": fam["D_max"],
                "Diámetros disponibles": ", ".join(
                    str(int(d)) if float(d).is_integer() else f"{d:.1f}"
                    for d in fam["diametros_disponibles"]
                ),
                "_fam": fam,
            }
        )
    return pd.DataFrame(rows)


def load_database_widget() -> Tuple[Optional[PumpDatabase], Optional[List[Dict]]]:
    uploaded_file = st.sidebar.file_uploader("Cargar Base Oficial (CSV)", type=["csv"])
    if uploaded_file is None:
        st.info("Carga la base CSV para habilitar la selección.")
        return None, None

    db = PumpDatabase()
    try:
        db.load_from_csv(uploaded_file)
        families = db.get_families()
        st.sidebar.success(f"Base cargada: {len(families)} familias activas")
        return db, families
    except Exception as exc:
        st.error(f"Error de integridad en CSV: {exc}")
        return None, None


# ==============================
# Resultados por punto hidráulico
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
        end_op = base_curve.q_max * trim_curve.ratio
        p_max = trim_curve.get_max_power(end_op, density=densidad, viscosity_cf=visco_cf)
        motor_kw = select_motor(p_max)
        npsh_status = npsha >= (npshr + NPSH_MARGIN_M)

        results.append(
            {
                "Serie (Marca)": fam["serie"],
                "Marca": fam["marca"],
                "Modelo": fam["modelo"],
                "Polos": fam["polos"],
                "RPM": fam["rpm"],
                "D_Impulsor (mm)": round(d_req, 1),
                "Q Op. (m3/h)": round(op_q, 2),
                "H Op. (m)": round(h_op, 2),
                "Eficiencia (%)": round(eta_op, 2),
                "Potencia (kW)": round(p_kw, 2),
                "NPSHr (m)": round(npshr, 2),
                "Status NPSH": npsh_status,
                "Motor IEC (kW)": motor_kw,
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


def npsh_style(val):
    if isinstance(val, bool) and not val:
        return "background-color: #fde2e1"
    return ""


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
                    f"Diámetro escogido = {selected_trim.diam:.1f} mm"
                    if selected_real_diam is None
                    else f"D = {selected_real_diam:.1f} mm"
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
            )
        )

    if metric == "H" and sys_curve is not None:
        q_lim = max([c.q_max for c in fam["curvas"]] + [q_req or 0.0])
        qq_sys = np.linspace(0.0, max(1.2 * q_lim, 1.0), 160)
        fig.add_trace(
            go.Scatter(
                x=qq_sys,
                y=[sys_curve.get_h(float(q)) for q in qq_sys],
                mode="lines",
                name="Curva del sistema",
                line=dict(width=2, color="#f59e0b", dash="dash"),
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
    )
    fig.update_yaxes(
        title_text=metric_label(metric),
        showgrid=True,
        gridcolor="rgba(200,200,200,0.35)",
        zeroline=False,
    )
    return fig


def render_characteristic_curves(
    fam: Dict,
    title_prefix: str,
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
                title=f"{title_prefix} · H-Q",
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
                title=f"{title_prefix} · Eficiencia-Q",
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
                title=f"{title_prefix} · Potencia-Q",
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
                title=f"{title_prefix} · NPSHr-Q",
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
                "Diámetro (mm)": curve.diam,
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
# Pantallas principales
# ==============================
def hydraulic_selection_view(families: List[Dict]) -> None:
    st.sidebar.header("1. Parámetros del Fluido")
    fluid_name = st.sidebar.text_input("Fluido", "Agua limpia")
    densidad = st.sidebar.number_input("Densidad (kg/m³)", value=1000.0, step=10.0)
    viscosidad = st.sidebar.number_input("Viscosidad cinemática (cSt)", value=1.0, step=0.5)

    st.sidebar.header("2. Punto de Operación")
    q_req = st.sidebar.number_input("Caudal solicitado Q (m³/h)", value=50.0, step=1.0)
    h_req = st.sidebar.number_input("Altura solicitada H (m)", value=30.0, step=1.0)
    npsha = st.sidebar.number_input("NPSH disponible (m)", value=10.0, step=0.5)

    st.sidebar.header("3. Curva del Sistema")
    usar_sistema = st.sidebar.checkbox("Considerar curva del sistema", value=True)
    h_est = st.sidebar.number_input("Carga estática (m)", value=0.0, disabled=not usar_sistema)
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
    st.caption(f"Fluido: {fluid_name} · Densidad: {densidad:.1f} kg/m³ · Viscosidad: {viscosidad:.1f} cSt")

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
        df_res.style.map(npsh_style, subset=["Status NPSH"]),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    labels = [
        f"{r['Serie (Marca)']} | {r['Modelo']} | {r['Polos']} polos | η={r['Eficiencia (%)']:.2f}% | D={r['D_Impulsor (mm)']:.1f} mm"
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

    detail_cols = st.columns(5)
    detail_cols[0].metric("Serie", selected["Serie (Marca)"])
    detail_cols[1].metric("Modelo", selected["Modelo"])
    detail_cols[2].metric("Diámetro escogido", f"{selected['D_Impulsor (mm)']:.1f} mm")
    detail_cols[3].metric("Punto operativo", f"Q={selected['Q Op. (m3/h)']:.2f} | H={selected['H Op. (m)']:.2f}")
    detail_cols[4].metric("NPSH", "OK" if selected["Status NPSH"] else "Revisar")

    render_characteristic_curves(
        fam=fam,
        title_prefix=f"{selected['Modelo']} · {selected['Polos']} polos",
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
        family_curve_summary_table(fam),
        use_container_width=True,
        hide_index=True,
    )


def manual_selection_view(families: List[Dict]) -> None:
    st.markdown("### Seleccion de Bombas Manual")
    catalog_df = build_catalog_df(families)
    if catalog_df.empty:
        st.warning("La base cargada no contiene familias utilizables.")
        return

    st.sidebar.header("Filtros selección manual")
    available_series = sorted([x for x in catalog_df["Serie (Marca)"].dropna().unique().tolist()])
    available_poles = sorted([int(x) for x in catalog_df["Polos"].dropna().unique().tolist()])
    available_dn = sorted([int(x) for x in catalog_df["Descarga DN"].dropna().unique().tolist()])

    filter_series = st.sidebar.multiselect("Serie (Marca)", available_series)
    filter_poles = st.sidebar.multiselect("Número de polos", available_poles)
    filter_dn = st.sidebar.multiselect("Diámetro de descarga", available_dn)
    model_search = st.sidebar.text_input("Buscar modelo")
    show_all_diameters = st.sidebar.radio(
        "Diámetros a mostrar",
        ["Todos los diámetros disponibles", "Solo un diámetro"],
        index=0,
    ) == "Todos los diámetros disponibles"

    filtered = catalog_df.copy()
    if filter_series:
        filtered = filtered[filtered["Serie (Marca)"].isin(filter_series)]
    if filter_poles:
        filtered = filtered[filtered["Polos"].isin(filter_poles)]
    if filter_dn:
        filtered = filtered[filtered["Descarga DN"].isin(filter_dn)]
    if model_search:
        filtered = filtered[filtered["Modelo"].astype(str).str.contains(model_search, case=False, na=False)]

    if filtered.empty:
        st.warning("No hay modelos que coincidan con los filtros aplicados.")
        return

    st.subheader("Lista completa de modelos")
    st.dataframe(filtered.drop(columns=["_fam"]), use_container_width=True, hide_index=True)

    selection_labels = [
        f"{row['Serie (Marca)']} | {row['Modelo']} | {int(row['Polos']) if pd.notna(row['Polos']) else '-'} polos | DN {int(row['Descarga DN']) if pd.notna(row['Descarga DN']) else '-'}"
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
            format_func=lambda x: f"{x:.1f} mm",
        )

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Serie", str(selected_row["Serie (Marca)"]))
    kpi_cols[1].metric("Modelo", str(selected_row["Modelo"]))
    kpi_cols[2].metric("Polos", str(int(selected_row["Polos"])) if pd.notna(selected_row["Polos"]) else "-")
    kpi_cols[3].metric("D mín. / D máx.", f"{selected_row['D mín. (mm)']:.1f} / {selected_row['D máx. (mm)']:.1f} mm")
    kpi_cols[4].metric("Cantidad diámetros", str(len(fam["diametros_disponibles"])))

    render_characteristic_curves(
        fam=fam,
        title_prefix=f"{selected_row['Modelo']} · {int(selected_row['Polos']) if pd.notna(selected_row['Polos']) else '-'} polos",
        show_all_diameters=show_all_diameters,
        selected_real_diam=selected_real_diam,
        density=1000.0,
        visco_cf=1.0,
    )

    st.markdown("#### Diámetros presentes en la base de datos")
    st.dataframe(
        family_curve_summary_table(fam),
        use_container_width=True,
        hide_index=True,
    )


def main_menu_view(families: List[Dict]) -> None:
    st.markdown("## Menu principal")
    st.markdown(
        """
        <div class='menu-card'>
            <h4>1. Selección de Bombas por punto hidraulico</h4>
            <div class='small-note'>Evalúa el punto Q-H requerido, calcula diámetro necesario y revisa curvas características.</div>
        </div>
        <br>
        <div class='menu-card'>
            <h4>2. Seleccion de Bombas Manual</h4>
            <div class='small-note'>Explora la lista completa de modelos, filtra por serie, polos o diámetro de descarga y revisa sus curvas.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    view = st.radio(
        "Ir a",
        ["Selección de Bombas por punto hidraulico", "Seleccion de Bombas Manual"],
        horizontal=True,
    )

    st.markdown("---")
    if view == "Selección de Bombas por punto hidraulico":
        hydraulic_selection_view(families)
    else:
        manual_selection_view(families)


# ==============================
# App principal
# ==============================
def app() -> None:
    if not st.session_state.authenticated:
        login_view()
        return

    render_top_header()

    with st.sidebar:
        st.success(f"Usuario conectado: {st.session_state.username}")
        if st.button("Cerrar sesión", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.rerun()
        st.markdown("---")

    db, families = load_database_widget()
    if not db or not families:
        return

    main_menu_view(families)


if __name__ == "__main__":
    app()
