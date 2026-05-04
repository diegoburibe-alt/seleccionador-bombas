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
VALID_PASSWORD = "Vogt1234"
APP_SUBTITLE = "Series N-NP-N(V)"
APP_TITLE = "Seleccionador Bombas Normalizadas"
BASE_DATABASE_FILENAME = "Base_Datos_wilo_sempa_grundfos.csv"

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


def display_serie_value(marca: str, serie: Optional[str] = None) -> str:
    marca_txt = str(marca).strip().lower()
    if "grundfos" in marca_txt:
        return "N"
    if "sempa" in marca_txt:
        return "NP"
    if "wilo" in marca_txt:
        return "N (V)"
    if serie is not None and str(serie).strip() not in ["", "-"] and not pd.isna(serie):
        return str(serie)
    return str(marca)


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
    def __init__(self, diam: float, puntos: List[Dict[str, float]], q_max_override: Optional[float] = None) -> None:
        self.diam = float(diam)
        self.puntos = sorted(puntos, key=lambda p: p["Q"])

        q_raw = np.array([p["Q"] for p in self.puntos], dtype=float)
        h_raw = np.array([p["H"] for p in self.puntos], dtype=float)
        eta_raw = np.array([p["eta"] for p in self.puntos], dtype=float)
        npsh_raw = np.array([p["NPSH"] for p in self.puntos], dtype=float)
        p_raw = np.array([p.get("P", np.nan) for p in self.puntos], dtype=float)

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
        q_max_calculado = float(np.max(self.unique_q))
        if q_max_override is not None and not pd.isna(q_max_override) and float(q_max_override) > 0:
            self.q_max = float(q_max_override)
        else:
            self.q_max = q_max_calculado

        valid_p_mask = ~np.isnan(p_raw)
        self.has_power_poly = bool(np.sum(valid_p_mask) >= 3)
        if self.has_power_poly:
            self.popt_p, _ = curve_fit(poly2, q_raw[valid_p_mask], p_raw[valid_p_mask])
        else:
            self.popt_p = None

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
        if self.has_power_poly and self.popt_p is not None:
            return max(0.0, float(poly2(np.array([q]), *self.popt_p)[0]))
        return 0.0


class TrimmedCurve:
    def __init__(self, base_curve: PumpCurveBase, trim_diam: float) -> None:
        self.base = base_curve
        self.diam = float(trim_diam)
        self.ratio = self.diam / self.base.diam
        self.q_min = self.base.q_min * self.ratio
        self.q_max = self.base.q_max * self.ratio

    def get_h(self, q: float) -> float:
        q_base = q / self.ratio
        return (self.ratio ** 2) * self.base.get_h(q_base)

    def get_eta(self, q: float) -> float:
        q_base = q / self.ratio
        return self.base.get_eta(q_base)

    def get_npshr(self, q: float) -> float:
        q_base = q / self.ratio
        return self.base.get_npshr(q_base) * (self.ratio ** 2)

    def get_power(self, q: float, density: float = 1000.0, viscosity_cf: float = 1.0) -> float:
        q_base = q / self.ratio
        return max(0.0, (self.ratio ** 3) * self.base.get_power(q_base, density=density, viscosity_cf=viscosity_cf))

    def get_max_power(self, end_q: float, density: float = 1000.0, viscosity_cf: float = 1.0) -> float:
        qq = np.linspace(max(0.1, 0.02 * end_q), max(end_q, 0.1), 120)
        powers = [self.get_power(qi, density=density, viscosity_cf=viscosity_cf) for qi in qq]
        return float(max(powers)) if powers else 0.0




class InterpolatedDiameterCurve:
    def __init__(
        self,
        curve_low: PumpCurveBase,
        curve_high: PumpCurveBase,
        interp_diam: float,
        q_max_est: Optional[float] = None,
    ) -> None:
        if curve_high.diam <= curve_low.diam:
            raise ValueError("Los diámetros vecinos deben estar ordenados de menor a mayor.")
        self.curve_low = curve_low
        self.curve_high = curve_high
        self.diam = float(interp_diam)
        self.d1 = float(curve_low.diam)
        self.d2 = float(curve_high.diam)
        self.lam = (self.diam - self.d1) / (self.d2 - self.d1)
        self.q_min = max(curve_low.q_min, curve_high.q_min)

        if q_max_est is not None and not pd.isna(q_max_est) and float(q_max_est) > 0:
            self.q_max = float(q_max_est)
        else:
            self.q_max = blend_q_limit(curve_low.q_max, curve_high.q_max, self.lam)

    def _blend(self, y1: float, y2: float) -> float:
        return float((1.0 - self.lam) * y1 + self.lam * y2)

    def _tail_equation_value(self, curve: PumpCurveBase, q: float, metric: str) -> float:
        q_data = np.array(curve.unique_q, dtype=float)
        q_end = float(q_data[-1])

        if metric == "ETA":
            if q <= q_end:
                return curve.get_eta(q)
            y_data = np.array(curve.eta_values, dtype=float)
            y_end = float(curve.get_eta(q_end))
        elif metric == "NPSH":
            if q <= q_end:
                return curve.get_npshr(q)
            y_data = np.array(curve.npsh_values, dtype=float)
            y_end = float(curve.get_npshr(q_end))
        else:
            return 0.0

        n_tail = min(10, len(q_data))
        if n_tail < 2:
            return y_end

        x_tail = q_data[-n_tail:] - q_end
        y_tail = y_data[-n_tail:]
        degree = 2 if n_tail >= 4 else 1

        try:
            coeffs = np.polyfit(x_tail, y_tail, degree)
            x_eval = float(q) - q_end
            y_poly = float(np.polyval(coeffs, x_eval))
            y_poly_at_end = float(np.polyval(coeffs, 0.0))
            y_est = y_end + (y_poly - y_poly_at_end)
        except Exception:
            y_est = y_end

        if metric == "ETA":
            return float(max(0.0, min(100.0, y_est)))

        if metric == "NPSH":
            return float(max(0.0, y_est))

        return float(y_est)

    def get_h(self, q: float) -> float:
        return self._blend(self.curve_low.get_h(q), self.curve_high.get_h(q))

    def get_eta(self, q: float) -> float:
        if q > float(self.curve_low.unique_q[-1]):
            eta_low = self._tail_equation_value(self.curve_low, q, "ETA")
        else:
            eta_low = self.curve_low.get_eta(q)
        eta_high = self.curve_high.get_eta(q)
        return self._blend(eta_low, eta_high)

    def get_npshr(self, q: float) -> float:
        if q > float(self.curve_low.unique_q[-1]):
            npsh_low = self._tail_equation_value(self.curve_low, q, "NPSH")
        else:
            npsh_low = self.curve_low.get_npshr(q)
        npsh_high = self.curve_high.get_npshr(q)
        return self._blend(npsh_low, npsh_high)

    def get_power(self, q: float, density: float = 1000.0, viscosity_cf: float = 1.0) -> float:
        return self._blend(
            self.curve_low.get_power(q, density=density, viscosity_cf=viscosity_cf),
            self.curve_high.get_power(q, density=density, viscosity_cf=viscosity_cf),
        )

    def get_max_power(self, end_q: float, density: float = 1000.0, viscosity_cf: float = 1.0) -> float:
        qq = np.linspace(max(0.1, self.q_min), max(end_q, 0.1), 120)
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
        has_pot_kw = "pot_kw" in df.columns
        qmax_col = None
        qmax_col_candidates = [
            "Qmax_m3/h", "Q_max_m3/h", "Qmáx_m3/h", "Q_máx_m3/h",
            "Qmax", "Q_max", "Q máx", "Q max", "qmax", "q_max",
        ]
        normalized_cols = {str(col).strip().lower(): col for col in df.columns}
        for candidate in qmax_col_candidates:
            key = candidate.strip().lower()
            if key in normalized_cols:
                qmax_col = normalized_cols[key]
                break

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

                q_max_override = None
                if qmax_col is not None:
                    qmax_values = [safe_float(v, np.nan) for v in group_d[qmax_col].tolist()]
                    qmax_values = [v for v in qmax_values if not np.isnan(v) and v > 0]
                    if qmax_values:
                        q_max_override = max(qmax_values)

                for _, row in group_d.iterrows():
                    q = safe_float(row["Q_m3/h"])
                    h = safe_float(row["H_m"])
                    eta = safe_float(row.get("h_%", row.get("eta", 0.0)), 0.0)
                    npsh = safe_float(row.get("NPSH_m", row.get("NPSH", 0.0)), 0.0)
                    p = safe_float(row.get("pot_kw", np.nan), np.nan) if has_pot_kw else np.nan
                    if not np.isnan(q) and not np.isnan(h):
                        puntos.append({"Q": q, "H": h, "eta": eta, "NPSH": npsh, "P": p})

                if len(puntos) >= 3:
                    try:
                        curvas.append(PumpCurveBase(d, puntos, q_max_override=q_max_override))
                    except Exception:
                        continue

            if not curvas:
                continue

            curvas = sorted(curvas, key=lambda c: c.diam)

            self.families.append(
                {
                    "serie": serie if serie else marca,
                    "serie_display": display_serie_value(marca, serie),
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
def get_curve_q_max(curve_obj) -> Optional[float]:
    if curve_obj is None:
        return None
    if hasattr(curve_obj, "q_max"):
        q_max = safe_float(getattr(curve_obj, "q_max"), np.nan)
        return None if np.isnan(q_max) else float(q_max)
    if hasattr(curve_obj, "base") and hasattr(curve_obj, "ratio"):
        q_max = safe_float(curve_obj.base.q_max, np.nan) * safe_float(curve_obj.ratio, np.nan)
        return None if np.isnan(q_max) else float(q_max)
    return None


def blend_q_limit(q_low: Optional[float], q_high: Optional[float], lam: float) -> Optional[float]:
    if q_low is None and q_high is None:
        return None
    if q_low is None:
        return float(q_high)
    if q_high is None:
        return float(q_low)
    return float((1.0 - lam) * q_low + lam * q_high)


def qmax_por_afinidad_desde_diametro_maximo(fam: Dict, d_req: float) -> Optional[float]:
    curvas = sorted(fam["curvas"], key=lambda c: c.diam)
    if not curvas:
        return None

    curva_dmax = curvas[-1]
    d_max = safe_float(curva_dmax.diam, np.nan)
    q_max_dmax = get_curve_q_max(curva_dmax)

    if q_max_dmax is None or np.isnan(d_max) or d_max <= 0:
        return q_max_dmax

    return float(q_max_dmax * (float(d_req) / d_max))


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

def find_interpolated_diameter_curve(
    fam: Dict,
    q_req: float,
    h_req: float,
) -> Tuple[Optional[float], Optional[object], Optional[PumpCurveBase], Optional[PumpCurveBase]]:
    curvas = sorted(fam["curvas"], key=lambda c: c.diam)

    for curve in curvas:
        q_max_curve = get_curve_q_max(curve)
        if q_max_curve is not None and q_req > q_max_curve:
            continue

        if abs(curve.get_h(q_req) - h_req) < 1e-8:
            return float(curve.diam), curve, curve, curve

    for curve_low, curve_high in zip(curvas[:-1], curvas[1:]):
        h_low = float(curve_low.get_h(q_req))
        h_high = float(curve_high.get_h(q_req))
        h_min = min(h_low, h_high)
        h_max = max(h_low, h_high)

        if h_min <= h_req <= h_max:
            if abs(h_high - h_low) < 1e-12:
                continue

            lam = (h_req - h_low) / (h_high - h_low)
            d_req = float(curve_low.diam + lam * (curve_high.diam - curve_low.diam))

            q_max_interp = qmax_por_afinidad_desde_diametro_maximo(fam, d_req)
            if q_max_interp is not None and q_req > q_max_interp:
                continue

            if abs(d_req - curve_low.diam) < 1e-8:
                return float(curve_low.diam), curve_low, curve_low, curve_low
            if abs(d_req - curve_high.diam) < 1e-8:
                return float(curve_high.diam), curve_high, curve_high, curve_high

            interp_curve = InterpolatedDiameterCurve(
                curve_low,
                curve_high,
                d_req,
                q_max_est=q_max_interp,
            )
            return d_req, interp_curve, curve_low, curve_high

    return None, None, None, None


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
        "manual_selected_table_row": 0,
        "manual_selected_point_q": None,
        "manual_selected_point_curve_diam": None,
        "manual_selected_point_model": None,
        "manual_selected_point_show_all": None,
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
def find_database_path() -> Optional[str]:
    possible_paths = [
        BASE_DATABASE_FILENAME,
        os.path.join(os.getcwd(), BASE_DATABASE_FILENAME),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), BASE_DATABASE_FILENAME),
        os.path.join("/mnt/data", BASE_DATABASE_FILENAME),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def load_database_widget() -> Optional[List[Dict]]:
    if st.session_state.loaded_families is None:
        db_path = find_database_path()

        if db_path is None:
            st.error(f"No se encontró la base de datos: {BASE_DATABASE_FILENAME}")
            return None

        db = PumpDatabase()
        try:
            db.load_from_csv(db_path)
            st.session_state.loaded_families = db.get_families()
        except Exception as exc:
            st.error(f"Error de integridad en CSV: {exc}")
            st.session_state.loaded_families = None
            return None

    families = st.session_state.loaded_families

    if families is not None:
        st.sidebar.success(f"Base cargada: {len(families)} familias activas")
        return families

    return None


def build_catalog_df(families: List[Dict]) -> pd.DataFrame:
    rows = []
    for fam in families:
        rows.append(
            {
                "Serie": fam["serie_display"],
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


def resolve_manual_curve(fam: Dict, requested_diam: float):
    for curve in fam["curvas"]:
        if abs(curve.diam - requested_diam) < 1e-9:
            return curve

    candidates = [c for c in fam["curvas"] if c.diam >= requested_diam]
    base_curve = candidates[0] if candidates else fam["curvas"][-1]
    return TrimmedCurve(base_curve, requested_diam)


def plot_family_metric(
    fam: Dict,
    metric: str,
    title: str,
    show_all_diameters: bool,
    selected_curve_obj=None,
    selected_real_diam: Optional[float] = None,
    op_q: Optional[float] = None,
    q_req: Optional[float] = None,
    h_req: Optional[float] = None,
    sys_curve: Optional[SystemCurve] = None,
    density: float = 1000.0,
    visco_cf: float = 1.0,
    black_curves: bool = False,
    smooth_curves: bool = False,
    highlight_curve_obj=None,
    highlight_q: Optional[float] = None,
) -> go.Figure:
    fig = go.Figure()

    base_color = "rgba(0,0,0,0.75)" if black_curves else "rgba(130,130,130,0.75)"
    selected_color = "#000000" if black_curves else "#0059aa"
    n_points = 320 if smooth_curves else 140
    n_points_selected = 420 if smooth_curves else 180
    line_shape = "spline" if smooth_curves else "linear"

    if show_all_diameters:
        for curve in fam["curvas"]:
            qq = np.linspace(curve.q_min, curve.q_max, n_points)
            fig.add_trace(
                go.Scatter(
                    x=qq,
                    y=curve_values(curve, metric, qq, density=density, visco_cf=visco_cf),
                    mode="lines",
                    name=f"D={curve.diam:.0f} mm",
                    line=dict(width=1.5, color=base_color, shape=line_shape),
                    hovertemplate="Q: %{x:.2f} m³/h<br>Valor: %{y:.2f}<extra></extra>",
                )
            )

    if selected_curve_obj is not None:
        q_max_plot = selected_curve_obj.q_max if hasattr(selected_curve_obj, "q_max") else selected_curve_obj.base.q_max * selected_curve_obj.ratio
        q_min_plot = selected_curve_obj.q_min if hasattr(selected_curve_obj, "q_min") else max(0.05, selected_curve_obj.base.q_min * selected_curve_obj.ratio)
        qq_sel = np.linspace(q_min_plot, q_max_plot, n_points_selected)

        fig.add_trace(
            go.Scatter(
                x=qq_sel,
                y=curve_values(selected_curve_obj, metric, qq_sel, density=density, visco_cf=visco_cf),
                mode="lines",
                name=f"D = {selected_real_diam:.0f} mm" if selected_real_diam is not None else "Curva seleccionada",
                line=dict(width=3, color=selected_color, shape=line_shape),
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

    if op_q is not None and selected_curve_obj is not None:
        if metric == "H":
            op_y = selected_curve_obj.get_h(op_q)
        elif metric == "ETA":
            op_y = selected_curve_obj.get_eta(op_q) * visco_cf
        elif metric == "P":
            op_y = selected_curve_obj.get_power(op_q, density=density, viscosity_cf=visco_cf)
        else:
            op_y = selected_curve_obj.get_npshr(op_q)

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

    if highlight_curve_obj is not None and highlight_q is not None:
        if metric == "H":
            high_y = highlight_curve_obj.get_h(highlight_q)
        elif metric == "ETA":
            high_y = highlight_curve_obj.get_eta(highlight_q) * visco_cf
        elif metric == "P":
            high_y = highlight_curve_obj.get_power(highlight_q, density=density, viscosity_cf=visco_cf)
        else:
            high_y = highlight_curve_obj.get_npshr(highlight_q)

        fig.add_trace(
            go.Scatter(
                x=[highlight_q],
                y=[high_y],
                mode="markers",
                name="Punto seleccionado",
                marker=dict(size=11, color="#dc2626"),
                hovertemplate="Q: %{x:.2f}<br>Valor: %{y:.2f}<extra></extra>",
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


def render_characteristic_curves_point(
    fam: Dict,
    show_all_diameters: bool,
    selected_curve_obj=None,
    selected_real_diam: Optional[float] = None,
    op_q: Optional[float] = None,
    q_req: Optional[float] = None,
    h_req: Optional[float] = None,
    sys_curve: Optional[SystemCurve] = None,
    density: float = 1000.0,
    visco_cf: float = 1.0,
    values_at_point: Optional[Dict[str, float]] = None,
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
                selected_curve_obj=selected_curve_obj,
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
        if values_at_point is not None:
            st.caption(f"Valor en el punto: {fmt1(values_at_point['H'])} m")

    with c2:
        st.plotly_chart(
            plot_family_metric(
                fam=fam,
                metric="ETA",
                title="Eficiencia",
                show_all_diameters=show_all_diameters,
                selected_curve_obj=selected_curve_obj,
                selected_real_diam=selected_real_diam,
                op_q=op_q,
                density=density,
                visco_cf=visco_cf,
            ),
            use_container_width=True,
        )
        if values_at_point is not None:
            st.caption(f"Valor en el punto: {fmt2(values_at_point['ETA'])} %")

    with c3:
        st.plotly_chart(
            plot_family_metric(
                fam=fam,
                metric="P",
                title="Potencia",
                show_all_diameters=show_all_diameters,
                selected_curve_obj=selected_curve_obj,
                selected_real_diam=selected_real_diam,
                op_q=op_q,
                density=density,
                visco_cf=visco_cf,
            ),
            use_container_width=True,
        )
        if values_at_point is not None:
            st.caption(f"Valor en el punto: {fmt2(values_at_point['P'])} kW")

    with c4:
        st.plotly_chart(
            plot_family_metric(
                fam=fam,
                metric="NPSH",
                title="NPSH",
                show_all_diameters=show_all_diameters,
                selected_curve_obj=selected_curve_obj,
                selected_real_diam=selected_real_diam,
                op_q=op_q,
                density=density,
                visco_cf=visco_cf,
            ),
            use_container_width=True,
        )
        if values_at_point is not None:
            st.caption(f"Valor en el punto: {fmt2(values_at_point['NPSH'])} m")


def render_manual_interactive_curves(
    fam: Dict,
    show_all_diameters: bool,
    selected_curve_obj=None,
    selected_real_diam: Optional[float] = None,
    density: float = 1000.0,
    visco_cf: float = 1.0,
    model_key: str = "",
) -> None:
    st.subheader("Curvas Características")
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    highlight_curve_obj = None
    highlight_q = None
    if (
        st.session_state.manual_selected_point_model == fam["modelo"]
        and st.session_state.manual_selected_point_show_all == show_all_diameters
        and st.session_state.manual_selected_point_q is not None
    ):
        highlight_q = float(st.session_state.manual_selected_point_q)
        curve_diam = st.session_state.manual_selected_point_curve_diam
        if show_all_diameters:
            for curve in fam["curvas"]:
                if abs(curve.diam - curve_diam) < 1e-6:
                    highlight_curve_obj = curve
                    break
        else:
            if selected_curve_obj is not None and abs(selected_curve_obj.diam - curve_diam) < 1e-6:
                highlight_curve_obj = selected_curve_obj

    with c1:
        fig_h = plot_family_metric(
            fam=fam,
            metric="H",
            title="Altura",
            show_all_diameters=show_all_diameters,
            selected_curve_obj=selected_curve_obj,
            selected_real_diam=selected_real_diam,
            density=density,
            visco_cf=visco_cf,
            black_curves=True,
            smooth_curves=True,
            highlight_curve_obj=highlight_curve_obj,
            highlight_q=highlight_q,
        )
        event = st.plotly_chart(
            fig_h,
            use_container_width=True,
            key=f"manual_h_plot_{model_key}",
            on_select="rerun",
            selection_mode=("points",),
        )

        try:
            selected_points = event.selection["points"] if event and event.selection else []
        except Exception:
            selected_points = []

        if selected_points:
            point = selected_points[0]
            selected_q = float(point["x"])
            curve_number = int(point["curve_number"])

            selected_curve_for_point = None
            if show_all_diameters:
                if 0 <= curve_number < len(fam["curvas"]):
                    selected_curve_for_point = fam["curvas"][curve_number]
            else:
                selected_curve_for_point = selected_curve_obj

            if selected_curve_for_point is not None:
                st.session_state.manual_selected_point_q = selected_q
                st.session_state.manual_selected_point_curve_diam = float(selected_curve_for_point.diam)
                st.session_state.manual_selected_point_model = fam["modelo"]
                st.session_state.manual_selected_point_show_all = show_all_diameters
                highlight_q = selected_q
                highlight_curve_obj = selected_curve_for_point

    with c2:
        st.plotly_chart(
            plot_family_metric(
                fam=fam,
                metric="ETA",
                title="Eficiencia",
                show_all_diameters=show_all_diameters,
                selected_curve_obj=selected_curve_obj,
                selected_real_diam=selected_real_diam,
                density=density,
                visco_cf=visco_cf,
                black_curves=True,
                smooth_curves=True,
                highlight_curve_obj=highlight_curve_obj,
                highlight_q=highlight_q,
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
                selected_curve_obj=selected_curve_obj,
                selected_real_diam=selected_real_diam,
                density=density,
                visco_cf=visco_cf,
                black_curves=True,
                smooth_curves=True,
                highlight_curve_obj=highlight_curve_obj,
                highlight_q=highlight_q,
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
                selected_curve_obj=selected_curve_obj,
                selected_real_diam=selected_real_diam,
                density=density,
                visco_cf=visco_cf,
                black_curves=True,
                smooth_curves=True,
                highlight_curve_obj=highlight_curve_obj,
                highlight_q=highlight_q,
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


def sanitize_filename(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip())
    return name.strip("_") or "modelo"


def family_curve_raw_data_df(fam: Dict) -> pd.DataFrame:
    rows = []
    for curve in fam["curvas"]:
        for point in curve.puntos:
            rows.append(
                {
                    "Serie": fam["serie_display"],
                    "Marca": fam["marca"],
                    "Modelo": fam["modelo"],
                    "RPM": fam["rpm"],
                    "Polos": fam["polos"],
                    "Diametro_mm": curve.diam,
                    "Q_m3/h": point.get("Q", np.nan),
                    "H_m": point.get("H", np.nan),
                    "pot_kw": point.get("P", np.nan),
                    "h_%": point.get("eta", np.nan),
                    "NPSH_m": point.get("NPSH", np.nan),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(by=["Diametro_mm", "Q_m3/h"]).reset_index(drop=True)


def render_curve_data_download(fam: Dict, key: str) -> None:
    curve_data = family_curve_raw_data_df(fam)
    if curve_data.empty:
        return

    csv_data = curve_data.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    file_name = f"datos_curva_{sanitize_filename(fam['serie_display'])}_{sanitize_filename(fam['modelo'])}.csv"

    st.download_button(
        label="Descargar datos de la curva del modelo",
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
        key=key,
        use_container_width=True,
    )


def selected_diameter_curve_data_df(
    fam: Dict,
    curve_obj,
    selected_diam: Optional[float],
    density: float = 1000.0,
    visco_cf: float = 1.0,
    n_points: int = 420,
) -> pd.DataFrame:
    if curve_obj is None:
        return pd.DataFrame()

    q_min = float(getattr(curve_obj, "q_min", 0.0))
    q_max = float(getattr(curve_obj, "q_max", q_min))

    if q_max <= q_min:
        q_values = np.array([q_min], dtype=float)
    else:
        q_values = np.linspace(q_min, q_max, n_points)

    if selected_diam is not None and not pd.isna(selected_diam):
        diam = float(selected_diam)
    else:
        diam = float(getattr(curve_obj, "diam", np.nan))

    rows = []
    for q in q_values:
        q_float = float(q)
        rows.append(
            {
                "Serie": fam["serie_display"],
                "Marca": fam["marca"],
                "Modelo": fam["modelo"],
                "RPM": fam["rpm"],
                "Polos": fam["polos"],
                "Diametro_mm": diam,
                "Q_m3/h": q_float,
                "H_m": curve_obj.get_h(q_float),
                "pot_kw": curve_obj.get_power(q_float, density=density, viscosity_cf=visco_cf),
                "h_%": curve_obj.get_eta(q_float) * visco_cf,
                "NPSH_m": curve_obj.get_npshr(q_float),
            }
        )

    return pd.DataFrame(rows)


def render_selected_diameter_data_download(
    fam: Dict,
    curve_obj,
    selected_diam: Optional[float],
    key: str,
    density: float = 1000.0,
    visco_cf: float = 1.0,
) -> None:
    curve_data = selected_diameter_curve_data_df(
        fam=fam,
        curve_obj=curve_obj,
        selected_diam=selected_diam,
        density=density,
        visco_cf=visco_cf,
    )
    if curve_data.empty:
        return

    diam_value = curve_data["Diametro_mm"].iloc[0]
    diam_txt = f"{float(diam_value):.2f}".rstrip("0").rstrip(".")
    csv_data = curve_data.to_csv(index=False, sep=";", decimal=".").encode("utf-8-sig")
    file_name = (
        f"datos_curva_{sanitize_filename(fam['serie_display'])}_"
        f"{sanitize_filename(fam['modelo'])}_D{sanitize_filename(diam_txt)}mm.csv"
    )

    st.download_button(
        label="Descargar datos del diámetro seleccionado",
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
        key=key,
        use_container_width=True,
    )


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
        d_req, interp_curve, curve_low, curve_high = find_interpolated_diameter_curve(fam, q_req, h_req)
        if d_req is None or interp_curve is None:
            continue

        q_max_curve = get_curve_q_max(interp_curve)
        if q_max_curve is not None and q_req > q_max_curve:
            continue

        if not (fam["D_min"] <= d_req <= fam["D_max"]):
            continue

        base_curve = interp_curve if isinstance(interp_curve, PumpCurveBase) else curve_high
        if base_curve is not None and base_curve.stable_q_min > q_req:
            continue

        op_q = find_operating_point(
            interp_curve,
            sys_curve,
            q_max=interp_curve.q_max,
        )
        if op_q is None:
            op_q = q_req

        h_op = interp_curve.get_h(op_q)
        eta_op = interp_curve.get_eta(op_q) * visco_cf
        p_kw = interp_curve.get_power(op_q, density=densidad, viscosity_cf=visco_cf)
        npshr = interp_curve.get_npshr(op_q)
        p_max = interp_curve.get_max_power(interp_curve.q_max, density=densidad, viscosity_cf=visco_cf)
        motor_kw = select_motor(p_max)
        npsh_status = npsha >= (npshr + NPSH_MARGIN_M)

        results.append(
            {
                "Serie": fam["serie_display"],
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
                "_trim": interp_curve,
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

    series_options = sorted(list({r["Serie"] for r in results}))
    selected_series = st.multiselect("Serie", series_options, key="hydraulic_series_filter")

    if selected_series:
        filtered_results = [r for r in results if r["Serie"] in selected_series]
    else:
        filtered_results = results

    if not filtered_results:
        st.warning("No hay bombas para la serie seleccionada.")
        return

    st.subheader("Bombas que cumplen con los requisitos")
    df_res = pd.DataFrame(filtered_results).drop(
        columns=["_trim", "_fam", "_sys_curve", "_q_req", "_h_req", "_visco_cf", "_densidad", "Motor IEC (kW)"]
    )

    ordered_cols = [
        "Serie",
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
        f"{r['Serie']} | {r['Modelo']} | {fmt0(r['Polos'])} polos | η={fmt2(r['Eficiencia (%)'])}% | D={fmt0(r['D_Impulsor (mm)'])} mm"
        for r in filtered_results
    ]
    selected_idx = st.selectbox(
        "Selecciona una bomba para revisar curvas",
        range(len(filtered_results)),
        format_func=lambda i: labels[i],
    )

    selected = filtered_results[selected_idx]
    fam = selected["_fam"]
    trim_curve = selected["_trim"]

    st.markdown(f"#### {selected['Modelo']} · {fmt0(selected['Polos'])} polos")

    detail_cols = st.columns(5)
    detail_cols[0].metric("Serie", selected["Serie"])
    detail_cols[1].metric("Modelo", selected["Modelo"])
    detail_cols[2].metric("Diámetro aproximado", f"{fmt0(selected['D_Impulsor (mm)'])} mm")
    detail_cols[3].metric(
        "Punto operativo",
        f"Q={fmt1(selected['Q Op. (m3/h)'])} | H={fmt1(selected['H Op. (m)'])}"
    )
    detail_cols[4].metric("NPSH", "OK" if selected["Status NPSH"] else "Revisar")

    selected_diam_exact = float(getattr(trim_curve, "diam", selected["D_Impulsor (mm)"]))
    download_cols = st.columns(2)
    with download_cols[0]:
        render_curve_data_download(
            fam=fam,
            key=f"download_hydraulic_{sanitize_filename(selected['Serie'])}_{sanitize_filename(selected['Modelo'])}_{selected_idx}",
        )
    with download_cols[1]:
        render_selected_diameter_data_download(
            fam=fam,
            curve_obj=trim_curve,
            selected_diam=selected_diam_exact,
            density=densidad,
            visco_cf=selected["_visco_cf"],
            key=f"download_hydraulic_diam_{sanitize_filename(selected['Serie'])}_{sanitize_filename(selected['Modelo'])}_{selected_idx}",
        )

    values_at_point = {
        "H": float(selected["H Op. (m)"]),
        "ETA": float(selected["Eficiencia (%)"]),
        "P": float(selected["Potencia (kW)"]),
        "NPSH": float(selected["NPSHr (m)"]),
    }

    render_characteristic_curves_point(
        fam=fam,
        show_all_diameters=show_all_diameters,
        selected_curve_obj=trim_curve,
        selected_real_diam=selected["D_Impulsor (mm)"],
        op_q=selected["Q Op. (m3/h)"],
        q_req=q_req,
        h_req=h_req,
        sys_curve=selected["_sys_curve"],
        density=densidad,
        visco_cf=selected["_visco_cf"],
        values_at_point=values_at_point,
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

        series_options = sorted([x for x in catalog_df["Serie"].dropna().unique().tolist()])
        model_options = sorted([x for x in catalog_df["Modelo"].dropna().unique().tolist()])
        poles_options = sorted([int(x) for x in catalog_df["Polos"].dropna().unique().tolist()])
        dn_options = sorted([int(x) for x in catalog_df["Descarga DN"].dropna().unique().tolist()])

        with st.expander("Serie", expanded=True):
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
        filtered = filtered[filtered["Serie"].isin(selected_series)]
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

        table_display = pd.DataFrame(
            {
                "Serie": filtered["Serie"],
                "Modelo": filtered["Modelo"],
                "Polos": filtered["Polos"].apply(fmt0),
                "Impulsor actual [mm]": filtered["D máx. (mm)"].apply(fmt0),
                "Descarga DN": filtered["Descarga DN"].apply(fmt0),
                "RPM": filtered["RPM"].apply(fmt0),
            }
        )

        event = st.dataframe(
            table_display,
            use_container_width=True,
            hide_index=True,
            key="manual_interactive_table",
            on_select="rerun",
            selection_mode="single-row",
        )

        try:
            selected_rows = event.selection["rows"] if event and event.selection else []
        except Exception:
            selected_rows = []

        if selected_rows:
            st.session_state.manual_selected_table_row = int(selected_rows[0])

        selection_index = min(st.session_state.manual_selected_table_row, len(filtered) - 1)

        st.markdown("---")

        show_all_diameters = st.radio(
            "Diámetros a mostrar",
            ["Todos los diámetros disponibles", "Solo un diámetro"],
            index=0,
            horizontal=True,
        ) == "Todos los diámetros disponibles"

        selected_row = filtered.iloc[selection_index]
        fam = selected_row["_fam"]

        selected_curve_obj = None
        selected_real_diam = None

        if not show_all_diameters:
            default_diam = int(round(selected_row["D máx. (mm)"]))
            selected_real_diam = st.number_input(
                "Diámetro a mostrar (mm)",
                min_value=float(int(round(selected_row["D mín. (mm)"]))),
                max_value=float(int(round(selected_row["D máx. (mm)"]))),
                value=float(default_diam),
                step=1.0,
                format="%.0f",
            )
            selected_curve_obj = resolve_manual_curve(fam, float(selected_real_diam))
        else:
            selected_real_diam = None
            selected_curve_obj = None

        st.markdown(f"#### {selected_row['Modelo']} · {fmt0(selected_row['Polos'])} polos")

        kpi_cols = st.columns(6)
        kpi_cols[0].metric("Serie", str(selected_row["Serie"]))
        kpi_cols[1].metric("Modelo", str(selected_row["Modelo"]))
        kpi_cols[2].metric("Polos", fmt0(selected_row["Polos"]))
        kpi_cols[3].metric("RPM", fmt0(selected_row["RPM"]))
        kpi_cols[4].metric("D mín. (mm)", fmt0(selected_row["D mín. (mm)"]))
        kpi_cols[5].metric("D máx. (mm)", fmt0(selected_row["D máx. (mm)"]))

        download_cols = st.columns(2)
        with download_cols[0]:
            render_curve_data_download(
                fam=fam,
                key=f"download_manual_{sanitize_filename(str(selected_row['Serie']))}_{sanitize_filename(str(selected_row['Modelo']))}_{selection_index}",
            )
        if selected_curve_obj is not None:
            with download_cols[1]:
                render_selected_diameter_data_download(
                    fam=fam,
                    curve_obj=selected_curve_obj,
                    selected_diam=selected_real_diam,
                    density=1000.0,
                    visco_cf=1.0,
                    key=f"download_manual_diam_{sanitize_filename(str(selected_row['Serie']))}_{sanitize_filename(str(selected_row['Modelo']))}_{selection_index}",
                )

        render_manual_interactive_curves(
            fam=fam,
            show_all_diameters=show_all_diameters,
            selected_curve_obj=selected_curve_obj,
            selected_real_diam=selected_real_diam,
            density=1000.0,
            visco_cf=1.0,
            model_key=f"{selected_row['Modelo']}_{fmt0(selected_row['Polos'])}",
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
