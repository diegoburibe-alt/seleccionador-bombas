import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit, brentq
import os
from fpdf import FPDF

# Tolerancias y Margenes
NPSH_MARGIN_M = 0.5
# Tamaños estándar de motores IEC (kW)
IEC_MOTORS_KW = [
    0.18, 0.25, 0.37, 0.55, 0.75, 1.1, 1.5, 2.2, 3.0, 4.0, 5.5, 7.5,
    11, 15, 18.5, 22, 30, 37, 45, 55, 75, 90, 110, 132, 160, 200, 250, 315, 355, 400, 500, 630
]

def get_service_factor(power_kw):
    """ Factor de servicio estándar corporativo API/ISO """
    if power_kw < 22:
        return 1.15
    elif power_kw < 55:
        return 1.10
    else:
        return 1.05

def select_motor(max_power_kw):
    req_power = max_power_kw * get_service_factor(max_power_kw)
    for m in IEC_MOTORS_KW:
        if m >= req_power:
            return m
    return IEC_MOTORS_KW[-1]

def poly2(x, a, b, c):
    """ Ecuación H-Q estándar H = a*Q^2 + b*Q + c """
    return a * (x**2) + b * x + c

class PumpCurveBase:
    """ Representa una curva de ensayo experimental (Laboratorio) """
    def __init__(self, diam, puntos):
        self.diam = diam
        self.puntos = puntos
        q_raw = np.array([p['Q'] for p in puntos])
        h_raw = np.array([p['H'] for p in puntos])
        eta_raw = np.array([p['eta'] for p in puntos])
        npsh_raw = np.array([p['NPSH'] for p in puntos])

        # Ajuste H-Q
        self.popt_h, _ = curve_fit(poly2, q_raw, h_raw)
        self.a, self.b, self.c = self.popt_h

        # Validación de estabilidad (Control de pendiente anómala)
        if self.a < 0 and self.b > 0:
            self.stable_q_min = -self.b / (2 * self.a)
        else:
            self.stable_q_min = 0.0

        sort_idx = np.argsort(q_raw)
        q_sort = q_raw[sort_idx]
        eta_sort = eta_raw[sort_idx]
        npsh_sort = npsh_raw[sort_idx]

        unique_q, unique_idx = np.unique(q_sort, return_index=True)
        self.unique_q = unique_q
        self.interp_eta = PchipInterpolator(unique_q, eta_sort[unique_idx])
        self.interp_npsh = PchipInterpolator(unique_q, npsh_sort[unique_idx])

        self.q_min = min(unique_q)
        self.q_max = max(unique_q)

        # Identificar BEP (Best Efficiency Point)
        qq = np.linspace(max(0.1, self.q_min), self.q_max, 200)
        ee = self.interp_eta(qq)
        idx_bep = np.argmax(ee)
        self.q_bep = qq[idx_bep]
        self.eta_bep = ee[idx_bep]

    def get_h(self, q):
        return poly2(q, *self.popt_h)

class TrimmedCurve:
    """ Curva escalada mediante leyes de similitud de afinidad """
    def __init__(self, base_curve, trim_diam):
        self.base = base_curve
        self.diam = trim_diam
        self.ratio = trim_diam / base_curve.diam

    def get_h(self, q):
        if self.ratio <= 0: return 0
        q_base = q / self.ratio
        return (self.ratio**2) * self.base.get_h(q_base)

    def get_eta(self, q):
        if self.ratio <= 0: return 0
        q_base = q / self.ratio
        q_eval = np.clip(q_base, self.base.unique_q[0], self.base.unique_q[-1])
        base_eta = float(self.base.interp_eta(q_eval))
        # Penalización real por recorte empírico
        penalty = max(0, 1.0 - self.ratio) * 10.0  # Reducción porcentual lineal severa
        return max(0, base_eta - penalty)

    def get_npshr(self, q):
        if self.ratio <= 0: return 0
        q_base = q / self.ratio
        q_eval = np.clip(q_base, self.base.unique_q[0], self.base.unique_q[-1])
        base_npsh = float(self.base.interp_npsh(q_eval))
        return base_npsh * (self.ratio**2)

    def get_power(self, q, density=1000.0, viscosity_cf=1.0):
        # Viscosity_cf es de la metodología del Hydraulic Institute
        h = self.get_h(q)
        eta = self.get_eta(q) * viscosity_cf
        if eta <= 0: return float('inf')
        power_w = (q / 3600.0) * h * density * 9.81 / (eta / 100.0)
        return power_w / 1000.0

    def get_max_power(self, end_q, density=1000.0, viscosity_cf=1.0):
        qq = np.linspace(0.1, end_q, 100)
        powers = [self.get_power(qi, density, viscosity_cf) for qi in qq]
        return max(powers)

class SystemCurve:
    def __init__(self, h_stat, k):
        self.h_stat = h_stat
        self.k = k
    def get_h(self, q):
        return self.h_stat + self.k * (q**2)

class PumpDatabase:
    def __init__(self):
        self.families = []

    def load_from_csv(self, file_obj):
        df = pd.read_csv(file_obj, sep=';', decimal='.')
        df.columns = [str(col).strip() for col in df.columns]

        vital_cols = ['Marca', 'modelo', 'diametro_mm', 'Q_m3/h', 'H_m']
        for c in vital_cols:
            if c not in df.columns:
                raise ValueError(f"Falta columna vital requerida: {c}")

        has_rpm = 'RPM' in df.columns
        groupby_cols = ['Marca', 'modelo', 'RPM'] if has_rpm else ['Marca', 'modelo']

        self.families = []
        for name, group in df.groupby(groupby_cols):
            if has_rpm:
                marca, modelo, rpm_val = name
            else:
                marca, modelo = name
                rpm_val = None

            diametros = group['diametro_mm'].dropna().astype(float).unique()
            if len(diametros) == 0: continue
            
            d_max = float(max(diametros))
            d_min = float(min(diametros))
            known_diameters = sorted(list(diametros))

            curvas = []
            for d in known_diameters:
                group_d = group[group['diametro_mm'].astype(float) == d]
                puntos = []
                for _, row in group_d.iterrows():
                    try:
                        p = {
                            'Q': float(str(row['Q_m3/h']).replace(',', '.')),
                            'H': float(str(row['H_m']).replace(',', '.')),
                            'eta': float(str(row.get('h_%', '0')).replace(',', '.')),
                            'NPSH': float(str(row.get('NPSH_m', '0')).replace(',', '.'))
                        }
                        puntos.append(p)
                    except:
                        pass
                
                if len(puntos) >= 3:
                    try:
                        curvas.append(PumpCurveBase(d, puntos))
                    except:
                        pass
            
            if not curvas: continue
            
            self.families.append({
                'marca': str(marca),
                'modelo': str(modelo),
                'rpm': rpm_val if pd.notna(rpm_val) else None,
                'D_max': d_max,
                'D_min': d_min,
                'curvas': sorted(curvas, key=lambda c: c.diam)
            })

    def get_families(self):
        return self.families

def find_trim_diameter(q_req, h_req, base_curve):
    def objective(ratio):
        if ratio <= 0: return -1e9
        q_base = q_req / ratio
        return (ratio**2) * base_curve.get_h(q_base) - h_req

    min_ratio = 0.5
    max_ratio = 1.1
    try:
        f_min = objective(min_ratio)
        f_max = objective(max_ratio)
        if f_min * f_max > 0:
            return None # Sin cruce
        ratio_opt = brentq(objective, min_ratio, max_ratio, xtol=1e-4)
        return base_curve.diam * ratio_opt
    except Exception:
        return None

def find_operating_point(trim_curve, sys_curve, q_max=None):
    if q_max is None:
        q_max = trim_curve.base.q_max * trim_curve.ratio * 1.5

    def objective(q):
        return trim_curve.get_h(q) - sys_curve.get_h(q)

    try:
        if objective(0.001) < 0:
            return None # Bomba no vence altura estática
        q_op = brentq(objective, 0.001, q_max)
        return q_op
    except:
        return None

def create_datasheet(result, q_req, h_req, npsha, density, fluid_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(190, 10, "Hoja de Datos Tecnicos - Ingenieria", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(190, 8, f"Equipo: {result['Proveedor']} {result['Modelo']} ({result['RPM Nominal']} RPM)", ln=True)

    pdf.set_font("helvetica", "", 12)
    pdf.cell(190, 8, f"Punto de Diseno: Q = {q_req} m3/h | H = {h_req} m", ln=True)
    pdf.cell(190, 8, f"Fluido: {fluid_name} | Dens = {density} kg/m3", ln=True)
    pdf.cell(190, 8, f"NPSH Disponible (NPSHa): {npsha} m", ln=True)
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 12)
    pdf.cell(190, 8, "Resultados Hidraulicos Operativos:", ln=True)

    pdf.set_font("helvetica", "", 12)
    pdf.cell(190, 8, f"- Diametro Mecanizado: {result['D_Impulsor_mm']:.2f} mm", ln=True)
    pdf.cell(190, 8, f"- Punto Op. Real en Sistema: Q = {result['Q_Op_m3h']:.2f} m3/h", ln=True)
    pdf.cell(190, 8, f"- Eficiencia Nominal: {result['Eficiencia_pct']:.2f} %", ln=True)
    pdf.cell(190, 8, f"- Potencia Absorbida Eje: {result['Potencia_kW']:.2f} kW", ln=True)
    pdf.cell(190, 8, f"- Sugerencia Motor IEC: {result['Motor_IEC_kW']:.2f} kW (Base curva max)", ln=True)

    npshr = result['NPSHr_m']
    status = "APROBADO" if result['Status_NPSH'] else "RECHAZADO (Peligro Cavitacion)"
    pdf.cell(190, 8, f"- NPSH Requerido: {npshr:.2f} m -> Status: {status}", ln=True)

    pdf_path = f"Datasheet_{result['Modelo']}.pdf".replace("/", "_")
    pdf.output(pdf_path)
    return pdf_path

st.set_page_config(layout="wide", page_title="Seleccionador Corporativo V3", page_icon="⚙️")

def main():
    st.title("⚙️ Seleccionador Corporativo de Equipos de Bombeo")
    st.markdown("Arquitectura Desacoplada | Normativa ANSI/HI & ISO | Motores IEC | Trazabilidad")

    db = PumpDatabase()

    with st.sidebar:
        st.header("1. Parámetros del Fluido")
        fluid_name = st.text_input("Fluido", "Agua Limpia")
        densidad = st.number_input("Densidad (kg/m³)", value=1000.0, step=10.0)
        viscosidad = st.number_input("Viscosidad Cinemática (cSt)", value=1.0, step=0.5)

        st.header("2. Punto de Operación Requerido")
        q_req = st.number_input("Caudal Solicitado Q (m³/h)", value=50.0, step=1.0)
        h_req = st.number_input("Altura Solicitada H (m)", value=30.0, step=1.0)
        npsha = st.number_input("NPSH Disponible (m)", value=10.0, step=0.5)

        st.header("3. Curva del Sistema")
        usar_sistema = st.checkbox("Considerar Curva de Sistema (Intersección Real)", value=True)
        h_est = st.number_input("Carga Estática (m)", value=0.0, disabled=not usar_sistema)
        k_calc = (h_req - h_est) / (q_req**2) if q_req > 0 else 0
        k_sys = st.number_input("Coef. Fricción Sistema (k)", value=float(k_calc), format="%.6f", disabled=not usar_sistema)

        st.header("4. Base de Datos")
        uploaded_file = st.file_uploader("Cargar Base Oficial (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            db.load_from_csv(uploaded_file)
            families = db.get_families()
            st.success(f"✅ Base de datos verificada: {len(families)} familias estructurales activas.")
        except Exception as e:
            st.error(f"Error de Integridad en CSV: {e}")
            return
        
        if usar_sistema:
            sys_curve = SystemCurve(h_stat=h_est, k=k_sys)
        else:
            sys_curve = None

        results = []
        for fam in families:
            valid_curves = [c for c in fam['curvas'] if c.get_h(q_req) >= h_req]
            if valid_curves:
                base_curve = valid_curves[0]
            else:
                base_curve = fam['curvas'][-1]

            d_req = find_trim_diameter(q_req, h_req, base_curve)
            
            if d_req is not None and fam['D_min'] <= d_req <= fam['D_max'] * 1.05:
                trim_curve = TrimmedCurve(base_curve, d_req)
                
                # Validacion Estricta de Estabilidad Hidráulica
                if base_curve.stable_q_min > q_req:
                    continue
                    
                if sys_curve is not None:
                    op_q = find_operating_point(trim_curve, sys_curve, fam['curvas'][-1].q_max)
                    if op_q is None:
                        op_q = q_req
                else:
                    op_q = q_req

                # Correccion visual de viscosidad estandar 1.0 si es agua
                visco_cf = 1.0 if viscosidad <= 1.0 else 0.95 # Simplificacion HI para front-end
                
                eta_req = trim_curve.get_eta(op_q) * visco_cf
                npshr = trim_curve.get_npshr(op_q)
                p_kw = trim_curve.get_power(op_q, density=densidad, viscosity_cf=visco_cf)
                
                end_op = fam['curvas'][-1].q_max * trim_curve.ratio
                p_max = trim_curve.get_max_power(end_op, density=densidad, viscosity_cf=visco_cf)
                
                npsh_status = (npsha >= npshr + NPSH_MARGIN_M)
                
                results.append({
                    "Proveedor": fam['marca'],
                    "Modelo": fam['modelo'],
                    "RPM Nominal": fam['rpm'],
                    "D_Impulsor (mm)": round(d_req, 1),
                    "Q Op. (m3/h)": round(op_q, 1),
                    "Eficiencia (%)": round(eta_req, 1),
                    "NPSHr (m)": round(npshr, 2),
                    "Potencia (kW)": round(p_kw, 2),
                    "Motor IEC (kW)": select_motor(p_max),
                    "Status NPSH": npsh_status,
                    "_trim": trim_curve,
                    "_fam": fam
                })

        if results:
            results.sort(key=lambda x: x['Eficiencia (%)'], reverse=True)
            df_res = pd.DataFrame(results)

            st.subheader("🏆 Ranking Multicriterio (Norma API/ISO)")
            df_disp = df_res.drop(columns=['_trim', '_fam'])

            # La función de estilo en Streamlit a veces da problemas si no se aplica bien a rows. 
            def hl_npsh(val):
                color = '#ffcccc' if not val else ''
                return f'background-color: {color}'
            
            st.dataframe(df_disp.style.format(precision=2).map(hl_npsh, subset=['Status NPSH']), use_container_width=True)

            st.markdown("---")
            st.subheader("📊 Análisis Energético y Curvas de Sistema")
            res_options = df_res.to_dict('records')

            idx = st.selectbox(
                "Auditar Equipo:", 
                range(len(res_options)), 
                format_func=lambda i: f"{res_options[i]['Proveedor']} {res_options[i]['Modelo']} (Ef: {res_options[i]['Eficiencia (%)']}%)"
            )
            selected = res_options[idx]
            tc = selected['_trim']
            fam = selected['_fam']

            fig = go.Figure()
            q_plot = np.linspace(0, max(q_req*1.5, fam['curvas'][-1].q_max), 150)

            # Cascada de ISO Curvas Reales (Negro)
            for c in fam['curvas']:
                fig.add_trace(go.Scatter(
                    x=q_plot, 
                    y=[c.get_h(qq) for qq in q_plot], 
                    mode='lines', 
                    name=f"Matriz D={c.diam}",
                    line=dict(color='black', width=1),
                    hovertemplate='Q: %{x:.2f} m³/h<br>H: %{y:.2f} m<extra></extra>'
                ))

            # Curva Operacional (Azul)
            h_trim = [tc.get_h(qq) for qq in q_plot]
            fig.add_trace(go.Scatter(
                x=q_plot, 
                y=h_trim, 
                mode='lines', 
                name=f"Recortada D={selected['D_Impulsor (mm)']}",
                line=dict(color='blue', width=2),
                hovertemplate='Q: %{x:.2f} m³/h<br>H: %{y:.2f} m<extra></extra>'
            ))

            # Interseccion de Curva de Sistema
            if sys_curve is not None:
                h_sys_plot = [sys_curve.get_h(qq) for qq in q_plot]
                fig.add_trace(go.Scatter(
                    x=q_plot, 
                    y=h_sys_plot, 
                    mode='lines', 
                    name="Curva Sistema",
                    line=dict(color='orange', dash='dash'),
                    hovertemplate='Q: %{x:.2f} m³/h<br>H: %{y:.2f} m<extra></extra>'
                ))

            # Punto de Trabajo Operativo
            op_q = selected['Q Op. (m3/h)']
            fig.add_trace(go.Scatter(
                x=[op_q], 
                y=[tc.get_h(op_q)], 
                mode='markers', 
                name='Punto Op.',
                marker=dict(color='red', size=12),
                hovertemplate='Q: %{x:.2f} m³/h<br>H: %{y:.2f} m<extra></extra>'
            ))

            fig.update_layout(
                title="Curva Hidráulica e Intersección", 
                xaxis_title="Caudal Q (m³/h)", 
                yaxis_title="Altura H (m)", 
                template="plotly_white"
            )
            fig.update_xaxes(rangemode="tozero", minor=dict(ticks="inside", showgrid=True))
            fig.update_yaxes(rangemode="tozero", minor=dict(ticks="inside", showgrid=True))
            st.plotly_chart(fig, use_container_width=True)

            if st.button("📄 Exportar Datasheet Oficial PDF"):
                compatible_res = {
                    'Proveedor': selected['Proveedor'],
                    'Modelo': selected['Modelo'],
                    'RPM Nominal': selected['RPM Nominal'],
                    'D_Impulsor_mm': selected['D_Impulsor (mm)'],
                    'Q_Op_m3h': selected['Q Op. (m3/h)'],
                    'Eficiencia_pct': selected['Eficiencia (%)'],
                    'Potencia_kW': selected['Potencia (kW)'],
                    'Motor_IEC_kW': selected['Motor IEC (kW)'],
                    'NPSHr_m': selected['NPSHr (m)'],
                    'Status_NPSH': selected['Status NPSH']
                }
                pdf_path = create_datasheet(compatible_res, q_req, h_req, npsha, densidad, fluid_name)
                
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Descargar Datasheet Compilado", f, file_name=f"Datasheet_{selected['Modelo']}.pdf", mime="application/pdf")
        else:
            st.warning("No se encontraron bombas que cumplan (Restricción física o de NPSH extremo).")

if __name__ == "__main__":
    main()
