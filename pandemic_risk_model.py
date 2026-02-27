import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AeroSafe: Indoor Pandemic Risk Modeller",
    page_icon="🦠",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.6rem;
    }
    .risk-low    { color: #2ecc71; font-size: 1.6rem; font-weight: 700; }
    .risk-medium { color: #f39c12; font-size: 1.6rem; font-weight: 700; }
    .risk-high   { color: #e74c3c; font-size: 1.6rem; font-weight: 700; }
    .section-header { font-size: 1.1rem; font-weight: 600; margin-top: 1rem; color: #aab; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Core simulation functions
# ─────────────────────────────────────────────

VENTILATION_MULTIPLIERS = {"Low": 1.0, "Medium": 0.6, "High": 0.3}
BASE_BETA = 0.3          # base daily transmission rate (well-mixed room)
GAMMA = 1 / 7            # recovery rate (7-day infectious period)
DAYS = 30


def mask_multiplier(compliance_pct: float) -> float:
    """Masks reduce transmission roughly proportional to compliance × effectiveness (~60%)."""
    effectiveness = 0.60
    reduction = (compliance_pct / 100) * effectiveness
    return max(1.0 - reduction, 0.05)


def density_multiplier(density: float) -> float:
    """
    density = persons / m²
    Reference threshold: 1 person / 4 m² = 0.25 p/m².
    Above reference → increases transmission; below → decreases.
    """
    reference = 0.25
    return max(density / reference, 0.1)


def compute_beta(ventilation: str, mask_pct: float, density: float) -> float:
    vent_m = VENTILATION_MULTIPLIERS[ventilation]
    mask_m = mask_multiplier(mask_pct)
    dens_m = density_multiplier(density)
    return BASE_BETA * vent_m * mask_m * dens_m


def run_sir(N: int, I0: int, beta: float, days: int = DAYS):
    S = np.zeros(days + 1)
    I = np.zeros(days + 1)
    R = np.zeros(days + 1)

    S[0] = max(N - I0, 0)
    I[0] = min(I0, N)
    R[0] = 0.0

    for t in range(days):
        new_infections = beta * S[t] * I[t] / N
        new_recoveries = GAMMA * I[t]
        new_infections = min(new_infections, S[t])
        new_recoveries = min(new_recoveries, I[t])

        S[t + 1] = max(S[t] - new_infections, 0)
        I[t + 1] = max(I[t] + new_infections - new_recoveries, 0)
        R[t + 1] = R[t] + new_recoveries

    return S, I, R


def classify_risk(peak_pct: float) -> tuple[str, str]:
    if peak_pct < 20:
        return "LOW", "risk-low"
    elif peak_pct < 50:
        return "MODERATE", "risk-medium"
    else:
        return "HIGH", "risk-high"


def find_safe_occupancy(I0: int, beta_per_person_density: dict, ventilation: str, mask_pct: float,
                         length: float, width: float, N_start: int) -> int:
    """Iteratively reduce N until peak infection < 20%."""
    floor_area = length * width
    for N_test in range(N_start, 0, -1):
        density = N_test / floor_area
        beta_test = compute_beta(ventilation, mask_pct, density)
        _, I_test, _ = run_sir(N_test, min(I0, N_test), beta_test)
        peak = I_test.max() / N_test * 100
        if peak < 20:
            return N_test
    return 1


# ─────────────────────────────────────────────
# UI – Sidebar inputs
# ─────────────────────────────────────────────
st.sidebar.markdown("<div style='margin-top: -3rem;'>", unsafe_allow_html=True)
st.sidebar.title("Room & Scenario Parameters")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown('<p class="section-header">Room Dimensions</p>', unsafe_allow_html=True)
length = st.sidebar.number_input("Length (m)", min_value=1.0, max_value=500.0, value=10.0, step=0.5)
width  = st.sidebar.number_input("Width (m)",  min_value=1.0, max_value=500.0, value=8.0,  step=0.5)
height = st.sidebar.number_input("Height (m)", min_value=1.5, max_value=20.0,  value=3.0,  step=0.1)

st.sidebar.markdown('<p class="section-header">Occupancy</p>', unsafe_allow_html=True)
total_occupants = st.sidebar.number_input("Total Occupants", min_value=2, max_value=10000, value=30, step=1)
initial_infected = st.sidebar.number_input("Initially Infected", min_value=1, max_value=total_occupants, value=1, step=1)

st.sidebar.markdown('<p class="section-header">Intervention Parameters</p>', unsafe_allow_html=True)
ventilation   = st.sidebar.selectbox("Ventilation Level", ["Low", "Medium", "High"], index=1)
mask_pct      = st.sidebar.slider("Mask Compliance (%)", 0, 100, 50)
exposure_hrs  = st.sidebar.slider("Daily Exposure Duration (hrs)", 1, 24, 8)

st.sidebar.markdown("---")
st.sidebar.caption("Comparison Scenario (optional)")
compare_on   = st.sidebar.checkbox("Show comparison curve")
if compare_on:
    vent2     = st.sidebar.selectbox("Ventilation (B)", ["Low", "Medium", "High"], index=2, key="v2")
    mask2     = st.sidebar.slider("Mask Compliance % (B)", 0, 100, 80, key="m2")

# ─────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────
errors = []
if initial_infected > total_occupants:
    errors.append("Initially infected cannot exceed total occupants.")
if length <= 0 or width <= 0 or height <= 0:
    errors.append("Room dimensions must be positive.")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

# ─────────────────────────────────────────────
# Derived values & simulation
# ─────────────────────────────────────────────
floor_area       = length * width
volume           = floor_area * height
density          = total_occupants / floor_area
exposure_factor  = exposure_hrs / 8  # scale beta relative to 8-hr reference workday

beta  = compute_beta(ventilation, mask_pct, density) * exposure_factor
S, I, R = run_sir(total_occupants, initial_infected, beta)

peak_infected     = I.max()
peak_pct          = peak_infected / total_occupants * 100
time_to_peak      = int(I.argmax())
risk_label, risk_css = classify_risk(peak_pct)

safe_occupancy = find_safe_occupancy(
    initial_infected, {}, ventilation, mask_pct, length, width, total_occupants
)

days_axis = np.arange(DAYS + 1)

# Comparison scenario
if compare_on:
    beta2 = compute_beta(vent2, mask2, density) * exposure_factor
    _, I2, _ = run_sir(total_occupants, initial_infected, beta2)

# ─────────────────────────────────────────────
# Main page layout
# ─────────────────────────────────────────────
st.title("🦠 Indoor Pandemic Risk Modeller")
st.markdown("Real-time SIR simulation to inform ventilation standards, occupancy planning and intervention strategies.")

# ── KPI row ──────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Room Volume", f"{volume:,.1f} m³")
col2.metric("Occupancy Density", f"{density:.2f} p/m²")
col3.metric("Effective β", f"{beta:.4f}")
col4.metric("Peak Infected", f"{peak_pct:.1f}% (day {time_to_peak})")
col5.metric("Safe Occupancy", f"{safe_occupancy} people")

# ── Risk badge ───────────────────────────────
st.markdown("---")
risk_col, info_col = st.columns([1, 3])
with risk_col:
    st.markdown(f"**Risk Classification**")
    st.markdown(f'<span class="{risk_css}">{risk_label}</span>', unsafe_allow_html=True)
with info_col:
    if risk_label == "LOW":
        st.success(f"Peak infection stays below 20%. Current settings are within safe operating parameters.")
    elif risk_label == "MODERATE":
        st.warning(f"Peak infection reaches {peak_pct:.1f}%. Consider improving ventilation, increasing mask compliance, or reducing occupancy to {safe_occupancy}.")
    else:
        st.error(f"Peak infection reaches {peak_pct:.1f}%. This scenario poses HIGH risk. Recommended maximum safe occupancy: **{safe_occupancy} people**.")

st.markdown("---")

# ── Infection curve ──────────────────────────
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=days_axis, y=S, name="Susceptible",
    line=dict(color="#3498db", width=2), fill="tozeroy", fillcolor="rgba(52,152,219,0.08)"
))
fig.add_trace(go.Scatter(
    x=days_axis, y=I, name="Infected (Scenario A)",
    line=dict(color="#e74c3c", width=2.5), fill="tozeroy", fillcolor="rgba(231,76,60,0.12)"
))
fig.add_trace(go.Scatter(
    x=days_axis, y=R, name="Recovered",
    line=dict(color="#2ecc71", width=2), fill="tozeroy", fillcolor="rgba(46,204,113,0.08)"
))

if compare_on:
    fig.add_trace(go.Scatter(
        x=days_axis, y=I2, name=f"Infected (Scenario B: {vent2} vent, {mask2}% mask)",
        line=dict(color="#f39c12", width=2.5, dash="dash")
    ))

fig.add_vline(x=time_to_peak, line_dash="dot", line_color="rgba(231,76,60,0.5)",
              annotation_text=f"Peak day {time_to_peak}", annotation_position="top right")

fig.add_hline(y=total_occupants * 0.20, line_dash="dash", line_color="rgba(243,156,18,0.6)",
              annotation_text="20% threshold", annotation_position="bottom right")

fig.update_layout(
    title="SIR Infection Curve – 30-Day Projection",
    xaxis_title="Day",
    yaxis_title="Number of People",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font=dict(color="#ccc"),
    margin=dict(l=40, r=40, t=60, b=40),
)
st.plotly_chart(fig, use_container_width=True)

# ── Detailed breakdown ───────────────────────
st.markdown("---")
detail_col1, detail_col2 = st.columns(2)

with detail_col1:
    st.subheader("Simulation Parameters")
    params = {
        "Room Dimensions": f"{length}m × {width}m × {height}m",
        "Floor Area": f"{floor_area:.1f} m²",
        "Room Volume": f"{volume:.1f} m³",
        "Total Occupants": total_occupants,
        "Initially Infected": initial_infected,
        "Ventilation": ventilation,
        "Mask Compliance": f"{mask_pct}%",
        "Exposure Duration": f"{exposure_hrs} hrs/day",
        "Occupancy Density": f"{density:.3f} persons/m²",
    }
    for k, v in params.items():
        st.markdown(f"**{k}:** {v}")

with detail_col2:
    st.subheader("Epidemiological Results")
    results = {
        "Base β (transmission rate)": f"{BASE_BETA}",
        "Ventilation Multiplier": f"{VENTILATION_MULTIPLIERS[ventilation]}",
        "Mask Multiplier": f"{mask_multiplier(mask_pct):.3f}",
        "Density Multiplier": f"{density_multiplier(density):.3f}",
        "Exposure Multiplier": f"{exposure_factor:.3f}",
        "Effective β": f"{beta:.5f}",
        "Recovery Rate γ": f"{GAMMA:.4f} (7-day period)",
        "Basic Reproduction R₀": f"{beta / GAMMA:.2f}",
        "Peak Infected": f"{int(peak_infected)} people ({peak_pct:.1f}%)",
        "Day of Peak": f"Day {time_to_peak}",
        "Risk Classification": risk_label,
        "Max Safe Occupancy": f"{safe_occupancy} people",
    }
    for k, v in results.items():
        st.markdown(f"**{k}:** {v}")

# ── Sensitivity table ────────────────────────
st.markdown("---")
st.subheader("Scenario Sensitivity – Ventilation vs Mask Compliance")

vent_levels = ["Low", "Medium", "High"]
mask_levels = [0, 25, 50, 75, 100]
table_data  = []

for v in vent_levels:
    row = []
    for m in mask_levels:
        b = compute_beta(v, m, density) * exposure_factor
        _, I_t, _ = run_sir(total_occupants, initial_infected, b)
        pk = I_t.max() / total_occupants * 100
        row.append(f"{pk:.1f}%")
    table_data.append(row)

import pandas as pd
df = pd.DataFrame(table_data, index=vent_levels,
                  columns=[f"{m}% masks" for m in mask_levels])
df.index.name = "Ventilation"

def colour_cell(val):
    pct = float(val.replace("%", ""))
    if pct < 20:
        return "background-color: #1a4731; color: #2ecc71"
    elif pct < 50:
        return "background-color: #3d2b05; color: #f39c12"
    else:
        return "background-color: #3d0a0a; color: #e74c3c"

st.dataframe(df.style.applymap(colour_cell), use_container_width=True)

st.caption(
    "Model assumes well-mixed air (WETA), discrete-time SIR, 7-day infectious period. "
    "β is adjusted by ventilation, mask compliance, occupancy density, and daily exposure duration. "
    "This tool is for educational and planning purposes — not a substitute for professional public health advice."
)
