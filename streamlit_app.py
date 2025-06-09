import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Digital Liver v2 – Bioactivated DILI Simulator", layout="wide")
st.title("🧬 Digital Liver v2: CYP450-Driven Drug-Induced Liver Injury Simulation")

st.markdown("""
This enhanced model simulates:
- **CYP450 bioactivation**
- **Mitochondrial dysfunction**
- **ROS, GSH depletion**
- **ALT/AST elevation**
- **Apoptosis, Necrosis, Fibrosis**
- **Idiosyncratic DILI risk**
""")

# Liver model function
def liver_dili_model(y, t, amp, dose, idio=0):
    drug, tox_met, gsh, ros, alt, ast, mito, chol, apop, necro, fib = y
    k_cyp = 0.04
    k_bio = 0.03
    k_gsh = 0.025
    k_ros = 0.03 * amp
    k_mito = 0.015 * amp
    k_clear = 0.01
    k_apop = 0.012 * amp
    k_necro = 0.008 * amp
    k_chol = 0.006 * amp
    k_fib = 0.005 * amp
    idiosync = 1 + np.sin(t/5) * idio  # fluctuating sensitivity

    d_drug = -k_cyp * drug
    d_tox_met = k_cyp * drug * k_bio - k_gsh * min(tox_met, gsh)
    d_gsh = -k_gsh * min(tox_met, gsh)
    d_ros = k_ros * tox_met - k_clear * ros
    d_alt = 0.012 * ros
    d_ast = 0.012 * ros
    d_mito = k_mito * (ros + tox_met)
    d_chol = k_chol * tox_met * idiosync
    d_apop = k_apop * ros + 0.01 * mito
    d_necro = k_necro * ros * idiosync
    d_fib = k_fib * (apop + necro)

    return [d_drug, d_tox_met, d_gsh, d_ros, d_alt, d_ast, d_mito, d_chol, d_apop, d_necro, d_fib]

# Sidebar Inputs
st.sidebar.header("Drug Input")
smiles_list = st.sidebar.text_area("Enter SMILES (one per line)", "CC(=O)NC1=CC=C(C=C1)O").splitlines()
dose = st.sidebar.slider("Dose Level", 0.1, 3.0, 1.0, 0.1)
duration = st.sidebar.slider("Simulation Time (h)", 12, 96, 48)
idiosync_on = st.sidebar.checkbox("Include Idiosyncratic Risk", value=True)

# Run simulation
if st.sidebar.button("Run Simulation"):
    for idx, smiles in enumerate(smiles_list):
        amp = 2.5 if any(x in smiles.lower() for x in ["cl", "br", "no2", "epoxide"]) else 1.0
        idioval = 1.0 if idiosync_on else 0.0

        y0 = [dose, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        t = np.linspace(0, duration, 300)
        sol = odeint(liver_dili_model, y0, t, args=(amp, dose, idioval))

        labels = ["Drug", "Toxic Metabolite", "GSH", "ROS", "ALT", "AST", "Mito Stress",
                  "Cholestasis", "Apoptosis", "Necrosis", "Fibrosis"]

        st.subheader(f"🧪 Compound {idx+1}: `{smiles}`")
        fig, ax = plt.subplots(figsize=(11, 5))
        for i in range(len(labels)):
            ax.plot(t, sol[:, i], label=labels[i])
        ax.set_title("Liver Toxicity Simulation")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Level")
        ax.legend()
        st.pyplot(fig)

        final = sol[-1]
        score = (
            0.18 * final[3] + 0.14 * final[4] + 0.12 * final[6] +
            0.12 * final[8] + 0.12 * final[9] + 0.18 * final[10] + 0.14 * final[7]
        )
        risk = "LOW" if score < 0.6 else "MODERATE" if score < 1.5 else "HIGH"

        st.markdown(f"### 🧬 DILI Score: `{score:.2f}` → **Risk: {risk}`**")

        # PDF generation
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Digital Liver v2: DILI Simulation Report", ln=1)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"SMILES: {smiles}", ln=1)
            pdf.cell(0, 10, f"Dose: {dose} | Time: {duration}h", ln=1)
            pdf.cell(0, 10, f"Toxicity Amplifier: {amp}", ln=1)
            pdf.cell(0, 10, f"DILI Score: {score:.2f} | Risk: {risk}", ln=1)
            pdf.output(tmp.name)

            with open(tmp.name, "rb") as file:
                st.download_button(
                    "📄 Download PDF Report",
                    data=file,
                    file_name=f"dili_report_{idx+1}.pdf",
                    mime="application/pdf"
                )
            os.unlink(tmp.name)
