import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Digital Liver v3 – Antioxidant-Aware DILI Simulator", layout="wide")
st.title("🧬 Digital Liver v3: Antioxidant-Aware Liver Injury Simulation")

st.markdown("""
This improved model includes:
- CYP450 bioactivation
- ROS and oxidative stress balance
- Mitochondrial damage and apoptosis
- Antioxidant behavior of compounds (e.g., luteolin, quercetin)
- DILI score + redox alert system
""")

def safe_text(text):
    return text.encode('latin-1', 'replace').decode('latin-1')

def liver_dili_model(y, t, amp, dose, is_antioxidant=False):
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

    antioxidant_clearance = 0.015 if is_antioxidant else 0.0

    d_drug = -k_cyp * drug
    d_tox_met = k_cyp * drug * k_bio - k_gsh * min(tox_met, gsh)
    d_gsh = -k_gsh * min(tox_met, gsh)
    d_ros = k_ros * tox_met - (k_clear + antioxidant_clearance) * ros
    d_alt = 0.012 * ros
    d_ast = 0.012 * ros
    d_mito = k_mito * (ros + tox_met)
    d_chol = k_chol * tox_met
    d_apop = k_apop * ros + 0.01 * mito
    d_necro = k_necro * ros
    d_fib = k_fib * (apop + necro)

    return [d_drug, d_tox_met, d_gsh, d_ros, d_alt, d_ast, d_mito, d_chol, d_apop, d_necro, d_fib]

def interpret_toxicity(marker):
    explanations = {
        "ROS": "High oxidative stress was observed, which can trigger widespread cellular damage.",
        "ALT": "Elevated ALT suggests liver cell leakage or membrane injury.",
        "Mito Stress": "Mitochondrial dysfunction can lead to energy depletion and trigger apoptosis.",
        "Apoptosis": "Programmed cell death was a major outcome, often seen in early DILI.",
        "Necrosis": "Cell death through necrosis indicates severe, uncontrolled damage.",
        "Fibrosis": "Chronic damage may result in fibrotic tissue formation.",
        "Cholestasis": "Bile flow disruption was prominent, which can lead to jaundice or hepatic inflammation."
    }
    return explanations.get(marker, "Toxic pathway contribution was observed.")

st.sidebar.header("Drug Input")
smiles_list = st.sidebar.text_area("Enter SMILES (one per line)", "CC(=O)NC1=CC=C(C=C1)O").splitlines()
dose = st.sidebar.slider("Dose Level", 0.1, 3.0, 1.0, 0.1)
duration = st.sidebar.slider("Simulation Time (h)", 12, 96, 48)

antioxidants = ["luteolin", "quercetin", "resveratrol", "curcumin", "naringenin"]

if st.sidebar.button("Run Simulation"):
    for idx, smiles in enumerate(smiles_list):
        smiles_lower = smiles.lower()
        amp = 2.5 if any(x in smiles_lower for x in ["cl", "br", "no2", "epoxide"]) else 1.0
        is_antiox = any(x in smiles_lower for x in antioxidants)

        y0 = [dose, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        t = np.linspace(0, duration, 300)
        sol = odeint(liver_dili_model, y0, t, args=(amp, dose, is_antiox))

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
        tox_markers = {
            "ROS": final[3],
            "ALT": final[4],
            "Mito Stress": final[6],
            "Cholestasis": final[7],
            "Apoptosis": final[8],
            "Necrosis": final[9],
            "Fibrosis": final[10]
        }
        dominant = max(tox_markers, key=tox_markers.get)
        explanation = interpret_toxicity(dominant)

        score = (
            0.20 * final[3] + 0.13 * final[4] + 0.13 * final[6] +
            0.13 * final[8] + 0.13 * final[9] + 0.18 * final[10] + 0.10 * final[7]
        )
        risk = "LOW" if score < 0.6 else "MODERATE" if score < 1.5 else "HIGH"

        st.markdown(f"### 🧬 DILI Score: `{score:.2f}` → **Risk: {risk}`**")
        st.markdown(f"**🔬 Dominant Toxicity Pathway:** `{dominant}`")
        st.info(f"🧠 {explanation}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, safe_text("Digital Liver v3: DILI Simulation Report"), ln=1)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, safe_text(f"SMILES: {smiles}"), ln=1)
            pdf.cell(0, 10, safe_text(f"Dose: {dose} | Time: {duration}h"), ln=1)
            pdf.cell(0, 10, safe_text(f"Toxicity Amplifier: {amp}"), ln=1)
            pdf.cell(0, 10, safe_text(f"DILI Score: {score:.2f} | Risk: {risk}"), ln=1)
            pdf.multi_cell(0, 10, safe_text(f"Main Toxicity: {dominant} → {explanation}"))
            pdf.output(tmp.name)

            with open(tmp.name, "rb") as file:
                st.download_button("📄 Download PDF Report", data=file, file_name=f"dili_report_{idx+1}.pdf", mime="application/pdf")
            os.unlink(tmp.name)
'''

# Save file
with open(enhanced_path, "w") as f:
    f.write(code_text)
