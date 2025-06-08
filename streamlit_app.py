import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
from mordred import Calculator, descriptors
from rdkit import Chem

st.set_page_config(page_title="Enhanced Digital Liver DILI Simulator", layout="wide")
st.title("🧬 Enhanced Digital Liver: Drug-Induced Liver Injury (DILI) Simulator")

st.markdown("""
This simulation models drug-induced liver injury (DILI) through multiple pathways:
- **Direct Hepatocyte Damage**
- **Immune-mediated injury**
- **Mitochondrial dysfunction**
- **Cholestasis**
- **Oxidative stress, apoptosis, necrosis, fibrosis**
""")

# ODE Model
def liver_dili_model(y, t, amp, dose):
    drug, metab, gsh, ros, alt, ast, dna, apoptosis, necrosis, chol, fibrosis = y
    k_cyp = 0.05
    k_metab = 0.04
    k_gsh = 0.03
    k_ros = 0.02 * amp
    k_clear = 0.01
    k_dna = 0.015 * amp
    k_apop = 0.01 * amp
    k_necro = 0.008 * amp
    k_chol = 0.006 * amp
    k_fibrosis = 0.005 * amp
    k_liver = 0.01 * amp

    d_drug = -k_cyp * drug
    d_metab = k_cyp * drug - k_gsh * min(metab, gsh)
    d_gsh = -k_gsh * min(metab, gsh)
    d_ros = k_ros * metab - k_clear * ros
    d_alt = k_liver * ros
    d_ast = k_liver * ros
    d_dna = k_dna * metab
    d_apop = k_apop * ros
    d_necro = k_necro * ros
    d_chol = k_chol * metab
    d_fibrosis = k_fibrosis * (apoptosis + necrosis)

    return [d_drug, d_metab, d_gsh, d_ros, d_alt, d_ast, d_dna, d_apop, d_necro, d_chol, d_fibrosis]

# Input section
st.sidebar.header("Drug Input")
smiles_list = st.sidebar.text_area("Enter SMILES (one per line)", "CC(=O)NC1=CC=C(C=C1)O").splitlines()
dose = st.sidebar.slider("Drug Dose Level (normalized)", 0.1, 3.0, 1.0, 0.1)
duration = st.sidebar.slider("Simulation Duration (hours)", 12, 96, 48)

if st.sidebar.button("Run Simulation for All Drugs"):
    st.success(f"✅ Running simulation for {len(smiles_list)} compounds")

    for idx, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            st.error(f"❌ Invalid SMILES: {smiles}")
            continue

        calc = Calculator(descriptors.TPSA, descriptors.MolWt, descriptors.LogP)
        result = calc(mol)
        try:
            mw = result["MolWt"]
            logp = result["LogP"]
            tpsa = result["TPSA"]
        except:
            st.warning(f"⚠️ Could not compute descriptors for {smiles}")
            continue

        amp = 2.5 if any(x in smiles.lower() for x in ["cl", "br", "no2", "n=o", "n#n"]) else 1.0

        y0 = [dose, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        t = np.linspace(0, duration, 300)
        sol = odeint(liver_dili_model, y0, t, args=(amp, dose))

        labels = ["Drug", "Metabolite", "GSH", "ROS", "ALT", "AST", "DNA Damage",
                  "Apoptosis", "Necrosis", "Cholestasis", "Fibrosis"]

        st.subheader(f"🧪 Compound {idx+1}: `{smiles}`")
        st.write(f"**MolWt**: {mw:.2f} | **LogP**: {logp:.2f} | **TPSA**: {tpsa:.2f} | **Amp:** {amp}")

        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(len(labels)):
            ax.plot(t, sol[:, i], label=labels[i])
        ax.set_title("Liver Injury Simulation")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Level")
        ax.legend()
        st.pyplot(fig)

        score = 0.2 * sol[-1][3] + 0.15 * sol[-1][6] + 0.15 * sol[-1][7] + 0.15 * sol[-1][8] + 0.15 * sol[-1][9] + 0.2 * sol[-1][10]
        risk = "LOW"
        if score > 1.5:
            risk = "HIGH"
        elif score > 0.75:
            risk = "MODERATE"

        st.markdown(f"### 🧬 DILI Score: `{score:.2f}` → **Risk Level: `{risk}`**")

        # PDF Report
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Digital Liver DILI Simulation Report", ln=1)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"SMILES: {smiles}", ln=1)
            pdf.cell(0, 10, f"MolWt: {mw:.2f} | LogP: {logp:.2f} | TPSA: {tpsa:.2f}", ln=1)
            pdf.cell(0, 10, f"Dose: {dose} | Duration: {duration}h", ln=1)
            pdf.cell(0, 10, f"Toxicity Amplifier: {amp}", ln=1)
            pdf.cell(0, 10, f"Final Score: {score:.2f} | Risk Level: {risk}", ln=1)
            pdf.output(tmp.name)

            with open(tmp.name, "rb") as file:
                st.download_button("📄 Download PDF Report", data=file, file_name=f"dili_report_{idx+1}.pdf", mime="application/pdf")
            os.unlink(tmp.name)
