import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import joblib
import json

# Load model, scaler, and metrics
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('metrics.json') as f:
    metrics = json.load(f)

# Feature extraction functions
def ligand_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(10)
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.MolMR(mol),
        Descriptors.RingCount(mol)
    ])

def target_features(seq):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    seq = seq.upper()
    total = len(seq)
    return np.array([seq.count(aa)/total if total > 0 else 0 for aa in amino_acids])

# Streamlit UI
st.set_page_config(page_title="Binding Affinity Predictor", layout="centered")

st.title("🧬 Binding Affinity Predictor")
st.markdown("Upload a CSV with **SMILES molecules** and input a **target protein sequence** to predict binding affinity (pKd).")

# Show metrics
st.sidebar.header("📊 Model Metrics")
for metric, value in metrics.items():
    st.sidebar.write(f"**{metric}**: `{value:.3f}`")

# File uploader
uploaded_file = st.file_uploader("📄 Upload CSV file with SMILES", type=["csv"])

# Target protein input
target_seq = st.text_area("🧪 Enter Target Protein Sequence", height=100)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if the CSV contains a column 'SMILES'
        if "SMILES" not in df.columns:
            st.error("CSV must contain a column named 'SMILES'.")
        else:
            if target_seq:
                st.subheader("🔍 Predicted Binding Affinities")
                t_feat = target_features(target_seq)

                for idx, row in df.iterrows():
                    smi = row["SMILES"]
                    l_feat = ligand_features(smi)
                    X = np.concatenate((l_feat, t_feat)).reshape(1, -1)
                    X_scaled = scaler.transform(X)
                    affinity = model.predict(X_scaled)[0]

                    st.markdown(f"**Molecule {idx+1}:** `{smi}`")
                    st.success(f"Predicted Affinity: **{affinity:.2f} pKd**")

                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        img = Draw.MolToImage(mol, size=(250, 250))
                        st.image(img, caption=f"Molecule {idx+1}", use_column_width=False)
                    st.markdown("---")
            else:
                st.warning("Please enter a target protein sequence.")
    
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty. Please upload a valid CSV file.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.caption("Made with ❤️ using RDKit, XGBoost, and Streamlit")
