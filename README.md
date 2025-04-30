Dataset Link:https://drive.google.com/file/d/1XtlIEYkZZygrfzHeIIuUU8_tfLdbPzv0/view
This project is a three-stage AI-powered drug discovery pipeline that:
1.	Generates novel drug-like molecules conditioned on protein sequences (Deep Learning).
2.	Predicts binding affinity between molecules and proteins (Machine Learning).
3.	Deploys an interactive web app for real-time predictions (Streamlit).
________________________________________
1. Molecule Generation with Deep Learning (1st Code)
Objective
•	Generate novel SMILES strings (drug-like molecules) that are likely to bind to a given protein target.
Key Components
•	Protein Encoder: Uses ESM-2 (Evolutionary Scale Modeling) to convert protein sequences into embeddings.
•	Drug Decoder: Uses GPT-2 (fine-tuned for SMILES generation) with cross-attention to condition molecule generation on protein embeddings.
Workflow
1.	Data Loading & Preprocessing
o	Input: BindingDB.csv (contains protein sequences, SMILES, and binding affinities).
o	Filters:
	Keeps only high-affinity interactions (Affinity ≤ 1000 nM).
	Removes invalid SMILES and large molecules (MolWt ≤ 600).
	Limits protein sequence length (≤ 512 amino acids).
2.	Model Architecture
o	Protein Encoder (ProteinEncoder)
	Uses ESM-2-tiny (facebook/esm2_t6_8M_UR50D) for efficiency.
	Extracts protein embeddings → projects to lower dimension (256-D).
o	Drug Decoder (DrugDecoder)
	Uses GPT-2 with cross-attention layers to incorporate protein context.
	Only 2 cross-attention layers (for memory efficiency).
3.	Training & Generation
o	Training: Minimizes cross-entropy loss for SMILES generation.
o	Generation:
	Given a protein sequence, the model generates multiple SMILES candidates.
	Uses top-k sampling (k=50) for diverse yet high-quality outputs.
4.	Output
o	A CSV file containing generated SMILES molecules.
________________________________________
2. Binding Affinity Prediction with XGBoost (2nd Code)
Objective
•	Predict how strongly a molecule (SMILES) binds to a protein (measured in pKd, where higher = stronger binding).
Key Components
•	Feature Extraction: Computes molecular descriptors (RDKit) + protein sequence features (amino acid composition).
•	XGBoost Model: Optimized using RandomizedSearchCV.
Workflow
1.	Feature Extraction
o	Ligand Features (SMILES → 10 features)
	MolWt, NumHDonors, NumHAcceptors, TPSA, MolLogP, etc.
o	Target Features (Protein Sequence → 20 features)
	Normalized amino acid frequencies (A, C, D, E, ..., Y).
2.	Model Training
o	Data Split: 80% train, 20% test.
o	Scaling: StandardScaler applied to features.
o	Hyperparameter Tuning:
	Uses RandomizedSearchCV to optimize n_estimators, max_depth, learning_rate, etc.
o	Evaluation Metrics:
	MAE, MSE, RMSE, R² (saved in metrics.json).
3.	Output
o	xgb_model.pkl: Trained XGBoost model.
o	scaler.pkl: Feature scaler for preprocessing.
o	metrics.json: Performance metrics.
________________________________________
3. Streamlit Web App for Deployment (3rd Code)
Objective
•	Provide a user-friendly interface to:
1.	Upload generated SMILES (from 1st step).
2.	Input a protein sequence.
3.	Predict binding affinities (using the XGBoost model).
Key Features
1.	File Upload
o	Accepts a CSV file with a SMILES column.
2.	Protein Sequence Input
o	Users paste a protein sequence (e.g., "MKWVTFISLLFLFSSAYSRGV...").
3.	Prediction & Visualization
o	For each SMILES:
	Computes ligand + protein features.
	Scales features using scaler.pkl.
	Predicts pKd using xgb_model.pkl.
	Displays molecule structure (RDKit visualization).
4.	Model Metrics
o	Shows MAE, MSE, RMSE, R² in the sidebar.
Workflow Integration
1.	Input → Generated SMILES (from 1st code).
2.	Prediction → Uses XGBoost model (from 2nd code).
3.	Output → Interactive predictions in a web app.
