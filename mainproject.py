import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, average_precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import google.generativeai as genai


# 2. GEMINI SETUP

api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
model_ai = genai.GenerativeModel("gemini-2.5-flash")

# 3. LOAD DATA

df = pd.read_csv(r"C:\Users\Vedant\Desktop\creditcard.csv")
df["risk"] = df["Class"].map({0: "LOW", 1: "HIGH"})
features = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
X = df[features]
y = df["Class"]

# 4. PREPROCESS & SPLIT

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Applying SMOTE to partially balance the training data...")
# Instead of fully balancing (1:1), we balance to 10% (0.1) which prevents Logistic Regression 
# from overfitting on synthetic data and destroying its Precision/F1 score.
smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"Before SMOTE: {sum(y_train==1)} fraud / {sum(y_train==0)} normal")
print(f"After SMOTE: {sum(y_train_resampled==1)} fraud / {sum(y_train_resampled==0)} normal")

# 5. TRAIN & EVALUATE MODELS

print("\nTraining models...")

log_model = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear')
log_model.fit(X_train_resampled, y_train_resampled)
log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

# XGBoost (New Main Model for High Performance on Imbalanced Data)
# We use a balanced scale_pos_weight rather than the absolute ratio, to avoid tanking Precision
# A common trick is math.sqrt(sum(negative) / sum(positive)) to balance Precision and Recall
scale_pos_w = np.sqrt(sum(y_train == 0) / sum(y_train == 1))

xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6, # Slightly deeper trees to catch more complex patterns
    scale_pos_weight=scale_pos_w, 
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)

# Custom Threshold Tuning for XGBoost
xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
# We'll use 0.5 (default) or slightly higher to improve precision. Let's stick to 0.5 for now to see balanced performance.
custom_threshold = 0.5 
xgb_pred = (xgb_prob >= custom_threshold).astype(int)

# Isolation Forest (anomaly detector)
# Note: Isolation Forest is unsupervised, so we train it only on normal data for better anomaly detection
# A contamination of 0.005 balances well to find anomalies without destroying Precision
iso_model = IsolationForest(
    contamination=0.005, 
    random_state=42,
    n_jobs=-1
)
iso_model.fit(X_train_scaled[y_train == 0]) 

# Custom threshold for Isolation Forest
iso_scores = iso_model.decision_function(X_test_scaled)
# We raise the threshold from 0.0 to 0.03 to balance the F1 (harmonic mean) perfectly
iso_pred = (iso_scores < 0.03).astype(int) 

# 5b. DEEP LEARNING AUTOENCODER (PyTorch)

print("Training Deep Learning Autoencoder on normal transactions...")

input_dim = X_train_scaled.shape[1]

# Define Autoencoder Architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, input_dim),
            nn.Tanh() # Tanh helps reconstruct scaled numerical data between -1 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

ae_model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(ae_model.parameters(), lr=0.005)

# We train ONLY on normal transactions (Class=0) so the network learns what "normal" looks like
X_train_normal_tensor = torch.FloatTensor(X_train_scaled[y_train == 0])

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = ae_model(X_train_normal_tensor)
    loss = criterion(outputs, X_train_normal_tensor)
    loss.backward()
    optimizer.step()

# Calculate Autoencoder Reconstruction Error on Test Set
X_test_tensor = torch.FloatTensor(X_test_scaled)
with torch.no_grad():
    ae_test_predictions = ae_model(X_test_tensor)
    
# Mean Squared Error for each individual transaction
ae_mse = torch.mean((X_test_tensor - ae_test_predictions) ** 2, dim=1).numpy()

# Set a threshold for what MSE is considered "Fraud"
ae_threshold = np.percentile(ae_mse, 95) # Top 5% highest errors are flagged
ae_pred = (ae_mse > ae_threshold).astype(int)

print("✅ Models trained\n")

# Evaluation Metrics
print("\n===== MODEL EVALUATION =====")
# Note: F1-Score is the harmonic mean of Precision and Recall. High F1 requires both.
print("Logistic Regression:")
print(f"Accuracy:  {accuracy_score(y_test, log_pred):.4f}")
print(f"Precision: {precision_score(y_test, log_pred):.4f} (When it predicts Fraud, how often is it right?)")
print(f"Recall:    {recall_score(y_test, log_pred):.4f} (Out of all actual Frauds, how many did it catch?)")
print(f"F1 Score:  {f1_score(y_test, log_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, log_prob):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, log_pred))

print("\nXGBoost (Main Model):")
print(f"Accuracy:  {accuracy_score(y_test, xgb_pred):.4f}")
print(f"Precision: {precision_score(y_test, xgb_pred):.4f} (When it predicts Fraud, how often is it right?)")
print(f"Recall:    {recall_score(y_test, xgb_pred):.4f} (Out of all actual Frauds, how many did it catch?)")
print(f"F1 Score:  {f1_score(y_test, xgb_pred):.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, xgb_prob):.4f}")
print(f"PR AUC:    {average_precision_score(y_test, xgb_prob):.4f} (Overall performance on rare fraud class)")
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))

print("\nIsolation Forest (Unsupervised Baseline):")
print(f"Accuracy:  {accuracy_score(y_test, iso_pred):.4f}")
print(f"Precision: {precision_score(y_test, iso_pred):.4f} (When it predicts Fraud, how often is it right?)")
print(f"Recall:    {recall_score(y_test, iso_pred):.4f} (Out of all actual Frauds, how many did it catch?)")
print(f"F1 Score:  {f1_score(y_test, iso_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, iso_pred))

print("\nDeep Learning Autoencoder (Unsupervised Neural Net):")
print(f"Accuracy:  {accuracy_score(y_test, ae_pred):.4f}")
print(f"Precision: {precision_score(y_test, ae_pred):.4f} (When it predicts Fraud, how often is it right?)")
print(f"Recall:    {recall_score(y_test, ae_pred):.4f} (Out of all actual Frauds, how many did it catch?)")
print(f"F1 Score:  {f1_score(y_test, ae_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, ae_pred))
print("============================\n")

# 6. FEATURE IMPORTANCE

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10] # Top 10 features
    
    print("\n📊 Top 10 Feature Importances (XGBoost):")
    for i in range(10):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Feature Importances in Fraud Detection (XGBoost)")
    plt.bar(range(10), importances[indices], align="center")
    plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45)
    plt.xlim([-1, 10])
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("📈 Feature importance plot saved as 'feature_importance.png'")

plot_feature_importance(xgb_model, features)

# 7. RISK ANALYSIS & ENSEMBLE DECISION

def analyze_transaction(row, recent_fraud_spike=False):

    row_df = pd.DataFrame([row], columns=features)
    row_scaled = scaler.transform(row_df)

    # 1. Logistic prediction (Probability) 
    log_p = log_model.predict_proba(row_scaled)[0][1] # Kept for evaluation, removed from fusion

    # 2. XGBoost prediction (Probability)
    xgb_p = xgb_model.predict_proba(row_scaled)[0][1]

    # 3. Isolation forest anomaly (-1 anomaly, 1 normal)
    anomaly_score = iso_model.decision_function(row_scaled)[0]
    # Convert anomaly score into a pseudo-probability (lower score = higher fraud risk)
    iso_prob = 1 / (1 + np.exp(anomaly_score * 10)) 
    
    anomaly = iso_model.predict(row_scaled)[0]
    
    # 4. Deep Learning Autoencoder Reconstruction Error
    row_tensor = torch.FloatTensor(row_scaled)
    with torch.no_grad():
        ae_pred = ae_model(row_tensor)
    ae_error = torch.mean((row_tensor - ae_pred) ** 2).item()
    # Normalize the error into a pseudo-probability using the 95th percentile threshold
    ae_prob = min(1.0, ae_error / (ae_threshold * 1.5))

    # MODEL FUSION (HYBRID RISK SCORE)
    final_risk_score = (
        0.50 * xgb_p +
        0.20 * log_p +
        0.15 * iso_prob +
        0.15 * ae_prob
    ) * 100

    # ⭐ DYNAMIC RISK SCORING (ADAPTIVE CALIBRATION)

    # If the system detects a sudden spike in recent fraud across the network, 
    # it becomes mathematically more strict, artificially boosting the risk score 
    # to catch borderline cases that it might normally let pass.
    if recent_fraud_spike:
        final_risk_score = min(100.0, final_risk_score * 1.25) # 25% Boost to Risk Score during an attack

    if final_risk_score > 80:
        risk_level = "CRITICAL"
    elif final_risk_score > 60:
        risk_level = "HIGH"
    elif final_risk_score > 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return risk_level, final_risk_score

# ======================
# 8. MULTI-AGENT GEMINI AI WORKFLOW (WITH RAG & RATE-LIMIT CACHING)
# ======================

# Simulated Threat Intelligence Database (RAG Source)
THREAT_DB = {
    "192.168.1.100": "Clean history. Known residential proxy.",
    "45.22.19.11": "WARNING: IP address associated with recent dark web credential stuffing botnet (AlphaBay leaks).",
    "103.44.2.99": "CRITICAL THREAT: Known exit node for Tor network. High correlation with anonymous cybercrime.",
    "77.100.5.55": "Clean history. Standard mobile carrier gateway.",
    "188.4.3.2": "WARNING: Merchant ID flagged in recent Telegram fraud channels for card tumbling."
}

MOCK_MERCHANTS = ["Amazon.com", "Unknown VPN Provider", "CryptoExchange LLC", "Local Grocery", "Luxury Watches Inc."]
MOCK_IPS = list(THREAT_DB.keys())

# Cache dictionaries to prevent hitting the 15 RPM free tier limit
_cache_ai_report = {}
_cache_sar = {}
_cache_customer = {}

def get_ai_explanation(sample, risk, confidence, merchant, ip_address):
    if risk == "LOW":
        return "Low risk transaction. No major concern."

    # RAG Retrieval: Look up the IP in the threat database
    rag_context = THREAT_DB.get(ip_address, "No threat intelligence records found for this IP.")

    cache_key = f"{risk}_{int(confidence/10)}_{ip_address}"
    if cache_key in _cache_ai_report:
        return _cache_ai_report[cache_key]

    prompt = f"""
You are an expert Anti-Money Laundering analyst with access to a Retrieval-Augmented Generation (RAG) threat database.

Transaction Details:
Amount: ${sample[-1]:.2f}
Time: {sample[0]:.2f}
Merchant: {merchant}
IP Address: {ip_address}
System Risk Level: {risk}
Hybrid Risk Score: {confidence:.1f}%

[RAG THREAT INTELLIGENCE FEED]:
{rag_context}

You have access to a Model Fusion setup (XGBoost + Logistic Regression + PyTorch Autoencoder + Isolation Forest). 
Analyze the data above and explain briefly:
1. Why this transaction tripped the ensemble detection models.
2. How the RAG Threat Intelligence context elevates the specific real-world AML risk severity this represents.
3. Suggested immediate action for the human analyst (e.g. freeze account, SAR filing, contact customer).

Format the output clearly with bullet points. Keep it professional, analytical, and short.
"""
    try:
        response = model_ai.generate_content(prompt)
        _cache_ai_report[cache_key] = response.text
        return response.text
    except Exception as e:
        print("⚠ Gemini error:", e)
        return (f"• TRANSACTION FLAGGED BY ENSEMBLE AI\n"
                f"• Risk Severity: {risk} (Score: {confidence:.1f}%)\n"
                f"• The Autoencoder detected anomalous reconstruction error while the Ensemble model confirmed irregular spending patterns.\n"
                f"• ACTION REQUIRED: Immediate manual review of account history recommended.")

def generate_network_alert(spike_info):
    prompt = f"""
You are an AI Security System monitoring a banking network. 
An unusual volume of high-risk transactions has been detected ({spike_info}).
Write a 1-2 sentence urgent alert to the security operations center (SOC) team advising them to enable strict dynamic risk scoring.
Keep it punchy and professional.
"""
    try:
        response = model_ai.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"🚨 SOC ALERT: Unusual volume of deep-pattern anomalies detected ({spike_info}). Recommend immediate transition to Strict Risk Calibration."

def generate_sar_narrative(sample, risk, confidence, ai_explanation):
    if risk == "LOW":
        return "Not applicable."
    
    cache_key = f"{risk}_{int(confidence/10)}"
    if cache_key in _cache_sar:
        return _cache_sar[cache_key]

    prompt = f"""
You are an expert Anti-Money Laundering (AML) compliance officer.
Based on the following transaction and AI analysis, draft a formal Suspicious Activity Report (SAR) narrative.

Transaction Amount: ${sample[-1]:.2f}
Time Metric: {sample[0]:.2f}
Risk Score: {confidence:.1f}%

Previous AI Analysis:
{ai_explanation}

The narrative should include:
- A formal introduction.
- Description of the suspicious activity.
- The reason for filing the SAR.
Draft this in a highly professional, legal, and compliance-oriented tone. Keep it under 150 words.
"""
    try:
        response = model_ai.generate_content(prompt)
        _cache_sar[cache_key] = response.text
        return response.text
    except Exception as e:
        return (f"Suspicious Activity Report (Internal Draft)\n\n"
                f"This report documents a {risk}-risk transaction of ${sample[-1]:.2f} flagged by the ensemble detection system with a confidence of {confidence:.1f}%. "
                f"The transaction deviated significantly from historical baselines across multiple vectors. Pending further investigation, "
                f"this activity warrants regulatory reporting under AML protocols.")

def draft_customer_alert_email(sample):
    cache_key = f"{int(sample[-1]/100)}"
    if cache_key in _cache_customer:
        return _cache_customer[cache_key]

    prompt = f"""
Draft a short, polite, and professional SMS/Email to a bank customer to verify a recent potentially suspicious transaction.
Transaction Amount: ${sample[-1]:.2f}
Do not mention fraud or AI directly, just ask them to verify if they made this transaction. Keep it under 50 words.
"""
    try:
        response = model_ai.generate_content(prompt)
        _cache_customer[cache_key] = response.text
        return response.text
    except Exception as e:
        return f"Govinda AI Security: Did you attempt a recent transaction of ${sample[-1]:.2f}? Reply 1 for YES, 2 for NO. If you do not reply, your card may be temporarily restricted."

# ======================
# 9. FASTAPI BACKEND SERVER
# ======================
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import random
import uuid

app = FastAPI(title="Fraud Intelligence Center API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TransactionRequest(BaseModel):
    time: float
    amount: float
    v_features: list[float] = [0.0] * 28 # Default to 0s if not provided
    strict_mode: bool = False

@app.get("/api/simulate")
async def simulate_transaction(strict: bool = False):
    """Fetches a random transaction from the dataset and analyzes it."""
    # 30% chance of pulling a fraud transaction for the demo to make it interesting
    is_fraud_demo = random.random() < 0.3
    
    if is_fraud_demo:
        fraud_txns = X[y == 1]
        idx = random.randint(0, len(fraud_txns) - 1)
        sample = fraud_txns.iloc[idx].values
    else:
        normal_txns = X[y == 0]
        idx = random.randint(0, len(normal_txns) - 1)
        sample = normal_txns.iloc[idx].values

    # Run Analysis
    risk_level, final_risk_score = analyze_transaction(sample, recent_fraud_spike=strict)

    # Assign random mock RAG metadata
    merchant = random.choice(MOCK_MERCHANTS)
    ip_address = random.choice(MOCK_IPS)

    response_data = {
        "id": str(uuid.uuid4())[:8],
        "amount": float(sample[-1]),
        "time": float(sample[0]),
        "risk_level": risk_level,
        "score": float(final_risk_score),
        "merchant": merchant,
        "ip_address": ip_address
    }

    # If high risk, generate AI reports
    if risk_level in ["HIGH", "CRITICAL"]:
        ai_report = get_ai_explanation(sample, risk_level, final_risk_score, merchant, ip_address)
        sar_draft = generate_sar_narrative(sample, risk_level, final_risk_score, ai_report)
        customer_msg = draft_customer_alert_email(sample)
        
        response_data.update({
            "ai_report": ai_report,
            "sar_draft": sar_draft,
            "customer_msg": customer_msg
        })

    return response_data

@app.post("/api/analyze/manual")
async def analyze_manual(request: TransactionRequest):
    """Analyzes a manually inputted transaction."""
    if len(request.v_features) != 28:
        raise HTTPException(status_code=400, detail="Must provide exactly 28 V features")
        
    sample = np.array([request.time] + request.v_features + [request.amount])
    risk_level, final_risk_score = analyze_transaction(sample, recent_fraud_spike=request.strict_mode)
    
    # Assign random mock RAG metadata
    merchant = random.choice(MOCK_MERCHANTS)
    ip_address = random.choice(MOCK_IPS)

    response_data = {
        "id": str(uuid.uuid4())[:8],
        "amount": request.amount,
        "time": request.time,
        "risk_level": risk_level,
        "score": float(final_risk_score),
        "merchant": merchant,
        "ip_address": ip_address
    }

    if risk_level in ["HIGH", "CRITICAL"]:
        ai_report = get_ai_explanation(sample, risk_level, final_risk_score, merchant, ip_address)
        sar_draft = generate_sar_narrative(sample, risk_level, final_risk_score, ai_report)
        customer_msg = draft_customer_alert_email(sample)
        
        response_data.update({
            "ai_report": ai_report,
            "sar_draft": sar_draft,
            "customer_msg": customer_msg
        })

    return response_data

from fastapi import UploadFile, File
import io

@app.post("/api/analyze/batch")
async def analyze_batch(file: UploadFile = File(...), strict_mode: bool = False):
    """Parses an uploaded CSV file and runs batch analysis"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")
        
    contents = await file.read()
    try:
        df_batch = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")
        
    # Check if required columns exist
    missing_cols = [col for col in features if col not in df_batch.columns]
    
    # Fill missing features with 0.0, even if all V features are missing
    for col in missing_cols:
        df_batch[col] = 0.0
            
    if "Amount" not in df_batch.columns or "Time" not in df_batch.columns:
        raise HTTPException(status_code=400, detail="Uploaded CSV must contain 'Time' and 'Amount' columns.")
        
    X_batch = df_batch[features]
    limit = min(len(X_batch), 1000)
    X_batch_limited = X_batch.iloc[:limit]
    
    results = []
    
    for i in range(len(X_batch_limited)):
        row_vals = X_batch_limited.iloc[i].values
        lvl, prob = analyze_transaction(row_vals, recent_fraud_spike=strict_mode)
        
        # Assign random mock RAG metadata
        merchant = random.choice(MOCK_MERCHANTS)
        ip_address = random.choice(MOCK_IPS)

        res = {
            "id": i,
            "time": float(row_vals[0]),
            "amount": float(row_vals[-1]),
            "risk_level": lvl,
            "score": float(prob),
            "merchant": merchant,
            "ip_address": ip_address
        }
        
        # If it's a high risk transaction, get an AI report for it right away
        if lvl in ['HIGH', 'CRITICAL']:
            # We'll put a small delay between requests to avoid totally blowing out 
            # the free Gemini tier rate limit if there's a huge batch of fraud
            import time
            time.sleep(1.5) 
            ai_report = get_ai_explanation(row_vals, lvl, prob, merchant, ip_address)
            res['ai_report'] = ai_report
            
        results.append(res)
        
    return {
        "status": "success",
        "total_analyzed": len(results),
        "total_flagged": len([r for r in results if r['risk_level'] in ['HIGH', 'CRITICAL']]),
        "highest_risk_analysis": None,
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting FastAPI Backend Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)