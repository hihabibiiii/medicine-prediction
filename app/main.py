from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Medicine Demand Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load trained model
# =========================
model = joblib.load("app/medicine_demand_model_v1.joblib")

# =========================
# Load & prepare data
# =========================
df = pd.read_csv("data/salesdaily.csv")
df['datum'] = pd.to_datetime(df['datum'])
df = df.sort_values('datum')

medicine_cols = [
    'M01AB','M01AE','N02BA','N02BE',
    'N05B','N05C','R03','R06'
]

long_df = df.melt(
    id_vars=['datum'],
    value_vars=medicine_cols,
    var_name='medicine',
    value_name='sales'
)

long_df = long_df.sort_values(['medicine', 'datum'])

# =========================
# Feature engineering
# =========================
def create_features(group):
    group['last_7d'] = group['sales'].rolling(7).sum()
    group['last_30d'] = group['sales'].rolling(30).sum()
    group['avg_30d'] = group['sales'].rolling(30).mean()
    group['std_30d'] = group['sales'].rolling(30).std()
    return group

long_df = long_df.groupby('medicine', group_keys=False).apply(create_features)
long_df.dropna(inplace=True)

FEATURES = ['last_7d', 'last_30d', 'avg_30d', 'std_30d']
THRESHOLD = 0.55

# =========================
# Routes
# =========================
@app.get("/medicines")
def get_medicines():
    medicines = sorted(long_df['medicine'].unique().tolist())
    return medicines
@app.get("/")
def root():
    return {"message": "Medicine Demand Prediction API is running"}

@app.get("/predict")
def predict():
    latest = long_df.groupby('medicine').tail(1)

    probs = model.predict_proba(latest[FEATURES])[:, 1]

    latest['probability'] = probs
    latest['action'] = np.where(
        latest['probability'] >= THRESHOLD,
        "Increase Stock",
        "Normal"
    )

    result = latest[['medicine', 'probability', 'action']]
    result = result.sort_values('probability', ascending=False)

    return result.to_dict(orient="records")
class MedicineInput(BaseModel):
    medicine: str
    last_7d: float
    last_30d: float
    avg_30d: float
    std_30d: float


@app.post("/predict_json")
def predict_json(data: MedicineInput):

    input_df = pd.DataFrame([{
        "last_7d": data.last_7d,
        "last_30d": data.last_30d,
        "avg_30d": data.avg_30d,
        "std_30d": data.std_30d
    }])

    prob = model.predict_proba(input_df)[0][1]

    action = "Increase Stock" if prob >= THRESHOLD else "Normal"

    return {
        "medicine": data.medicine,
        "probability": round(float(prob), 4),
        "action": action
    }

