# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib, os

RANDOM_STATE = 42
DATA_PATH    = "data/adult 3.csv"
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "salary_predictor_corrected.pkl")

# --- helper ----------------------------------------------------------
def simulate_salary(label: str) -> int:
    """Generate a realistic INR salary from the <=50K / >50K label."""
    return (
        np.random.randint(300_000, 500_001)   # ₹ 3 L – ₹ 5 L
        if label == "<=50k"
        else np.random.randint(500_001, 1_500_001)  # ₹ 5 L – ₹ 15 L
    )

# --- data loader -----------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # 1️⃣ normalise column names
    df.columns = df.columns.str.strip().str.lower()

    # 2️⃣ strip whitespace & lower‑case the income labels
    df["income"] = df["income"].str.strip().str.lower()

    # 3️⃣ replace “?” with NaN
    df.replace("?", np.nan, inplace=True)

    # 4️⃣ keep only valid income rows
    df = df[df["income"].isin(["<=50k", ">50k"])].copy()

    # 5️⃣ simulate numeric salary & drop the original label
    df["salary"] = df["income"].apply(simulate_salary)
    df.drop("income", axis=1, inplace=True)

    # 6️⃣ minimal imputation so we don’t lose all rows
    numeric_cols = ["age", "hours-per-week", "capital-gain", "capital-loss"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = ["education", "occupation", "gender", "race", "marital-status"]
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    print("✅ Data loaded – rows:", len(df))
    return df

# --- pipeline --------------------------------------------------------
def build_pipeline() -> Pipeline:
    numeric_features = ["age", "hours-per-week", "capital-gain", "capital-loss"]
    categorical_features = ["education", "occupation", "gender", "race", "marital-status"]

    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), numeric_features),
         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)]
    )

    return Pipeline(
        [("preprocessor", preprocessor),
         ("regressor", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))]
    )

def train_and_save(X_train, y_train):
    pipe = build_pipeline()

    # (optional) quick grid search
    param_grid = {
        "regressor__max_depth": [10, 20],
        "regressor__min_samples_split": [2, 5],
    }
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")

    return best_model

# --- main ------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)

    X = df[
        ["age", "education", "occupation", "gender", "race",
         "marital-status", "hours-per-week", "capital-gain", "capital-loss"]
    ]
    y = df["salary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = train_and_save(X_train, y_train)

    # sanity‑check prediction
    sample = pd.DataFrame(
        [{
            "age": 30, "education": "bachelors", "occupation": "tech-support",
            "gender": "male", "race": "white", "marital-status": "never-married",
            "hours-per-week": 40, "capital-gain": 0, "capital-loss": 0
        }]
    )
    print("💰 Example prediction: ₹{:,.2f}".format(model.predict(sample)[0]))

if __name__ == "__main__":
    main()
