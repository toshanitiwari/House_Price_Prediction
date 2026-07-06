import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------- Page config ----------
st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

FEATURES = [
    "Overall Qual",
    "Gr Liv Area",
    "Total Bsmt SF",
    "Garage Cars",
    "Garage Area",
    "Year Built",
    "Full Bath",
    "TotRms AbvGrd",
    "Neighborhood",
]
TARGET = "SalePrice"


# ---------- Load data & train model (cached so it runs only once) ----------
@st.cache_resource
def load_data_and_train_model():
    data = pd.read_csv("houses.csv")
    data = data[FEATURES + [TARGET]]

    # Fill missing values
    num_cols = data.select_dtypes(include=["int64", "float64"]).columns
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    cat_cols = data.select_dtypes(include=["object", "string"]).columns
    data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

    data_encoded = pd.get_dummies(data, drop_first=True)

    X = data_encoded.drop(TARGET, axis=1)
    y = data_encoded[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    neighborhoods = sorted(data["Neighborhood"].unique().tolist())
    defaults = {
        "Overall Qual": int(data["Overall Qual"].median()),
        "Gr Liv Area": int(data["Gr Liv Area"].median()),
        "Total Bsmt SF": int(data["Total Bsmt SF"].median()),
        "Garage Cars": int(data["Garage Cars"].median()),
        "Garage Area": int(data["Garage Area"].median()),
        "Year Built": int(data["Year Built"].median()),
        "Full Bath": int(data["Full Bath"].median()),
        "TotRms AbvGrd": int(data["TotRms AbvGrd"].median()),
    }

    return model, X.columns, neighborhoods, defaults, rmse, r2


model, train_columns, neighborhoods, defaults, rmse, r2 = load_data_and_train_model()

# ---------- UI ----------
st.title("🏠 House Price Prediction")
st.write("Enter house details below to predict its sale price using a Random Forest regression model.")

col1, col2 = st.columns(2)

with col1:
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, defaults["Overall Qual"])
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=6000, value=defaults["Gr Liv Area"])
    total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=4000, value=defaults["Total Bsmt SF"])
    garage_cars = st.slider("Garage Capacity (cars)", 0, 5, defaults["Garage Cars"])
    garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=defaults["Garage Area"])

with col2:
    year_built = st.number_input("Year Built", min_value=1870, max_value=2026, value=defaults["Year Built"])
    full_bath = st.slider("Full Bathrooms", 0, 5, defaults["Full Bath"])
    tot_rms = st.slider("Total Rooms Above Ground", 2, 15, defaults["TotRms AbvGrd"])
    neighborhood = st.selectbox("Neighborhood", neighborhoods)

if st.button("Predict Price"):
    input_dict = {
        "Overall Qual": overall_qual,
        "Gr Liv Area": gr_liv_area,
        "Total Bsmt SF": total_bsmt_sf,
        "Garage Cars": garage_cars,
        "Garage Area": garage_area,
        "Year Built": year_built,
        "Full Bath": full_bath,
        "TotRms AbvGrd": tot_rms,
        "Neighborhood": neighborhood,
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)

    predicted_price = model.predict(input_encoded)[0]

    st.success(f"### Predicted House Price: ${predicted_price:,.0f}")

st.markdown("---")
st.caption(f"Model: Random Forest Regressor | R² Score: {r2:.3f} | RMSE: ${rmse:,.0f}")
st.caption("Dataset: Ames Housing Dataset (Kaggle)")
