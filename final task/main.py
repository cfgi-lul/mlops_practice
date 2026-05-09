from pathlib import Path

import streamlit as st

from app.inference import FEATURE_NAMES, predict_price


st.set_page_config(page_title="Final task: housing predictor", page_icon=":bar_chart:")
st.title("Housing Price Predictor")
st.caption("MLOps final task demo: Streamlit + pretrained model + tests + CI/CD.")

with st.form("predict_form"):
    st.subheader("Input features")
    area = st.number_input("Area (m2)", min_value=10.0, max_value=500.0, value=60.0, step=1.0)
    rooms = st.number_input("Rooms", min_value=1.0, max_value=10.0, value=2.0, step=1.0)
    floor = st.number_input("Floor", min_value=1.0, max_value=50.0, value=5.0, step=1.0)
    age = st.number_input("Building age (years)", min_value=0.0, max_value=150.0, value=10.0, step=1.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = [area, rooms, floor, age]
    try:
        prediction = predict_price(row)
        st.success(f"Predicted price: {prediction:,.2f}")
    except FileNotFoundError:
        model_path = Path(__file__).parent / "model" / "linear_model.pkl"
        st.error(f"Model file is missing: {model_path}")
    except Exception as exc:  # pragma: no cover - defensive UI branch
        st.error(f"Inference failed: {exc}")

with st.expander("Expected features"):
    st.write(FEATURE_NAMES)
