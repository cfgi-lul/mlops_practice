from pathlib import Path

import pandas as pd
import streamlit as st

from app.inference import FEATURE_NAMES, predict_price


st.set_page_config(page_title="Final task: housing predictor", page_icon=":bar_chart:")
st.title("Housing Price Predictor")
st.caption("MLOps final task demo: Streamlit + pretrained model + tests + CI/CD.")

DEFAULT_ROW = {"area_m2": 60.0, "rooms": 2.0, "floor": 5.0, "building_age": 10.0, "prediction": None}
TABLE_COLUMNS = ["area_m2", "rooms", "floor", "building_age", "prediction"]
SIGNATURE_COLUMN = "_pred_signature"


def build_signature(row: pd.Series) -> str:
    return "|".join(f"{float(row[col]):.4f}" for col in FEATURE_NAMES)


def add_default_row() -> None:
    table = st.session_state.prediction_table.copy()
    default_signature = build_signature(pd.Series(DEFAULT_ROW))
    table.loc[len(table)] = {**DEFAULT_ROW, SIGNATURE_COLUMN: default_signature}
    st.session_state.prediction_table = table


def ensure_table_initialized() -> None:
    if "prediction_table" not in st.session_state:
        default_signature = build_signature(pd.Series(DEFAULT_ROW))
        st.session_state.prediction_table = pd.DataFrame([{**DEFAULT_ROW, SIGNATURE_COLUMN: default_signature}])


def clear_predictions_for_changed_rows(table: pd.DataFrame) -> pd.DataFrame:
    updated = table.copy()
    previous = st.session_state.prediction_table.copy()

    # Compare only feature columns: if any feature changed in a row, cached prediction is stale.
    previous_features = previous[FEATURE_NAMES].reset_index(drop=True)
    current_features = updated[FEATURE_NAMES].reset_index(drop=True)
    changed_mask = current_features.ne(previous_features).any(axis=1)

    updated.loc[changed_mask, "prediction"] = None
    updated[SIGNATURE_COLUMN] = updated.apply(build_signature, axis=1)
    return updated


def predict_missing_rows(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    updated = table.copy()
    predicted_count = 0
    for idx in updated.index:
        if pd.notna(updated.loc[idx, "prediction"]):
            continue
        features = [float(updated.loc[idx, feature]) for feature in FEATURE_NAMES]
        updated.loc[idx, "prediction"] = round(predict_price(features), 2)
        updated.loc[idx, SIGNATURE_COLUMN] = build_signature(updated.loc[idx])
        predicted_count += 1
    return updated, predicted_count


ensure_table_initialized()

st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] [role="columnheader"]:last-child,
    div[data-testid="stDataFrame"] [role="gridcell"]:last-child {
        position: sticky;
        right: 0;
        z-index: 5;
        background: var(--background-color);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.subheader("Prediction table")
controls_left, controls_right = st.columns([1, 1])
with controls_left:
    st.button("Add row", on_click=add_default_row, use_container_width=True)
with controls_right:
    predict_clicked = st.button("Predict missing", use_container_width=True)

edited_table = st.data_editor(
    st.session_state.prediction_table[TABLE_COLUMNS],
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    disabled=["prediction"],
    column_config={
        "area_m2": st.column_config.NumberColumn("Area (m2)", min_value=10.0, max_value=500.0, step=1.0),
        "rooms": st.column_config.NumberColumn("Rooms", min_value=1.0, max_value=10.0, step=1.0),
        "floor": st.column_config.NumberColumn("Floor", min_value=1.0, max_value=50.0, step=1.0),
        "building_age": st.column_config.NumberColumn("Building age (years)", min_value=0.0, max_value=150.0, step=1.0),
        "prediction": st.column_config.NumberColumn("Prediction", format="%.2f"),
    },
)

table_with_meta = edited_table.copy()
table_with_meta[SIGNATURE_COLUMN] = st.session_state.prediction_table[SIGNATURE_COLUMN].values
table_with_meta = clear_predictions_for_changed_rows(table_with_meta)

if predict_clicked:
    try:
        table_with_meta, predicted_count = predict_missing_rows(table_with_meta)
        st.session_state.prediction_table = table_with_meta
        st.success(f"Predicted rows: {predicted_count}")
        st.rerun()
    except FileNotFoundError:
        model_path = Path(__file__).parent / "model" / "linear_model.pkl"
        st.error(f"Model file is missing: {model_path}")
    except Exception as exc:  # pragma: no cover - defensive UI branch
        st.error(f"Inference failed: {exc}")

st.session_state.prediction_table = table_with_meta

with st.expander("Expected features"):
    st.write(FEATURE_NAMES)
