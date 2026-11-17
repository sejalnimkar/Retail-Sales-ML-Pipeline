import pandas as pd
import streamlit as st

from src.components.data_ingestion import period_label
from src.pipeline.predict_pipeline import PredictPipeline

st.set_page_config(page_title="Retail Sales Forecaster", layout="centered")

st.title("Montgomery County Retail Sales Forecaster")
st.markdown(
    "Forecast an item's monthly retail sales from its own sales history. "
    "Pick a supplier and item, then the month to forecast."
)


@st.cache_resource
def load_pipeline() -> PredictPipeline:
    return PredictPipeline()


try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m src.pipeline.train_pipeline --data <path to csv>` first.")
    st.stop()

items = pipeline.items()

st.divider()
col1, col2 = st.columns(2)

with col1:
    supplier = st.selectbox("Supplier", options=sorted(items["SUPPLIER"].dropna().unique()))
    supplier_items = items[items["SUPPLIER"] == supplier].sort_values("ITEM DESCRIPTION")
    item_code = st.selectbox(
        "Item",
        options=supplier_items["ITEM CODE"],
        format_func=lambda code: (
            f"{supplier_items.set_index('ITEM CODE').loc[code, 'ITEM DESCRIPTION']} ({code})"
        ),
    )

history = pipeline.history(item_code)
last_period = int(history["LAST_PERIOD"])
data_end = int(history["DATA_END_PERIOD"])
# Offer the six months after the data ends.
choices = [data_end + i for i in range(1, 7)]

with col2:
    target = st.selectbox("Month to forecast", options=choices, format_func=period_label)
    st.caption(f"Last recorded month for this item: {period_label(last_period)}.")

st.divider()

if st.button("Forecast retail sales", use_container_width=True):
    prediction = pipeline.predict(item_code, year=target // 12, month=target % 12 + 1)
    st.success(f"**Forecast: {prediction:,.1f} units**")

st.subheader("What the forecast is based on")
st.dataframe(
    pd.DataFrame(
        {
            "Value": {
                "Item type": history["ITEM TYPE"],
                "Retail sales, last recorded month (units)": f"{history['PREV_RETAIL_SALES']:,.1f}",
                "Warehouse sales, last recorded month (units)": f"{history['PREV_WAREHOUSE_SALES']:,.1f}",
                "Average monthly retail sales (units)": f"{history['ITEM_MEAN_RETAIL_PRIOR']:,.1f}",
                "Months of history": int(history["ITEM_MONTHS_OBSERVED"]),
            }
        }
    ),
    use_container_width=True,
)
st.caption("Model accuracy on held-out months is reported in reports/results.md.")
