from datetime import datetime
import joblib
import pandas as pd
import requests
import streamlit as st
from config import API_KEY, MANILA_ROADS, estimate_vehicle_volume

st.set_page_config(page_title="Manila Live Traffic Classifier", layout="wide")


@st.cache_resource
def load_artifacts():
    return joblib.load("congestion_model.pkl"), joblib.load("preprocessor.pkl")


def fetch_and_predict(model, preprocessor):
    records = []
    for road_name, lat, lon, lanes, cap_per_lane in MANILA_ROADS:
        url = (
            f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
            f"absolute/10/json?point={lat},{lon}&key={API_KEY}"
        )
        try:
            data = requests.get(url, timeout=10).json().get("flowSegmentData", {})
            curr_tt = data.get("currentTravelTime")
            curr_spd = data.get("currentSpeed")
            free_spd = data.get("freeFlowSpeed")

            if curr_tt and curr_spd and free_spd:
                volume = estimate_vehicle_volume(curr_spd, free_spd, lanes, cap_per_lane)
                records.append({
                    "Road_Name": road_name,
                    "Vehicle_Volume": volume,
                    "Travel_Time_Sec": curr_tt,
                    "Current_Speed_Kph": curr_spd,
                    "lat": lat,
                    "lon": lon,
                })
        except Exception as e:
            st.warning(f"Error fetching {road_name}: {e}")

    df = pd.DataFrame(records)
    if df.empty:
        return df

    X_live = preprocessor.transform(df[["Road_Name", "Vehicle_Volume", "Travel_Time_Sec"]])
    df["Prediction"] = model.predict(X_live)
    df["Congestion_Prob_%"] = (model.predict_proba(X_live)[:, 1] * 100).round(1)
    df["Status"] = df["Prediction"].map({1: "Yes (1) - Congested", 0: "No (0) - Clear"})
    df["color"] = df["Prediction"].map({1: "#FF4B4B", 0: "#00CC66"})
    df["size"] = df["Vehicle_Volume"] * 0.08
    return df


st.title("Manila Multi-Road Traffic Congestion Classification")
st.caption("Live Traffic Classification using Vehicle Volume & Travel Time")

model, preprocessor = load_artifacts()

with st.sidebar:
    st.header("Live Feed Controls")
    auto_refresh = st.toggle("Auto-Refresh (Every 30s)", value=False)
    st.button("Refresh Live Traffic Now", type="primary", use_container_width=True)

tab_live, tab_eval = st.tabs(["Live Manila Traffic Monitor", "Model Evaluation & Graphs"])

with tab_live:
    @st.fragment(run_every=30 if auto_refresh else None)
    def render_live():
        df_live = fetch_and_predict(model, preprocessor)
        if df_live.empty:
            st.error("Unable to fetch live data. Verify your TomTom API key in config.py.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Congested Roads (1)", f"{int((df_live['Prediction'] == 1).sum())} / {len(df_live)}")
        c2.metric("Clear Roads (0)", f"{int((df_live['Prediction'] == 0).sum())} / {len(df_live)}")
        c3.metric("Avg Travel Time", f"{int(df_live['Travel_Time_Sec'].mean())} sec")
        c4.metric("Last Updated", datetime.now().strftime("%I:%M:%S %p"))

        st.divider()
        map_col, table_col = st.columns([1.1, 1.2])

        with map_col:
            st.subheader("Live Manila Congestion Map")
            st.caption("Red = Yes (1) Congested | Green = No (0) Clear")
            st.map(df_live, latitude="lat", longitude="lon", color="color", size="size", zoom=11)

        with table_col:
            st.subheader("Live Road Predictions")
            cols = ["Road_Name", "Vehicle_Volume", "Travel_Time_Sec", "Current_Speed_Kph", "Congestion_Prob_%", "Status"]
            st.dataframe(
                df_live[cols].sort_values(by="Congestion_Prob_%", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    render_live()

with tab_eval:
    st.subheader("Model Evaluation Graphs")
    g1, g2 = st.columns(2)
    with g1:
        st.image("confusion_matrix.png", caption="Confusion Matrix")
    with g2:
        st.image("roc_curve.png", caption="ROC Curve")