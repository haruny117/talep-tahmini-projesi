import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Talep Tahmini & Stok Optimizasyonu", layout="wide")

st.title("📦 Perakende Talep Tahmini & Stok Optimizasyonu")
st.caption("Rossmann Store Sales verisi üzerinde XGBoost tabanlı talep tahmini ve stok optimizasyonu")

@st.cache_data
def load_data():
    predictions = pd.read_csv("test_predictions.csv", parse_dates=["Date"])
    stock_plan = pd.read_csv("stock_plan.csv")
    return predictions, stock_plan

predictions, stock_plan = load_data()

st.sidebar.header("Filtreler")
store_list = sorted(predictions["Store"].unique())
selected_store = st.sidebar.selectbox("Mağaza seç", store_list)

tab1, tab2 = st.tabs(["📈 Talep Tahmini", "📊 Stok Planı"])

with tab1:
    st.subheader(f"Mağaza {selected_store} — Gerçek vs Tahmin Edilen Satış")

    store_data = predictions[predictions["Store"] == selected_store].sort_values("Date")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(store_data["Date"], store_data["Sales"], label="Gerçek Satış", marker="o")
    ax.plot(store_data["Date"], store_data["XGB_Pred"], label="Model Tahmini", marker="x")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Satış")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    mape = (abs(store_data["Sales"] - store_data["XGB_Pred"]) / store_data["Sales"].replace(0, pd.NA)).mean() * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("Ortalama Gerçek Satış", f"{store_data['Sales'].mean():.0f}")
    col2.metric("Ortalama Tahmin", f"{store_data['XGB_Pred'].mean():.0f}")
    col3.metric("MAPE", f"%{mape:.1f}")

with tab2:
    st.subheader(f"Mağaza {selected_store} — Stok Önerisi")

    store_stock = stock_plan[stock_plan["Store"] == selected_store].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Günlük Ortalama Talep", f"{store_stock['AvgDailyDemand']:.0f}")
    col2.metric("Model Tabanlı Reorder Point", f"{store_stock['ModelReorderPoint']:.0f}")
    col3.metric("Naive Tabanlı Reorder Point", f"{store_stock['NaiveReorderPoint']:.0f}",
                delta=f"{store_stock['ModelReorderPoint'] - store_stock['NaiveReorderPoint']:.0f}",
                delta_color="inverse")

    st.markdown("---")
    st.markdown("### Tüm Mağazalar — Reorder Point Karşılaştırması")
    st.dataframe(
        stock_plan.sort_values("AvgDailyDemand", ascending=False),
        use_container_width=True,
        height=400
    )

    savings_pct = (stock_plan["NaiveReorderPoint"] - stock_plan["ModelReorderPoint"]).sum() / stock_plan["NaiveReorderPoint"].sum() * 100
    st.info(f"📉 Model tabanlı yaklaşım, tüm mağazalar genelinde ortalama **%{savings_pct:.1f}** daha az reorder point (envanter yatırımı) gerektiriyor.")
    