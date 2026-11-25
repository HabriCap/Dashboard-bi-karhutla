import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# KONFIGURASI HALAMAN & GAYA MINIMALIS (FOREST GREEN PALETTE)
# ==============================================================================
st.set_page_config(
    page_title="Dashboard BI Karhutla, Gambut & Kualitas Udara Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY_GREEN = "#4C5C2D"
ACCENT_GREEN = "#6A7E3F"
ALERT_RED = "#C0392B"
NEUTRAL_GRAY = "#95A5A6"

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #2C351A; font-weight: 700; }
    div[data-testid="stMetric"] {
        background-color: #F7F9F4;
        border-left: 4px solid #4C5C2D;
        padding: 12px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# LOAD DATA MART (CACHED UNTUK PERFORMA CEPAT)
# ==============================================================================
DATA_DIR = "streamlit_data"

@st.cache_data
def load_all_data():
    prov = pd.read_parquet(os.path.join(DATA_DIR, "mart_provinsi_bulanan.parquet"))
    gambut = pd.read_parquet(os.path.join(DATA_DIR, "mart_gambut_bulanan.parquet"))
    top_khg = pd.read_parquet(os.path.join(DATA_DIR, "mart_top_khg.parquet"))
    peta = pd.read_parquet(os.path.join(DATA_DIR, "mart_peta_hotspot_sample.parquet"))
    ispu = pd.read_parquet(os.path.join(DATA_DIR, "mart_ispu_kesehatan.parquet"))
    pred = pd.read_parquet(os.path.join(DATA_DIR, "mart_prediksi_karhutla.parquet"))
    fi = pd.read_parquet(os.path.join(DATA_DIR, "mart_feature_importance.parquet"))
    
    prov["periode"] = pd.to_datetime(prov["tahun"].astype(str) + "-" + prov["bulan"].astype(str).str.zfill(2) + "-01")
    gambut["periode"] = pd.to_datetime(gambut["tahun"].astype(str) + "-" + gambut["bulan"].astype(str).str.zfill(2) + "-01")
    return prov, gambut, top_khg, peta, ispu, pred, fi

try:
    df_prov, df_gambut, df_top_khg, df_peta, df_ispu, df_pred, df_fi = load_all_data()
except Exception as e:
    st.error(f"Gagal memuat data dari folder '{DATA_DIR}/'. Pastikan semua file .parquet sudah di-push ke GitHub. Detail: {e}")
    st.stop()

# ==============================================================================
# SIDEBAR FILTER GLOBAL
# ==============================================================================
st.sidebar.title("Filter Analisis")
tahun_list = sorted(df_prov["tahun"].unique())
tahun_range = st.sidebar.select_slider(
    "Rentang Tahun Observasi",
    options=tahun_list,
    value=(min(tahun_list), max(tahun_list))
)

prov_list = ["Semua Provinsi"] + sorted(df_prov["provinsi_id"].dropna().unique().tolist())
selected_prov = st.sidebar.selectbox("Fokus Provinsi", prov_list)

enso_list = ["Semua Fase"] + sorted(df_prov["status_enso"].dropna().unique().tolist())
selected_enso = st.sidebar.selectbox("Status Iklim Global (ENSO)", enso_list)

# Terapkan Filter ke Dataset Provinsi
mask_prov = (df_prov["tahun"] >= tahun_range[0]) & (df_prov["tahun"] <= tahun_range[1])
if selected_prov != "Semua Provinsi":
    mask_prov &= (df_prov["provinsi_id"] == selected_prov)
if selected_enso != "Semua Fase":
    mask_prov &= (df_prov["status_enso"] == selected_enso)
df_prov_filt = df_prov[mask_prov].copy()

# ==============================================================================
# HEADER UTAMA
# ==============================================================================
st.title("Dashboard Business Intelligence: Anomali Iklim, Karhutla Gambut & Dampak Kesehatan")
st.caption("Analisis Integrasi Multi-Sumber: Indeks Iklim Global (ONI & DMI), Satelit NASA FIRMS, Peta KHG Nasional, dan Kualitas Udara SPKU.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Iklim & Tren Hotspot",
    "2. Kerentanan Lahan Gambut",
    "3. Kualitas Udara & Beban ISPA",
    "4. Peringatan Dini (Prediktif)"
])

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW & PEMICU IKLIM
# ==============================================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Titik Panas (SNPP)", f"{int(df_prov_filt['hotspots_snpp'].sum()):,}")
    c2.metric("Titik Panas High Confidence", f"{int(df_prov_filt['high_confidence_snpp'].sum()):,}")
    c3.metric("Total Daya Radiasi Api (FRP)", f"{df_prov_filt['frp_sum_snpp'].sum():,.0f} MW")
    c4.metric("Rata-rata Anomali ONI", f"{df_prov_filt['oni_anom'].mean():.2f} °C")

    st.markdown("---")
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.subheader("Korelasi Lonjakan Titik Panas dengan Anomali El Niño (ONI)")
        tren_iklim = df_prov_filt.groupby("periode", as_index=False).agg({
            "hotspots_snpp": "sum",
            "oni_anom": "mean",
            "dmi_value": "mean"
        })
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(
            go.Bar(x=tren_iklim["periode"], y=tren_iklim["hotspots_snpp"], name="Titik Panas", marker_color=ACCENT_GREEN, opacity=0.75),
            secondary_y=False
        )
        fig_dual.add_trace(
            go.Scatter(x=tren_iklim["periode"], y=tren_iklim["oni_anom"], name="Anomali ONI (°C)", line=dict(color=ALERT_RED, width=2.5)),
            secondary_y=True
        )
        fig_dual.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
        fig_dual.update_yaxes(title_text="Jumlah Titik Panas", secondary_y=False)
        fig_dual.update_yaxes(title_text="Indeks ONI (°C)", secondary_y=True)
        st.plotly_chart(fig_dual, use_container_width=True)

    with col_right:
        st.subheader("Top 10 Provinsi Rawan Karhutla")
        top10_p = df_prov_filt.groupby("provinsi_id", as_index=False)["hotspots_snpp"].sum().sort_values("hotspots_snpp", ascending=True).tail(10)
        fig_bar_p = px.bar(
            top10_p, x="hotspots_snpp", y="provinsi_id", orientation="h",
            labels={"hotspots_snpp": "Total Titik Panas", "provinsi_id": ""},
            color_discrete_sequence=[PRIMARY_GREEN]
        )
        fig_bar_p.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_bar_p, use_container_width=True)

# ==============================================================================
# TAB 2: KERENTANAN LAHAN GAMBUT (KHG)
# ==============================================================================
with tab2:
    mask_g = (df_gambut["tahun"] >= tahun_range[0]) & (df_gambut["tahun"] <= tahun_range[1])
    df_g_filt = df_gambut[mask_g].copy()
    df_g_filt["Tipe Lahan"] = df_g_filt["is_lahan_gambut"].map({1: "Lahan Gambut (KHG)", 0: "Non-Gambut"})

    total_all = df_g_filt["jumlah_hotspot"].sum()
    total_peat = df_g_filt[df_g_filt["is_lahan_gambut"] == 1]["jumlah_hotspot"].sum()
    pct_peat = (total_peat / total_all * 100) if total_all > 0 else 0

    g1, g2, g3 = st.columns(3)
    g1.metric("Titik Panas di Lahan Gambut", f"{int(total_peat):,}")
    g2.metric("Proporsi Kebakaran Gambut", f"{pct_peat:.2f}%")
    g3.metric("KHG Terparah (Kode)", df_top_khg.iloc[0]["kode_khg"] if not df_top_khg.empty else "-")

    st.markdown("---")
    c_g1, c_g2 = st.columns([1.3, 1])

    with c_g1:
        st.subheader("Perbandingan Tren Kebakaran: Lahan Gambut vs Non-Gambut")
        fig_area = px.area(
            df_g_filt, x="periode", y="jumlah_hotspot", color="Tipe Lahan",
            color_discrete_map={"Lahan Gambut (KHG)": PRIMARY_GREEN, "Non-Gambut": NEUTRAL_GRAY},
            labels={"periode": "Periode", "jumlah_hotspot": "Jumlah Titik Panas"}
        )
        fig_area.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_area, use_container_width=True)

    with c_g2:
        st.subheader("Top 10 Kode KHG Paling Rawan Kebakaran")
        fig_khg = px.bar(
            df_top_khg.head(10).sort_values("jumlah_hotspot", ascending=True),
            x="jumlah_hotspot", y="kode_khg", orientation="h",
            color_discrete_sequence=[ALERT_RED],
            labels={"jumlah_hotspot": "Akumulasi Titik Panas", "kode_khg": "Kode KHG"}
        )
        fig_khg.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_khg, use_container_width=True)

    st.subheader("Sebaran Spasial Titik Panas Terbaru (Sampel 2023–2026)")
    df_peta_plot = df_peta.copy()
    df_peta_plot["Status Lahan"] = df_peta_plot["is_lahan_gambut"].map({1: "Lahan Gambut", 0: "Non-Gambut"})
    fig_map = px.scatter_mapbox(
        df_peta_plot, lat="latitude", lon="longitude", color="Status Lahan",
        color_discrete_map={"Lahan Gambut": ALERT_RED, "Non-Gambut": ACCENT_GREEN},
        zoom=3.8, center={"lat": -2.2, "lon": 117.5}, height=420, opacity=0.65
    )
    fig_map.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# ==============================================================================
# TAB 3: KUALITAS UDARA (ISPU) & DAMPAK KESEHATAN ISPA
# ==============================================================================
with tab3:
    df_ispu_filt = df_ispu.copy()
    if selected_prov != "Semua Provinsi":
        df_ispu_filt = df_ispu_filt[df_ispu_filt["provinsi_id"] == selected_prov]

    i1, i2, i3 = st.columns(3)
    i1.metric("Rata-rata PM2.5", f"{df_ispu_filt['pm25_concentration_ugm3'].mean():.1f} µg/m³")
    i2.metric("Rata-rata Indeks ISPU", f"{df_ispu_filt['ispu_calculated_index'].mean():.1f}")
    i3.metric("Skor Proyeksi Beban ISPA", f"{df_ispu_filt['projected_ispa_clinic_burden_score'].mean():.2f} / 10")

    st.markdown("---")
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.subheader("Dampak Stagnasi Atmosfer & Kabut Asap terhadap PM2.5")
        df_ispu_filt["Status Kabut Asap"] = df_ispu_filt["wildfire_smoke_plume_flag"].map({True: "Terdeteksi Asap Karhutla", False: "Non-Asap Karhutla"})
        fig_scat = px.scatter(
            df_ispu_filt.sample(n=min(3000, len(df_ispu_filt)), random_state=42),
            x="atmospheric_stagnation_index", y="pm25_concentration_ugm3",
            color="Status Kabut Asap",
            color_discrete_map={"Terdeteksi Asap Karhutla": ALERT_RED, "Non-Asap Karhutla": ACCENT_GREEN},
            labels={"atmospheric_stagnation_index": "Indeks Stagnasi Atmosfer (0-10)", "pm25_concentration_ugm3": "Konsentrasi PM2.5 (µg/m³)"},
            opacity=0.65
        )
        fig_scat.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_scat, use_container_width=True)

    with col_i2:
        st.subheader("Rata-rata Skor Beban Faskes ISPA per Provinsi")
        prov_ispa = df_ispu.groupby("provinsi_id", as_index=False)["projected_ispa_clinic_burden_score"].mean().sort_values("projected_ispa_clinic_burden_score", ascending=True).tail(12)
        fig_ispa_bar = px.bar(
            prov_ispa, x="projected_ispa_clinic_burden_score", y="provinsi_id", orientation="h",
            color_discrete_sequence=[PRIMARY_GREEN],
            labels={"projected_ispa_clinic_burden_score": "Skor Beban Faskes ISPA (0-10)", "provinsi_id": ""}
        )
        fig_ispa_bar.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_ispa_bar, use_container_width=True)

# ==============================================================================
# TAB 4: PERINGATAN DINI & FEATURE IMPORTANCE (RANDOM FOREST)
# ==============================================================================
with tab4:
    st.subheader("Sistem Peringatan Dini Harian Risiko Karhutla (Data Uji 2024–2026)")
    p1, p2 = st.columns([1, 1.4])

    with p1:
        st.markdown("**Faktor Cuaca & Iklim Paling Berpengaruh**")
        fig_fi = px.bar(
            df_fi.sort_values("importance", ascending=True),
            x="importance", y="fitur", orientation="h",
            color_discrete_sequence=[ACCENT_GREEN],
            labels={"importance": "Feature Importance", "fitur": ""}
        )
        fig_fi.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_fi, use_container_width=True)

    with p2:
        st.markdown("**Daftar Peringatan Risiko Tinggi Terbaru (`Probabilitas >= 50%`)**")
        df_alert = df_pred[df_pred["status_prediksi"] == "Rawan Tinggi"].sort_values(["tanggal", "probabilitas_karhutla"], ascending=[False, False])
        if selected_prov != "Semua Provinsi":
            df_alert = df_alert[df_alert["provinsi_id"] == selected_prov]
        st.dataframe(
            df_alert[["tanggal", "provinsi_id", "probabilitas_karhutla", "hotspots_snpp", "curah_hujan", "oni_anom"]].head(100),
            use_container_width=True,
            height=380
        )
