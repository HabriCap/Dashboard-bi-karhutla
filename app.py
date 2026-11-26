import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
# KONFIGURASI HALAMAN & GAYA
# ==============================================================================
st.set_page_config(
    page_title="Dashboard BI Karhutla, Gambut & Kualitas Udara Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)

PRIMARY_GREEN = "#4C5C2D"
ACCENT_GREEN = "#6A7E3F"
ALERT_RED = "#C84B31"
NEUTRAL_GRAY = "#95A5A6"

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #2C351A; font-weight: 700; }
    div[data-testid="stMetric"],
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #FFFFFF !important;
    }
    div[data-testid="stMetric"] {
        background-color: #1E2124;
        border: 1px solid #6A7E3F;
        border-left: 4px solid #6A7E3F;
        padding: 12px 16px;
        border-radius: 6px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: #A3A8B8 !important;
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
    frp_sample = pd.read_parquet(os.path.join(DATA_DIR, "mart_frp_sample.parquet"))
    ispu = pd.read_parquet(os.path.join(DATA_DIR, "mart_ispu_kesehatan.parquet"))
    pred = pd.read_parquet(os.path.join(DATA_DIR, "mart_prediksi_karhutla.parquet"))
    fi = pd.read_parquet(os.path.join(DATA_DIR, "mart_feature_importance.parquet"))
    
    prov["periode"] = pd.to_datetime(prov["tahun"].astype(str) + "-" + prov["bulan"].astype(str).str.zfill(2) + "-01")
    gambut["periode"] = pd.to_datetime(gambut["tahun"].astype(str) + "-" + gambut["bulan"].astype(str).str.zfill(2) + "-01")
    return prov, gambut, top_khg, peta, frp_sample, ispu, pred, fi


def show_visual_guide(title, terms):
    with st.expander(f"Panduan membaca: {title}"):
        st.table(pd.DataFrame(terms, columns=["Istilah / elemen", "Penjelasan"]))


try:
    df_prov, df_gambut, df_top_khg, df_peta, df_frp_sample, df_ispu, df_pred, df_fi = load_all_data()
except Exception as e:
    st.error(f"Gagal memuat data dari folder '{DATA_DIR}/'. Pastikan semua file .parquet sudah di-push ke GitHub. Detail: {e}")
    st.stop()

# ==============================================================================
# SIDEBAR FILTER GLOBAL
# ==============================================================================
st.sidebar.title("Filter Data")
tahun_list = sorted(
    set(df_prov["tahun"].unique()).union(df_peta["tahun"].unique())
)
tahun_range = st.sidebar.select_slider(
    "Rentang tahun",
    options=tahun_list,
    value=(min(tahun_list), max(tahun_list))
)

prov_list = ["Semua Provinsi"] + sorted(df_prov["provinsi_id"].dropna().unique().tolist())
selected_prov = st.sidebar.selectbox("Provinsi", prov_list)

enso_list = ["Semua Fase"] + sorted(df_prov["status_enso"].dropna().unique().tolist())
selected_enso = st.sidebar.selectbox("Fase ENSO (suhu laut Pasifik)", enso_list)
st.sidebar.caption(
    "Data hotspot mencakup 2016–2026; data iklim provinsi tersedia hingga 2025."
)

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
st.title("Dashboard Karhutla: Hotspot, Gambut, Kualitas Udara, dan Risiko")
st.caption("Atur rentang tahun, provinsi, dan fase ENSO melalui filter di sebelah kiri untuk menyesuaikan analisis.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Hotspot & Iklim",
    "2. Hotspot di Lahan Gambut",
    "3. Kualitas Udara & Kesehatan",
    "4. Prediksi Risiko"
])

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW & PEMICU IKLIM
# ==============================================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total hotspot terdeteksi (SNPP)", f"{int(df_prov_filt['hotspots_snpp'].sum()):,}")
    c2.metric("Hotspot dengan keyakinan tinggi (confidence)", f"{int(df_prov_filt['high_confidence_snpp'].sum()):,}")
    c3.metric("Total fire radiative power (FRP)", f"{df_prov_filt['frp_sum_snpp'].sum():,.0f} MW")
    c4.metric("Rata-rata anomali ONI", f"{df_prov_filt['oni_anom'].mean():.2f} °C")
    show_visual_guide("angka ringkasan", [
        ("Hotspot (titik panas)", "Anomali suhu yang dideteksi satelit dan dapat mengindikasikan kebakaran. Hotspot bukan pengukuran luas area terbakar."),
        ("Confidence tinggi", "Deteksi dengan tingkat keyakinan lebih tinggi berdasarkan karakteristik sinyal satelit."),
        ("Fire radiative power (FRP)", "Estimasi daya radiasi dari sumber panas dalam megawatt (MW); nilai lebih tinggi menunjukkan sinyal panas lebih kuat."),
        ("Anomali ONI", "Penyimpangan suhu permukaan laut Pasifik dari kondisi normal; berkaitan dengan variasi iklim seperti El Niño dan La Niña."),
    ])

    st.markdown("---")
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.subheader("Tren anomali suhu Pasifik dan jumlah hotspot")
        tren_iklim = df_prov_filt.groupby("periode", as_index=False).agg({
            "hotspots_snpp": "sum",
            "oni_anom": "mean",
            "dmi_value": "mean"
        })
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(
            go.Bar(x=tren_iklim["periode"], y=tren_iklim["hotspots_snpp"], name="Jumlah hotspot", marker_color=ACCENT_GREEN, opacity=0.75),
            secondary_y=False
        )
        fig_dual.add_trace(
            go.Scatter(x=tren_iklim["periode"], y=tren_iklim["oni_anom"], name="Anomali ONI (°C)", line=dict(color=ALERT_RED, width=2.5)),
            secondary_y=True
        )
        fig_dual.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
        fig_dual.update_yaxes(title_text="Jumlah hotspot", secondary_y=False)
        fig_dual.update_yaxes(title_text="Anomali ONI (°C)", secondary_y=True)
        st.plotly_chart(fig_dual, use_container_width=True)
        show_visual_guide("grafik cuaca dan titik panas", [
            ("Sumbu mendatar", "Bulan dan tahun pengamatan."),
            ("Batang hijau", "Jumlah hotspot yang terdeteksi pada bulan tersebut."),
            ("Garis merah", "Anomali suhu permukaan laut Pasifik yang dirangkum dalam indeks ONI."),
            ("Cara membaca", "Lihat apakah jumlah titik panas dan garis suhu naik pada waktu yang sama. Ini menunjukkan hubungan waktu, bukan bukti bahwa satu hal pasti menyebabkan yang lain."),
        ])

    with col_right:
        st.subheader("10 provinsi dengan jumlah hotspot tertinggi")
        top10_p = df_prov_filt.groupby("provinsi_id", as_index=False)["hotspots_snpp"].sum().sort_values("hotspots_snpp", ascending=True).tail(10)
        fig_bar_p = px.bar(
            top10_p, x="hotspots_snpp", y="provinsi_id", orientation="h",
            labels={"hotspots_snpp": "Jumlah hotspot", "provinsi_id": "Provinsi"},
            color_discrete_sequence=[PRIMARY_GREEN], template="plotly_dark"
        )
        fig_bar_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_bar_p, use_container_width=True)
        show_visual_guide("peringkat provinsi", [
            ("Nama provinsi", "Wilayah yang dibandingkan."),
            ("Panjang batang", "Jumlah hotspot yang tercatat; batang lebih panjang menunjukkan lebih banyak deteksi."),
            ("Catatan", "Peringkat berdasarkan jumlah deteksi, bukan luas area terbakar atau tingkat risiko."),
        ])

    st.markdown("---")
    climate_col, rain_col = st.columns(2)
    with climate_col:
        st.subheader("Estimasi hotspot harian menurut fase ENSO dan IOD")
        if {"status_enso", "status_iod", "periode"}.issubset(df_prov_filt.columns):
            hotspot_col = "jumlah_hotspot" if "jumlah_hotspot" in df_prov_filt.columns else (
                "hotspots_snpp" if "hotspots_snpp" in df_prov_filt.columns else None
            )
            if hotspot_col:
                daily_data = df_prov_filt.copy()
                daily_data["rata_hotspot_harian"] = (
                    daily_data[hotspot_col] / daily_data["periode"].dt.days_in_month
                )
                daily_summary = daily_data.groupby(
                    ["status_enso", "status_iod"], as_index=False
                )["rata_hotspot_harian"].mean()
                enso_order = ["El Nino", "Netral", "La Nina"]
                iod_order = ["Negative IOD", "Netral", "Positive IOD"]
                daily_summary["status_enso"] = pd.Categorical(
                    daily_summary["status_enso"], categories=enso_order, ordered=True
                )
                daily_summary = daily_summary.sort_values("status_enso")
                fig_enso_iod = px.bar(
                    daily_summary, x="status_enso", y="rata_hotspot_harian",
                    color="status_iod", barmode="group",
                    category_orders={"status_enso": enso_order, "status_iod": iod_order},
                    color_discrete_map={
                        "Negative IOD": NEUTRAL_GRAY,
                        "Netral": ACCENT_GREEN,
                        "Positive IOD": ALERT_RED,
                    },
                    labels={
                        "status_enso": "Fase ENSO",
                        "rata_hotspot_harian": "Estimasi hotspot per hari",
                        "status_iod": "Fase IOD",
                    },
                    template="plotly_dark",
                )
                fig_enso_iod.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=380, margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_enso_iod, use_container_width=True)
                st.caption(
                    "Perkiraan per hari dibuat dari jumlah bulanan dibagi jumlah hari dalam bulan. Data sumber hanya tersedia per bulan."
                )
                show_visual_guide("estimasi hotspot menurut fase iklim", [
                    ("Fase ENSO", "Klasifikasi kondisi suhu permukaan laut di Samudra Pasifik: El Niño, netral, atau La Niña."),
                    ("Fase IOD", "Klasifikasi kondisi suhu permukaan laut di Samudra Hindia: negatif, netral, atau positif."),
                    ("Tinggi batang", "Estimasi rata-rata hotspot per hari untuk kombinasi fase ENSO dan IOD tersebut."),
                    ("Warna", "Membedakan fase IOD pada setiap kelompok ENSO."),
                ])
            else:
                st.info("Kolom jumlah hotspot tidak tersedia pada data terfilter.")
        else:
            st.info("Kolom ENSO, IOD, atau periode tidak tersedia pada data terfilter.")

    with rain_col:
        st.subheader("Curah hujan dan jumlah hotspot per provinsi")
        hotspot_col = "jumlah_hotspot" if "jumlah_hotspot" in df_prov_filt.columns else (
            "hotspots_snpp" if "hotspots_snpp" in df_prov_filt.columns else None
        )
        scatter_cols = {"provinsi_id", "status_enso", "curah_hujan"}
        if scatter_cols.issubset(df_prov_filt.columns) and hotspot_col:
            scatter_summary = df_prov_filt.groupby(
                ["provinsi_id", "status_enso"], as_index=False
            ).agg(
                curah_hujan=("curah_hujan", "mean"),
                total_hotspot=(hotspot_col, "sum"),
            )
            fig_rain = px.scatter(
                scatter_summary, x="curah_hujan", y="total_hotspot",
                color="status_enso", hover_name="provinsi_id",
                hover_data={"provinsi_id": True, "total_hotspot": True},
                color_discrete_map={
                    "El Nino": ALERT_RED, "Netral": NEUTRAL_GRAY, "La Nina": ACCENT_GREEN
                },
                labels={
                    "curah_hujan": "Curah hujan rata-rata",
                    "total_hotspot": "Jumlah hotspot",
                    "status_enso": "Fase ENSO",
                    "provinsi_id": "Provinsi",
                },
                template="plotly_dark",
            )
            fig_rain.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_rain, use_container_width=True)
            show_visual_guide("curah hujan dan hotspot", [
                ("Setiap titik", "Satu provinsi pada satu fase ENSO."),
                ("Sumbu X", "Rata-rata curah hujan; posisi lebih ke kanan menunjukkan curah hujan lebih tinggi."),
                ("Sumbu Y", "Jumlah hotspot yang tercatat; posisi lebih tinggi menunjukkan lebih banyak deteksi."),
                ("Warna", "Menunjukkan fase ENSO."),
                ("Detail saat diarahkan", "Menampilkan nama provinsi dan jumlah hotspot."),
            ])
        else:
            st.info("Kolom provinsi, ENSO, curah hujan, atau hotspot tidak tersedia pada data terfilter.")

# ==============================================================================
# TAB 2: KERENTANAN LAHAN GAMBUT (KHG)
# ==============================================================================
def filter_tab2_data(data):
    filtered = data[
        data["tahun"].between(tahun_range[0], tahun_range[1])
    ].copy()
    if selected_prov != "Semua Provinsi":
        filtered = filtered[filtered["provinsi_id"] == selected_prov]
    if selected_enso != "Semua Fase":
        filtered = filtered[filtered["status_enso"] == selected_enso]
    return filtered


with tab2:
    st.caption(
        "Seluruh visualisasi tab ini mengikuti filter rentang tahun, provinsi, dan fase ENSO."
    )
    if selected_enso != "Semua Fase" and tahun_range[1] >= 2026:
        st.caption(
            "Data tahun 2026 belum memiliki klasifikasi ENSO; data tahun tersebut hanya "
            "muncul saat fase ENSO diatur ke Semua Fase."
        )
    df_g_filt = filter_tab2_data(df_gambut)
    df_g_filt["Tipe Lahan"] = df_g_filt["is_lahan_gambut"].map({1: "Di gambut", 0: "Di luar gambut"})

    total_all = df_g_filt["jumlah_hotspot"].sum()
    total_peat = df_g_filt[df_g_filt["is_lahan_gambut"] == 1]["jumlah_hotspot"].sum()
    pct_peat = (total_peat / total_all * 100) if total_all > 0 else 0
    df_top_khg_filt = filter_tab2_data(df_top_khg)
    top_khg_summary = (
        df_top_khg_filt.groupby("kode_khg", as_index=False)["jumlah_hotspot"]
        .sum()
        .sort_values("jumlah_hotspot", ascending=False)
    )

    g1, g2, g3 = st.columns(3)
    g1.metric("Hotspot terdeteksi di lahan gambut", f"{int(total_peat):,}")
    g2.metric("Proporsi hotspot di gambut", f"{pct_peat:.2f}%")
    g3.metric(
        "KHG dengan jumlah hotspot tertinggi",
        top_khg_summary.iloc[0]["kode_khg"] if not top_khg_summary.empty else "-",
    )
    show_visual_guide("angka ringkasan lahan gambut", [
        ("Lahan gambut (KHG)", "Lahan gambut yang berada di dalam peta Kesatuan Hidrologis Gambut (KHG)."),
        ("Proporsi hotspot di gambut", "Persentase hotspot yang terdeteksi di lahan gambut dibandingkan seluruh hotspot dalam data."),
        ("Kode KHG", "Kode untuk mengenali satu wilayah bentang lahan gambut."),
    ])

    st.markdown("---")
    c_g1, c_g2 = st.columns([1.3, 1])

    with c_g1:
        st.subheader("Tren jumlah hotspot di lahan gambut dan non-gambut")
        if not df_g_filt.empty:
            trend_by_month = df_g_filt.groupby(
                ["periode", "Tipe Lahan"], as_index=False
            )["jumlah_hotspot"].sum()
            fig_area = px.area(
                trend_by_month, x="periode", y="jumlah_hotspot", color="Tipe Lahan",
                color_discrete_map={"Di gambut": PRIMARY_GREEN, "Di luar gambut": NEUTRAL_GRAY},
                labels={"periode": "Periode", "jumlah_hotspot": "Jumlah hotspot"},
                template="plotly_dark"
            )
            fig_area.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_area, use_container_width=True)
        else:
            st.info("Tidak ada data hotspot untuk kombinasi filter yang dipilih.")
        show_visual_guide("tren hotspot menurut kategori lahan", [
            ("Sumbu X", "Bulan dan tahun pengamatan."),
            ("Sumbu Y", "Jumlah hotspot yang terdeteksi."),
            ("Warna hijau / abu-abu", "Membedakan hotspot di lahan gambut dan non-gambut."),
            ("Area berwarna", "Menunjukkan perubahan jumlah dari waktu ke waktu; nilai setiap kategori dijumlahkan per bulan."),
        ])

    with c_g2:
        st.subheader("10 wilayah KHG dengan jumlah hotspot tertinggi")
        if not top_khg_summary.empty:
            fig_khg = px.bar(
                top_khg_summary.head(10).sort_values("jumlah_hotspot", ascending=True),
                x="jumlah_hotspot", y="kode_khg", orientation="h",
                color_discrete_sequence=[ALERT_RED],
                labels={"jumlah_hotspot": "Jumlah hotspot", "kode_khg": "Kode KHG"},
                template="plotly_dark"
            )
            fig_khg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_khg, use_container_width=True)
        else:
            st.info("Tidak ada data wilayah KHG untuk kombinasi filter yang dipilih.")
        show_visual_guide("peringkat wilayah gambut", [
            ("Kode wilayah", "Identitas wilayah gambut pada peta KHG."),
            ("Panjang batang", "Jumlah hotspot pada wilayah tersebut; batang lebih panjang menunjukkan lebih banyak deteksi."),
            ("Catatan", "Jumlah hotspot tidak secara langsung menunjukkan luas area terbakar."),
        ])

    st.markdown("---")
    peat_col, insight_col = st.columns(2)
    with peat_col:
        st.subheader("Distribusi fire radiative power (FRP) menurut kategori lahan")
        df_frp = filter_tab2_data(df_frp_sample)
        if {"is_lahan_gambut", "frp"}.issubset(df_frp.columns):
            df_frp["Tipe Lahan"] = df_frp["is_lahan_gambut"].map(
                {1: "Di gambut", 0: "Di luar gambut"}
            )
            df_frp["frp"] = pd.to_numeric(df_frp["frp"], errors="coerce")
            df_frp = df_frp.dropna(subset=["Tipe Lahan", "frp"])
            if not df_frp.empty:
                fig_frp = px.box(
                    df_frp, x="Tipe Lahan", y="frp", color="Tipe Lahan",
                    points=False,
                    color_discrete_map={
                        "Di gambut": ACCENT_GREEN, "Di luar gambut": NEUTRAL_GRAY
                    },
                    labels={"frp": "Fire radiative power (MW)", "Tipe Lahan": "Kategori lahan"},
                    template="plotly_dark",
                )
                fig_frp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=380, showlegend=False, margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_frp, use_container_width=True)
                st.caption(
                    "Boxplot memakai sampel acak hingga 50.000 titik pada filter aktif. "
                    "Rata-rata FRP menggunakan agregat seluruh data."
                )
                show_visual_guide("distribusi FRP menurut kategori lahan", [
                    ("FRP (MW)", "Estimasi daya radiasi api dari satelit; MW (megawatt) adalah satuan pengukurannya."),
                    ("Garis di dalam kotak", "Median: sekitar separuh observasi berada di bawah dan separuh di atas nilai ini."),
                    ("Kotak", "Rentang antarkuartil yang merangkum 50% observasi di bagian tengah distribusi."),
                    ("Perbandingan", "Kotak atau median yang lebih tinggi menunjukkan nilai FRP yang cenderung lebih besar."),
                ])
            else:
                st.info("Tidak ada observasi FRP untuk kombinasi filter yang dipilih.")
        else:
            st.info("Kolom FRP atau kategori lahan tidak tersedia pada data sampel.")

    with insight_col:
        st.subheader("Hubungan proporsi hotspot gambut dengan total hotspot")
        peat_by_month = df_g_filt.pivot_table(
            index="periode", columns="is_lahan_gambut",
            values="jumlah_hotspot", aggfunc="sum", fill_value=0,
        )
        if 1 in peat_by_month.columns and 0 in peat_by_month.columns:
            monthly_total = peat_by_month[1] + peat_by_month[0]
            peat_share = peat_by_month[1].div(monthly_total.replace(0, np.nan))
            paired = pd.DataFrame(
                {"proporsi_gambut": peat_share, "total_hotspot": monthly_total}
            ).dropna()
            corr = (
                paired["proporsi_gambut"].corr(paired["total_hotspot"])
                if len(paired) > 1 else np.nan
            )
            if pd.notna(corr):
                strength = "lemah" if abs(corr) < 0.3 else (
                    "sedang" if abs(corr) < 0.7 else "kuat"
                )
                direction = "positif" if corr >= 0 else "negatif"
                st.metric("Korelasi proporsi hotspot gambut dan total hotspot", f"{corr:.2f}")
                st.caption(
                    f"Hubungannya {direction} dan {strength} pada {len(paired)} bulan yang tersedia. "
                    "Ini menunjukkan dua hal bergerak bersama, bukan bukti sebab-akibat."
                )
                show_visual_guide("angka hubungan gambut dan titik panas", [
                    ("Angka mendekati +1", "Keduanya cenderung naik atau turun bersama."),
                    ("Angka mendekati 0", "Tidak terlihat pola gerak bersama yang jelas."),
                    ("Angka mendekati -1", "Saat satu naik, yang lain cenderung turun."),
                    ("Batas data", "Perhitungan memakai agregat bulanan untuk provinsi, rentang tahun, dan fase ENSO sesuai filter aktif. Korelasi menunjukkan asosiasi, bukan sebab-akibat."),
                ])
            else:
                st.info("Data bulanan tidak cukup untuk menghitung korelasi.")
        else:
            st.info("Kedua kategori lahan harus tersedia untuk menghitung korelasi.")

        frp_summary = df_g_filt.groupby("is_lahan_gambut").agg(
            jumlah_observasi_frp=("frp_observations", "sum"),
            total_frp=("total_frp", "sum"),
        )
        frp_means = frp_summary["total_frp"].div(
            frp_summary["jumlah_observasi_frp"].replace(0, np.nan)
        )
        frp_gambut = frp_means.get(1, np.nan)
        frp_non_gambut = frp_means.get(0, np.nan)
        metric_cols = st.columns(2)
        metric_cols[0].metric(
            "Rata-rata kekuatan panas di gambut",
            f"{frp_gambut:.2f} MW" if pd.notna(frp_gambut) else "N/A",
        )
        metric_cols[1].metric(
            "Rata-rata kekuatan panas di luar gambut",
            f"{frp_non_gambut:.2f} MW" if pd.notna(frp_non_gambut) else "N/A",
        )
        st.info(
            "Data partikel halus di udara (PM2.5) tidak menunjukkan apakah pengukurannya berasal "
            "dari area gambut atau bukan. Karena itu, perbandingan PM2.5 antara kedua jenis lahan "
            "belum bisa dihitung dengan benar."
        )
        show_visual_guide("rata-rata kekuatan panas", [
            ("Nilai gambut", "Rata-rata FRP dari seluruh observasi yang berada di gambut pada filter aktif."),
            ("Nilai luar gambut", "Rata-rata FRP dari seluruh observasi yang berada di luar gambut pada filter aktif."),
            ("MW", "Satuan kekuatan panas yang terdeteksi satelit."),
            ("Cakupan", "Rata-rata dihitung dari agregat seluruh data; hanya boxplot distribusi yang memakai sampel acak."),
        ])
    st.subheader("Peta persebaran hotspot pada lahan gambut dan non-gambut")
    df_peta_plot = filter_tab2_data(df_peta)
    df_peta_plot["Status Lahan"] = df_peta_plot["is_lahan_gambut"].map(
        {1: "Lahan gambut", 0: "Non-gambut"}
    )
    if not df_peta_plot.empty:
        fig_map = px.scatter_map(
            df_peta_plot, lat="latitude", lon="longitude", color="Status Lahan",
            hover_data={
                "tahun": True,
                "bulan": True,
                "frp": ":.2f",
                "kode_khg": True,
                "latitude": False,
                "longitude": False,
            },
            labels={
                "tahun": "Tahun",
                "bulan": "Bulan",
                "frp": "FRP (MW)",
                "kode_khg": "Kode KHG",
                "Status Lahan": "Kategori lahan",
            },
            color_discrete_map={"Lahan gambut": ALERT_RED, "Non-gambut": ACCENT_GREEN},
            zoom=3.8, center={"lat": -2.2, "lon": 117.5}, height=420, opacity=0.65
        )
        fig_map.update_layout(map_style="carto-positron", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption(
            f"Peta menampilkan {len(df_peta_plot):,} titik dari sampel berstrata berdasarkan provinsi, "
            "tahun, bulan, dan kategori lahan. Semua filter aktif diterapkan; jumlah marker bukan "
            "jumlah total hotspot."
        )
        show_visual_guide("peta persebaran hotspot", [
            ("Setiap marker", "Satu observasi hotspot dari sampel data; bukan jumlah seluruh kejadian."),
            ("Warna merah", "Lokasi observasi yang diklasifikasikan berada di lahan gambut."),
            ("Warna hijau", "Lokasi observasi yang diklasifikasikan berada di luar lahan gambut."),
            ("FRP (MW)", "Estimasi daya radiasi api pada observasi tersebut; bukan luas area terbakar."),
            ("Batas interpretasi", "Peta menunjukkan lokasi sampel. Marker yang berdekatan belum tentu berasal dari satu kejadian kebakaran yang sama."),
        ])
    else:
        st.info("Tidak ada observasi hotspot pada rentang tahun yang dipilih.")

# ==============================================================================
# TAB 3: KUALITAS UDARA (ISPU) & DAMPAK KESEHATAN ISPA
# ==============================================================================
with tab3:
    df_ispu_period = df_ispu[
        df_ispu["tahun"].between(tahun_range[0], tahun_range[1])
    ].copy()
    df_ispu_filt = df_ispu_period.copy()
    if selected_prov != "Semua Provinsi":
        df_ispu_filt = df_ispu_filt[df_ispu_filt["provinsi_id"] == selected_prov]

    def format_ispu_mean(column, decimals, suffix=""):
        value = df_ispu_filt[column].mean()
        return f"{value:.{decimals}f}{suffix}" if pd.notna(value) else "N/A"

    i1, i2, i3 = st.columns(3)
    i1.metric("Rata-rata konsentrasi PM2.5", format_ispu_mean("pm25_concentration_ugm3", 1, " µg/m³"))
    i2.metric("Rata-rata indeks kualitas udara", format_ispu_mean("ispu_calculated_index", 1))
    i3.metric("Estimasi skor beban layanan ISPA", format_ispu_mean("projected_ispa_clinic_burden_score", 2, " / 10"))
    show_visual_guide("angka ringkasan udara dan kesehatan", [
        ("Partikulat halus (PM2.5)", "Partikel sangat kecil dari debu atau asap yang dapat masuk jauh ke paru-paru. Nilai lebih tinggi berarti konsentrasi terukur lebih besar."),
        ("Indeks kualitas udara", "Indikator ringkas kondisi pencemaran udara; nilai lebih tinggi berarti kualitas udara lebih buruk."),
        ("Skor beban layanan ISPA", "Estimasi kebutuhan layanan kesehatan gangguan pernapasan pada skala 0–10, bukan jumlah pasien."),
    ])

    st.markdown("---")
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.subheader("Indeks stagnasi atmosfer, asap karhutla, dan konsentrasi PM2.5")
        if not df_ispu_filt.empty:
            df_ispu_plot = df_ispu_filt.copy()
            df_ispu_plot["Status Kabut Asap"] = df_ispu_plot[
                "wildfire_smoke_plume_flag"
            ].map({True: "Terdeteksi Asap Karhutla", False: "Non-Asap Karhutla"})
            fig_scat = px.scatter(
                df_ispu_plot.sample(n=min(3000, len(df_ispu_plot)), random_state=42),
                x="atmospheric_stagnation_index", y="pm25_concentration_ugm3",
                color="Status Kabut Asap",
                color_discrete_map={"Terdeteksi Asap Karhutla": ALERT_RED, "Non-Asap Karhutla": ACCENT_GREEN},
                labels={"atmospheric_stagnation_index": "Indeks stagnasi atmosfer (0–10)", "pm25_concentration_ugm3": "Konsentrasi PM2.5 (µg/m³)"},
                opacity=0.65
            )
            fig_scat.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_scat, use_container_width=True)
            show_visual_guide("stagnasi atmosfer dan konsentrasi PM2.5", [
                ("Setiap titik", "Satu pengamatan kualitas udara."),
                ("Sumbu X", "Indeks stagnasi atmosfer; nilai lebih tinggi menunjukkan pergerakan udara lebih terbatas."),
                ("Sumbu Y", "Konsentrasi partikulat halus PM2.5 yang terukur."),
                ("Merah / hijau", "Pengamatan dengan / tanpa indikasi plume asap karhutla."),
                ("Interpretasi", "Bandingkan konsentrasi PM2.5 pada tingkat stagnasi atau kelompok asap yang berbeda. Pola ini tidak membuktikan hubungan sebab-akibat."),
            ])
        else:
            st.info("Tidak ada data kualitas udara untuk filter yang dipilih.")

    with col_i2:
        st.subheader("Peringkat hari dengan indikasi asap karhutla")
        if not df_ispu_period.empty:
            daily_smoke = (
                df_ispu_period.groupby(["provinsi_id", "tanggal"], as_index=False)
                .agg(indikasi_asap=("wildfire_smoke_plume_flag", "any"))
            )
            prov_smoke = (
                daily_smoke.groupby("provinsi_id", as_index=False)
                .agg(
                    hari_terindikasi_asap=("indikasi_asap", "sum"),
                    hari_diamati=("tanggal", "size"),
                )
            )
            prov_smoke["persentase_hari_asap"] = (
                prov_smoke["hari_terindikasi_asap"]
                .div(prov_smoke["hari_diamati"])
                .mul(100)
            )
            prov_smoke = (
                prov_smoke.sort_values("persentase_hari_asap", ascending=True)
                .tail(12)
            )
            fig_smoke_rank = px.bar(
                prov_smoke,
                x="persentase_hari_asap",
                y="provinsi_id",
                orientation="h",
                color_discrete_sequence=[ALERT_RED],
                hover_data={
                    "hari_terindikasi_asap": True,
                    "hari_diamati": True,
                    "persentase_hari_asap": ":.1f",
                },
                labels={
                    "persentase_hari_asap": "Hari dengan indikasi asap (%)",
                    "provinsi_id": "Provinsi",
                    "hari_terindikasi_asap": "Hari terindikasi asap",
                    "hari_diamati": "Hari diamati",
                },
            )
            fig_smoke_rank.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
            fig_smoke_rank.update_xaxes(ticksuffix="%")
            st.plotly_chart(fig_smoke_rank, use_container_width=True)
            st.caption(
                "Peringkat memakai semua provinsi pada rentang tahun terpilih. "
                "Satu hari dihitung terindikasi jika ada minimal satu catatan asap "
                "di provinsi tersebut."
            )
            show_visual_guide("peringkat indikasi asap karhutla", [
                ("Panjang batang", "Persentase tanggal yang memiliki setidaknya satu catatan terindikasi asap."),
                ("Hari terindikasi asap", "Jumlah tanggal unik dengan indikasi asap di provinsi tersebut."),
                ("Hari diamati", "Jumlah tanggal unik yang memiliki data; jumlah hari ini dapat berbeda antarprovinsi."),
                ("Batas interpretasi", "Indikasi asap bukan jumlah kebakaran dan tidak membuktikan bahwa asap menyebabkan dampak kesehatan."),
            ])
        else:
            st.info("Tidak ada data kualitas udara pada rentang tahun yang dipilih.")

        st.markdown("**Rata-rata saat terindikasi asap dan non-asap**")
        if not df_ispu_filt.empty:
            smoke_comparison = (
                df_ispu_filt.groupby("wildfire_smoke_plume_flag", as_index=False)
                .agg(
                    jumlah_pengamatan=("tanggal", "size"),
                    rata_rata_pm25=("pm25_concentration_ugm3", "mean"),
                    rata_rata_skor_ispa=("projected_ispa_clinic_burden_score", "mean"),
                )
            )
            smoke_comparison["Status asap"] = smoke_comparison[
                "wildfire_smoke_plume_flag"
            ].map({True: "Terindikasi asap", False: "Non-asap"})
            smoke_comparison = smoke_comparison[
                ["Status asap", "jumlah_pengamatan", "rata_rata_pm25", "rata_rata_skor_ispa"]
            ].rename(columns={
                "jumlah_pengamatan": "Jumlah pengamatan",
                "rata_rata_pm25": "Rata-rata PM2.5 (µg/m³)",
                "rata_rata_skor_ispa": "Rata-rata skor ISPA (0–10)",
            })
            smoke_comparison["Rata-rata PM2.5 (µg/m³)"] = smoke_comparison[
                "Rata-rata PM2.5 (µg/m³)"
            ].round(1)
            smoke_comparison["Rata-rata skor ISPA (0–10)"] = smoke_comparison[
                "Rata-rata skor ISPA (0–10)"
            ].round(2)
            st.dataframe(smoke_comparison, use_container_width=True, hide_index=True)
            st.caption(
                "Perbandingan mengikuti filter Provinsi di sidebar. "
                "Skor ISPA adalah estimasi, bukan jumlah pasien."
            )
        else:
            st.info("Tidak ada data untuk membandingkan kelompok asap pada filter yang dipilih.")

# ==============================================================================
# TAB 4: PERINGATAN DINI & FEATURE IMPORTANCE (RANDOM FOREST)
# ==============================================================================
with tab4:
    st.subheader("Prediksi risiko karhutla harian")
    p1, p2 = st.columns([1, 1.4])

    with p1:
        st.markdown("**Kepentingan fitur pada model prediksi**")
        df_fi_plot = df_fi.copy()
        df_fi_plot["fitur"] = (
            df_fi_plot["fitur"].astype(str).str.replace("_", " ", regex=False).str.capitalize()
        )
        fig_fi = px.bar(
            df_fi_plot.sort_values("importance", ascending=True),
            x="importance", y="fitur", orientation="h",
            color_discrete_sequence=[ACCENT_GREEN],
            labels={"importance": "Feature importance", "fitur": "Fitur"},
            template="plotly_dark"
        )
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_fi, use_container_width=True)
        show_visual_guide("feature importance model prediksi", [
            ("Fitur", "Variabel cuaca atau kondisi yang digunakan model untuk menghasilkan prediksi."),
            ("Panjang batang", "Nilai feature importance; semakin tinggi nilainya, semakin besar kontribusi relatif fitur pada model."),
            ("Catatan", "Feature importance tidak menunjukkan bahwa fitur tersebut menyebabkan kebakaran."),
        ])

    with p2:
        st.markdown("**Wilayah yang kemungkinan terdampak karhutla**")
        df_alert = df_pred.copy()
        if "tanggal" in df_alert.columns:
            df_alert["tanggal"] = pd.to_datetime(df_alert["tanggal"], errors="coerce")
            df_alert = df_alert[
                (df_alert["tanggal"].dt.year >= tahun_range[0])
                & (df_alert["tanggal"].dt.year <= tahun_range[1])
            ]
        if selected_prov != "Semua Provinsi":
            if "provinsi_id" in df_alert.columns:
                df_alert = df_alert[df_alert["provinsi_id"] == selected_prov]
        if "probabilitas_karhutla" in df_alert.columns:
            prob = pd.to_numeric(df_alert["probabilitas_karhutla"], errors="coerce")
            df_alert["prediksi_status"] = np.where(
                prob >= 0.50, "Rawan Tinggi", "Normal"
            )
            df_alert = df_alert[
                (prob >= 0.50) & prob.notna()
            ].copy()
        elif "status_prediksi" in df_alert.columns:
            df_alert["prediksi_status"] = df_alert["status_prediksi"]
            df_alert = df_alert[
                df_alert["prediksi_status"].eq("Rawan Tinggi")
            ].copy()

        alert_prob = (
            pd.to_numeric(df_alert["probabilitas_karhutla"], errors="coerce")
            if "probabilitas_karhutla" in df_alert.columns else pd.Series(dtype=float)
        )
        alert_prob = alert_prob.dropna()
        alert_kpi1, alert_kpi2, alert_kpi3 = st.columns(3)
        alert_kpi1.metric("Jumlah peringatan risiko tinggi", f"{len(df_alert):,}")
        alert_kpi2.metric(
            "Rata-rata peluang risiko pada daftar",
            f"{alert_prob.mean():.1%}" if not alert_prob.empty else "N/A",
        )
        affected_regions = (
            df_alert["provinsi_id"].nunique()
            if "provinsi_id" in df_alert.columns else 0
        )
        alert_kpi3.metric("Provinsi berisiko", f"{affected_regions:,}")

        if "tanggal" in df_alert.columns:
            month_names = {
                1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
                5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
                9: "September", 10: "Oktober", 11: "November", 12: "Desember",
            }
            df_alert["tahun"] = df_alert["tanggal"].dt.year
            df_alert["kuartal"] = "Q" + df_alert["tanggal"].dt.quarter.astype(str)
            df_alert["bulan"] = df_alert["tanggal"].dt.month.map(month_names)
            df_alert["hari"] = df_alert["tanggal"].dt.day
        df_alert = df_alert.sort_values(
            "probabilitas_karhutla", ascending=False
        ) if "probabilitas_karhutla" in df_alert.columns else df_alert
        alert_columns = [
            c for c in [
                "provinsi_id", "tahun", "kuartal", "bulan", "hari",
                "curah_hujan", "suhu_maks", "hotspots_snpp", "oni_anom",
                "probabilitas_karhutla", "prediksi_status",
            ] if c in df_alert.columns
        ]
        alert_table = df_alert[alert_columns].copy().rename(columns={
            "provinsi_id": "Provinsi",
            "tahun": "Tahun",
            "kuartal": "Kuartal",
            "bulan": "Bulan",
            "hari": "Hari",
            "curah_hujan": "Curah hujan",
            "suhu_maks": "Suhu maksimum (°C)",
            "probabilitas_karhutla": "Probabilitas prediksi risiko (%)",
            "hotspots_snpp": "Hotspot terdeteksi",
            "oni_anom": "Anomali ONI (°C)",
        })
        if "Probabilitas prediksi risiko (%)" in alert_table.columns:
            alert_table["Probabilitas prediksi risiko (%)"] = (
                pd.to_numeric(alert_table["Probabilitas prediksi risiko (%)"], errors="coerce") * 100
            ).round(1)
        st.dataframe(alert_table.head(100), use_container_width=True, height=380)
        st.caption(
            "Tabel menampilkan hingga 100 prediksi dengan probabilitas tertinggi. "
            "Suhu maksimum diambil dari data iklim harian untuk tanggal dan provinsi yang sama."
        )
        show_visual_guide("ringkasan dan daftar peringatan", [
            ("prediksi_status", "Rawan Tinggi jika probabilitas prediksi minimal 50%; selain itu Normal. Ini adalah estimasi, bukan kepastian kejadian."),
            ("Provinsi berisiko", "Jumlah provinsi unik yang memiliki setidaknya satu prediksi berstatus Rawan Tinggi pada filter aktif."),
            ("Probabilitas prediksi", "Estimasi model untuk setiap baris; probabilitas 60% bukan kepastian bahwa kebakaran akan terjadi."),
            ("Suhu maksimum", "Suhu maksimum harian (°C) dari data iklim pada tanggal dan provinsi yang sama; penambahan kolom ini tidak mengubah probabilitas prediksi."),
            ("Hotspot terdeteksi", "Jumlah deteksi panas satelit pada tanggal dan wilayah tersebut."),
            ("Curah hujan", "Akumulasi hujan tercatat; interpretasikan bersama variabel lain, bukan sebagai faktor tunggal."),
            ("Anomali ONI", "Penyimpangan suhu permukaan laut Pasifik dari kondisi normal."),
        ])
