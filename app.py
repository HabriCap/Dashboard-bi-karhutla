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
    ispu = pd.read_parquet(os.path.join(DATA_DIR, "mart_ispu_kesehatan.parquet"))
    pred = pd.read_parquet(os.path.join(DATA_DIR, "mart_prediksi_karhutla.parquet"))
    fi = pd.read_parquet(os.path.join(DATA_DIR, "mart_feature_importance.parquet"))
    
    prov["periode"] = pd.to_datetime(prov["tahun"].astype(str) + "-" + prov["bulan"].astype(str).str.zfill(2) + "-01")
    gambut["periode"] = pd.to_datetime(gambut["tahun"].astype(str) + "-" + gambut["bulan"].astype(str).str.zfill(2) + "-01")
    return prov, gambut, top_khg, peta, ispu, pred, fi


def show_visual_guide(title, terms):
    with st.expander(f"Cara membaca: {title}"):
        st.table(pd.DataFrame(terms, columns=["Istilah / bagian", "Arti sederhana"]))


try:
    df_prov, df_gambut, df_top_khg, df_peta, df_ispu, df_pred, df_fi = load_all_data()
except Exception as e:
    st.error(f"Gagal memuat data dari folder '{DATA_DIR}/'. Pastikan semua file .parquet sudah di-push ke GitHub. Detail: {e}")
    st.stop()

# ==============================================================================
# SIDEBAR FILTER GLOBAL
# ==============================================================================
st.sidebar.title("Pilih Data")
tahun_list = sorted(df_prov["tahun"].unique())
tahun_range = st.sidebar.select_slider(
    "Rentang tahun",
    options=tahun_list,
    value=(min(tahun_list), max(tahun_list))
)

prov_list = ["Semua Provinsi"] + sorted(df_prov["provinsi_id"].dropna().unique().tolist())
selected_prov = st.sidebar.selectbox("Provinsi", prov_list)

enso_list = ["Semua Fase"] + sorted(df_prov["status_enso"].dropna().unique().tolist())
selected_enso = st.sidebar.selectbox("Kondisi ENSO (suhu laut Pasifik)", enso_list)

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
st.title("Dashboard Titik Panas, Gambut, Udara, dan Risiko Kebakaran")
st.caption("Gunakan pilihan tahun, provinsi, dan kondisi iklim di sebelah kiri untuk melihat data yang sesuai.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Cuaca & Titik Panas",
    "2. Kebakaran di Lahan Gambut",
    "3. Udara & Kesehatan",
    "4. Perkiraan Risiko"
])

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW & PEMICU IKLIM
# ==============================================================================
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah titik panas terdeteksi", f"{int(df_prov_filt['hotspots_snpp'].sum()):,}")
    c2.metric("Titik panas dengan keyakinan tinggi", f"{int(df_prov_filt['high_confidence_snpp'].sum()):,}")
    c3.metric("Total kekuatan panas terukur", f"{df_prov_filt['frp_sum_snpp'].sum():,.0f} MW")
    c4.metric("Rata-rata perubahan suhu Pasifik", f"{df_prov_filt['oni_anom'].mean():.2f} °C")
    show_visual_guide("angka ringkasan", [
        ("Titik panas", "Tanda dari satelit bahwa ada lokasi yang sangat panas; ini petunjuk kebakaran, bukan hitungan luas area terbakar."),
        ("Keyakinan tinggi", "Titik panas yang sinyal satelitnya dinilai lebih meyakinkan."),
        ("Kekuatan panas (MW)", "Ukuran energi panas yang terdeteksi. Nilai lebih besar berarti sinyal panas lebih kuat."),
        ("Perubahan suhu Pasifik", "Perbedaan suhu laut Pasifik dari kondisi biasanya; dapat berkaitan dengan musim kering atau basah."),
    ])

    st.markdown("---")
    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        st.subheader("Perubahan suhu laut Pasifik dan jumlah titik panas dari waktu ke waktu")
        tren_iklim = df_prov_filt.groupby("periode", as_index=False).agg({
            "hotspots_snpp": "sum",
            "oni_anom": "mean",
            "dmi_value": "mean"
        })
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(
            go.Bar(x=tren_iklim["periode"], y=tren_iklim["hotspots_snpp"], name="Jumlah titik panas", marker_color=ACCENT_GREEN, opacity=0.75),
            secondary_y=False
        )
        fig_dual.add_trace(
            go.Scatter(x=tren_iklim["periode"], y=tren_iklim["oni_anom"], name="Perubahan suhu Pasifik (°C)", line=dict(color=ALERT_RED, width=2.5)),
            secondary_y=True
        )
        fig_dual.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12))
        fig_dual.update_yaxes(title_text="Jumlah titik panas", secondary_y=False)
        fig_dual.update_yaxes(title_text="Perubahan suhu Pasifik (°C)", secondary_y=True)
        st.plotly_chart(fig_dual, use_container_width=True)
        show_visual_guide("grafik cuaca dan titik panas", [
            ("Sumbu mendatar", "Bulan dan tahun pengamatan."),
            ("Batang hijau", "Jumlah titik panas yang terdeteksi pada bulan tersebut."),
            ("Garis merah", "Perubahan suhu permukaan laut Pasifik dari keadaan biasanya (ONI)."),
            ("Cara membaca", "Lihat apakah jumlah titik panas dan garis suhu naik pada waktu yang sama. Ini menunjukkan hubungan waktu, bukan bukti bahwa satu hal pasti menyebabkan yang lain."),
        ])

    with col_right:
        st.subheader("10 provinsi dengan titik panas terbanyak")
        top10_p = df_prov_filt.groupby("provinsi_id", as_index=False)["hotspots_snpp"].sum().sort_values("hotspots_snpp", ascending=True).tail(10)
        fig_bar_p = px.bar(
            top10_p, x="hotspots_snpp", y="provinsi_id", orientation="h",
            labels={"hotspots_snpp": "Jumlah titik panas", "provinsi_id": "Provinsi"},
            color_discrete_sequence=[PRIMARY_GREEN], template="plotly_dark"
        )
        fig_bar_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_bar_p, use_container_width=True)
        show_visual_guide("peringkat provinsi", [
            ("Nama provinsi", "Wilayah yang dibandingkan."),
            ("Panjang batang", "Jumlah titik panas yang tercatat; batang lebih panjang berarti jumlahnya lebih banyak."),
            ("Catatan", "Peringkat ini berdasarkan jumlah deteksi, bukan ukuran luas kebakaran atau tingkat bahaya."),
        ])

    st.markdown("---")
    climate_col, rain_col = st.columns(2)
    with climate_col:
        st.subheader("Rata-rata titik panas per hari menurut kondisi laut")
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
                        "status_enso": "Kondisi Samudra Pasifik",
                        "rata_hotspot_harian": "Perkiraan titik panas per hari",
                        "status_iod": "Kondisi Samudra Hindia",
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
                show_visual_guide("rata-rata titik panas menurut kondisi laut", [
                    ("El Nino / Netral / La Nina", "Kelompok kondisi suhu laut di Samudra Pasifik."),
                    ("IOD negatif / netral / positif", "Kelompok kondisi suhu laut di Samudra Hindia."),
                    ("Tinggi batang", "Perkiraan rata-rata titik panas per hari untuk gabungan kondisi yang sama."),
                    ("Warna", "Membedakan kondisi Samudra Hindia di dalam setiap kelompok Pasifik."),
                ])
            else:
                st.info("Kolom jumlah hotspot tidak tersedia pada data terfilter.")
        else:
            st.info("Kolom ENSO, IOD, atau periode tidak tersedia pada data terfilter.")

    with rain_col:
        st.subheader("Curah hujan dan jumlah titik panas di tiap provinsi")
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
                    "total_hotspot": "Jumlah titik panas",
                    "status_enso": "Kondisi Pasifik",
                    "provinsi_id": "Provinsi",
                },
                template="plotly_dark",
            )
            fig_rain.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=380, margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig_rain, use_container_width=True)
            show_visual_guide("curah hujan dan titik panas", [
                ("Setiap titik", "Satu provinsi dalam satu kelompok kondisi Pasifik."),
                ("Posisi mendatar", "Rata-rata curah hujan; titik lebih ke kanan berarti hujan lebih tinggi."),
                ("Posisi tegak", "Jumlah titik panas yang tercatat; titik lebih tinggi berarti jumlahnya lebih banyak."),
                ("Warna titik", "Kondisi suhu laut Pasifik pada kelompok tersebut."),
                ("Nama saat diarahkan", "Menampilkan provinsi dan jumlah titik panas."),
            ])
        else:
            st.info("Kolom provinsi, ENSO, curah hujan, atau hotspot tidak tersedia pada data terfilter.")

# ==============================================================================
# TAB 2: KERENTANAN LAHAN GAMBUT (KHG)
# ==============================================================================
with tab2:
    mask_g = (df_gambut["tahun"] >= tahun_range[0]) & (df_gambut["tahun"] <= tahun_range[1])
    df_g_filt = df_gambut[mask_g].copy()
    df_g_filt["Tipe Lahan"] = df_g_filt["is_lahan_gambut"].map({1: "Di gambut", 0: "Di luar gambut"})

    total_all = df_g_filt["jumlah_hotspot"].sum()
    total_peat = df_g_filt[df_g_filt["is_lahan_gambut"] == 1]["jumlah_hotspot"].sum()
    pct_peat = (total_peat / total_all * 100) if total_all > 0 else 0
    has_province_gambut = "provinsi_id" in df_g_filt.columns

    g1, g2, g3 = st.columns(3)
    g1.metric("Titik panas di lahan gambut", f"{int(total_peat):,}")
    g2.metric("Bagian titik panas di gambut", f"{pct_peat:.2f}%")
    g3.metric("Wilayah gambut dengan titik panas terbanyak", df_top_khg.iloc[0]["kode_khg"] if not df_top_khg.empty else "-")
    show_visual_guide("angka ringkasan lahan gambut", [
        ("Lahan gambut (KHG)", "Lahan gambut yang berada di dalam peta Kesatuan Hidrologis Gambut (KHG)."),
        ("Bagian titik panas di gambut", "Persentase titik panas yang lokasinya berada di gambut dibandingkan seluruh titik panas pada data."),
        ("Kode KHG", "Kode untuk mengenali satu wilayah bentang lahan gambut."),
    ])

    st.markdown("---")
    c_g1, c_g2 = st.columns([1.3, 1])

    with c_g1:
        st.subheader("Perubahan jumlah titik panas di gambut dan di luar gambut")
        fig_area = px.area(
            df_g_filt, x="periode", y="jumlah_hotspot", color="Tipe Lahan",
            color_discrete_map={"Di gambut": PRIMARY_GREEN, "Di luar gambut": NEUTRAL_GRAY},
            labels={"periode": "Waktu", "jumlah_hotspot": "Jumlah titik panas"},
            template="plotly_dark"
        )
        fig_area.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_area, use_container_width=True)
        show_visual_guide("tren titik panas di lahan", [
            ("Sumbu mendatar", "Bulan dan tahun pengamatan."),
            ("Sumbu tegak", "Jumlah titik panas yang terdeteksi."),
            ("Warna hijau / abu-abu", "Membedakan titik panas di gambut dan di luar gambut."),
            ("Luas berwarna", "Menunjukkan perubahan jumlah dari waktu ke waktu; perhatikan bahwa jumlah tiap kategori dijumlahkan per bulan."),
        ])

    with c_g2:
        st.subheader("10 wilayah gambut dengan titik panas terbanyak")
        fig_khg = px.bar(
            df_top_khg.head(10).sort_values("jumlah_hotspot", ascending=True),
            x="jumlah_hotspot", y="kode_khg", orientation="h",
            color_discrete_sequence=[ALERT_RED],
            labels={"jumlah_hotspot": "Jumlah titik panas", "kode_khg": "Kode wilayah gambut"},
            template="plotly_dark"
        )
        fig_khg.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_khg, use_container_width=True)
        show_visual_guide("peringkat wilayah gambut", [
            ("Kode wilayah", "Identitas wilayah gambut pada peta KHG."),
            ("Panjang batang", "Jumlah titik panas pada wilayah itu; batang lebih panjang berarti lebih banyak deteksi."),
            ("Catatan", "Banyaknya titik panas tidak langsung menunjukkan luas area yang terbakar."),
        ])

    st.markdown("---")
    peat_col, insight_col = st.columns(2)
    with peat_col:
        st.subheader("Sebaran kekuatan panas yang terdeteksi: gambut dan luar gambut")
        df_frp = df_peta.copy()
        if "tahun" in df_frp.columns:
            df_frp = df_frp[
                (df_frp["tahun"] >= tahun_range[0]) & (df_frp["tahun"] <= tahun_range[1])
            ].copy()
        if "provinsi_id" in df_frp.columns and selected_prov != "Semua Provinsi":
            df_frp = df_frp[df_frp["provinsi_id"] == selected_prov]
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
                    labels={"frp": "Kekuatan panas terukur (MW)", "Tipe Lahan": "Jenis lahan"},
                    template="plotly_dark",
                )
                fig_frp.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    height=380, showlegend=False, margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_frp, use_container_width=True)
                show_visual_guide("kekuatan panas di tiap jenis lahan", [
                    ("FRP / MW", "FRP adalah perkiraan kekuatan panas dari satelit; MW (megawatt) adalah satuan ukurannya."),
                    ("Garis di dalam kotak", "Nilai tengah: kira-kira separuh data berada di bawah dan separuh di atasnya."),
                    ("Kotak", "Rentang bagian tengah data, sehingga terlihat sebaran nilai yang umum."),
                    ("Cara membandingkan", "Kotak atau garis tengah yang lebih tinggi menunjukkan nilai FRP yang cenderung lebih besar."),
                ])
            else:
                st.info("Tidak ada observasi FRP pada rentang tahun yang dipilih.")
        else:
            st.info("Kolom FRP atau kategori lahan tidak tersedia pada data sampel.")

    with insight_col:
        st.subheader("Apakah perubahan di gambut berjalan seiring dengan total titik panas?")
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
                st.metric("Angka hubungan antara bagian titik panas gambut dan total", f"{corr:.2f}")
                st.caption(
                    f"Hubungannya {direction} dan {strength} pada {len(paired)} bulan yang tersedia. "
                    "Ini menunjukkan dua hal bergerak bersama, bukan bukti sebab-akibat."
                )
                show_visual_guide("angka hubungan gambut dan titik panas", [
                    ("Angka mendekati +1", "Keduanya cenderung naik atau turun bersama."),
                    ("Angka mendekati 0", "Tidak terlihat pola gerak bersama yang jelas."),
                    ("Angka mendekati -1", "Saat satu naik, yang lain cenderung turun."),
                    ("Batas data", "Perhitungan memakai data bulanan gabungan seluruh provinsi, karena data gambut tidak mencantumkan provinsi."),
                ])
            else:
                st.info("Data bulanan tidak cukup untuk menghitung korelasi.")
        else:
            st.info("Kedua kategori lahan harus tersedia untuk menghitung korelasi.")

        frp_means = (
            df_frp.groupby("Tipe Lahan")["frp"].mean()
            if {"Tipe Lahan", "frp"}.issubset(df_frp.columns) else pd.Series(dtype=float)
        )
        frp_gambut = frp_means.get("Di gambut", np.nan)
        frp_non_gambut = frp_means.get("Di luar gambut", np.nan)
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
            ("Nilai gambut", "Rata-rata kekuatan panas satelit pada sampel titik yang berada di gambut."),
            ("Nilai luar gambut", "Rata-rata kekuatan panas satelit pada sampel titik di luar gambut."),
            ("MW", "Satuan kekuatan panas yang terdeteksi satelit."),
            ("Sampel", "Angka dihitung dari sebagian titik panas yang tersedia, bukan seluruh kejadian."),
        ])
        if selected_prov != "Semua Provinsi" and not has_province_gambut:
            st.caption(
                "Data bulanan gambut dan sampel FRP tidak memiliki kolom provinsi; analisis ini "
                "tetap agregat nasional untuk filter provinsi."
            )

    st.subheader("Peta lokasi titik panas pada gambut dan di luar gambut")
    df_peta_plot = df_peta.copy()
    df_peta_plot["Status Lahan"] = df_peta_plot["is_lahan_gambut"].map({1: "Di gambut", 0: "Di luar gambut"})
    fig_map = px.scatter_map(
        df_peta_plot, lat="latitude", lon="longitude", color="Status Lahan",
        color_discrete_map={"Di gambut": ALERT_RED, "Di luar gambut": ACCENT_GREEN},
        zoom=3.8, center={"lat": -2.2, "lon": 117.5}, height=420, opacity=0.65
    )
    fig_map.update_layout(map_style="carto-positron", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_map, use_container_width=True)
    show_visual_guide("peta titik panas", [
        ("Setiap titik", "Satu lokasi titik panas dalam sampel data."),
        ("Warna merah", "Lokasi yang berada di lahan gambut."),
        ("Warna hijau", "Lokasi yang berada di luar lahan gambut."),
        ("Peta", "Membantu melihat sebaran lokasi; titik berdekatan belum tentu satu kebakaran yang sama."),
    ])

# ==============================================================================
# TAB 3: KUALITAS UDARA (ISPU) & DAMPAK KESEHATAN ISPA
# ==============================================================================
with tab3:
    df_ispu_filt = df_ispu.copy()
    if selected_prov != "Semua Provinsi":
        df_ispu_filt = df_ispu_filt[df_ispu_filt["provinsi_id"] == selected_prov]

    i1, i2, i3 = st.columns(3)
    i1.metric("Rata-rata partikel halus di udara", f"{df_ispu_filt['pm25_concentration_ugm3'].mean():.1f} µg/m³")
    i2.metric("Rata-rata tingkat pencemaran udara", f"{df_ispu_filt['ispu_calculated_index'].mean():.1f}")
    i3.metric("Perkiraan kebutuhan layanan gangguan pernapasan", f"{df_ispu_filt['projected_ispa_clinic_burden_score'].mean():.2f} / 10")
    show_visual_guide("angka ringkasan udara dan kesehatan", [
        ("Partikel halus (PM2.5)", "Debu atau asap sangat kecil yang dapat masuk jauh ke paru-paru. Angka lebih tinggi berarti lebih banyak partikel terukur."),
        ("Tingkat pencemaran udara", "Angka ringkasan kondisi udara; nilai lebih tinggi berarti kualitas udara lebih buruk."),
        ("Kebutuhan layanan gangguan pernapasan", "Perkiraan skor kebutuhan layanan kesehatan pada skala 0–10, bukan jumlah pasien."),
    ])

    st.markdown("---")
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.subheader("Udara yang sulit bergerak, asap, dan partikel halus")
        df_ispu_filt["Status Kabut Asap"] = df_ispu_filt["wildfire_smoke_plume_flag"].map({True: "Terdeteksi Asap Karhutla", False: "Non-Asap Karhutla"})
        fig_scat = px.scatter(
            df_ispu_filt.sample(n=min(3000, len(df_ispu_filt)), random_state=42),
            x="atmospheric_stagnation_index", y="pm25_concentration_ugm3",
            color="Status Kabut Asap",
            color_discrete_map={"Terdeteksi Asap Karhutla": ALERT_RED, "Non-Asap Karhutla": ACCENT_GREEN},
            labels={"atmospheric_stagnation_index": "Seberapa sulit udara bergerak (0-10)", "pm25_concentration_ugm3": "Partikel halus di udara (µg/m³)"},
            opacity=0.65
        )
        fig_scat.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_scat, use_container_width=True)
        show_visual_guide("udara sulit bergerak dan partikel halus", [
            ("Setiap titik", "Satu pengamatan kualitas udara."),
            ("Posisi mendatar", "Skor keadaan udara yang sulit bergerak; nilai lebih tinggi berarti udara lebih stagnan."),
            ("Posisi tegak", "Jumlah partikel halus PM2.5 yang terukur."),
            ("Merah / hijau", "Pengamatan yang ditandai ada / tidak ada jejak asap kebakaran."),
            ("Cara membaca", "Lihat apakah nilai partikel halus cenderung lebih tinggi pada skor stagnasi atau kelompok asap tertentu. Pola ini bukan bukti sebab-akibat."),
        ])

    with col_i2:
        st.subheader("Perkiraan kebutuhan layanan gangguan pernapasan per provinsi")
        prov_ispa = df_ispu.groupby("provinsi_id", as_index=False)["projected_ispa_clinic_burden_score"].mean().sort_values("projected_ispa_clinic_burden_score", ascending=True).tail(12)
        fig_ispa_bar = px.bar(
            prov_ispa, x="projected_ispa_clinic_burden_score", y="provinsi_id", orientation="h",
            color_discrete_sequence=[PRIMARY_GREEN],
            labels={"projected_ispa_clinic_burden_score": "Skor perkiraan kebutuhan layanan (0-10)", "provinsi_id": "Provinsi"}
        )
        fig_ispa_bar.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_ispa_bar, use_container_width=True)
        show_visual_guide("peringkat kebutuhan layanan kesehatan", [
            ("Nama provinsi", "Wilayah yang dibandingkan."),
            ("Panjang batang", "Skor perkiraan kebutuhan layanan untuk gangguan pernapasan; batang lebih panjang berarti skor lebih tinggi."),
            ("Skala 0–10", "Skor ringkasan, bukan jumlah pasien yang datang ke fasilitas kesehatan."),
        ])

# ==============================================================================
# TAB 4: PERINGATAN DINI & FEATURE IMPORTANCE (RANDOM FOREST)
# ==============================================================================
with tab4:
    st.subheader("Perkiraan risiko kebakaran harian")
    p1, p2 = st.columns([1, 1.4])

    with p1:
        st.markdown("**Faktor yang paling banyak dipakai dalam perkiraan risiko**")
        df_fi_plot = df_fi.copy()
        df_fi_plot["fitur"] = (
            df_fi_plot["fitur"].astype(str).str.replace("_", " ", regex=False).str.capitalize()
        )
        fig_fi = px.bar(
            df_fi_plot.sort_values("importance", ascending=True),
            x="importance", y="fitur", orientation="h",
            color_discrete_sequence=[ACCENT_GREEN],
            labels={"importance": "Seberapa banyak faktor dipakai model", "fitur": "Faktor"},
            template="plotly_dark"
        )
        fig_fi.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_fi, use_container_width=True)
        show_visual_guide("faktor dalam perkiraan risiko", [
            ("Nama faktor", "Informasi cuaca atau kondisi yang dipertimbangkan sistem."),
            ("Panjang batang", "Seberapa banyak faktor tersebut digunakan oleh model untuk membuat perkiraan."),
            ("Penting", "Batang panjang bukan berarti faktor itu pasti menyebabkan kebakaran."),
        ])

    with p2:
        st.markdown("**Hari dan wilayah dengan perkiraan risiko tinggi**")
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
            risk_high = prob >= 0.50
            if "status_prediksi" in df_alert.columns:
                risk_high |= df_alert["status_prediksi"].eq("Rawan Tinggi")
            df_alert = df_alert[risk_high].copy()
        elif "status_prediksi" in df_alert.columns:
            df_alert = df_alert[df_alert["status_prediksi"] == "Rawan Tinggi"].copy()

        alert_prob = (
            pd.to_numeric(df_alert["probabilitas_karhutla"], errors="coerce")
            if "probabilitas_karhutla" in df_alert.columns else pd.Series(dtype=float)
        )
        alert_prob = alert_prob.dropna()
        alert_kpi1, alert_kpi2 = st.columns(2)
        alert_kpi1.metric("Jumlah peringatan risiko tinggi", f"{len(df_alert):,}")
        alert_kpi2.metric(
            "Rata-rata peluang risiko pada daftar",
            f"{alert_prob.mean():.1%}" if not alert_prob.empty else "N/A",
        )
        sort_cols = [c for c in ["tanggal", "probabilitas_karhutla"] if c in df_alert.columns]
        if sort_cols:
            df_alert = df_alert.sort_values(
                sort_cols,
                ascending=[False] * len(sort_cols),
            )
        alert_columns = [
            c for c in [
                "tanggal", "provinsi_id", "probabilitas_karhutla",
                "hotspots_snpp", "curah_hujan", "oni_anom",
            ] if c in df_alert.columns
        ]
        alert_table = df_alert[alert_columns].copy().rename(columns={
            "tanggal": "Tanggal",
            "provinsi_id": "Provinsi",
            "probabilitas_karhutla": "Perkiraan peluang risiko (%)",
            "hotspots_snpp": "Titik panas terdeteksi",
            "curah_hujan": "Curah hujan",
            "oni_anom": "Perubahan suhu Pasifik (°C)",
        })
        if "Perkiraan peluang risiko (%)" in alert_table.columns:
            alert_table["Perkiraan peluang risiko (%)"] = (
                pd.to_numeric(alert_table["Perkiraan peluang risiko (%)"], errors="coerce") * 100
            ).round(1)
        st.dataframe(alert_table.head(100), use_container_width=True, height=380)
        show_visual_guide("angka ringkasan dan daftar peringatan", [
            ("Jumlah peringatan", "Banyaknya hari dalam data terpilih yang berstatus risiko tinggi atau memiliki peluang minimal 50%."),
            ("Peluang risiko", "Perkiraan sistem untuk setiap baris; misalnya 60% berarti model menilai risiko lebih tinggi daripada 40%. Ini bukan kepastian kejadian."),
            ("Titik panas terdeteksi", "Jumlah deteksi panas satelit pada hari dan wilayah tersebut."),
            ("Curah hujan", "Jumlah hujan yang tercatat; baca bersama faktor lain, bukan sebagai penentu tunggal."),
            ("Perubahan suhu Pasifik", "Perbedaan suhu laut Pasifik dari kondisi biasanya."),
        ])
