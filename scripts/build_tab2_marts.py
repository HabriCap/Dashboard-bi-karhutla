from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "streamlit_data"
SOURCE_PATH = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path.home() / "Downloads" / "clean_hotspots_gambut.parquet"
)

PROVINCE_NAMES = {
    "Aceh": "Aceh",
    "Bali": "Bali",
    "Bangka-Belitung Islands": "Kepulauan Bangka Belitung",
    "Banten": "Banten",
    "Bengkulu": "Bengkulu",
    "Central Java": "Jawa Tengah",
    "Central Kalimantan": "Kalimantan Tengah",
    "Central Sulawesi": "Sulawesi Tengah",
    "East Java": "Jawa Timur",
    "East Kalimantan": "Kalimantan Timur",
    "East Nusa Tenggara": "Nusa Tenggara Timur",
    "Gorontalo": "Gorontalo",
    "Jakarta Special Capital Region": "DKI Jakarta",
    "Jambi": "Jambi",
    "Lampung": "Lampung",
    "Maluku": "Maluku",
    "North Kalimantan": "Kalimantan Utara",
    "North Maluku": "Maluku Utara",
    "North Sulawesi": "Sulawesi Utara",
    "North Sumatra": "Sumatera Utara",
    "Papua": "Papua",
    "Riau": "Riau",
    "Riau Islands": "Kepulauan Riau",
    "South Kalimantan": "Kalimantan Selatan",
    "South Sulawesi": "Sulawesi Selatan",
    "South Sumatra": "Sumatera Selatan",
    "Southeast Sulawesi": "Sulawesi Tenggara",
    "Special Region of Yogyakarta": "DI Yogyakarta",
    "West Java": "Jawa Barat",
    "West Kalimantan": "Kalimantan Barat",
    "West Nusa Tenggara": "Nusa Tenggara Barat",
    "West Papua": "Papua Barat",
    "West Sulawesi": "Sulawesi Barat",
    "West Sumatra": "Sumatera Barat",
}


def main():
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Data hotspot tidak ditemukan: {SOURCE_PATH}")

    raw = pd.read_parquet(
        SOURCE_PATH,
        columns=[
            "latitude",
            "longitude",
            "province",
            "frp",
            "kode_khg",
            "is_lahan_gambut",
            "tahun",
            "bulan",
        ],
    )
    raw["provinsi_id"] = raw["province"].map(PROVINCE_NAMES)
    if raw["provinsi_id"].isna().any():
        unknown = sorted(raw.loc[raw["provinsi_id"].isna(), "province"].unique())
        raise ValueError(f"Nama provinsi belum dipetakan: {unknown}")

    climate = pd.read_parquet(
        DATA_DIR / "mart_provinsi_bulanan.parquet",
        columns=["tahun", "bulan", "provinsi_id", "status_enso", "status_iod"],
    )
    climate_keys = ["tahun", "bulan", "provinsi_id"]
    if climate.duplicated(climate_keys).any():
        raise ValueError("Data iklim memiliki kunci provinsi/tahun/bulan duplikat.")

    raw = raw.merge(climate, on=climate_keys, how="left", validate="many_to_one")
    trend_dimensions = [
        "tahun",
        "bulan",
        "provinsi_id",
        "status_enso",
        "status_iod",
        "is_lahan_gambut",
    ]
    trend = (
        raw.groupby(trend_dimensions, dropna=False, as_index=False)
        .agg(
            jumlah_hotspot=("is_lahan_gambut", "size"),
            frp_observations=("frp", "count"),
            total_frp=("frp", "sum"),
        )
        .sort_values(trend_dimensions, na_position="last")
    )

    khg_rows = raw[
        raw["is_lahan_gambut"].eq(1) & raw["kode_khg"].notna()
    ]
    khg_dimensions = [
        "tahun",
        "bulan",
        "provinsi_id",
        "status_enso",
        "status_iod",
        "kode_khg",
    ]
    khg = (
        khg_rows.groupby(khg_dimensions, dropna=False, as_index=False)
        .agg(
            jumlah_hotspot=("kode_khg", "size"),
            total_frp=("frp", "sum"),
        )
        .sort_values(khg_dimensions, na_position="last")
    )

    sample_dimensions = [
        "provinsi_id",
        "tahun",
        "bulan",
        "is_lahan_gambut",
    ]
    sample_parts = [
        group.sample(n=min(2, len(group)), random_state=42 + index)
        for index, (_, group) in enumerate(
            raw.groupby(sample_dimensions, sort=False, dropna=False)
        )
    ]
    map_sample = pd.concat(sample_parts, ignore_index=True)
    map_sample = map_sample[
        [
            "latitude",
            "longitude",
            "tahun",
            "bulan",
            "provinsi_id",
            "status_enso",
            "status_iod",
            "is_lahan_gambut",
            "kode_khg",
            "frp",
        ]
    ].sort_values(sample_dimensions)
    frp_sample = raw.sample(n=min(50000, len(raw)), random_state=42).sort_values(
        ["provinsi_id", "tahun", "bulan"]
    )
    frp_sample = frp_sample[
        [
            "latitude",
            "longitude",
            "tahun",
            "bulan",
            "provinsi_id",
            "status_enso",
            "status_iod",
            "is_lahan_gambut",
            "kode_khg",
            "frp",
        ]
    ]

    trend.to_parquet(DATA_DIR / "mart_gambut_bulanan.parquet", index=False)
    khg.to_parquet(DATA_DIR / "mart_top_khg.parquet", index=False)
    map_sample.to_parquet(DATA_DIR / "mart_peta_hotspot_sample.parquet", index=False)
    frp_sample.to_parquet(DATA_DIR / "mart_frp_sample.parquet", index=False)
    print(
        f"Built tab 2 marts from {len(raw):,} source records: "
        f"{len(trend):,} monthly groups, {len(khg):,} KHG groups, "
        f"{len(map_sample):,} map sample points, {len(frp_sample):,} FRP sample points."
    )


if __name__ == "__main__":
    main()
