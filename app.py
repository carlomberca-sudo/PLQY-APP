import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import re

st.set_page_config(page_title="PLQY Analyzer", layout="wide")

APP_DIR = Path(__file__).parent
DEFAULT_CC_DIR = APP_DIR / "correction_curves"


# -----------------------------
# Helpers
# -----------------------------

def load_spectrum(uploaded_file):
    """
    Expected format similar to Fluorolog export:
    skip first row, read up to 1024 rows, columns:
    [?, channel, intensity]
    """
    if uploaded_file is None:
        return None, None, None

    data = np.loadtxt(uploaded_file, skiprows=1, max_rows=1024)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Spectrum file must contain at least 3 columns.")

    channel = data[:, 1]
    intensity = data[:, 2]
    return data, channel, intensity


def load_single_correction_curve(file_obj, wl_axis):
    """
    Accepts one CSV correction curve.
    Supports two cases:
    1) first column = wavelength, second column = correction (%)
    2) malformed / non-wavelength first column -> resample by index
    """
    cc_raw = np.genfromtxt(
        file_obj,
        delimiter=",",
        usecols=(0, 1),
        dtype=float,
        invalid_raise=False,
    )

    cc_raw = np.atleast_2d(cc_raw)
    cc_x = cc_raw[:, 0]
    cc_y = cc_raw[:, 1]

    mask = (~np.isnan(cc_x)) & (~np.isnan(cc_y))
    cc_x = cc_x[mask]
    cc_y = cc_y[mask] * 0.01  # % -> fraction

    if cc_y.size < 2:
        raise ValueError("Correction curve file did not yield usable numeric data.")

    looks_like_wavelength = (
        (cc_x.min() > 100)
        and (cc_x.max() < 2000)
        and np.all(np.diff(cc_x) > 0)
    )

    if looks_like_wavelength:
        cc_interp = np.interp(wl_axis, cc_x, cc_y, left=cc_y[0], right=cc_y[-1])
    else:
        x_old = np.arange(cc_y.size)
        x_new = np.linspace(0, cc_y.size - 1, wl_axis.size)
        cc_interp = np.interp(x_new, x_old, cc_y)

    return cc_interp


def build_wavelength_axis(channel, center_wavelength, grating_number):
    if grating_number == 1:
        g = 0.4196
    else:
        g = 0.4192
    wl = np.array([center_wavelength - ((i - 513) * g) for i in channel], dtype=float)
    return wl


def compute_plqy(sample_i, ref_i, cc, wl, integration_boundary):
    dif = sample_i - ref_i
    dif_correct = dif * cc * wl

    idx = int(np.argmin(np.abs(wl - integration_boundary)))

    area_em = np.trapezoid(dif_correct[:idx], wl[:idx])
    area_abs = np.trapezoid(dif_correct[idx:], wl[idx:])

    if area_abs == 0:
        raise ZeroDivisionError("Absorption area is zero, cannot compute PLQY.")

    plqy = (-area_em / area_abs) * 100

    return {
        "dif": dif,
        "dif_correct": dif_correct,
        "integration_boundary": integration_boundary,
        "integration_index": idx,
        "area_em": area_em,
        "area_abs": area_abs,
        "plqy": plqy,
    }


def results_dataframe(wl, sample_i, ref_i, dif, dif_correct):
    return pd.DataFrame(
        {
            "Wavelength_nm": wl,
            "Sample_Intensity": sample_i,
            "Reference_Intensity": ref_i,
            "Difference_Uncorrected": dif,
            "Difference_Corrected": dif_correct,
        }
    )


def list_default_correction_files():
    if not DEFAULT_CC_DIR.exists():
        return []
    return sorted([p for p in DEFAULT_CC_DIR.iterdir() if p.suffix.lower() in {".csv", ".txt"}])


def correction_file_matches(name, grating_number, center_wavelength, filter_number):
    """
    Match names like:
    CC_2022_UVVIS_G1_Cen550_f4.csv
    """
    name_lower = name.lower()
    return (
        f"g{grating_number}".lower() in name_lower
        and f"cen{center_wavelength}".lower() in name_lower
        and f"f{filter_number}".lower() in name_lower
    )


def select_uploaded_correction_file(uploaded_files, grating_number, center_wavelength, filter_number):
    matches = []
    for f in uploaded_files:
        fname = getattr(f, "name", "")
        if correction_file_matches(fname, grating_number, center_wavelength, filter_number):
            matches.append(f)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No uploaded correction curve matched G{grating_number}, Cen{center_wavelength}, f{filter_number}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple uploaded correction curves matched G{grating_number}, Cen{center_wavelength}, f{filter_number}: "
            + ", ".join([m.name for m in matches])
        )

    return matches[0], matches[0].name


def select_default_correction_file(grating_number, center_wavelength, filter_number):
    files = list_default_correction_files()
    matches = []

    for path in files:
        if correction_file_matches(path.name, grating_number, center_wavelength, filter_number):
            matches.append(path)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No default correction curve matched G{grating_number}, Cen{center_wavelength}, f{filter_number}."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple default correction curves matched G{grating_number}, Cen{center_wavelength}, f{filter_number}: "
            + ", ".join([m.name for m in matches])
        )

    return matches[0], matches[0].name


# -----------------------------
# UI
# -----------------------------

st.title("PLQY Analyzer")
st.caption(
    "Upload sample and reference files. Then the app selects the correct correction curve based on grating, center wavelength, and filter."
)

left, right = st.columns([1, 1.5], gap="large")

with left:
    st.subheader("Inputs")

    sample_file = st.file_uploader(
        "1. Drop sample file", type=["txt", "csv", "dat"], key="sample"
    )
    ref_file = st.file_uploader(
        "2. Drop reference file", type=["txt", "csv", "dat"], key="ref"
    )

    cc_source = st.radio(
        "3. Correction curves source",
        options=["Upload correction files now", "Use default files stored in app"],
    )

    cc_files = []
    cc_names_preview = []

    if cc_source == "Upload correction files now":
        cc_files = st.file_uploader(
            "Drop correction curve files",
            type=["csv", "txt"],
            accept_multiple_files=True,
            key="cc_multi",
        )
        if cc_files:
            cc_names_preview = [f.name for f in cc_files]
            st.caption(f"Loaded {len(cc_files)} correction files")
            st.write(cc_names_preview)
    else:
        default_cc_files = list_default_correction_files()
        cc_names_preview = [p.name for p in default_cc_files]
        st.caption(f"Default correction files found in app: {len(default_cc_files)}")
        if default_cc_files:
            st.write(cc_names_preview)

    center_wavelength = st.number_input(
        "4. Center wavelength (nm)",
        min_value=200,
        max_value=1200,
        value=550,
        step=1,
    )

    excitation_wavelength = st.number_input(
        "5. Excitation wavelength (nm)",
        min_value=200,
        max_value=1200,
        value=390,
        step=1,
    )

    grating_number = st.selectbox("6. Grating", options=[1, 2], index=0)
    filter_number = st.selectbox("7. Filter", options=[1, 2, 3, 4], index=3)

    default_boundary = excitation_wavelength + 50
    integration_boundary = st.number_input(
        "8. Integration boundary (nm)",
        min_value=200.0,
        max_value=1200.0,
        value=float(default_boundary),
        step=1.0,
        help="Manual split between emission and absorption areas.",
    )

    run = st.button("Run analysis", type="primary", use_container_width=True)

with right:
    st.subheader("Results")

    if run:
        try:
            _, sample_channel, sample_i = load_spectrum(sample_file)
            _, ref_channel, ref_i = load_spectrum(ref_file)

            if sample_channel is None or ref_channel is None:
                st.error("Please upload sample and reference files.")
                st.stop()

            if len(sample_channel) != len(ref_channel):
                st.error("Sample and reference files do not have the same number of points.")
                st.stop()

            wl = build_wavelength_axis(sample_channel, center_wavelength, grating_number)

            if cc_source == "Upload correction files now":
                if not cc_files:
                    st.error("Please upload the correction curve files.")
                    st.stop()

                selected_cc_file, selected_cc_name = select_uploaded_correction_file(
                    cc_files,
                    grating_number=grating_number,
                    center_wavelength=center_wavelength,
                    filter_number=filter_number,
                )
                try:
                    selected_cc_file.seek(0)
                except Exception:
                    pass
                cc = load_single_correction_curve(selected_cc_file, wl)

            else:
                selected_cc_path, selected_cc_name = select_default_correction_file(
                    grating_number=grating_number,
                    center_wavelength=center_wavelength,
                    filter_number=filter_number,
                )
                with open(selected_cc_path, "rb") as f:
                    cc = load_single_correction_curve(f, wl)

            res = compute_plqy(
                sample_i=sample_i,
                ref_i=ref_i,
                cc=cc,
                wl=wl,
                integration_boundary=integration_boundary,
            )
            df = results_dataframe(wl, sample_i, ref_i, res["dif"], res["dif_correct"])

            tab1, tab2, tab3 = st.tabs(
                [
                    "PLQY value",
                    "Raw + processed graphs",
                    "Processed data + integration",
                ]
            )

            with tab1:
                c1, c2, c3 = st.columns(3)
                c1.metric("PLQY (%)", f"{res['plqy']:.2f}")
                c2.metric("Emission area", f"{res['area_em']:.4g}")
                c3.metric("Absorption area", f"{res['area_abs']:.4g}")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Parameter": [
                                "Center wavelength (nm)",
                                "Excitation wavelength (nm)",
                                "Integration boundary (nm)",
                                "Grating",
                                "Filter",
                                "Correction file used",
                            ],
                            "Value": [
                                center_wavelength,
                                excitation_wavelength,
                                res["integration_boundary"],
                                grating_number,
                                filter_number,
                                selected_cc_name,
                            ],
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            with tab2:
                fig1, ax1 = plt.subplots(figsize=(8, 4.5))
                ax1.plot(wl, sample_i, label="sample")
                ax1.plot(wl, ref_i, label="reference")
                ax1.set_title("Raw spectra")
                ax1.set_xlabel("Wavelength (nm)")
                ax1.set_ylabel("Intensity")
                ax1.grid(True)
                ax1.legend()
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots(figsize=(8, 4.5))
                ax2.plot(wl, res["dif"], label="uncorrected")
                ax2.plot(wl, res["dif_correct"], label="corrected")
                ax2.axvline(
                    res["integration_boundary"],
                    linestyle="--",
                    label="integration boundary",
                )
                ax2.set_title("Processed spectra")
                ax2.set_xlabel("Wavelength (nm)")
                ax2.set_ylabel("Signal")
                ax2.grid(True)
                ax2.legend()
                st.pyplot(fig2)

            with tab3:
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                ax3.plot(wl, res["dif_correct"], label="corrected")
                ax3.axvline(
                    res["integration_boundary"], linestyle="--", label="boundary"
                )
                ax3.fill_between(
                    wl[: res["integration_index"]],
                    res["dif_correct"][: res["integration_index"]],
                    alpha=0.3,
                    label="emission area",
                )
                ax3.fill_between(
                    wl[res["integration_index"] :],
                    res["dif_correct"][res["integration_index"] :],
                    alpha=0.3,
                    label="absorption area",
                )
                ax3.set_title("Corrected data with integration split")
                ax3.set_xlabel("Wavelength (nm)")
                ax3.set_ylabel("Corrected signal")
                ax3.grid(True)
                ax3.legend()
                st.pyplot(fig3)

                st.dataframe(df, use_container_width=True)
                st.write("Correction file used:")
                st.write(selected_cc_name)

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download processed data (CSV)",
                    data=csv_bytes,
                    file_name="plqy_processed_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error while processing files: {e}")
    else:
        st.info("Upload the files, set the wavelengths, and click 'Run analysis'.")
