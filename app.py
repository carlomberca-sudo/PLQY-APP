import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="PLQY Analyzer", layout="wide")


def check_password() -> bool:
    """Simple shared-password gate using Streamlit secrets."""
    expected = None
    try:
        expected = st.secrets["app_password"]
    except Exception:
        return True  # allow local testing before secrets are configured

    if st.session_state.get("authenticated", False):
        return True

    st.title("PLQY Analyzer")
    st.subheader("Private access")
    entered = st.text_input("Password", type="password")
    if st.button("Login", use_container_width=True):
        if entered == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


def load_spectrum(uploaded_file):
    if uploaded_file is None:
        raise ValueError("Missing spectrum file.")

    uploaded_file.seek(0)
    data = np.loadtxt(uploaded_file, skiprows=1, max_rows=1024)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Spectrum file must contain at least 3 columns after the header row.")

    channel = data[:, 1]
    intensity = data[:, 2]
    return channel, intensity


def load_correction_curve(uploaded_file, wl_axis):
    if uploaded_file is None:
        raise ValueError("Missing correction curve file.")

    uploaded_file.seek(0)
    cc_raw = np.genfromtxt(
        uploaded_file,
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
    cc_y = cc_y[mask] * 0.01

    if cc_y.size < 2:
        raise ValueError("Correction curve file did not yield usable numeric data.")

    looks_like_wavelength = (
        (cc_x.min() > 100)
        and (cc_x.max() < 2000)
        and np.all(np.diff(cc_x) > 0)
    )

    if looks_like_wavelength:
        return np.interp(wl_axis, cc_x, cc_y, left=cc_y[0], right=cc_y[-1])

    x_old = np.arange(cc_y.size)
    x_new = np.linspace(0, cc_y.size - 1, wl_axis.size)
    return np.interp(x_new, x_old, cc_y)


def build_wavelength_axis(channel, center_wavelength, grating_number):
    g = 0.4196 if grating_number == 1 else 0.4192
    return np.array([center_wavelength - ((i - 513) * g) for i in channel], dtype=float)


def compute_plqy(sample_i, ref_i, cc, wl, integration_boundary_nm):
    dif = sample_i - ref_i
    dif_correct = dif * cc * wl

    idx = int(np.argmin(np.abs(wl - integration_boundary_nm)))
    area_em = np.trapezoid(dif_correct[:idx], wl[:idx])
    area_abs = np.trapezoid(dif_correct[idx:], wl[idx:])

    if area_abs == 0:
        raise ZeroDivisionError("Absorption area is zero, cannot compute PLQY.")

    plqy = (-area_em / area_abs) * 100
    return {
        "dif": dif,
        "dif_correct": dif_correct,
        "integration_boundary": integration_boundary_nm,
        "integration_index": idx,
        "area_em": area_em,
        "area_abs": area_abs,
        "plqy": plqy,
    }


def build_results_table(wl, sample_i, ref_i, dif, dif_correct):
    return pd.DataFrame(
        {
            "Wavelength_nm": wl,
            "Sample_Intensity": sample_i,
            "Reference_Intensity": ref_i,
            "Difference_Uncorrected": dif,
            "Difference_Corrected": dif_correct,
        }
    )


check_password()

st.title("PLQY Analyzer")
st.caption("Upload the sample, reference, and correction files. Then inspect the PLQY value, raw spectra, and integration split.")

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.subheader("Inputs")
    sample_file = st.file_uploader("1. Drop sample file", type=["txt", "csv", "dat"], key="sample")
    ref_file = st.file_uploader("2. Drop reference file", type=["txt", "csv", "dat"], key="reference")
    cc_file = st.file_uploader("3. Drop correction curve file", type=["csv", "txt"], key="correction")

    center_wavelength = st.number_input("4. Center wavelength (nm)", min_value=200, max_value=1200, value=550, step=1)
    excitation_wavelength = st.number_input("5. Excitation wavelength (nm)", min_value=200, max_value=1200, value=390, step=1)
    grating_number = st.selectbox("Grating", options=[1, 2], index=0)

    default_boundary = excitation_wavelength + 50
    integration_boundary_nm = st.number_input(
        "Manual integration boundary (nm)",
        min_value=200.0,
        max_value=1200.0,
        value=float(default_boundary),
        step=1.0,
        help="This separates emission and absorption. You can move it manually.",
    )

    run = st.button("Run analysis", type="primary", use_container_width=True)

with right:
    st.subheader("Results")

    if run:
        try:
            sample_channel, sample_i = load_spectrum(sample_file)
            ref_channel, ref_i = load_spectrum(ref_file)

            if len(sample_channel) != len(ref_channel):
                st.error("Sample and reference files do not have the same number of points.")
                st.stop()

            wl = build_wavelength_axis(sample_channel, center_wavelength, grating_number)
            cc = load_correction_curve(cc_file, wl)
            res = compute_plqy(sample_i, ref_i, cc, wl, integration_boundary_nm)
            df = build_results_table(wl, sample_i, ref_i, res["dif"], res["dif_correct"])

            tab1, tab2, tab3 = st.tabs([
                "PLQY value",
                "Raw + processed graphs",
                "Processed data + integration",
            ])

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
                                "Manual integration boundary (nm)",
                                "Grating",
                            ],
                            "Value": [
                                center_wavelength,
                                excitation_wavelength,
                                integration_boundary_nm,
                                grating_number,
                            ],
                        }
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

            with tab2:
                fig1, ax1 = plt.subplots(figsize=(8, 4.5))
                ax1.plot(wl, sample_i, label="Sample")
                ax1.plot(wl, ref_i, label="Reference")
                ax1.set_title("Raw spectra")
                ax1.set_xlabel("Wavelength (nm)")
                ax1.set_ylabel("Intensity")
                ax1.grid(True)
                ax1.legend()
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots(figsize=(8, 4.5))
                ax2.plot(wl, res["dif"], label="Uncorrected")
                ax2.plot(wl, res["dif_correct"], label="Corrected")
                ax2.axvline(res["integration_boundary"], linestyle="--", label="Integration boundary")
                ax2.set_title("Processed spectra")
                ax2.set_xlabel("Wavelength (nm)")
                ax2.set_ylabel("Signal")
                ax2.grid(True)
                ax2.legend()
                st.pyplot(fig2)

            with tab3:
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                ax3.plot(wl, res["dif_correct"], label="Corrected")
                ax3.axvline(res["integration_boundary"], linestyle="--", label="Boundary")
                ax3.fill_between(wl[:res["integration_index"]], res["dif_correct"][:res["integration_index"]], alpha=0.25, label="Emission area")
                ax3.fill_between(wl[res["integration_index"]:], res["dif_correct"][res["integration_index"]:], alpha=0.25, label="Absorption area")
                ax3.set_title("Corrected data with integration split")
                ax3.set_xlabel("Wavelength (nm)")
                ax3.set_ylabel("Corrected signal")
                ax3.grid(True)
                ax3.legend()
                st.pyplot(fig3)

                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download processed data (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="plqy_processed_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Processing failed: {e}")
    else:
        st.info("Upload the files, set the parameters, and click Run analysis.")
