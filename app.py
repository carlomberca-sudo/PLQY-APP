import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

st.set_page_config(page_title="PLQY Batch Analyzer", layout="wide")

APP_DIR = Path(__file__).parent
DEFAULT_CC_DIR = APP_DIR / "correction_curves"

if "batch_results_ready" not in st.session_state:
    st.session_state.batch_results_ready = False

if "batch_results_df" not in st.session_state:
    st.session_state.batch_results_df = pd.DataFrame()

if "batch_wide_summary_df" not in st.session_state:
    st.session_state.batch_wide_summary_df = pd.DataFrame()

if "batch_warnings_df" not in st.session_state:
    st.session_state.batch_warnings_df = pd.DataFrame()

if "batch_parsed_df" not in st.session_state:
    st.session_state.batch_parsed_df = pd.DataFrame()

if "batch_details" not in st.session_state:
    st.session_state.batch_details = {}


# -----------------------------
# Helpers
# -----------------------------

def extract_excitation(filename: str):
    for part in filename.split("_"):
        if part.lower().startswith("exc"):
            digits = "".join(filter(str.isdigit, part))
            if digits:
                return int(digits)
    return None


def extract_sample_name(filename: str):
    if "_Exc" in filename:
        return filename.split("_Exc")[0]
    if "_exc" in filename:
        return filename.split("_exc")[0]
    return Path(filename).stem


def load_spectrum(uploaded_file):
    data = np.loadtxt(uploaded_file, skiprows=1, max_rows=1024)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Spectrum file must contain at least 3 columns.")
    channel = data[:, 1]
    intensity = data[:, 2]
    return data, channel, intensity


def load_single_correction_curve(file_obj, wl_axis):
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
    cc_y = cc_y[mask] * 0.01

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
    g = 0.4196 if grating_number == 1 else 0.4192
    return np.array([center_wavelength - ((i - 513) * g) for i in channel], dtype=float)


def correction_file_matches(name, grating_number, center_wavelength, filter_number):
    name_lower = name.lower()
    return (
        f"g{grating_number}".lower() in name_lower
        and f"cen{center_wavelength}".lower() in name_lower
        and f"f{filter_number}".lower() in name_lower
    )


def list_default_correction_files():
    if not DEFAULT_CC_DIR.exists():
        return []
    return sorted([p for p in DEFAULT_CC_DIR.iterdir() if p.suffix.lower() in {".csv", ".txt"}])


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


def build_wide_summary(results_df: pd.DataFrame):
    if results_df.empty:
        return pd.DataFrame()

    tmp = results_df.copy()
    tmp["Exc Label"] = tmp["Excitation (nm)"].apply(lambda x: f"EXC {int(x)}")
    wide = tmp.pivot_table(index="Sample", columns="Exc Label", values="PLQY (%)", aggfunc="first")
    wide = wide.reset_index()
    exc_cols = sorted([c for c in wide.columns if c != "Sample"], key=lambda x: int(x.split()[-1]))
    return wide[["Sample"] + exc_cols]


# -----------------------------
# UI
# -----------------------------

st.title("PLQY Batch Analyzer")
st.caption(
    "Upload many measurement files, define how references are recognized, and compute PLQY for the whole batch."
)

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.subheader("Inputs")

    measurement_files = st.file_uploader(
        "1. Drop all measurement .txt files",
        type=["txt", "dat", "csv"],
        accept_multiple_files=True,
        key="measurement_files",
    )

    reference_keyword = st.text_input(
        "2. Reference keyword",
        value="REF",
        help="Files whose sample name contains this keyword will be treated as references.",
    )

    cc_source = st.radio(
        "3. Correction curves source",
        options=["Upload correction files now", "Use default files stored in app"],
    )

    cc_files = []
    if cc_source == "Upload correction files now":
        cc_files = st.file_uploader(
            "Drop correction curve files",
            type=["csv", "txt"],
            accept_multiple_files=True,
            key="cc_multi_batch",
        )
        if cc_files:
            st.caption(f"Loaded {len(cc_files)} correction files")
            st.write([f.name for f in cc_files])
    else:
        default_cc_files = list_default_correction_files()
        st.caption(f"Default correction files found in app: {len(default_cc_files)}")
        if default_cc_files:
            st.write([p.name for p in default_cc_files])

    grating_number = st.selectbox("4. Grating", options=[1, 2], index=0)
    filter_number = st.selectbox("5. Filter", options=[1, 2, 3, 4], index=3)
    default_boundary_offset = st.number_input(
        "6. Integration boundary offset from excitation (nm)",
        min_value=-200,
        max_value=300,
        value=50,
        step=1,
    )

    run = st.button("Run batch analysis", type="primary", width="stretch")

with right:
    st.subheader("Results")

    if run:
        try:
            if not measurement_files:
                st.error("Please upload the measurement files.")
                st.stop()

            samples = defaultdict(dict)
            refs = defaultdict(list)
            parsed_files = []
            warnings = []
            results = []
            details = {}

            # Parse filenames and group files
            for uploaded in measurement_files:
                filename = uploaded.name
                exc = extract_excitation(filename)
                sample_name = extract_sample_name(filename)

                if exc is None:
                    warnings.append({
                        "Type": "Filename parsing",
                        "File": filename,
                        "Message": "Could not extract excitation wavelength from filename.",
                    })
                    continue

                parsed_files.append({
                    "File": filename,
                    "Sample": sample_name,
                    "Excitation (nm)": exc,
                    "Reference?": reference_keyword.lower() in sample_name.lower(),
                })

                try:
                    uploaded.seek(0)
                except Exception:
                    pass

                if reference_keyword.lower() in sample_name.lower():
                    refs[exc].append(uploaded)
                else:
                    samples[sample_name][exc] = uploaded

            parsed_df = pd.DataFrame(parsed_files)

            # Main batch computation
            for sample_name, exc_map in samples.items():
                for exc, sample_file in sorted(exc_map.items()):
                    if exc not in refs or not refs[exc]:
                        warnings.append({
                            "Type": "Missing reference",
                            "File": sample_file.name,
                            "Message": f"No reference found for excitation {exc} nm.",
                        })
                        continue

                    if len(refs[exc]) > 1:
                        warnings.append({
                            "Type": "Multiple references",
                            "File": sample_file.name,
                            "Message": f"Multiple references found for excitation {exc} nm. Using the first one: {refs[exc][0].name}",
                        })

                    ref_file = refs[exc][0]

                    try:
                        try:
                            sample_file.seek(0)
                            ref_file.seek(0)
                        except Exception:
                            pass

                        _, sample_channel, sample_i = load_spectrum(sample_file)
                        _, ref_channel, ref_i = load_spectrum(ref_file)

                        if len(sample_channel) != len(ref_channel):
                            raise ValueError("Sample and reference files do not have the same number of points.")

                        inferred_center = None
                        for part in sample_file.name.split("_"):
                            if part.lower().startswith("cen"):
                                digits = "".join(filter(str.isdigit, part))
                                if digits:
                                    inferred_center = int(digits)
                                    break

                        if inferred_center is None:
                            raise ValueError(f"Could not extract center wavelength from filename: {sample_file.name}")

                        wl = build_wavelength_axis(sample_channel, inferred_center, grating_number)

                        if cc_source == "Upload correction files now":
                            if not cc_files:
                                raise ValueError("No correction curve files were uploaded.")
                            selected_cc_file, selected_cc_name = select_uploaded_correction_file(
                                cc_files,
                                grating_number=grating_number,
                                center_wavelength=inferred_center,
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
                                center_wavelength=inferred_center,
                                filter_number=filter_number,
                            )
                            with open(selected_cc_path, "rb") as f:
                                cc = load_single_correction_curve(f, wl)

                        integration_boundary = exc + default_boundary_offset

                        res = compute_plqy(
                            sample_i=sample_i,
                            ref_i=ref_i,
                            cc=cc,
                            wl=wl,
                            integration_boundary=integration_boundary,
                        )

                        results.append({
                            "Sample": sample_name,
                            "Excitation (nm)": exc,
                            "Center (nm)": inferred_center,
                            "Reference file": ref_file.name,
                            "Correction file": selected_cc_name,
                            "PLQY (%)": round(res["plqy"], 2),
                            "Emission area": res["area_em"],
                            "Absorption area": res["area_abs"],
                            "Integration boundary (nm)": integration_boundary,
                        })

                        details[(sample_name, exc)] = {
                            "wl": wl,
                            "sample_i": sample_i,
                            "ref_i": ref_i,
                            "dif": res["dif"],
                            "dif_correct": res["dif_correct"],
                            "integration_boundary": integration_boundary,
                            "integration_index": res["integration_index"],
                            "reference_file": ref_file.name,
                            "correction_file": selected_cc_name,
                        }

                    except Exception as e:
                        warnings.append({
                            "Type": "Processing error",
                            "File": sample_file.name,
                            "Message": str(e),
                        })

            results_df = pd.DataFrame(results)
            if not results_df.empty:
                results_df = results_df.sort_values(by=["Sample", "Excitation (nm)"])

            warnings_df = pd.DataFrame(warnings)
            wide_summary_df = build_wide_summary(results_df)

            st.session_state.batch_results_df = results_df
            st.session_state.batch_wide_summary_df = wide_summary_df
            st.session_state.batch_warnings_df = warnings_df
            st.session_state.batch_parsed_df = parsed_df
            st.session_state.batch_details = details
            st.session_state.batch_results_ready = True

        except Exception as e:
            st.error(f"Error while processing files: {e}")

    if st.session_state.batch_results_ready:
        results_df = st.session_state.batch_results_df
        wide_summary_df = st.session_state.batch_wide_summary_df
        warnings_df = st.session_state.batch_warnings_df
        parsed_df = st.session_state.batch_parsed_df
        details = st.session_state.batch_details

        tab1, tab2, tab3, tab4 = st.tabs([
            "PLQY summary",
            "Long table",
            "Graphs",
            "Warnings / parsed files",
        ])

        with tab1:
            st.subheader("Wide PLQY summary")
            if wide_summary_df.empty:
                st.info("No results were generated.")
            else:
                st.dataframe(wide_summary_df, width="stretch")
                csv_bytes = wide_summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download wide summary (CSV)",
                    data=csv_bytes,
                    file_name="plqy_batch_summary_wide.csv",
                    mime="text/csv",
                    width="stretch",
                )

        with tab2:
            st.subheader("Detailed results")
            if results_df.empty:
                st.info("No detailed results available.")
            else:
                st.dataframe(results_df, width="stretch")
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download detailed results (CSV)",
                    data=csv_bytes,
                    file_name="plqy_batch_results_long.csv",
                    mime="text/csv",
                    width="stretch",
                )

        with tab3:
            st.subheader("Per-sample graphs")
            if not details:
                st.info("No graphable results available.")
            else:
                graph_options = [
                    f"{sample} | EXC {exc}"
                    for sample, exc in sorted(details.keys(), key=lambda x: (x[0], x[1]))
                ]
                selected_graph = st.selectbox(
                    "Select result to inspect",
                    options=graph_options,
                    key="batch_graph_selector",
                )
                selected_sample, selected_exc = selected_graph.split(" | EXC ")
                selected_exc = int(selected_exc)
                d = details[(selected_sample, selected_exc)]

                fig1, ax1 = plt.subplots(figsize=(8, 4.5))
                ax1.plot(d["wl"], d["sample_i"], label="sample")
                ax1.plot(d["wl"], d["ref_i"], label="reference")
                ax1.set_title(f"Raw spectra — {selected_sample} — EXC {selected_exc}")
                ax1.set_xlabel("Wavelength (nm)")
                ax1.set_ylabel("Intensity")
                ax1.grid(True)
                ax1.legend()
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots(figsize=(8, 4.5))
                ax2.plot(d["wl"], d["dif"], label="uncorrected")
                ax2.plot(d["wl"], d["dif_correct"], label="corrected")
                ax2.axvline(d["integration_boundary"], linestyle="--", label="integration boundary")
                ax2.set_title(f"Processed spectra — {selected_sample} — EXC {selected_exc}")
                ax2.set_xlabel("Wavelength (nm)")
                ax2.set_ylabel("Signal")
                ax2.grid(True)
                ax2.legend()
                st.pyplot(fig2)

                fig3, ax3 = plt.subplots(figsize=(8, 5))
                ax3.plot(d["wl"], d["dif_correct"], label="corrected")
                ax3.axvline(d["integration_boundary"], linestyle="--", label="boundary")
                ax3.fill_between(
                    d["wl"][: d["integration_index"]],
                    d["dif_correct"][: d["integration_index"]],
                    alpha=0.3,
                    label="emission area",
                )
                ax3.fill_between(
                    d["wl"][d["integration_index"] :],
                    d["dif_correct"][d["integration_index"] :],
                    alpha=0.3,
                    label="absorption area",
                )
                ax3.set_title(f"Corrected data with integration split — {selected_sample} — EXC {selected_exc}")
                ax3.set_xlabel("Wavelength (nm)")
                ax3.set_ylabel("Corrected signal")
                ax3.grid(True)
                ax3.legend()
                st.pyplot(fig3)

                st.write("Reference file used:", d["reference_file"])
                st.write("Correction file used:", d["correction_file"])

        with tab4:
            st.subheader("Warnings")
            if warnings_df.empty:
                st.success("No warnings.")
            else:
                st.dataframe(warnings_df, width="stretch")

            st.subheader("Parsed uploaded files")
            if parsed_df.empty:
                st.info("No files were parsed.")
            else:
                st.dataframe(parsed_df, width="stretch")

    else:
        st.info("Upload the measurement files, define the reference keyword, and click 'Run batch analysis'.")
