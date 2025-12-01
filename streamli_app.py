import streamlit as st
import tempfile
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pylinac.ct import CatPhanBase


# ---------------------------
# CatPhan 500 custom pylinac class
# ---------------------------
class CatPhan500(CatPhanBase):
    catphan_name = "CatPhan 500"

    # Maps modules to slice locations (approx – modify as needed)
    # You may refine using actual DICOM Z positions
    module_z_positions = {
        "CTP401": -60,
        "CTP528": -30,
        "CTP515": 0,
        "CTP486": 30
    }

    def classify_module(self, z_pos):
        # Find closest module by Z distance
        diffs = {m: abs(z_pos - v) for m, v in self.module_z_positions.items()}
        return min(diffs, key=diffs.get)

    def analyze(self):
        results = {}

        for img, z in zip(self.images, self.z_positions):
            module_name = self.classify_module(z)
            pixel_array = img

            # Implement custom analysis per module
            if module_name == "CTP401":
                results["CTP401"] = self.analyze_ctp401(pixel_array)
            elif module_name == "CTP528":
                results["CTP528"] = self.analyze_ctp528(pixel_array)
            elif module_name == "CTP515":
                results["CTP515"] = self.analyze_ctp515(pixel_array)
            elif module_name == "CTP486":
                results["CTP486"] = self.analyze_ctp486(pixel_array)

        return results

    # ---------------------------
    # Custom module analysis functions
    # ---------------------------

    def analyze_ctp401(self, img):
        """Uniformity and noise."""
        center = img[img.shape[0]//4:3*img.shape[0]//4,
                     img.shape[1]//4:3*img.shape[1]//4]

        mean_val = np.mean(center)
        noise_val = np.std(center)

        return {"mean": mean_val, "noise": noise_val}

    def analyze_ctp528(self, img):
        """Very simplified MTF estimation."""
        proj = np.mean(img, axis=0)
        mtf = np.fft.fft(proj)
        return {"mtf": np.abs(mtf[:50])}

    def analyze_ctp515(self, img):
        """Low-contrast detectability (simplified)."""
        low_region = img[200:300, 200:300]
        mean = np.mean(low_region)
        std = np.std(low_region)
        snr = mean / std
        return {"snr": snr}

    def analyze_ctp486(self, img):
        """Simple geometric check."""
        centerline = img[:, img.shape[1]//2]
        peaks = np.where(centerline > np.percentile(centerline, 99))[0]
        return {"num_peaks": len(peaks)}


# ---------------------------
# Streamlit UI
# ---------------------------

st.title("CatPhan 500 DICOM Analyzer (pylinac custom)")

uploaded_files = st.file_uploader(
    "Carica le immagini DICOM del CatPhan 500",
    type=["dcm", "dicom"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file caricati.")

    with st.spinner("Lettura DICOM..."):
        dicoms = []
        positions = []

        for file in uploaded_files:
            ds = pydicom.dcmread(file)
            dicoms.append(ds.pixel_array.astype(float))
            positions.append(float(ds.ImagePositionPatient[2]))

        # Ordina per posizione z
        sorted_idx = np.argsort(positions)
        dicoms = [dicoms[i] for i in sorted_idx]
        positions = [positions[i] for i in sorted_idx]

    st.write("Analisi in corso...")

    # Crea analizzatore
    cp = CatPhan500()
    cp.images = dicoms
    cp.z_positions = positions

    results = cp.analyze()

    st.header("Risultati")

    # Visualizzazione dei moduli
    for module, data in results.items():
        st.subheader(module)
        st.json(data)

        # Grafico MTF
        if module == "CTP528":
            fig, ax = plt.subplots()
            ax.plot(data["mtf"])
            ax.set_title("MTF (stima)")
            st.pyplot(fig)

