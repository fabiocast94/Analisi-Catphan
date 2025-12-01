import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, filters

st.set_page_config(page_title="CTP401 QC Dashboard", layout="wide")
st.title("Catphan 500 - CTP401 Analysis")

uploaded_files = st.file_uploader("Carica le immagini DICOM", type="dcm", accept_multiple_files=True)

def load_dicom(file):
    ds = pydicom.dcmread(file)
    return ds.pixel_array, ds

def show_image(img, title="Image"):
    st.subheader(title)
    st.image(img, cmap="gray", use_column_width=True)

def calc_roi_stats(img, roi_coords):
    x1, y1, x2, y2 = roi_coords
    roi = img[y1:y2, x1:x2]
    mean = np.mean(roi)
    std = np.std(roi)
    return mean, std, roi

def plot_line_profile(img, line_coords):
    x1, y1, x2, y2 = line_coords
    profile = img[y1:y2, x1:x2].mean(axis=0)
    st.line_chart(profile)

def detect_edges(img):
    edges = filters.sobel(img)
    return edges

if uploaded_files:
    st.success(f"{len(uploaded_files)} file caricati")
    images = []
    for f in uploaded_files:
        img, ds = load_dicom(f)
        images.append((img, ds))
        show_image(img, title=f"Slice: {f.name}")
    
    st.markdown("---")
    st.header("1️⃣ Sensitometry (Linearity)")
    st.write("Seleziona ROI su ogni inserto e calcola la media dei valori CT")
    # Esempio ROI - da adattare a ogni materiale
    roi_coords = st.text_input("Inserisci ROI [x1,y1,x2,y2] separati da virgola", "50,50,100,100")
    roi_coords = [int(x) for x in roi_coords.split(",")]
    for img, ds in images:
        mean, std, roi = calc_roi_stats(img, roi_coords)
        st.write(f"Slice {ds.SOPInstanceUID}: Mean={mean:.2f}, Std={std:.2f}")
        st.image(roi, caption="ROI", cmap="gray")
    
    st.markdown("---")
    st.header("2️⃣ Scan Slice Geometry / Slice Sensitivity Profile")
    st.write("Traccia un profilo lungo lo slice per calcolare FWHM")
    # Linea di esempio
    line_coords = [50, 100, 200, 100]  # x1, y1, x2, y2
    for img, ds in images:
        plot_line_profile(img, line_coords)
    
    st.markdown("---")
    st.header("3️⃣ Pixel (Matrix) Size")
    st.write("Misura oggetto noto e calcola dimensione pixel")
    obj_size_mm = st.number_input("Dimensione reale oggetto (mm)", value=25.0)
    pixel_count = st.number_input("Numero pixel oggetto", value=50)
    pixel_size = obj_size_mm / pixel_count
    st.write(f"Pixel size stimato: {pixel_size:.2f} mm/pixel")
    
    st.markdown("---")
    st.header("4️⃣ Circular Symmetry")
    for img, ds in images:
        edges = detect_edges(img)
        contours = measure.find_contours(edges, 0.1)
        st.write(f"Slice {ds.SOPInstanceUID}: {len(contours)} contorni trovati")
        plt.imshow(edges, cmap="gray")
        st.pyplot(plt)
    
    st.markdown("---")
    st.header("5️⃣ Phantom Position Verification & 6️⃣ Patient Alignment System Check")
    st.write("Misurare offset dal centro (da implementare ROI/marker automatici)")
    
    st.markdown("---")
    st.header("7️⃣ Scan Incrementation")
    st.write("Verifica distanza tra slice successive")
    slice_positions = [float(ds.SliceLocation) for img, ds in images if hasattr(ds, "SliceLocation")]
    slice_positions.sort()
    increments = np.diff(slice_positions)
    st.write(f"Distanze tra slice: {increments}")

st.markdown("---")
st.write("Template base per CTP401 QC - da adattare ai tuoi protocolli specifici")
