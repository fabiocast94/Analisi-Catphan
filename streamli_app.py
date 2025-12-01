import streamlit as st
import zipfile
import io
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, filters

st.set_page_config(page_title="CTP401 QC Dashboard", layout="wide")
st.title("Catphan 500 - CTP401 Analysis (ZIP Upload)")

uploaded_zip = st.file_uploader("Carica un file ZIP con tutti i DICOM", type="zip")

def load_dicom_from_zip(zip_file):
    images = []
    with zipfile.ZipFile(zip_file) as z:
        for file_name in z.namelist():
            if file_name.lower().endswith(".dcm"):
                with z.open(file_name) as f:
                    ds = pydicom.dcmread(f)
                    images.append((ds.pixel_array, ds))
    return images

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

if uploaded_zip:
    images = load_dicom_from_zip(uploaded_zip)
    st.success(f"{len(images)} DICOM trovati nel ZIP")
    
    for i, (img, ds) in enumerate(images):
        show_image(img, title=f"Slice {i+1}: {ds.SOPInstanceUID}")

    st.markdown("---")
    st.header("1️⃣ Sensitometry (Linearity)")
    roi_coords = st.text_input("Inserisci ROI [x1,y1,x2,y2]", "50,50,100,100")
    roi_coords = [int(x) for x in roi_coords.split(",")]
    for img, ds in images:
        mean, std, roi = calc_roi_stats(img, roi_coords)
        st.write(f"Slice {ds.SOPInstanceUID}: Mean={mean:.2f}, Std={std:.2f}")
        st.image(roi, caption="ROI", cmap="gray")
    
    st.markdown("---")
    st.header("2️⃣ Scan Slice Geometry / Slice Sensitivity Profile")
    line_coords = [50, 100, 200, 100]  # esempio
    for img, ds in images:
        plot_line_profile(img, line_coords)
    
    st.markdown("---")
    st.header("3️⃣ Pixel (Matrix) Size")
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
    slice_positions = [float(ds.SliceLocation) for img, ds in images if hasattr(ds, "SliceLocation")]
    slice_positions.sort()
    increments = np.diff(slice_positions)
    st.write(f"Distanze tra slice: {increments}")

st.markdown("---")
st.write("Template base per CTP401 QC da ZIP - da adattare ai protocolli specifici")
