import streamlit as st
import zipfile
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, filters

st.set_page_config(page_title="CTP401 QC Dashboard", layout="wide")
st.title("Catphan 500 - CTP401 Analysis (ZIP Upload)")

# ---------- Funzioni ----------
def load_dicom_from_zip(zip_file):
    images = []
    with zipfile.ZipFile(zip_file) as z:
        for file_name in z.namelist():
            if file_name.lower().endswith(".dcm"):
                with z.open(file_name) as f:
                    ds = pydicom.dcmread(f)
                    images.append((ds.pixel_array, ds))
    return images

def normalize_image(img):
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    if img.ndim == 3:
        img = img[0,:,:]
    img_norm = img.astype(np.float32)
    img_norm -= img_norm.min()
    if img_norm.max() != 0:
        img_norm /= img_norm.max()
    img_norm = (img_norm * 255).astype(np.uint8)
    return img_norm

def show_image(img, title="Image"):
    img_norm = normalize_image(img)
    st.subheader(title)
    st.image(img_norm, use_column_width=True)

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

def find_marker_slice(images, marker_roi=(0, 0, 20, 50), threshold=200):
    """
    Trova la slice corretta basandosi su un marker bianco nella zona sinistra.
    marker_roi: (x1, y1, x2, y2) del ROI dove cercare il marker
    threshold: soglia di pixel per considerare marker presente
    """
    x1, y1, x2, y2 = marker_roi
    for img, ds in images:
        if img.ndim == 3:
            img_slice = img[0,:,:]
        else:
            img_slice = img
        roi = img_slice[y1:y2, x1:x2]
        roi_mean = np.mean(roi)
        if roi_mean >= threshold:
            return img_slice, ds
    return None, None

# ---------- Caricamento ZIP ----------
uploaded_zip = st.file_uploader("Carica un file ZIP con tutti i DICOM", type="zip")

if uploaded_zip is not None:
    images = load_dicom_from_zip(uploaded_zip)
    st.success(f"{len(images)} DICOM trovati nel ZIP")
    
    # Trova la slice corretta con marker
    img_correct, ds_correct = find_marker_slice(images)
    if img_correct is None:
        st.error("Non è stata trovata la slice corretta con il marker bianco!")
    else:
        st.success(f"Slice selezionata: {ds_correct.SOPInstanceUID}")
        show_image(img_correct, title=f"Slice corretta: {ds_correct.SOPInstanceUID}")

        # ---------- Analisi CTP401 ----------
        st.markdown("---")
        st.header("1️⃣ Sensitometry (Linearity)")
        roi_coords = st.text_input("Inserisci ROI [x1,y1,x2,y2]", "50,50,100,100")
        roi_coords = [int(x) for x in roi_coords.split(",")]
        mean, std, roi = calc_roi_stats(img_correct, roi_coords)
        st.write(f"Mean={mean:.2f}, Std={std:.2f}")
        st.image(np.nan_to_num(roi, nan=0.0), caption="ROI", use_column_width=True)

        st.markdown("---")
        st.header("2️⃣ Scan Slice Geometry / Slice Sensitivity Profile")
        line_coords = [50, 100, 200, 100]  # esempio
        plot_line_profile(img_correct, line_coords)
        
        st.markdown("---")
        st.header("3️⃣ Pixel (Matrix) Size")
        obj_size_mm = st.number_input("Dimensione reale oggetto (mm)", value=25.0)
        pixel_count = st.number_input("Numero pixel oggetto", value=50)
        pixel_size = obj_size_mm / pixel_count
        st.write(f"Pixel size stimato: {pixel_size:.2f} mm/pixel")
        
        st.markdown("---")
        st.header("4️⃣ Circular Symmetry")
        edges = detect_edges(img_correct)
        contours = measure.find_contours(edges, 0.1)
        st.write(f"{len(contours)} contorni trovati")
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

else:
    st.info("Attendere il caricamento del file ZIP")
