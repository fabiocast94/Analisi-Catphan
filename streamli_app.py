import streamlit as st
import io
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import zipfile
import os
import cv2

# ---------------------------
# Custom CTP401 Module with automatic slice detection via ramps
# ---------------------------
class CTP401:
    def __init__(self, dicom_files, roi_offsets=None):
        self.dicom_files = dicom_files
        # roi_offsets: dict with insert name -> (dx, dy, w, h) relative to phantom center
        self.roi_offsets = roi_offsets or {
            'Teflon': (0, -50, 20, 20),
            'Aria': (0, 50, 20, 20),
            'Acrilico': (-50, 0, 20, 20),
            'LDPE': (50, 0, 20, 20)
        }
        self.results = {}

    def detect_ctp401_slice(self):
        # Heuristic: find slice with 4 ramps using edge detection
        max_ramps = 0
        selected_idx = 0
        for i, f in enumerate(self.dicom_files):
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            edges = cv2.Canny(img, 50, 150)
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            n_lines = len(lines) if lines is not None else 0
            if n_lines > max_ramps:
                max_ramps = n_lines
                selected_idx = i
        self.slice_idx = selected_idx
        self.ds_slice = pydicom.dcmread(self.dicom_files[self.slice_idx])
        self.img_slice = self.ds_slice.pixel_array.astype(float)

    def find_phantom_center(self):
        img = self.img_slice
        thresh = np.percentile(img, 80)
        mask = img > thresh
        y, x = np.where(mask)
        cx = int(np.mean(x))
        cy = int(np.mean(y))
        self.phantom_center = (cx, cy)

    def sensitometry_linearity(self):
        self.detect_ctp401_slice()
        self.find_phantom_center()
        fig, ax = plt.subplots()
        ax.imshow(self.img_slice, cmap='gray')
        sensi_results = {}
        cx, cy = self.phantom_center
        for name, (dx, dy, w, h) in self.roi_offsets.items():
            x1 = cx + dx
            y1 = cy + dy
            x2 = x1 + w
            y2 = y1 + h
            roi = self.img_slice[y1:y2, x1:x2]
            sensi_results[name] = {
                'mean': float(np.mean(roi)),
                'std': float(np.std(roi))
            }
            rect = plt.Rectangle((x1, y1), w, h, edgecolor='red', facecolor='none', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x1, y1-5, name, color='red', fontsize=8)
        self.results['sensitometry'] = sensi_results
        self.fig = fig

    def analyze(self):
        self.sensitometry_linearity()
        return self.results

# ---------------------------
# Streamlit App
# ---------------------------
st.title('CatPhan 500 Analyzer - CTP401 Sensitometry')

uploaded_zip = st.file_uploader('Upload ZIP file with all DICOM images', type=['zip'])

if uploaded_zip is not None:
    with tempfile.TemporaryDirectory() as tmpdirname:
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)

        dicom_files = []
        for root, _, files in os.walk(tmpdirname):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, f))

        if not dicom_files:
            st.error("Nessun file DICOM trovato nel ZIP.")
        else:
            dicom_files.sort()
            ctp401 = CTP401(dicom_files)
            results = ctp401.analyze()

            st.header('CTP401 Sensitometry Results')
            st.json(results)

            st.subheader('ROI Visualization')
            st.pyplot(ctp401.fig)

            if st.button('Generate PDF Report'):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial','B',16)
                pdf.cell(0,10,'CatPhan 500 - CTP401 Sensitometry Report',ln=True)
                pdf.set_font('Arial','',12)
                pdf.cell(0,6,'Sensitometry:', ln=True)
                for name, val in results['sensitometry'].items():
                    pdf.cell(0,6,f"{name}: mean={val['mean']}, std={val['std']}", ln=True)
                buf = io.BytesIO()
                pdf.output(buf)
                buf.seek(0)
                st.download_button('Download PDF Report', data=buf, file_name='ctp401_report.pdf', mime='application/pdf')
else:
    st.info('Upload a ZIP file with all DICOM images to start analysis.')
