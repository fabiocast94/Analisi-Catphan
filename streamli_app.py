import streamlit as st
import io
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import zipfile
import os

# ---------------------------
# Custom CTP401 Module with automatic slice selection and ROI placement
# ---------------------------
class CTP401:
    def __init__(self, dicom_files, roi_offsets=None):
        self.dicom_files = dicom_files
        # roi_offsets: dict with insert name -> (x_offset, y_offset, width, height) relative to phantom center
        self.roi_offsets = roi_offsets or {
            'Teflon': (-40, -20, 20, 20),
            'Aria': (-10, -20, 20, 20),
            'Acrilico': (20, -20, 20, 20),
            'LDPE': (50, -20, 20, 20)
        }
        self.results = {}

    def find_central_slice(self):
        # Select slice with max std in central area (heuristic for insert visibility)
        max_std = -1
        selected_idx = 0
        for i, f in enumerate(self.dicom_files):
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(float)
            h, w = img.shape
            # central 100x100 region
            roi = img[h//2-50:h//2+50, w//2-50:w//2+50]
            s = np.std(roi)
            if s > max_std:
                max_std = s
                selected_idx = i
        self.central_slice_idx = selected_idx
        self.central_ds = pydicom.dcmread(self.dicom_files[self.central_slice_idx])
        self.central_img = self.central_ds.pixel_array.astype(float)

    def find_phantom_center(self):
        img = self.central_img
        # threshold to find bright regions (phantom body)
        thresh = np.percentile(img, 80)
        mask = img > thresh
        y, x = np.where(mask)
        cx = int(np.mean(x))
        cy = int(np.mean(y))
        self.phantom_center = (cx, cy)

    def sensitometry_linearity(self):
        self.find_central_slice()
        self.find_phantom_center()
        fig, ax = plt.subplots()
        ax.imshow(self.central_img, cmap='gray')
        sensi_results = {}
        cx, cy = self.phantom_center
        for name, (dx, dy, w, h) in self.roi_offsets.items():
            x1 = cx + dx
            y1 = cy + dy
            x2 = x1 + w
            y2 = y1 + h
            roi = self.central_img[y1:y2, x1:x2]
            sensi_results[name] = {
                'mean': float(np.mean(roi)),
                'std': float(np.std(roi))
            }
            # draw rectangle
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
        # Extract ZIP
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)

        # List all DICOM files recursively (case-insensitive)
        dicom_files = []
        for root, _, files in os.walk(tmpdirname):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dicom_files.append(os.path.join(root, f))

        if not dicom_files:
            st.error("Nessun file DICOM trovato nel ZIP.")
        else:
            dicom_files.sort()  # optionally sort by name

            # Analyze CTP401
            ctp401 = CTP401(dicom_files)
            results = ctp401.analyze()

            # Display results
            st.header('CTP401 Sensitometry Results')
            st.json(results)

            # Show image with ROI
            st.subheader('ROI Visualization')
            st.pyplot(ctp401.fig)

            # PDF report generation
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
