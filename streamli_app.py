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
# Custom CTP401 Module - Sensitometry only with ROI visualization
# ---------------------------
class CTP401:
    def __init__(self, dicom_files, inserts=None):
        self.dicom_files = dicom_files
        # inserts: dict with insert name -> ROI coords (x1,y1,x2,y2) on central slice
        self.inserts = inserts or {}
        self.results = {}

    def sensitometry_linearity(self):
        # Use central slice for analysis
        mid_idx = len(self.dicom_files)//2
        ds = pydicom.dcmread(self.dicom_files[mid_idx])
        img = ds.pixel_array.astype(float)

        sensi_results = {}
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        for name, coords in self.inserts.items():
            x1, y1, x2, y2 = coords
            roi = img[y1:y2, x1:x2]
            sensi_results[name] = {
                'mean': float(np.mean(roi)),
                'std': float(np.std(roi))
            }
            # Draw ROI rectangle
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', linewidth=1.5)
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

            # Define insert ROIs (example coordinates, user should adjust per phantom)
            inserts = {
                'Teflon': (30,30,50,50),
                'Aria': (60,30,80,50),
                'Acrilico': (90,30,110,50),
                'LDPE': (120,30,140,50)
            }

            # Analyze CTP401
            ctp401 = CTP401(dicom_files, inserts=inserts)
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
