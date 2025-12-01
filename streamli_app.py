import streamlit as st
import io
import pydicom
import numpy as np
from fpdf import FPDF
import tempfile
import zipfile
import os

# ---------------------------
# Custom CTP401 Module
# ---------------------------
class CTP401:
    def __init__(self, image):
        self.image = image
        self.name = 'CTP401'

    def analyze(self):
        h, w = self.image.shape
        cx, cy = w//2, h//2
        roi = self.image[cy-25:cy+25, cx-25:cx+25]  # central 50x50 ROI
        mean = float(np.mean(roi))
        noise = float(np.std(roi))
        return {'mean': mean, 'noise': noise}

# ---------------------------
# Streamlit App
# ---------------------------
st.title('CatPhan 500 Analyzer - CTP401 Module')

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
            mid_idx = len(dicom_files) // 2
            ds = pydicom.dcmread(dicom_files[mid_idx])
            img = ds.pixel_array.astype(float)

            # Analyze CTP401
            ctp401 = CTP401(img)
            res_ctp401 = ctp401.analyze()

            # Display results
            st.header('CTP401 Results')
            st.json(res_ctp401)

            # PDF report generation
            if st.button('Generate PDF Report'):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial','B',16)
                pdf.cell(0,10,'CatPhan 500 - CTP401 Report',ln=True)
                pdf.set_font('Arial','',12)
                for k, v in res_ctp401.items():
                    pdf.cell(0,6,f'{k}: {v}', ln=True)
                buf = io.BytesIO()
                pdf.output(buf)
                buf.seek(0)
                st.download_button('Download PDF Report', data=buf, file_name='ctp401_report.pdf', mime='application/pdf')
else:
    st.info('Upload a ZIP file with all DICOM images to start analysis.')
