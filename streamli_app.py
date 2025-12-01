import streamlit as st
import io
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from pylinac import CatPhan

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
st.title('CatPhan 500 Analyzer with pylinac')

uploaded_files = st.file_uploader('Upload DICOM files (CatPhan 500)', type=['dcm','dicom'], accept_multiple_files=True)

if uploaded_files:
    dicom_paths = []
    for f in uploaded_files:
        with open(f.name, 'wb') as temp_file:
            temp_file.write(f.read())
            dicom_paths.append(f.name)

    st.info('Creating CatPhan object...')
    cp = CatPhan(dicom_paths, name='CatPhan500')

    st.info('Analyzing modules with pylinac...')
    cp.analyze()

    # Custom CTP401 analysis (use central slice)
    mid_idx = len(dicom_paths)//2
    ds = pydicom.dcmread(dicom_paths[mid_idx])
    img = ds.pixel_array.astype(float)
    ctp401 = CTP401(img)
    res_ctp401 = ctp401.analyze()

    # Collect results
    results = {
        'CTP401': res_ctp401
    }

    # Add pylinac module results
    if hasattr(cp, 'ctp528'):
        results['CTP528'] = {
            'mtf': cp.ctp528.mtf['mtf'].tolist() if cp.ctp528.mtf else None
        }
    if hasattr(cp, 'ctp515'):
        results['CTP515'] = {
            'rods': len(cp.ctp515.rods) if cp.ctp515.rods else 0
        }
    if hasattr(cp, 'ctp486'):
        results['CTP486'] = {
            'num_markers': len(cp.ctp486.markers) if cp.ctp486.markers else 0
        }

    st.header('Results')
    st.json(results)

    # MTF plot for CTP528
    if 'CTP528' in results and results['CTP528']['mtf'] is not None:
        fig, ax = plt.subplots()
        ax.plot(results['CTP528']['mtf'])
        ax.set_title('CTP528 MTF')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('MTF')
        st.pyplot(fig)

    # PDF report generation
    if st.button('Generate PDF Report'):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial','B',16)
        pdf.cell(0,10,'CatPhan 500 Report',ln=True)
        pdf.set_font('Arial','',12)
        for mod, res in results.items():
            pdf.cell(0,8,f'Module: {mod}', ln=True)
            if isinstance(res, dict):
                for k,v in res.items():
                    pdf.cell(0,6,f'  {k}: {v}', ln=True)
        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)
        st.download_button('Download PDF Report', data=buf, file_name='catphan500_report.pdf', mime='application/pdf')
else:
    st.info('Upload DICOM files to start analysis.')
