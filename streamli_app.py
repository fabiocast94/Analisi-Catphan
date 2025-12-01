import streamlit as st
import io
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from pylinac.ct import CatPhan, CTP528, CTP515, CTP486, CTPModule

# ---------------------------
# Custom CTP401 Module
# ---------------------------
class CTP401(CTPModule):
    def __init__(self, catphan, name='CTP401'):
        super().__init__(catphan, name)
        self.roi_size = 50  # size of central ROI for uniformity/noise

    def analyze(self):
        img = self.image  # assumes central slice
        h, w = img.shape
        cx, cy = w//2, h//2
        half = self.roi_size // 2
        roi = img[cy-half:cy+half, cx-half:cx+half]
        mean = float(np.mean(roi))
        noise = float(np.std(roi))
        self.results = {'mean': mean, 'noise': noise}
        return self.results

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

    st.info('Creating CatPhan object with custom CTP401 module...')

    # Create CatPhan with default modules
    cp = CatPhan(dicom_paths, name='CatPhan500')

    # Add custom CTP401 module
    cp.modules.append(CTP401(cp))

    st.info('Analyzing modules...')
    results = {}

    # Analyze each module
    for module in cp.modules:
        try:
            res = module.analyze()
            results[module.name] = res
        except Exception as e:
            results[module.name] = f'Error: {e}'

    st.header('Results')
    st.json(results)

    # Display MTF plot for CTP528
    if 'CTP528' in results:
        module_528 = next((m for m in cp.modules if isinstance(m, CTP528)), None)
        if module_528:
            fig, ax = plt.subplots()
            ax.plot(module_528.mtf['freq'], module_528.mtf['mtf'])
            ax.set_title('CTP528 MTF')
            ax.set_xlabel('cycles/mm')
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
            else:
                pdf.cell(0,6,str(res), ln=True)
        buf = io.BytesIO()
        pdf.output(buf)
        buf.seek(0)
        st.download_button('Download PDF Report', data=buf, file_name='catphan500_report.pdf', mime='application/pdf')
else:
    st.info('Upload DICOM files to start analysis.')
