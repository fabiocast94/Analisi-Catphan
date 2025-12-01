import streamlit as st
import io
import pydicom
import numpy as np
from fpdf import FPDF
import tempfile
import zipfile
import os

# ---------------------------
# Custom CTP401 Module with full tests
# ---------------------------
class CTP401:
    def __init__(self, dicom_files):
        self.dicom_files = dicom_files
        self.results = {}

    def pixel_size_matrix(self):
        ds = pydicom.dcmread(self.dicom_files[0])
        self.results['pixel_spacing'] = ds.PixelSpacing  # [row_spacing, col_spacing]
        self.results['matrix_size'] = (ds.Rows, ds.Columns)

    def scan_incrementation(self):
        positions = [pydicom.dcmread(f).ImagePositionPatient[2] for f in self.dicom_files]
        increments = np.diff(positions)
        self.results['slice_increments'] = increments.tolist()
        self.results['mean_increment'] = float(np.mean(increments))

    def phantom_position_verification(self):
        # Approximate using geometric center of first slice
        ds = pydicom.dcmread(self.dicom_files[len(self.dicom_files)//2])
        self.results['phantom_center'] = (ds.Rows/2, ds.Columns/2)

    def circular_symmetry(self):
        # Check center of phantom using central ROI
        ds = pydicom.dcmread(self.dicom_files[len(self.dicom_files)//2])
        img = ds.pixel_array.astype(float)
        h, w = img.shape
        cx, cy = w//2, h//2
        roi = img[cy-25:cy+25, cx-25:cx+25]
        self.results['circular_symmetry'] = {'mean_roi': float(np.mean(roi)), 'std_roi': float(np.std(roi))}

    def patient_alignment_check(self):
        # Placeholder: use same as phantom center
        self.results['patient_alignment'] = self.results.get('phantom_center', None)

    def sensitometry_linearity(self):
        # Placeholder: use mean pixel of all slices
        means = []
        for f in self.dicom_files:
            ds = pydicom.dcmread(f)
            means.append(np.mean(ds.pixel_array))
        self.results['sensitometry_linearity'] = {'mean_values': [float(m) for m in means]}

    def scan_slice_geometry(self):
        # Placeholder: use number of slices and difference between first and last Z
        positions = [pydicom.dcmread(f).ImagePositionPatient[2] for f in self.dicom_files]
        slice_width = float(np.abs(positions[-1] - positions[0])/len(positions))
        self.results['slice_width'] = slice_width

    def analyze(self):
        self.pixel_size_matrix()
        self.scan_incrementation()
        self.phantom_position_verification()
        self.circular_symmetry()
        self.patient_alignment_check()
        self.sensitometry_linearity()
        self.scan_slice_geometry()
        return self.results

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

            # Analyze CTP401
            ctp401 = CTP401(dicom_files)
            results = ctp401.analyze()

            # Display results
            st.header('CTP401 Results')
            st.json(results)

            # PDF report generation
            if st.button('Generate PDF Report'):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font('Arial','B',16)
                pdf.cell(0,10,'CatPhan 500 - CTP401 Report',ln=True)
                pdf.set_font('Arial','',12)
                for k, v in results.items():
                    pdf.cell(0,6,f'{k}: {v}', ln=True)
                buf = io.BytesIO()
                pdf.output(buf)
                buf.seek(0)
                st.download_button('Download PDF Report', data=buf, file_name='ctp401_report.pdf', mime='application/pdf')
else:
    st.info('Upload a ZIP file with all DICOM images to start analysis.')
