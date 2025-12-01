"""
Streamlit app: CatPhan 500 analyzer
Features:
- Accepts multiple DICOMs
- Auto-detect phantom center and module slice locations
- MTF calculation (ESF -> LSF -> MTF)
- Low-contrast rod detection and CNR/SNR
- Geometric checks for CTP486
- PDF report generation

Requirements (put in requirements.txt):
streamlit
pydicom
numpy
scipy
matplotlib
opencv-python
fpdf
pylinac

Notes:
- This is an advanced, practical implementation that trades off absolute clinical validation for robustness and ease-of-use.
- Tweak thresholds/parameters for your scanner/phantom if needed.

"""

import streamlit as st
import tempfile
import os
import io
import math
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import fftpack
from scipy.signal import savgol_filter
import cv2
from fpdf import FPDF
# pylinac is imported for compatibility / optional utilities
try:
    from pylinac.core.image import Image
except Exception:
    Image = None

st.set_page_config(page_title="CatPhan 500 Analyzer", layout="wide")

# -------------------------------
# Utility functions
# -------------------------------

def read_dicom_files(uploaded_files):
    dicoms = []
    zpos = []
    for f in uploaded_files:
        data = pydicom.dcmread(f)
        if hasattr(data, 'pixel_array'):
            arr = data.pixel_array.astype(np.float32)
            # rescale if needed
            if hasattr(data, 'RescaleIntercept') and hasattr(data, 'RescaleSlope'):
                arr = arr * float(data.RescaleSlope) + float(data.RescaleIntercept)
            dicoms.append(arr)
            # attempt to get z position
            try:
                z = float(data.ImagePositionPatient[2])
            except Exception:
                # fallback to InstanceNumber
                z = float(getattr(data, 'InstanceNumber', len(dicoms)))
            zpos.append(z)
    # sort by z
    idx = np.argsort(zpos)
    dicoms = [dicoms[i] for i in idx]
    zpos = [zpos[i] for i in idx]
    return dicoms, zpos


def normalize_img(img):
    a = img.astype(np.float32)
    a -= np.min(a)
    rng = np.max(a)
    if rng > 0:
        a /= rng
    return (a * 255).astype(np.uint8)


def find_phantom_center(img, blur=9, canny_thresh=(50,150)):
    # convert to uint8
    iu = normalize_img(img)
    b = cv2.GaussianBlur(iu, (blur, blur), 0)
    edges = cv2.Canny(b, canny_thresh[0], canny_thresh[1])
    # Hough circle detection
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=50, maxRadius=1500)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        return int(x), int(y), int(r)
    # fallback: centroid of largest contour
    thresh = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = iu.shape
        return w//2, h//2, min(w,h)//3
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M['m00'] == 0:
        h, w = iu.shape
        return w//2, h//2, min(w,h)//3
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    ((x,y), r) = cv2.minEnclosingCircle(c)
    return cx, cy, int(r)


def cluster_z_positions(zpos, n_clusters=4):
    # Simple clustering by gap detection to find module planes
    zs = np.array(zpos)
    diffs = np.diff(zs)
    # find large gaps
    gap_indices = np.where(diffs > (np.median(diffs) * 3))[0]
    boundaries = [0] + (gap_indices + 1).tolist() + [len(zs)]
    groups = []
    for i in range(len(boundaries)-1):
        s = boundaries[i]
        e = boundaries[i+1]
        groups.append(list(range(s,e)))
    # if we have more/less than expected, try kmeans-like grouping
    if len(groups) != n_clusters:
        # simple kmeans on z values
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(zs.reshape(-1,1))
        groups = []
        for k in range(n_clusters):
            groups.append(list(np.where(kmeans.labels_==k)[0]))
        # sort groups by mean z
        groups = sorted(groups, key=lambda g: np.mean(zs[g]))
    return groups

# -------------------------------
# Analysis: MTF (ESF -> LSF -> MTF)
# -------------------------------

def compute_mtf_from_roi(roi, edge_direction='vertical'):
    """
    roi: 2D numpy image containing a sharp edge
    edge_direction: 'vertical' if edge is vertical edge (edge runs vertical -> profile along x)
    returns: freq (normalized), mtf
    """
    # project to 1D ESF
    if edge_direction == 'vertical':
        # average rows to get ESF across columns
        esf = np.mean(roi, axis=0)
    else:
        esf = np.mean(roi, axis=1)
    # smooth ESF
    esf_s = savgol_filter(esf, window_length=51 if esf.size>51 else (esf.size//2*2+1), polyorder=3)
    # differentiate to get LSF
    lsf = np.gradient(esf_s)
    # window LSF to reduce noise
    win = np.hanning(len(lsf))
    lsf_w = lsf * win
    # fft
    mtf = np.abs(fftpack.fft(lsf_w))
    mtf = mtf[:len(mtf)//2]
    mtf /= mtf[0] if mtf[0] != 0 else 1
    freqs = np.linspace(0, 0.5, len(mtf))  # cycles/pixel (nyquist=0.5)
    return freqs, mtf

# -------------------------------
# Low contrast (CTP515): detect discs and compute CNR
# -------------------------------

def detect_circular_rods(img, dp=1.2, minDist=10, param1=50, param2=10, minR=2, maxR=40):
    iu = normalize_img(img)
    # Use HoughCircles
    circles = cv2.HoughCircles(iu, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minR, maxRadius=maxR)
    if circles is None:
        return []
    circles = np.uint16(np.around(circles[0]))
    return circles.tolist()


def compute_cnr(img, circles, bg_mask=None):
    results = []
    for c in circles:
        x, y, r = c
        rr, cc = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (rr - y)**2 + (cc - x)**2 <= (r-1)**2
        rod_vals = img[mask]
        # background annulus
        ann = ((rr - y)**2 + (cc - x)**2 <= (r+6)**2) & (~mask)
        bg_vals = img[ann]
        mean_rod = np.mean(rod_vals)
        std_bg = np.std(bg_vals)
        cnr = (mean_rod - np.mean(bg_vals)) / (std_bg + 1e-6)
        results.append({'x':int(x),'y':int(y),'r':int(r),'mean':float(mean_rod),'std_bg':float(std_bg),'cnr':float(cnr)})
    return results

# -------------------------------
# Geometric check (CTP486)
# -------------------------------

def geometric_test(img, center, radius):
    # Example geometric check: detect high-contrast markers along circle
    iu = normalize_img(img)
    h, w = iu.shape
    cx, cy = center
    # sample intensity along multiple radii
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    profile = []
    for a in angles:
        x = (cx + (radius * 0.9) * np.cos(a)).astype(int)
        y = (cy + (radius * 0.9) * np.sin(a)).astype(int)
        if 0 <= x < w and 0 <= y < h:
            profile.append(iu[y,x])
        else:
            profile.append(0)
    profile = np.array(profile)
    peaks = np.where(profile > np.percentile(profile, 95))[0]
    return {'num_markers_detected': int(len(peaks)), 'marker_angles_deg': peaks.tolist()}

# -------------------------------
# PDF Report
# -------------------------------

def create_pdf_report(results, dicom_preview, mtf_plot_buf=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'CatPhan 500 Analysis Report', ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    # Summary table
    for module, res in results.items():
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, f"Module: {module}", ln=True)
        pdf.set_font("Arial", size=11)
        if isinstance(res, dict):
            for k,v in res.items():
                if isinstance(v, (list, np.ndarray)):
                    pdf.multi_cell(0, 6, f" - {k}: [array length {len(v)}]")
                else:
                    pdf.cell(0,6,f" - {k}: {v}", ln=True)
        pdf.ln(2)

    # Add preview image
    if dicom_preview is not None:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0,8,'Preview image (center slice)', ln=True)
        # convert preview (np.uint8) to JPG in memory
        buf = io.BytesIO()
        plt.imsave(buf, dicom_preview, cmap='gray', format='jpg')
        buf.seek(0)
        pdf.image(buf, x=10, y=30, w=pdf.w - 20)

    # MTF plot
    if mtf_plot_buf is not None:
        pdf.add_page()
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0,8,'MTF plot', ln=True)
        pdf.image(mtf_plot_buf, x=10, y=30, w=pdf.w - 20)

    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# -------------------------------
# Streamlit interface
# -------------------------------

st.title("CatPhan 500 - Advanced Analyzer")
st.markdown("Carica immagini DICOM (slice CT) del CatPhan 500. L'app tenta di individuare automaticamente i moduli e calcolare MTF, CNR e test geometrici.")

uploaded_files = st.file_uploader("Carica file DICOM (multipli)", type=['dcm','dicom'], accept_multiple_files=True)

if uploaded_files:
    with st.spinner('Lettura immagini...'):
        dicoms, zpos = read_dicom_files(uploaded_files)

    st.success(f"{len(dicoms)} slice(s) lette. Z-range: {min(zpos):.2f} - {max(zpos):.2f}")

    # show center slice preview
    mid_idx = len(dicoms)//2
    center_img = dicoms[mid_idx]
    cx, cy, cr = find_phantom_center(center_img)

    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader('Preview (center slice)')
        figp, axp = plt.subplots(figsize=(4,4))
        axp.imshow(center_img, cmap='gray')
        axp.scatter([cx], [cy], c='r')
        circle = plt.Circle((cx, cy), cr, color='r', fill=False, linewidth=1)
        axp.add_patch(circle)
        axp.axis('off')
        st.pyplot(figp)

    with col2:
        st.subheader('Detected center & radius')
        st.write({'cx':cx,'cy':cy,'radius':cr})

    # cluster z positions to modules
    groups = cluster_z_positions(zpos, n_clusters=4)
    st.subheader('Module slice groups')
    for i, g in enumerate(groups):
        st.write(f"Module {i+1}: {len(g)} slice(s), indices: {g}")

    # Analysis per group
    results = {}

    for i, g in enumerate(groups):
        # pick central slice of group
        idx = g[len(g)//2]
        img = dicoms[idx]
        module_name = f"CTP_group_{i+1}"
        # Heuristics to map groups to module names by size/texture
        h, w = img.shape
        # CTP528 likely contains high-frequency resolution patterns -> find edge content
        # We'll attempt: if many edges -> assume CTP528
        edges = cv2.Canny(normalize_img(img), 60, 150)
        edge_density = np.mean(edges>0)

        if edge_density > 0.02:
            # treat as resolution module
            module_key = 'CTP528' if 'CTP528' not in results else module_name
            # select ROI near detected phantom side to find edge artifact for ESF
            # extract a narrow vertical strip at ~25% from center
            x0 = max(10, cx - int(cr*0.8))
            x1 = min(w-10, cx - int(cr*0.4))
            y0 = max(10, cy - int(cr*0.6))
            y1 = min(h-10, cy + int(cr*0.6))
            roi = img[y0:y1, x0:x1]
            # attempt vertical edge detection via Sobel
            sob = np.abs(cv2.Sobel(normalize_img(roi), cv2.CV_64F, 1, 0, ksize=3))
            # choose column with max gradient
            col_sum = np.mean(sob, axis=0)
            edge_col = np.argmax(col_sum)
            # create narrow ROI around edge column
            roi_edge = roi[:, max(0,edge_col-40):min(roi.shape[1],edge_col+40)]
            freqs, mtf = compute_mtf_from_roi(roi_edge, edge_direction='vertical')
            results['CTP528'] = {'freqs': freqs.tolist(), 'mtf': mtf.tolist(), 'edge_density': float(edge_density)}
        else:
            # check for low-contrast: many small circular rods
            circles = detect_circular_rods(img, dp=1.2, minDist=6, param1=50, param2=8, minR=3, maxR=30)
            if len(circles) >= 6:
                module_key = 'CTP515'
                cnr_res = compute_cnr(img, circles)
                results['CTP515'] = {'num_rods': len(circles), 'rods': cnr_res}
            else:
                # fallback uniformity/noise (CTP401) or geometric (CTP486)
                # if low texture and large uniform area -> CTP401
                if edge_density < 0.005:
                    module_key = 'CTP401'
                    # compute uniformity on central ROI
                    cy0 = max(0, cy - cr//3)
                    cy1 = min(h, cy + cr//3)
                    cx0 = max(0, cx - cr//3)
                    cx1 = min(w, cx + cr//3)
                    center_roi = img[cy0:cy1, cx0:cx1]
                    meanv = float(np.mean(center_roi))
                    noise = float(np.std(center_roi))
                    results['CTP401'] = {'mean': meanv, 'noise': noise}
                else:
                    module_key = 'CTP486'
                    geom = geometric_test(img, (cx,cy), cr)
                    results['CTP486'] = geom

    st.header('Results')
    st.json(results)

    # show MTF plot if available
    mtf_plot_buf = None
    if 'CTP528' in results:
        freqs = np.array(results['CTP528']['freqs'])
        mtf = np.array(results['CTP528']['mtf'])
        figm, axm = plt.subplots()
        axm.plot(freqs, mtf)
        axm.set_xlabel('cycles/pixel')
        axm.set_ylabel('MTF')
        axm.set_title('Estimated MTF')
        axm.grid(True)
        buf = io.BytesIO()
        figm.savefig(buf, format='jpg')
        buf.seek(0)
        mtf_plot_buf = buf
        st.pyplot(figm)

    # Create PDF report
    if st.button('Generate PDF report'):
        with st.spinner('Creazione report PDF...'):
            preview = normalize_img(center_img)
            pdf_buf = create_pdf_report(results, preview, mtf_plot_buf)
            st.success('Report generato')
            st.download_button('Download report (PDF)', data=pdf_buf, file_name='catphan500_report.pdf', mime='application/pdf')

    st.info('Se i risultati non sembrano corretti, prova a caricare solo le slice del singolo modulo oppure regola i parametri di rilevamento (Hough, thresholds) nel codice).')
else:
    st.info('Carica le immagini DICOM del CatPhan 500 per iniziare l\'analisi.')
