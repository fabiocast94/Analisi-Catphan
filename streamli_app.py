def find_marker_slice(images, marker_roi=(0, 0, 20, 50), threshold=200):
    """
    Trova la slice corretta basandosi su un marker bianco nella zona sinistra.
    marker_roi: (x1, y1, x2, y2) del ROI dove cercare il marker
    threshold: soglia di pixel per considerare marker presente
    """
    x1, y1, x2, y2 = marker_roi
    for img, ds in images:
        # Prendi primo slice se 3D
        if img.ndim == 3:
            img_slice = img[0,:,:]
        else:
            img_slice = img
        
        roi = img_slice[y1:y2, x1:x2]
        roi_mean = np.mean(roi)
        if roi_mean >= threshold:
            return img_slice, ds
    return None, None

# ---------- Uso nel template ----------
if uploaded_zip:
    images = load_dicom_from_zip(uploaded_zip)
    st.success(f"{len(images)} DICOM trovati nel ZIP")
    
    # Trova la slice giusta con marker
    img_correct, ds_correct = find_marker_slice(images)
    if img_correct is None:
        st.error("Non è stata trovata la slice corretta con il marker bianco!")
    else:
        st.success(f"Slice selezionata: {ds_correct.SOPInstanceUID}")
        show_image(img_correct, title=f"Slice corretta: {ds_correct.SOPInstanceUID}")

        # Ora tutte le analisi (sensitometry, slice geometry, ecc.) si fanno solo su questa slice
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
