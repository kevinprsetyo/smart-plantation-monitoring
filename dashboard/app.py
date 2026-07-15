"""
app.py
======
Dashboard Streamlit untuk Smart Plantation Safety Monitoring System.

Tampilan:
    Tab 1 - Image Detection  : Upload gambar, proses, tampilkan hasil deteksi PPE.
    Tab 2 - Video Detection  : Upload video, proses, preview output, timeline, AI report.
    Tab 3 - Violation Gallery: Galeri seluruh snapshot pelanggaran yang tersimpan.

Alur komunikasi:
    Streamlit (app.py) <--HTTP--> FastAPI (backend/main.py)

    Upload gambar  : POST /predict-image
    Upload video   : POST /predict-video
    Preview video  : GET  /video/{filename}
    Daftar galeri  : GET  /gallery
    Serve snapshot : GET  /snapshot/{category}/{filename}

Cara menjalankan:
    streamlit run dashboard/app.py

Catatan:
    FastAPI harus berjalan di http://127.0.0.1:8000 sebelum Streamlit dijalankan.
    Timeout POST video diatur ke 1800 detik (30 menit) untuk video panjang.
    OLLAMA_API_KEY harus diset di environment FastAPI untuk mengaktifkan AI Report.
"""

import streamlit as st
import requests
import pandas as pd


# ---------------------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------------------

API_BASE       = "http://127.0.0.1:8000"
GALLERY_COLS   = 3   # jumlah kolom pada tampilan galeri


st.set_page_config(
    page_title="Smart Plantation Safety Monitoring",
    layout="wide"
)

st.title("Smart Plantation Safety Monitoring System")


# ---------------------------------------------------------------------------
# TAB NAVIGASI
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "Image Detection",
    "Video Detection",
    "Violation Gallery"
])


# ===========================================================================
# TAB 1 — IMAGE DETECTION
# ===========================================================================

with tab1:

    st.header("Image Detection")

    uploaded_image = st.file_uploader(
        "Upload Image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="image"
    )

    if uploaded_image:

        st.subheader("Input Image")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_image, width=800)

        if st.button("Process Image", key="process_image"):

            with st.spinner("Memproses gambar..."):

                # Reset ke awal file sebelum dikirim ke API
                uploaded_image.seek(0)

                files = {
                    "file": (
                        uploaded_image.name,
                        uploaded_image,
                        uploaded_image.type
                    )
                }

                try:
                    response = requests.post(
                        f"{API_BASE}/predict-image",
                        files=files,
                        timeout=120
                    )
                    response.raise_for_status()
                    data = response.json()

                except requests.exceptions.ConnectionError:
                    st.error("Tidak bisa terhubung ke API. Pastikan FastAPI berjalan di port 8000.")
                    st.stop()

                except requests.exceptions.Timeout:
                    st.error("Request timeout. Coba lagi dengan gambar berukuran lebih kecil.")
                    st.stop()

                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e.response.text}")
                    st.stop()

            st.success("Gambar berhasil diproses.")

            # ---------------------------------------------------------------
            # OUTPUT IMAGE
            # ---------------------------------------------------------------
            st.subheader("Output Image")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(data["output_image"], width=800)

            # ---------------------------------------------------------------
            # PPE STATISTICS
            # ---------------------------------------------------------------
            st.subheader("PPE Statistics")

            stats = data["ppe_statistics"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Safe Worker",    stats["safe_worker"])
            c2.metric("No Helmet",      stats["no_helmet"])
            c3.metric("No Safety Vest", stats["no_safety_vest"])

            # ---------------------------------------------------------------
            # WORKER SAFETY STATUS
            # ---------------------------------------------------------------
            st.subheader("Worker Safety Status")

            if data["violations"]:
                df = pd.DataFrame(data["violations"])
                st.dataframe(df, width="stretch")
            else:
                st.info("Tidak ada worker terdeteksi.")

            # ---------------------------------------------------------------
            # OBJECT COUNT
            # ---------------------------------------------------------------
            st.subheader("Object Count")

            obj = data["object_count"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Person",      obj["person"])
            c2.metric("Helmet",      obj["helmet"])
            c3.metric("Safety Vest", obj["safety_vest"])
            c4.metric("Truck",       obj["truck"])

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Forklift",  obj["forklift"])
            c2.metric("Excavator", obj["excavator"])
            c3.metric("Fire",      obj["fire"])
            c4.metric("Smoke",     obj["smoke"])

            # ---------------------------------------------------------------
            # AI SAFETY REPORT
            # Model mengembalikan teks dalam format Markdown, sehingga
            # ditampilkan menggunakan st.markdown() agar heading, tabel,
            # dan teks tebal ter-render dengan benar.
            # ---------------------------------------------------------------
            st.subheader("AI Safety Report")

            ai_report = data.get("ai_report", "")

            if ai_report:
                with st.container(border=True):
                    st.markdown(ai_report)
            else:
                st.info("AI Safety Report tidak tersedia.")


# ===========================================================================
# TAB 2 — VIDEO DETECTION
# ===========================================================================

with tab2:

    st.header("Video Detection")

    uploaded_video = st.file_uploader(
        "Upload Video (MP4)",
        type=["mp4"],
        key="video"
    )

    if uploaded_video:

        st.subheader("Input Video")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.video(uploaded_video)

        if st.button("Process Video", key="process_video"):

            with st.spinner("Memproses video... (estimasi waktu tergantung panjang video)"):

                # Reset ke awal file sebelum dikirim ke API
                uploaded_video.seek(0)

                files = {
                    "file": (
                        uploaded_video.name,
                        uploaded_video,
                        "video/mp4"
                    )
                }

                try:
                    response = requests.post(
                        f"{API_BASE}/predict-video",
                        files=files,
                        timeout=1800
                    )
                    response.raise_for_status()
                    data = response.json()

                except requests.exceptions.ConnectionError:
                    st.error("Tidak bisa terhubung ke API. Pastikan FastAPI berjalan di port 8000.")
                    st.stop()

                except requests.exceptions.Timeout:
                    st.error("Request timeout. Video terlalu panjang atau server tidak responsif.")
                    st.stop()

                except requests.exceptions.HTTPError as e:
                    st.error(f"API error: {e.response.text}")
                    st.stop()

            st.success("Video berhasil diproses.")

            # ---------------------------------------------------------------
            # PPE STATISTICS
            # ---------------------------------------------------------------
            st.subheader("PPE Statistics")

            stats = data["statistics"]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Worker",   stats["total_person"])
            c2.metric("Safe Worker",    stats["safe_worker"])
            c3.metric("No Helmet",      stats["no_helmet"])
            c4.metric("No Safety Vest", stats["no_safety_vest"])
            c5.metric("Fire Alert",     stats["fire_alert"])

            # ---------------------------------------------------------------
            # OUTPUT VIDEO PREVIEW
            # Fetch video dari FastAPI endpoint sebagai bytes agar tidak
            # bergantung pada path file sistem.
            # ---------------------------------------------------------------
            st.subheader("Output Video")

            output_filename = data["output_video"]
            video_url       = f"{API_BASE}/video/{output_filename}"

            with st.spinner("Memuat preview video..."):
                try:
                    video_response = requests.get(video_url, timeout=300)
                    video_response.raise_for_status()
                    video_bytes = video_response.content

                    if len(video_bytes) == 0:
                        st.error("File video output kosong.")
                    else:
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.video(video_bytes)

                except requests.exceptions.RequestException as e:
                    st.warning(
                        f"Tidak bisa memuat preview video: {e}\n\n"
                        "Gunakan tombol Download di bawah untuk mengunduh video."
                    )

            # ---------------------------------------------------------------
            # TIMELINE ALERT
            # Menampilkan event pelanggaran secara kronologis.
            # Hanya event dengan perubahan status yang ditampilkan.
            # ---------------------------------------------------------------
            st.subheader("Timeline Alert")

            timeline_events = data.get("timeline_events", [])

            if timeline_events:
                df_timeline = pd.DataFrame(timeline_events)

                # Urutkan kolom agar tampilan konsisten
                df_timeline = df_timeline[["time", "track_id", "status"]]
                df_timeline.columns = ["Time", "Worker ID", "Status"]

                st.dataframe(df_timeline, width="stretch")
            else:
                st.info("Tidak ada event pelanggaran yang tercatat dalam video ini.")

            # ---------------------------------------------------------------
            # WORKER SAFETY STATUS
            # ---------------------------------------------------------------
            if "violations" in data:

                st.subheader("Worker Safety Status")

                if data["violations"]:
                    df_violations = pd.DataFrame(data["violations"])
                    st.dataframe(df_violations, width="stretch")
                else:
                    st.info("Tidak ada worker terdeteksi dalam video.")

            # ---------------------------------------------------------------
            # OBJECT COUNT
            # ---------------------------------------------------------------
            if "object_count" in data:

                st.subheader("Object Count (Maksimum per Frame)")

                obj = data["object_count"]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Person",      obj["person"])
                c2.metric("Helmet",      obj["helmet"])
                c3.metric("Safety Vest", obj["safety_vest"])
                c4.metric("Truck",       obj["truck"])

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Forklift",  obj["forklift"])
                c2.metric("Excavator", obj["excavator"])
                c3.metric("Fire",      obj["fire"])
                c4.metric("Smoke",     obj["smoke"])

            # ---------------------------------------------------------------
            # AI SAFETY REPORT
            # Model mengembalikan teks dalam format Markdown, sehingga
            # ditampilkan menggunakan st.markdown() agar heading, tabel,
            # dan teks tebal ter-render dengan benar.
            # ---------------------------------------------------------------
            st.subheader("AI Safety Report")

            ai_report = data.get("ai_report", "")

            if ai_report:
                with st.container(border=True):
                    st.markdown(ai_report)
            else:
                st.info("AI Safety Report tidak tersedia.")

            # ---------------------------------------------------------------
            # DOWNLOAD OUTPUT VIDEO
            # ---------------------------------------------------------------
            st.subheader("Download")

            try:
                dl_response = requests.get(video_url, timeout=300)
                dl_response.raise_for_status()

                st.download_button(
                    label="Download Processed Video (H.264 MP4)",
                    data=dl_response.content,
                    file_name=output_filename,
                    mime="video/mp4"
                )

            except requests.exceptions.RequestException as e:
                st.warning(f"Tidak bisa menyiapkan file download: {e}")


# ===========================================================================
# TAB 3 — VIOLATION GALLERY
# ===========================================================================

with tab3:

    st.header("Violation Gallery")
    st.caption(
        "Snapshot frame pelanggaran yang tersimpan selama pemrosesan video. "
        "Snapshot baru muncul setelah video berhasil diproses."
    )

    # Refresh button agar galeri bisa diperbarui tanpa reload halaman
    if st.button("Refresh Galeri", key="refresh_gallery"):
        st.rerun()

    # -----------------------------------------------------------------------
    # FETCH DAFTAR SNAPSHOT DARI API
    # -----------------------------------------------------------------------
    try:
        gallery_response = requests.get(f"{API_BASE}/gallery", timeout=30)
        gallery_response.raise_for_status()
        gallery_data = gallery_response.json()

    except requests.exceptions.ConnectionError:
        st.error("Tidak bisa terhubung ke API. Pastikan FastAPI berjalan di port 8000.")
        st.stop()

    except requests.exceptions.RequestException as e:
        st.error(f"Gagal memuat galeri: {e}")
        st.stop()

    # -----------------------------------------------------------------------
    # TAMPILKAN GALERI PER KATEGORI PELANGGARAN
    # -----------------------------------------------------------------------

    # Urutan tampilan kategori (dari yang paling kritis)
    CATEGORY_ORDER = ["NO_HELMET", "NO_SAFETY_VEST", "FIRE_ALERT", "SMOKE_ALERT"]

    total_snapshots = sum(
        len(gallery_data[cat]["files"])
        for cat in CATEGORY_ORDER
        if cat in gallery_data
    )

    if total_snapshots == 0:
        st.info(
            "Belum ada snapshot pelanggaran. "
            "Proses video terlebih dahulu untuk menghasilkan snapshot."
        )

    else:
        st.write(f"Total snapshot: **{total_snapshots}** gambar")
        st.divider()

        for category_label in CATEGORY_ORDER:

            if category_label not in gallery_data:
                continue

            category_info = gallery_data[category_label]
            folder_name   = category_info["folder"]
            files         = category_info["files"]

            if not files:
                continue

            st.subheader(f"{category_label}  ({len(files)} snapshot)")

            # Tampilkan gambar dalam grid GALLERY_COLS kolom
            for row_start in range(0, len(files), GALLERY_COLS):

                row_files = files[row_start : row_start + GALLERY_COLS]
                cols      = st.columns(GALLERY_COLS)

                for col_idx, filename in enumerate(row_files):

                    with cols[col_idx]:

                        snapshot_url = (
                            f"{API_BASE}/snapshot/{folder_name}/{filename}"
                        )

                        # Fetch gambar dari API sebagai bytes
                        try:
                            img_response = requests.get(
                                snapshot_url,
                                timeout=30
                            )
                            img_response.raise_for_status()
                            img_bytes = img_response.content

                            # Tampilkan preview gambar
                            st.image(
                                img_bytes,
                                width="stretch"
                            )

                            # Nama file (disingkat jika terlalu panjang)
                            display_name = filename if len(filename) <= 30 else filename[:27] + "..."
                            st.caption(display_name)

                            # Tombol download per gambar
                            st.download_button(
                                label="Download",
                                data=img_bytes,
                                file_name=filename,
                                mime="image/jpeg",
                                key=f"dl_{category_label}_{filename}"
                            )

                        except requests.exceptions.RequestException:
                            st.warning(f"Gagal memuat: {filename}")

            st.divider()