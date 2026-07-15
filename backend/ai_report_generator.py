"""
ai_report_generator.py
======================
Modul pembuat laporan keselamatan kerja otomatis menggunakan Ollama Cloud API.

Laporan dihasilkan oleh model bahasa besar (LLM) berdasarkan statistik PPE
yang dikumpulkan selama pemrosesan video. Laporan mencakup ringkasan,
persentase kepatuhan, penilaian risiko, dan rekomendasi.

Konfigurasi:
    Model    : gpt-oss:120b (Ollama Cloud)
    API Host : https://ollama.com
    Auth     : Bearer token dari environment variable OLLAMA_API_KEY

Penggunaan:
    from backend.ai_report_generator import generate_report

    statistics = {
        "total_person":   10,
        "safe_worker":    7,
        "no_helmet":      2,
        "no_safety_vest": 1,
        "fire_alert":     0,
        "smoke_alert":    0
    }
    report = generate_report(statistics)

Error Handling:
    Fungsi ini TIDAK melempar exception ke pemanggil. Jika terjadi
    kesalahan (API key tidak ada, koneksi gagal, timeout, dll),
    pesan error deskriptif akan dikembalikan sebagai string biasa
    sehingga server tidak crash.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from ollama import Client

# Muat otomatis file .env dari root project (dua level di atas file ini)
# Sehingga OLLAMA_API_KEY tersedia via os.environ tanpa perlu set manual di terminal
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE)


# ---------------------------------------------------------------------------
# KONSTANTA
# ---------------------------------------------------------------------------

MODEL_NAME   = "gpt-oss:120b"
OLLAMA_HOST  = "https://ollama.com"
REQUEST_TIMEOUT = 120   # detik


def _build_prompt(statistics: dict) -> str:
    """
    Susun prompt analisis keselamatan dari statistik PPE.

    Args:
        statistics (dict): Statistik PPE hasil video processing.

    Returns:
        str: Prompt lengkap yang siap dikirim ke LLM.
    """
    total_person   = statistics.get("total_person",   0)
    safe_worker    = statistics.get("safe_worker",    0)
    no_helmet      = statistics.get("no_helmet",      0)
    no_safety_vest = statistics.get("no_safety_vest", 0)
    fire_alert     = statistics.get("fire_alert",     0)
    smoke_alert    = statistics.get("smoke_alert",    0)

    return (
        "Kamu adalah asisten petugas keselamatan kerja.\n\n"
        "Analisis statistik keselamatan kerja berikut dari rekaman video:\n\n"
        f"Total pekerja terdeteksi: {total_person}\n"
        f"Pekerja aman: {safe_worker}\n"
        f"Pekerja tanpa helm: {no_helmet}\n"
        f"Pekerja tanpa rompi keselamatan: {no_safety_vest}\n"
        f"Alert kebakaran: {fire_alert}\n"
        f"Alert asap: {smoke_alert}\n\n"
        "Buat laporan dalam Bahasa Indonesia yang mencakup:\n\n"
        "1. Ringkasan Keselamatan\n"
        "2. Persentase Kepatuhan\n"
        "3. Penilaian Risiko\n"
        "4. Rekomendasi\n\n"
        "Gunakan bahasa yang profesional dan ringkas."
    )


def generate_report(statistics: dict) -> str:
    """
    Hasilkan laporan keselamatan kerja otomatis menggunakan Ollama Cloud API.

    Fungsi ini memvalidasi environment variable, membangun prompt,
    mengirim request ke Ollama Cloud, dan mengembalikan teks laporan.
    Semua error ditangani secara internal dan dikembalikan sebagai pesan
    string — fungsi tidak pernah melempar exception ke pemanggil.

    Args:
        statistics (dict): Statistik PPE dari StatisticsEngine.get_statistics().
            Kunci yang diharapkan:
                - total_person   (int)
                - safe_worker    (int)
                - no_helmet      (int)
                - no_safety_vest (int)
                - fire_alert     (int)
                - smoke_alert    (int)

    Returns:
        str: Laporan keselamatan dalam bahasa Inggris yang dihasilkan oleh LLM.
             Jika terjadi kesalahan, mengembalikan pesan deskriptif (bukan exception).

    Examples:
        >>> stats = {"total_person": 10, "safe_worker": 7, "no_helmet": 2,
        ...          "no_safety_vest": 1, "fire_alert": 0, "smoke_alert": 0}
        >>> report = generate_report(stats)
        >>> print(report)
        Safety Summary
        A total of 10 workers were detected...
    """
    # -----------------------------------------------------------------------
    # VALIDASI API KEY
    # -----------------------------------------------------------------------
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()

    if not api_key:
        return (
            "AI Safety Report tidak tersedia.\n\n"
            "Environment variable OLLAMA_API_KEY belum diset.\n"
            "Tambahkan OLLAMA_API_KEY ke environment sistem dan restart server."
        )

    # -----------------------------------------------------------------------
    # BANGUN PROMPT
    # -----------------------------------------------------------------------
    prompt = _build_prompt(statistics)

    # -----------------------------------------------------------------------
    # KIRIM REQUEST KE OLLAMA CLOUD
    # -----------------------------------------------------------------------
    try:
        client = Client(
            host=OLLAMA_HOST,
            headers={"Authorization": f"Bearer {api_key}"}
        )

        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        report_text = response.message.content

        if not report_text or not report_text.strip():
            return "AI Safety Report: Model mengembalikan respons kosong."

        return report_text.strip()

    except ConnectionError as e:
        return (
            f"AI Safety Report tidak dapat dibuat.\n\n"
            f"Gagal terhubung ke Ollama Cloud ({OLLAMA_HOST}).\n"
            f"Detail: {str(e)}"
        )

    except TimeoutError:
        return (
            f"AI Safety Report tidak dapat dibuat.\n\n"
            f"Request ke Ollama Cloud melebihi batas waktu ({REQUEST_TIMEOUT} detik)."
        )

    except Exception as e:
        return (
            f"AI Safety Report tidak dapat dibuat.\n\n"
            f"Error: {str(e)}"
        )


# ---------------------------------------------------------------------------
# IMAGE REPORT
# ---------------------------------------------------------------------------

def _build_image_prompt(statistics: dict) -> str:
    """
    Susun prompt analisis keselamatan dari statistik PPE hasil deteksi gambar.

    Args:
        statistics (dict): Statistik PPE hasil image detection.

    Returns:
        str: Prompt lengkap yang siap dikirim ke LLM.
    """
    safe_worker    = statistics.get("safe_worker",    0)
    no_helmet      = statistics.get("no_helmet",      0)
    no_safety_vest = statistics.get("no_safety_vest", 0)
    total_worker   = safe_worker + no_helmet + no_safety_vest

    object_count   = statistics.get("object_count", {})
    fire           = object_count.get("fire",  0)
    smoke          = object_count.get("smoke", 0)

    return (
        "Kamu adalah asisten petugas keselamatan kerja.\n\n"
        "Analisis hasil deteksi APD (Alat Pelindung Diri) berikut dari sebuah gambar:\n\n"
        f"Total pekerja terdeteksi: {total_worker}\n"
        f"Pekerja aman: {safe_worker}\n"
        f"Pekerja tanpa helm: {no_helmet}\n"
        f"Pekerja tanpa rompi keselamatan: {no_safety_vest}\n"
        f"Api terdeteksi: {fire}\n"
        f"Asap terdeteksi: {smoke}\n\n"
        "Buat laporan dalam Bahasa Indonesia yang mencakup:\n\n"
        "1. Ringkasan Keselamatan\n"
        "2. Persentase Kepatuhan\n"
        "3. Penilaian Risiko\n"
        "4. Rekomendasi\n\n"
        "Gunakan bahasa yang profesional dan ringkas."
    )


def generate_image_report(statistics: dict) -> str:
    """
    Hasilkan laporan keselamatan untuk hasil deteksi gambar menggunakan Ollama Cloud API.

    Fungsi ini memvalidasi environment variable, membangun prompt khusus gambar,
    mengirim request ke Ollama Cloud, dan mengembalikan teks laporan.
    Semua error ditangani secara internal — fungsi tidak pernah melempar exception.

    Args:
        statistics (dict): Statistik PPE dari predict_image endpoint.
            Kunci yang diharapkan:
                - safe_worker    (int)
                - no_helmet      (int)
                - no_safety_vest (int)
                - object_count   (dict) dengan kunci 'fire' dan 'smoke'

    Returns:
        str: Laporan keselamatan dalam bahasa Inggris yang dihasilkan oleh LLM.
             Jika terjadi kesalahan, mengembalikan pesan deskriptif (bukan exception).
    """
    # -----------------------------------------------------------------------
    # VALIDASI API KEY
    # -----------------------------------------------------------------------
    api_key = os.environ.get("OLLAMA_API_KEY", "").strip()

    if not api_key:
        return (
            "AI Safety Report tidak tersedia.\n\n"
            "Environment variable OLLAMA_API_KEY belum diset.\n"
            "Tambahkan OLLAMA_API_KEY ke environment sistem dan restart server."
        )

    # -----------------------------------------------------------------------
    # BANGUN PROMPT KHUSUS GAMBAR
    # -----------------------------------------------------------------------
    prompt = _build_image_prompt(statistics)

    # -----------------------------------------------------------------------
    # KIRIM REQUEST KE OLLAMA CLOUD
    # -----------------------------------------------------------------------
    try:
        client = Client(
            host=OLLAMA_HOST,
            headers={"Authorization": f"Bearer {api_key}"}
        )

        response = client.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

        report_text = response.message.content

        if not report_text or not report_text.strip():
            return "AI Safety Report: Model mengembalikan respons kosong."

        return report_text.strip()

    except ConnectionError as e:
        return (
            f"AI Safety Report tidak dapat dibuat.\n\n"
            f"Gagal terhubung ke Ollama Cloud ({OLLAMA_HOST}).\n"
            f"Detail: {str(e)}"
        )

    except TimeoutError:
        return (
            f"AI Safety Report tidak dapat dibuat.\n\n"
            f"Request ke Ollama Cloud melebihi batas waktu ({REQUEST_TIMEOUT} detik)."
        )

    except Exception as e:
        return (
            f"AI Safety Report tidak dapat dibuat.\n\n"
            f"Error: {str(e)}"
        )
