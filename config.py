# config.py
import os
import sys
import logging
from dotenv import load_dotenv

# Muat variabel .env
load_dotenv()

# Konfigurasi Debug & Log
IS_DEBUG = os.environ.get("DEBUG_MODE", "False").lower() in ("true", "1", "yes")
LOG_LEVEL = logging.INFO if IS_DEBUG else logging.ERROR

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app_failover.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Ambil konfigurasi GCP dari env (Gunakan default value sebagai cadangan)
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "default-project-id")
GCP_SECRET_ID = os.environ.get("GCP_SECRET_ID", "gemini-api-key")

# Konstruksi URL vLLM secara dinamis dari env
VLLM_EXTERNAL_IP = os.environ.get("VLLM_EXTERNAL_IP", "127.0.0.1")
VLLM_PORT = os.environ.get("VLLM_PORT", "8000")
VLLM_BASE_URL = f"http://{VLLM_EXTERNAL_IP}:{VLLM_PORT}/v1"

def get_gemini_api_key():
    """Mengambil API Key Gemini secara berlapis."""
    try:
        from google.cloud import secretmanager
        
        client = secretmanager.SecretManagerServiceClient()
        # Menggunakan variabel yang sudah diambil dari .env
        name = f"projects/{GCP_PROJECT_ID}/secrets/{GCP_SECRET_ID}/versions/latest"
        
        response = client.access_secret_version(request={"name": name})
        logging.info("🔐 [JALUR 1] Sukses menarik API Key dari Google Secret Manager.")
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logging.warning(f"Jalur Secret Manager dilewati atau belum diizinkan: {str(e)}")

    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
