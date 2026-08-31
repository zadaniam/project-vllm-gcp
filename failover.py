import os
import sys
import time
import logging
from openai import OpenAI, APIConnectionError, APITimeoutError
from google import genai
from google.genai import errors
# Import library untuk membaca file .env
from dotenv import load_dotenv

# Muat variabel dari file .env ke dalam sistem (Hanya aktif saat di MacBook lokal)
load_dotenv()

# =====================================================================
# CONFIG LOGGER (Membaca status DEBUG_MODE dari .env / Cloud Environment)
# =====================================================================
IS_DEBUG = os.environ.get("DEBUG_MODE", "False").lower() in ("true", "1", "yes")
log_level = logging.INFO if IS_DEBUG else logging.ERROR

logging.basicConfig(
    level=log_level,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app_failover.log"), # File log tetap mencatat di background
        logging.StreamHandler(sys.stdout)        # Output ke terminal disesuaikan levelnya
    ]
)

# =====================================================================
# STRATEGI FALLBACK: GOOGLE SECRET MANAGER VS LOKAL .ENV
# =====================================================================
def get_gemini_api_key():
    """
    Otomatis mendeteksi lingkungan. Jika di server GCE, ambil dari Secret Manager.
    Jika gagal/di MacBook lokal, otomatis fallback membaca file .env Anda.
    """
    # 1. Coba jalur industri Google Cloud Secret Manager (Saat dideploy di GCE)
    try:
        from google.cloud import secretmanager
        
        # Sesuai nama project di Google Cloud Console Anda
        PROJECT_ID = "my-first-project" 
        SECRET_ID = "gemini-api-key"
        
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/{SECRET_ID}/versions/latest"
        
        response = client.access_secret_version(request={"name": name})
        logging.info("🔐 [GCE ENVIRONMENT] API Key Gemini ditarik langsung dari Secret Manager.")
        return response.payload.data.decode("UTF-8")
        
    except Exception as e:
        # 2. Jika gagal/error (Saat debugging di VS Code MacBook Anda)
        logging.info("💻 [LOCAL ENVIRONMENT] Mencari kunci API dari berkas .env lokal...")
        api_key_lokal = os.environ.get("GEMINI_API_KEY")
        return api_key_lokal

# =====================================================================
# CONFIG SERVER & CLOUD ENGINE
# =====================================================================
VLLM_EXTERNAL_IP = "136.85.107.67"
VLLM_BASE_URL = f"http://{VLLM_EXTERNAL_IP}:8000/v1" 

# Inisialisasi Klien OpenAI (vLLM)
vllm_client = OpenAI(base_url=VLLM_BASE_URL, api_key="vllm-placeholder", timeout=5.0)

# Ambil API Key dengan sistem deteksi otomatis ganda
GEMINI_API_KEY = get_gemini_api_key()
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

class ChatSessionWithFailover:
    def __init__(self):
        self.history = []
        self.system_instruction = "Anda adalah asisten AI yang ramah, profesional, dan membantu."

    def get_vllm_messages(self, current_user_prompt: str):
        vllm_formatted = []
        if not self.history:
            combined_first_prompt = f"{self.system_instruction}\n\nPertanyaan pengguna:\n{current_user_prompt}"
            vllm_formatted.append({"role": "user", "content": combined_first_prompt})
        else:
            for msg in self.history:
                vllm_formatted.append({"role": msg["role"], "content": msg["content"]})
            vllm_formatted.append({"role": "user", "content": current_user_prompt})
        return vllm_formatted

    def get_gemini_contents(self, current_user_prompt: str):
        gemini_contents = []
        for msg in self.history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        gemini_contents.append({"role": "user", "parts": [{"text": current_user_prompt}]})
        return gemini_contents

    def send_message_stream(self, user_prompt: str):
        assistant_response = ""

        # -----------------------------------------------------------------
        # TRY: JALUR UTAMA (vLLM - Gemma 2B dengan History)
        # -----------------------------------------------------------------
        try:
            logging.info(f"Menghubungi server vLLM ({VLLM_EXTERNAL_IP}) via Streaming...")
            vllm_messages = self.get_vllm_messages(user_prompt)
            
            stream = vllm_client.chat.completions.create(
                model="google/gemma-2-2b-it",
                messages=vllm_messages,
                stream=True
            )
            
            if IS_DEBUG:
                print("🤖 AI (vLLM): ", end="")
            else:
                print("🤖 AI: ", end="")
                
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    assistant_response += token
                    yield token
            
            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": assistant_response})
            logging.info("Sukses menggunakan vLLM untuk sesi ini.")
            return

        except (APIConnectionError, APITimeoutError) as e:
            logging.warning(f"Koneksi vLLM Bermasalah ({type(e).__name__}). Mengalihkan ke Gemini...")
        except Exception as e:
            logging.error(f"Eror internal pada parser vLLM: {str(e)}. Mengalihkan ke Gemini...")

        # -----------------------------------------------------------------
        # CATCH: JALUR CADANGAN (Gemini API dengan History)
        # -----------------------------------------------------------------
        try:
            logging.warning("🚨 Mencoba mengaktifkan failover ke API Google Gemini...")
            gemini_contents = self.get_gemini_contents(user_prompt)
            
            # Menggunakan gemini-2.5-flash sebagai standar industri produksi terkini
            response_stream = gemini_client.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=gemini_contents,
                config={
                    "system_instruction": self.system_instruction,
                    "temperature": 0.7
                }
            )
            
            if IS_DEBUG:
                print("🤖 AI (Gemini Backup): ", end="")
            else:
                print("🤖 AI: ", end="")
                
            for chunk in response_stream:
                token = chunk.text
                assistant_response += token
                yield token

            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": assistant_response})
            logging.info("Sukses memulihkan layanan menggunakan cadangan Gemini.")

        except Exception as e:
            logging.critical(f"Gagal total! Seluruh sistem utama dan cadangan lumpuh: {str(e)}")
            yield "Maaf, sistem kami sedang mengalami gangguan."

# =====================================================================
# SIMULASI CONVERSATIONAL DI TERMINAL
# =====================================================================
if __name__ == "__main__":
    chat = ChatSessionWithFailover()
    
    status_debug = "AKTIF 🟢" if IS_DEBUG else "NONAKTIF 🔴"
    print(f"⚙️  Mode Debugging: {status_debug}")
    print("🤖 Sistem Chat Siap! Ketik 'keluar' untuk berhenti.\n")

    while True:
        user_input = input("\n👤 Anda: ")
        if user_input.lower() == 'keluar':
            print("Sampai jumpa!")
            break
            
        if not user_input.strip():
            continue

        for chunk in chat.send_message_stream(user_input):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()
