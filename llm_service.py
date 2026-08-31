# llm_service.py
import logging
from openai import OpenAI, APIConnectionError, APITimeoutError
from google import genai
import config

# Inisialisasi Klien
vllm_client = OpenAI(
    base_url=config.VLLM_BASE_URL, 
    api_key="vllm-placeholder", 
    timeout=5.0
)
gemini_client = genai.Client(api_key=config.get_gemini_api_key())

class ChatSessionWithFailover:
    def __init__(self):
        self.history = []
        self.system_instruction = "Anda adalah asisten AI yang ramah, profesional, dan membantu."

    def _get_vllm_messages(self, current_user_prompt: str):
        vllm_formatted = [{"role": "system", "content": self.system_instruction}]
        for msg in self.history:
            vllm_formatted.append({"role": msg["role"], "content": msg["content"]})
        vllm_formatted.append({"role": "user", "content": current_user_prompt})
        return vllm_formatted

    def _get_gemini_contents(self, current_user_prompt: str):
        gemini_contents = []
        for msg in self.history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        gemini_contents.append({"role": "user", "parts": [{"text": current_user_prompt}]})
        return gemini_contents

    def send_message_stream(self, user_prompt: str):
        assistant_response = ""

        # 1. JALUR UTAMA: vLLM
        try:
            logging.info(f"Menghubungi server vLLM ({config.VLLM_EXTERNAL_IP}) via Streaming...")
            vllm_messages = self._get_vllm_messages(user_prompt)
            
            stream = vllm_client.chat.completions.create(
                model="google/gemma-2-2b-it",
                messages=vllm_messages,
                stream=True
            )
            
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
            logging.warning(f"Koneksi vLLM Bermasalah ({type(e).__name__}). Mengalihkan...")
        except Exception as e:
            logging.error(f"Eror internal pada parser vLLM: {str(e)}. Mengalihkan...")

        # 2. JALUR FAILOVER: Gemini API
        try:
            logging.warning("🚨 Mencoba mengaktifkan failover ke API Google Gemini...")
            gemini_contents = self._get_gemini_contents(user_prompt)
            assistant_response = "" 
            
            response_stream = gemini_client.models.generate_content_stream(
                model="gemini-3.5-flash-lite",
                contents=gemini_contents,
                config={
                    "system_instruction": self.system_instruction,
                    "temperature": 0.7
                }
            )
            
            for chunk in response_stream:
                token = chunk.text
                assistant_response += token
                yield token

            self.history.append({"role": "user", "content": user_prompt})
            self.history.append({"role": "assistant", "content": assistant_response})
            logging.info("Sukses memulihkan layanan menggunakan cadangan Gemini.")

        except Exception as e:
            logging.critical(f"Gagal total! Seluruh sistem lumpuh: {str(e)}")
            yield "Maaf, sistem kami sedang mengalami gangguan."
