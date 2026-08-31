# app.py
import config
from llm_service import ChatSessionWithFailover

def run_terminal_ui():
    chat = ChatSessionWithFailover()
    
    status_debug = "AKTIF 🟢" if config.IS_DEBUG else "NONAKTIF 🔴"
    print(f"⚙️  Mode Debugging: {status_debug}")
    print("🤖 Sistem Chat Modular Siap! Ketik 'keluar' untuk berhenti.\n")

    while True:
        user_input = input("\n👤 Anda: ")
        if user_input.lower() == 'keluar':
            print("Sampai jumpa!")
            break
            
        if not user_input.strip():
            continue

        # Tampilkan Label AI sesuai mode debug
        if config.IS_DEBUG:
            print("🤖 AI (vLLM/Gemini): ", end="", flush=True)
        else:
            print("🤖 AI: ", end="", flush=True)

        # Mengonsumsi token streaming dari core service
        for chunk in chat.send_message_stream(user_input):
            print(chunk, end="", flush=True)
        print()

if __name__ == "__main__":
    run_terminal_ui()
