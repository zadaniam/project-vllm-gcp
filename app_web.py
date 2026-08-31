# app_web.py
import streamlit as st
import config
from llm_service import ChatSessionWithFailover

# 1. Atur Konfigurasi Halaman Web
st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Chatbot dengan Failover")

# Tampilkan status mode debug di pojok kanan atas/sidebar jika aktif
if config.IS_DEBUG:
    st.sidebar.warning("🟢 Mode Debugging Aktif")

# 2. Inisialisasi Sesi Chat di Server Streamlit (Agar history tidak ter-reset setiap user mengetik)
if "chat_session" not in st.session_state:
    st.session_state.chat_session = ChatSessionWithFailover()

# Referensi cepat ke objek chat session
chat = st.session_state.chat_session

# 3. Tampilkan Histori Obrolan yang Sudah Ada ke Layar Web
for msg in chat.history:
    role_label = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role_label):
        st.markdown(msg["content"])

# 4. Tangani Input Pengguna Baru
if user_input := st.chat_input("Ketik pesan Anda di sini..."):
    
    # Tampilkan pesan user yang baru diketik ke layar
    with st.chat_message("user"):
        st.markdown(user_input)

    # Tampilkan kontainer kosong untuk respons AI (Efek streaming)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Panggil generator streaming dari llm_service
        # Streamlit akan memperbarui teks di layar setiap kali token baru masuk
        for chunk in chat.send_message_stream(user_input):
            full_response += chunk
            # Tambahkan visual cursor ▌ agar mirip ChatGPT asli saat mengetik
            message_placeholder.markdown(full_response + "▌")
            
        # Tampilkan hasil akhir tanpa cursor setelah streaming selesai
        message_placeholder.markdown(full_response)
