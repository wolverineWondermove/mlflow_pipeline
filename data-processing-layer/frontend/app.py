import streamlit as st
import requests

st.title("📂 Excel/CSV → HuggingFace Dataset Uploader")

uploaded_file = st.file_uploader("⬆️ Upload CSV or Excel file", type=["csv", "xlsx"])
repo_name = st.text_input("🚩 HuggingFace Repo Name", placeholder="e.g., username/dataset_name")
hf_token = st.text_input("🔑 HuggingFace Token", type='password', placeholder="hf_xxx...")

if st.button("🚀 Upload and Process"):
    if uploaded_file and repo_name and hf_token:
        with st.spinner('Uploading and processing your data...'):
            files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
            data = {"repo_name": repo_name, "hf_token": hf_token}

            try:
                res = requests.post("http://localhost:8000/uploadfile/", files=files, data=data, timeout=120)
                if res.status_code == 200:
                    result = res.json()
                    st.success(f"✅ Success! Uploaded {result['uploaded_records']} records.")
                    st.markdown(f"[🌐 View Dataset]({result['url']})")
                else:
                    result = res.json()
                    st.error(f"❌ Error: {result.get('message', 'An unknown error occurred.')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to backend service at port 8000. Is backend running?")
            except requests.exceptions.Timeout:
                st.error("⌛ Request timeout. Try smaller files or increase timeout.")
            except Exception as e:
                st.error(f"❌ An unexpected error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please fill out all fields and upload your file before proceeding.")