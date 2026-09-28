import streamlit as st
import requests
import time

API_URL = "http://127.0.0.1:8000"

st.title("Osteo Inference")

uploaded_file = st.file_uploader("Upload DICOM", type=None)

if uploaded_file:
    st.write("Uploading file...")

    # Step 1: Upload to API
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{API_URL}/api/upload", files={"file": uploaded_file})

    if response.status_code != 200:
        st.error("Upload failed")
    else:
        job_id = response.json()["job_id"]
        st.success(f"Uploaded! Job ID: {job_id}")

        # Step 2: Poll for results
        status_placeholder = st.empty()

        while True:
            res = requests.post(
                f"{API_URL}/api/results",
                json={"job_id": job_id}
            )

            data = res.json()

            if data["status"] == "Completed":
                st.success("Inference complete!")

                findings = data["findings"]
                st.json(findings)
                break

            elif data["status"] == "Error":
                st.error(data.get("error_message", "Error occurred"))
                break

            else:
                status_placeholder.info(f"Status: {data['status']}")
                time.sleep(2)