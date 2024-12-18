import streamlit as st
from predict import predict
import pandas as pd
import tempfile
import os

df = pd.read_excel("info.xlsx")

def get_info(name: str):
    return df[df.기종 == name]

st.title("비행기 기종 판별기")
st.write("---")

uploaded_file = st.file_uploader("사진을 선택하세요", ["jpg", "png"])
st.write("---")

if uploaded_file is not None:
    try:
        with st.spinner("기종을 판별하고 있습니다..."):
            temp_dir = tempfile.TemporaryDirectory()
            temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            plane_type = predict(temp_filepath)
            info = get_info(plane_type)
            info
            st.image(temp_filepath)
            st.write(plane_type)   
    except:
        pass