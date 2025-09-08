import streamlit as st

st.title("테스트 앱")
st.write("Streamlit이 정상적으로 작동합니다!")

if st.button("클릭하세요"):
    st.balloons()
    st.success("성공!")