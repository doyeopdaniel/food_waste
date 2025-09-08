import streamlit as st
import pandas as pd
import numpy as np

st.title("디버그 테스트")

uploaded_file = st.file_uploader("CSV 파일 선택", type=['csv'])

if uploaded_file is not None:
    try:
        st.write("파일 읽기 시작...")
        df = pd.read_csv(uploaded_file)
        st.write(f"파일 읽기 완료: {len(df)}행")
        
        st.write("컬럼 확인:")
        st.write(df.columns.tolist())
        
        st.write("첫 5행:")
        st.write(df.head())
        
        required_columns = ['시간(min)', 'pH', '온도', '당농도', '질소농도', '교반속도', '탄소배출(kgCO2)', '비용(원)']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"누락된 컬럼: {missing_columns}")
        else:
            st.success("모든 필수 컬럼이 있습니다!")
            
            latest_row = df.iloc[-1]
            st.write("마지막 행:")
            st.write(latest_row)
            
    except Exception as e:
        st.error(f"오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())