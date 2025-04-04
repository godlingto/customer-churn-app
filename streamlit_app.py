import streamlit as st
import numpy as np
import pandas as pd
import joblib

# 모델, 스케일러, 피처 목록 불러오기
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

st.title('고객 이탈 예측 시스템')

# 사용자 입력 받기
gender = st.selectbox('성별', ['Male', 'Female'])
senior = st.selectbox('노인 여부', ['Yes', 'No'])
partner = st.selectbox('배우자 여부', ['Yes', 'No'])
dependents = st.selectbox('부양가족 여부', ['Yes', 'No'])
tenure = st.slider('계약 개월 수', 0, 72, 12)
phoneservice = st.selectbox('전화 서비스 사용 여부', ['Yes', 'No'])
internetservice = st.selectbox('인터넷 서비스', ['DSL', 'Fiber optic', 'No'])
contract = st.selectbox('계약 유형', ['Month-to-month', 'One year', 'Two year'])
monthlycharges = st.number_input('월 요금', value=70.0)
totalcharges = st.number_input('총 요금', value=1000.0)
paymentmethod = st.selectbox('결제 방법', [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])

# 입력값 딕셔너리 구성
user_input = {
    'gender': gender,
    'SeniorCitizen': 1 if senior == 'Yes' else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phoneservice,
    'InternetService': internetservice,
    'Contract': contract,
    'MonthlyCharges': monthlycharges,
    'TotalCharges': totalcharges,
    'PaymentMethod': paymentmethod
}

# 입력값 → DataFrame
input_df = pd.DataFrame([user_input])

# 동일한 전처리 적용 (One-Hot Encoding)
input_encoded = pd.get_dummies(input_df)

# 누락된 컬럼 채우기
for col in feature_names:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# 컬럼 순서 맞추기
input_encoded = input_encoded[feature_names]

# 스케일링
input_scaled = scaler.transform(input_encoded)

# 예측
if st.button('예측하기'):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    if pred == 1:
        st.error(f'⚠️ 이 고객은 이탈할 확률이 높습니다. (확률: {prob:.2f})')
    else:
        st.success(f'✅ 이 고객은 이탈할 확률이 낮습니다. (확률: {prob:.2f})')
