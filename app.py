import streamlit as st
import joblib
import pandas as pd

model = joblib.load('linear_model_clean.pkl')
scaler = joblib.load('scaler_clean.pkl')

st.title("Ứng dụng Dự đoán Giá Nhà")
st.write("Nhập các thông số cơ bản, hệ thống sẽ tự tính toán các biến phụ thuộc.")

col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Kinh độ (longitude):", value=0)
    housing_median_age = st.number_input("Tuổi thọ nhà (housing_median_age):", value=0)
    total_bedrooms = st.number_input("Tổng số phòng ngủ (total_bedrooms):", value=0)
    households = st.number_input("Số hộ gia đình (households):", value=1.0, min_value=1.0)

    ocean_proximity = st.selectbox(
        "Vị trí so với biển (ocean_proximity):",
        ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    )

with col2:
    latitude = st.number_input("Vĩ độ (latitude):", value=0)
    total_rooms = st.number_input("Tổng số phòng (total_rooms):", value=0)
    population = st.number_input("Dân số (population):", value=0)
    median_income = st.number_input("Thu nhập trung bình (median_income):", value=0)

if st.button("Dự đoán"):
    rooms_per_household = total_rooms / households
    population_per_household = population / households

    data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity_INLAND': 1.0 if ocean_proximity == 'INLAND' else 0.0,
        'ocean_proximity_ISLAND': 1.0 if ocean_proximity == 'ISLAND' else 0.0,
        'ocean_proximity_NEAR BAY': 1.0 if ocean_proximity == 'NEAR BAY' else 0.0,
        'ocean_proximity_NEAR OCEAN': 1.0 if ocean_proximity == 'NEAR OCEAN' else 0.0,
        'ocean_proximity_<1H OCEAN': 1.0 if ocean_proximity == '<1H OCEAN' else 0.0,
        'rooms_per_household': rooms_per_household,
        'population_per_household': population_per_household
    }

    input_df = pd.DataFrame([data])

    input_df = input_df[scaler.feature_names_in_]

    # Chuẩn hóa và dự đoán
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    st.success(f"Giá nhà dự kiến: ${prediction[0]:,.2f}")