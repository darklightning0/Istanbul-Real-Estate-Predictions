
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('./datasets/model_input_dataset.csv')

model = tf.keras.models.load_model('./model_final.keras')

price_min = 20000
price_max = 4000000
gsm_min = 1
gsm_max = 950
items = {0: 'Boş', 1: 'Eşyalı'}
nors = {0: '1', 1: '1+1', 2: '2+0', 3: '2+1', 4: '2+2', 5: '3+1', 6: '3+2', 7: '4+1', 8: '4+2', 9: '5', 10: '5+1', 11: '5+2', 12: '5+3', 13: '5+4', 14: '6+1', 15: '6+2', 16: '6+3', 17: '6+4', 18: '7+1', 19: '7+2', 20: '7+3', 21: '8+'}
flocs = {0: '1. Kat', 1: '10-20. Kat', 2: '10. Kat', 3: '11. Kat', 4: '12. Kat', 5: '13. Kat', 6: '14. Kat', 7: '15. Kat', 8: '16. Kat', 9: '17. Kat', 10: '18. Kat', 11: '19. Kat', 12: '2. Kat', 13: '20-30. Kat', 14: '20. Kat', 15: '21. Kat', 16: '22. Kat', 17: '23. Kat', 18: '24. Kat', 19: '25. Kat', 20: '26. Kat', 21: '27. Kat', 22: '28. Kat', 23: '29. Kat', 24: '3. Kat', 25: '30-40. Kat', 26: '30. Kat', 27: '31. Kat', 28: '33. Kat', 29: '35. Kat', 30: '36. Kat', 31: '38. Kat', 32: '4. Kat', 33: '40+. Kat', 34: '5. Kat', 35: '6. Kat', 36: '7. Kat', 37: '8. Kat', 38: '9. Kat', 39: 'Bahçe Dublex', 40: 'Bahçe Katı', 41: 'Düz Giriş', 42: 'Kot 1 (-1). Kat', 43: 'Kot 2 (-2). Kat', 44: 'Kot 3 (-3). Kat', 45: 'Müstakil', 46: 'Tam Bodrum', 47: 'Villa Tipi', 48: 'Yarı Bodrum', 49: 'Yüksek Bodrum', 50: 'Yüksek Giriş', 51: 'Çatı Dubleks', 52: 'Çatı Katı'}
districts = {0: 'Adalar', 1: 'Arnavutköy', 2: 'Atasehir', 3: 'Avcılar', 4: 'Bağcılar', 5: 'Bahçelievler', 6: 'Bakırköy', 7: 'Başakşehir', 8: 'Bayrampaşa', 9: 'Beşiktaş', 10: 'Beykoz', 11: 'Beylikdüzü', 12: 'Beyoğlu', 13: 'Büyükçekmece', 14: 'Çatalca', 15: 'Çekmekoy', 16: 'Esenler', 17: 'Esenyurt', 18: 'Eyüpsultan', 19: 'Fatih', 20: 'Gaziosmanpaşa', 21: 'Güngören', 22: 'Kadıköy', 23: 'Kağıthane', 24: 'Kartal', 25: 'Küçükçekmece', 26: 'Maltepe', 27: 'Pendik', 28: 'Sancaktepe', 29: 'Sarıyer', 30: 'Sile', 31: 'Silivri', 32: 'Sisli', 33: 'Sultanbeyli', 34: 'Sultangazi', 35: 'Tuzla', 36: 'Ümraniye', 37: 'Üskudar', 38: 'Zeytinburnu'}

st.title('Istanbul Real Estate Price Prediction')

st.sidebar.header('Features')

def user_input_features():
    district = st.sidebar.selectbox('District', list(districts.values()))
    gsm = st.sidebar.slider('Gross Square Meters', float(gsm_min), float(gsm_max), float(data['GrossSquareMeters'].mean()))
    nfb = st.sidebar.slider('Number of Floors of Building', int(data['NumberFloorsofBuilding'].min()), int(data['NumberFloorsofBuilding'].max()), int(data['NumberFloorsofBuilding'].mean()))
    item_status = st.sidebar.selectbox('Item Status', list(items.values()))
    nob = st.sidebar.slider('Number of Bathrooms', int(data['NumberOfBathrooms'].min()), int(data['NumberOfBathrooms'].max()), int(data['NumberOfBathrooms'].mean()))
    nor = st.sidebar.selectbox('Number of Rooms', list(nors.values()))
    floc = st.sidebar.selectbox('Floor Location', list(flocs.values()))

    features = pd.DataFrame({
        'district': [district],
        'GrossSquareMeters': [gsm],
        'NumberFloorsofBuilding': [nfb],
        'ItemStatus': [item_status],
        'NumberOfBathrooms': [nob],
        'NumberOfRooms': [nor],
        'FloorLocation': [floc]
    })
    return features

df = user_input_features()

df['district'] = df['district'].apply(lambda x: list(districts.keys())[list(districts.values()).index(x)])
df['ItemStatus'] = df['ItemStatus'].apply(lambda x: list(items.keys())[list(items.values()).index(x)])
df['NumberOfRooms'] = df['NumberOfRooms'].apply(lambda x: list(nors.keys())[list(nors.values()).index(x)])
df['FloorLocation'] = df['FloorLocation'].apply(lambda x: list(flocs.keys())[list(flocs.values()).index(x)])

prediction = model.predict(df.values)

st.subheader('Predicted Price')
st.write(f'The predicted price is: {prediction[0][0] * (price_max - price_min) + price_min:,.2f} TL')
