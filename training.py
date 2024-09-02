import pandas as pd
import tensorflow as tf   
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial import cKDTree

data = pd.read_csv('./datasets/model_input_dataset.csv')

inp = data.drop(columns=['price']).values
tar = data['price'].values

inp_train, inp_temp, tar_train, tar_temp = train_test_split(inp, tar, test_size=0.4, random_state=43)
inp_val, inp_test, tar_val, tar_test = train_test_split(inp_temp, tar_temp, test_size=0.5, random_state=43)
#model = tf.keras.models.load_model('./model_5000_1024x4.h5')

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu', input_shape=(inp_train.shape[1],)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(  1, activation='relu')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001), loss='mape', metrics=['mae'])

history = model.fit(inp_train, tar_train, validation_data=(inp_val, tar_val), epochs=1000, batch_size=512)

test_loss, test_mae = model.evaluate(inp_test, tar_test)
model.save('./model_final.keras')
print(model.summary())
print(f'Test Loss (MSE): {test_loss:.4f}')
print(f'Test Mean Absolute Error (MAE): {test_mae:.4f}')

num_samples = 10000
district_range = (data['district'].min(), data['district'].max())
gsm_range = (data['GrossSquareMeters'].min(), data['GrossSquareMeters'].max())
nfb_range = (data['NumberFloorsofBuilding'].min(), data['NumberFloorsofBuilding'].max())
item_range = (data['ItemStatus'].min(), data['ItemStatus'].max())
nob_range = (data['NumberOfBathrooms'].min(), data['NumberOfBathrooms'].max())
nor_range = (data['NumberOfRooms'].min(), data['NumberOfRooms'].max())
fl_range = (data['FloorLocation'].min(), data['FloorLocation'].max())

random_districts = np.random.randint(district_range[0], district_range[1] + 1, num_samples)
random_gsm = np.random.uniform(gsm_range[0], gsm_range[1], num_samples)
random_nfb = np.random.randint(nfb_range[0], nfb_range[1] + 1, num_samples)
random_items = np.random.randint(item_range[0], item_range[1] + 1, num_samples)
random_nob = np.random.randint(nob_range[0], nob_range[1] + 1, num_samples)
random_nor = np.random.randint(nor_range[0], nor_range[1] + 1, num_samples)
random_floc = np.random.randint(fl_range[0], fl_range[1] + 1, num_samples)

random_inp = np.column_stack((random_districts, random_gsm, random_nfb, random_items, random_nob, random_nor, random_floc))

predicted_prices = model.predict(random_inp).flatten()

price_min = 20000
price_max = 4000000
gsm_min = 1
gsm_max = 950
items = {0: 'Boş', 1: 'Eşyalı'}
nors = {0: '1', 1: '1+1', 2: '2+0', 3: '2+1', 4: '2+2', 5: '3+1', 6: '3+2', 7: '4+1', 8: '4+2', 9: '5', 10: '5+1', 11: '5+2', 12: '5+3', 13: '5+4', 14: '6+1', 15: '6+2', 16: '6+3', 17: '6+4', 18: '7+1', 19: '7+2', 20: '7+3', 21: '8+'}
flocs = {0: '1. Kat', 1: '10-20. Kat', 2: '10. Kat', 3: '11. Kat', 4: '12. Kat', 5: '13. Kat', 6: '14. Kat', 7: '15. Kat', 8: '16. Kat', 9: '17. Kat', 10: '18. Kat', 11: '19. Kat', 12: '2. Kat', 13: '20-30. Kat', 14: '20. Kat', 15: '21. Kat', 16: '22. Kat', 17: '23. Kat', 18: '24. Kat', 19: '25. Kat', 20: '26. Kat', 21: '27. Kat', 22: '28. Kat', 23: '29. Kat', 24: '3. Kat', 25: '30-40. Kat', 26: '30. Kat', 27: '31. Kat', 28: '33. Kat', 29: '35. Kat', 30: '36. Kat', 31: '38. Kat', 32: '4. Kat', 33: '40+. Kat', 34: '5. Kat', 35: '6. Kat', 36: '7. Kat', 37: '8. Kat', 38: '9. Kat', 39: 'Bahçe Dublex', 40: 'Bahçe Katı', 41: 'Düz Giriş', 42: 'Kot 1 (-1). Kat', 43: 'Kot 2 (-2). Kat', 44: 'Kot 3 (-3). Kat', 45: 'Müstakil', 46: 'Tam Bodrum', 47: 'Villa Tipi', 48: 'Yarı Bodrum', 49: 'Yüksek Bodrum', 50: 'Yüksek Giriş', 51: 'Çatı Dubleks', 52: 'Çatı Katı'}
districts = {0: 'adalar', 1: 'arnavutkoy', 2: 'atasehir', 3: 'avcilar', 4: 'bagcilar', 5: 'bahcelievler', 6: 'bakirkoy', 7: 'basaksehir', 8: 'bayrampasa', 9: 'besiktas', 10: 'beykoz', 11: 'beylikduzu', 12: 'beyoglu', 13: 'buyukcekmece', 14: 'catalca', 15: 'cekmekoy', 16: 'esenler', 17: 'esenyurt', 18: 'eyupsultan', 19: 'fatih', 20: 'gaziosmanpasa', 21: 'gungoren', 22: 'kadikoy', 23: 'kagithane', 24: 'kartal', 25: 'kucukcekmece', 26: 'maltepe', 27: 'pendik', 28: 'sancaktepe', 29: 'sariyer', 30: 'sile', 31: 'silivri', 32: 'sisli', 33: 'sultanbeyli', 34: 'sultangazi', 35: 'tuzla', 36: 'umraniye', 37: 'uskudar', 38: 'zeytinburnu'}
item_names = [items[code] for code in random_items]
district_names = [districts[code] for code in random_districts]
nor_names = [nors[code] for code in random_nor]
floc_names = [flocs[code] for code in random_floc]

predicted_df = pd.DataFrame({
    'district': district_names,
    'GrossSquareMeters': random_gsm * (gsm_max - gsm_min) + gsm_min,
    'NumberFloorsofBuilding': random_nfb,
    'ItemStatus': item_names,
    'NumberOfBathrooms': random_nob,
    'NumberOfRooms': nor_names,
    'FloorLocation': floc_names,
    'PredictedPrice': predicted_prices
})

predicted_df.to_csv('prediction.csv', index=False)

print('Predictions saved to prediction.csv')

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MAE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=tar_test, y=model.predict(inp_test).flatten())
plt.plot([tar_test.min(), tar_test.max()], [tar_test.min(), tar_test.max()], 'k--', lw=2)
plt.title('Predicted vs. Actual Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=tar_test, y=model.predict(inp_test).flatten())
plt.plot([tar_train.min(), tar_train.max()], [tar_train.min(), tar_train.max()], 'k--', lw=2)
plt.title('Predicted vs. Actual Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()
