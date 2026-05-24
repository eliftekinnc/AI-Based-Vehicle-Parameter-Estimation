import numpy as np
import pandas as pd
import mat73
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models

#Veri Yükle

print("Yeni veri seti yükleniyor...")
data_dict = mat73.loadmat('otonom_arac_dataset.mat')
dataset = data_dict['dataset']

# X (Girdi) ve y (Hedef) Hazırlığı
X_list = []
y_list = []

# Kaç adet örnek olduğunu anlamak için 'inputs'un uzunluğuna bakıyoruz
num_samples = len(dataset['inputs'])

print(f"Toplam {num_samples} örnek işleniyor...")

for i in range(num_samples):
    inputs = dataset['inputs'][i]  # [delta(streering input), Fx]
    outputs = dataset['outputs'][i]  # [vy, r]
    states = dataset['true_states'][i]  # [vx, vy, r]

    combined = np.column_stack([inputs, outputs, states])

    X_list.append(combined)
    y_list.append(dataset['params'][i])  # [m, Iz(atalet), Cf(on lastik sertliği), Cr(arka lastik sertligi)]

X = np.array(X_list)
y = np.array(y_list)



print(f"Veri başarıyla işlendi. Girdi boyutu: {X.shape}")

# 3. Veri Bölme
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42)

# 4.Normalization
scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train.reshape(-1, 7)).reshape(X_train.shape)
X_val_s = scaler_X.transform(X_val.reshape(-1, 7)).reshape(X_val.shape)
X_test_s = scaler_X.transform(X_test.reshape(-1, 7)).reshape(X_test.shape)

scaler_y = MinMaxScaler()  # Parametreleri 0-1 arasına çekelim
y_train_s = scaler_y.fit_transform(y_train)
y_val_s = scaler_y.transform(y_val)
y_test_s = scaler_y.transform(y_test)

# 5. 1D-CNN Model Mimarisi
model = models.Sequential([
    layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=(101, 7)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),

    layers.Conv1D(64, kernel_size=3, activation='relu'),
    layers.GlobalAveragePooling1D(),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(4, activation='linear')  # Çıktı: 4 parametre
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 6. Eğitim
print("\nEğitim başlıyor...")
history = model.fit(X_train_s, y_train_s,
                    epochs=100,
                    batch_size=16,
                    validation_data=(X_val_s, y_val_s),
                    verbose=1)

# 7. Sonuçları Değerlendir
y_pred_s = model.predict(X_test_s)
y_pred = scaler_y.inverse_transform(y_pred_s)  # 0-1'den gerçek değerlere dön

# İlk 5 Test Örneğini Kıyaslayalım
results = pd.DataFrame({
    'Gerçek Kütle (m)': y_test[:5, 0],
    'Tahmin Kütle (m)': y_pred[:5, 0],
    'Gerçek Sertlik (Cf)': y_test[:5, 2],
    'Tahmin Sertlik (Cf)': y_pred[:5, 2]
})
print("\n--- TEST SONUÇLARI ---")
print(results)

# Eğitim Grafiği
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Hata Oranı Değişimi')
plt.show()