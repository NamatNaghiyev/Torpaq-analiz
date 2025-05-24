import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# 1. JSON faylını oxuyuruq
with open(r'C:\Users\PC\OneDrive\İş masası\Agrotech\AgroTech_ML\response_1744827666092.json') as f:
    json_data = json.load(f)

# 2. DataFrame düzəldirik
df = pd.DataFrame(json_data['content'])

# 3. NaN dəyərləri sıfırla əvəz edirik
df.fillna(0, inplace=True)

# 4. Sensor dataları X kimi ayırırıq
X = df[['temperature', 'humidity', 'soil', 'pressure', 'gasResistance']]

# 5. Bitki növləri və uyğunluq aralığı
bitki_melumatlari = {
    'Taxıl': {'temp': (15, 25), 'humidity': (30, 60)},
    'Pambıq': {'temp': (20, 30), 'humidity': (40, 70)},
    'Alma': {'temp': (10, 20), 'humidity': (50, 80)},
    'Nar': {'temp': (18, 28), 'humidity': (40, 70)},
    'Heyva': {'temp': (12, 22), 'humidity': (60, 90)}
}

# 6. Bitki haqqında becərmə məlumatları
bitki_becermesi = {
    'Taxıl': """
    ✅ Əkin vaxtı: Payız və yaz aylarında.
    ✅ Torpaq: Orta humuslu, yaxşı şumlanmış torpaq.
    ✅ Suvarma: Yağışla kifayətlənir, amma quraqlıqda əlavə suvarma lazımdır.
    ✅ Gübrələmə: Azot, fosfor əsaslı gübrələr tövsiyə olunur.
    ✅ Məhsul yığımı: Yayın ortalarında kombaynla yığılır.
    """,

    'Pambıq': """
    ✅ Əkin vaxtı: Aprel-may aylarında.
    ✅ Torpaq: Qumlu və gil torpaqlar ideal sayılır.
    ✅ Suvarma: Tez-tez və bol suvarma tələb edir.
    ✅ Gübrələmə: Azot və kaliumla zəngin gübrələr tövsiyə olunur.
    ✅ Məhsul yığımı: Avqust-sentyabr arası əl ilə və ya texnika ilə.
    """,

    'Alma': """
    ✅ Əkin vaxtı: Payız və ya yaz ayları.
    ✅ Torpaq: Dərin, nəm saxlayan, zəngin torpaqlar.
    ✅ Suvarma: Həftədə 1 dəfə dərin suvarma lazımdır.
    ✅ Gübrələmə: Orqanik və kompleks gübrələrlə.
    ✅ Məhsul yığımı: Sentyabr-oktyabr aylarında.
    """,

    'Nar': """
    ✅ Əkin vaxtı: Yazın əvvəllərində.
    ✅ Torpaq: Qumlu və ya gil torpaq, yaxşı drenajlı.
    ✅ Suvarma: 7-10 gündə bir suvarma.
    ✅ Gübrələmə: Hər yaz və yayda azotlu gübrələr.
    ✅ Məhsul yığımı: Oktyabr və noyabr aylarında.
    """,

    'Heyva': """
    ✅ Əkin vaxtı: Payız və ya erkən yaz.
    ✅ Torpaq: Qidalandırıcı və dərin torpaqlar.
    ✅ Suvarma: Ayda 2-3 dəfə, xüsusən quraqlıqda.
    ✅ Gübrələmə: Əkin zamanı orqanik gübrə, sonra kompleks gübrə.
    ✅ Məhsul yığımı: Sentyabrın sonu - Oktyabrın əvvəli.
    """
}

# 7. Uyğun bitki qaytarma funksiyası
def uygun_bitki(temp, hum):
    for bitki, shert in bitki_melumatlari.items():
        if shert['temp'][0] <= temp <= shert['temp'][1] and shert['humidity'][0] <= hum <= shert['humidity'][1]:
            return bitki
    return "Uyğun deyil"

# 8. Hər sətrə uyğun bitki tapırıq
df['bitki'] = df.apply(lambda row: uygun_bitki(row['temperature'], row['humidity']), axis=1)

# 9. Uyğun olmayanları çıxarırıq
df = df[df['bitki'] != "Uyğun deyil"]

# 10. Bitki sütununu kodlaşdırırıq
df['bitki_code'] = df['bitki'].astype('category').cat.codes
categories = df['bitki'].astype('category').cat.categories  

# 11. X və y
X = df[['temperature', 'humidity', 'soil', 'pressure', 'gasResistance']]
y = df['bitki_code']

# 12. Standartlaşdırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 13. Model qururuq
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(len(categories), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 14. Modeli tren edirik
history = model.fit(X_scaled, y, epochs=100, batch_size=32, validation_split=0.2)

# 15. Qiymətləndirmə
loss, acc = model.evaluate(X_scaled, y)
print(f"Ziyana bax: {loss:.4f}, Dəqiqlik: {acc:.4f}")

# 16. Yeni data üçün prediksiya
new_data = np.array([[23.5, 60.0, 1600, 1020, 70]])
new_scaled = scaler.transform(new_data)

prediction = model.predict(new_scaled)
predicted_index = np.argmax(prediction)
predicted_bitki = categories[predicted_index]

print(f"\n🌱 Bu torpaq üçün ən uyğun bitki: {predicted_bitki}")
print(f"\n📋 Becərmə təlimatı:\n{bitki_becermesi.get(predicted_bitki, 'Təlimat tapılmadı.')}")

# 17. Training prosesi
history_dict = model.history.history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.plot(history_dict['loss'], label='Train Loss', color='r')
ax1.plot(history_dict['val_loss'], label='Validation Loss', color='b')
ax1.set_title('Loss over Epochs')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(history_dict['accuracy'], label='Train Accuracy', color='r')
ax2.plot(history_dict['val_accuracy'], label='Validation Accuracy', color='b')
ax2.set_title('Accuracy over Epochs')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()

# 18. Confusion Matrix və Classification Report
y_pred = model.predict(X_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y, y_pred_classes)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

print("\nClassification Report:")
print(classification_report(y, y_pred_classes, target_names=categories))

# 19. Modelin saxlanması
model.save('bitki_novleri_model.h5')
print("\n💾 Model saxlanıldı: 'bitki_novleri_model.h5'")
