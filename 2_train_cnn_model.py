import os
import numpy as np
import pandas as pd
import tensorflow as tf
import unicodedata
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import random

# 🚀 GPU Ayarları
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print("✅ GPU kullanımına izin verildi!")
    except RuntimeError as e:
        print(f"❌ GPU kullanımına izin verilemedi: {e}")
else:
    print("❌ GPU bulunamadı.")

# 📘 **Excel Dosyasını Yükle**
from google.colab import files
uploaded = files.upload()
excel_filename = list(uploaded.keys())[0]

try:
    excel_data = pd.read_excel(excel_filename, engine='openpyxl')
    print(f"✅ Excel dosyası başarıyla okundu: {excel_filename}")
except Exception as e:
    print(f"❌ Hata oluştu: {e}. CSV olarak deniyoruz.")
    try:
        excel_data = pd.read_csv(excel_filename, encoding='utf-8')
    except Exception as e:
        print(f"❌ CSV olarak da okunamadı. Hata: {e}")

# **Ad Soyad ve Patoloji Sütunlarını Al**
tumor_mapping = excel_data[['Ad Soyad', 'Patoloji']].dropna()

def clean_text(text):
    if isinstance(text, str):
        text = text.strip().lower()
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    return text

tumor_mapping['Ad Soyad Temiz'] = tumor_mapping['Ad Soyad'].apply(clean_text)

# 📘 **Google Drive'a Bağlan**
from google.colab import drive
drive.mount('/content/drive')

# 📘 **Resimlerin Bulunduğu Dizini Tanımla**
data_path = '/content/drive/MyDrive/PNG/'

image_paths = []
image_labels = []
patient_names = []

# **Tüm Hasta Klasörlerini Tara**
for patient_folder in os.listdir(data_path):
    patient_path = os.path.join(data_path, patient_folder)
    if os.path.isdir(patient_path):
        patient_name_cleaned = clean_text(patient_folder)
        tumor_type = tumor_mapping.loc[tumor_mapping['Ad Soyad Temiz'] == patient_name_cleaned, 'Patoloji'].values
        if len(tumor_type) > 0:
            tumor_type = str(tumor_type[0])
        else:
            print(f"❌ Hasta için tümör türü bulunamadı: {patient_folder}")
            continue

        for root, subdirs, files in os.walk(patient_path):
            for file_name in files:
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file_name)
                    image_paths.append(file_path)
                    image_labels.append(tumor_type)
                    patient_names.append(patient_folder)

# **Etiketleri Kategorik Formatta Dönüştür**
label_encoder = LabelEncoder()
image_labels = label_encoder.fit_transform(image_labels)
image_labels = to_categorical(image_labels)
tumor_types = label_encoder.classes_

print(f"🔍 Bulunan Tümör Türleri: {tumor_types}")
print(f"🔢 Toplam Tümör Türü Sayısı: {len(tumor_types)}")
print(f"✅ {len(image_paths)} resim bulundu!")

# **Hastaları bölme işlemi**
unique_patients = list(set(patient_names))
random.shuffle(unique_patients)

train_patients = unique_patients[:22]
val_patients = unique_patients[22:26]
test_patients = unique_patients[26:29]

# **Hasta bazlı veri setini oluştur**
X_train_paths, y_train = [], []
X_val_paths, y_val = [], []
X_test_paths, y_test = [], []

for i, patient_name in enumerate(patient_names):
    if patient_name in train_patients:
        X_train_paths.append(image_paths[i])
        y_train.append(image_labels[i])
    elif patient_name in val_patients:
        X_val_paths.append(image_paths[i])
        y_val.append(image_labels[i])
    elif patient_name in test_patients:
        X_test_paths.append(image_paths[i])
        y_test.append(image_labels[i])

# 📘 **Paralel Veri Yükleme ve Prefetch Kullanımı**
def load_and_preprocess_image(path, label):
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [128, 128])
        image = image / 255.0
        return image, label
    except Exception as e:
        print(f"❌ Hatalı resim atlandı: {path}, Hata: {e}")
        return None, None  # Hatalı dosya için None döner

def create_dataset(paths, labels, batch_size=512):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda x, y: x is not None and y is not None)  # Hatalı veriyi çıkar
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(X_train_paths, y_train, batch_size=512)
val_dataset = create_dataset(X_val_paths, y_val, batch_size=512)
test_dataset = create_dataset(X_test_paths, y_test, batch_size=512)

# **CNN Modelini Eğitmek İçin Kullan**
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

num_classes = image_labels.shape[1]  # Düzeltilmiş num_classes

# **CNN Modeli Tanımla**
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    verbose=1,  # Ekranda ilerlemeyi göster
    callbacks=[early_stopping, checkpoint]
)

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Kayıp (Loss): {test_loss}")
print(f"Test Doğruluk (Accuracy): {test_accuracy}")
