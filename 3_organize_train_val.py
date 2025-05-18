import os
import shutil
import random

# Ana klasör ve hedef klasörler
base_dir = '/content/drive/MyDrive/organized_tumors'  # Tümör türleri klasörü
train_dir = '/content/drive/MyDrive/organized_tumors_split/train'  # Eğitim seti
val_dir = '/content/drive/MyDrive/organized_tumors_split/val'  # Doğrulama seti

# Eğitim ve doğrulama oranı
train_split = 0.8

# Eğitim ve doğrulama için hedef klasörleri oluştur
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Ana klasör içindeki her tümör türü klasörünü işle
for tumor_type in os.listdir(base_dir):
    tumor_type_path = os.path.join(base_dir, tumor_type)

    if os.path.isdir(tumor_type_path):
        # Eğitim ve doğrulama klasörlerini oluştur
        tumor_train_dir = os.path.join(train_dir, tumor_type)
        tumor_val_dir = os.path.join(val_dir, tumor_type)
        os.makedirs(tumor_train_dir, exist_ok=True)
        os.makedirs(tumor_val_dir, exist_ok=True)

        # Hasta klasörlerini al
        patient_folders = [f for f in os.listdir(tumor_type_path) if os.path.isdir(os.path.join(tumor_type_path, f))]

        # Karıştır ve ayır
        random.shuffle(patient_folders)
        split_index = int(len(patient_folders) * train_split)
        train_patients = patient_folders[:split_index]
        val_patients = patient_folders[split_index:]

        # Eğitim seti için taşımalar
        for patient in train_patients:
            patient_path = os.path.join(tumor_type_path, patient)
            for root, dirs, files in os.walk(patient_path):
                for file in files:
                    if file.endswith('.png'):  # Sadece PNG dosyalarını taşı
                        src = os.path.join(root, file)
                        dst = os.path.join(tumor_train_dir, file)
                        shutil.copy(src, dst)

        # Doğrulama seti için taşımalar
        for patient in val_patients:
            patient_path = os.path.join(tumor_type_path, patient)
            for root, dirs, files in os.walk(patient_path):
                for file in files:
                    if file.endswith('.png'):  # Sadece PNG dosyalarını taşı
                        src = os.path.join(root, file)
                        dst = os.path.join(tumor_val_dir, file)
                        shutil.copy(src, dst)

print("Veriler başarıyla eğitim ve doğrulama setlerine ayrıldı.")
