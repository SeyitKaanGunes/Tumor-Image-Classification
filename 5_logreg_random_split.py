# Gerekli kütüphaneleri yükleyin
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

# Veri dizini
data_dir = '/content/drive/My Drive/PNG'  # Kendi veri klasörünüzün yolu

# Hastalık isimleri ve klasör adları
disease1_name = 'osman yazıcı'  # Meningotelyal Meningiom (Etiket: 0)
disease2_name = 'gülseren'  # Transizyonel Meningiom (Etiket: 1)

# Görüntülerin boyutunu belirleme
img_size = (128, 128)  # Tüm görüntüler bu boyuta yeniden boyutlandırılacak

# Boş listeler oluşturma
X = []
y = []

# Hastalık 1'in görüntülerini yükleme (Etiket: 0)
disease1_path = os.path.join(data_dir, disease1_name)
if os.path.exists(disease1_path):
    for filename in os.listdir(disease1_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(disease1_path, filename)
            try:
                img = Image.open(img_path).convert('L')  # Gri tonlamaya çevirme
                img = img.resize(img_size)
                img_array = np.array(img).flatten()  # Görüntüyü düzleştirme
                X.append(img_array)
                y.append(0)  # Hastalık 1 etiketi
            except Exception as e:
                print(f"Görüntü yükleme hatası {img_path}: {e}")
else:
    print(f"{disease1_path} klasörü bulunamadı.")

# Hastalık 2'nin görüntülerini yükleme (Etiket: 1)
disease2_path = os.path.join(data_dir, disease2_name)
if os.path.exists(disease2_path):
    for filename in os.listdir(disease2_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(disease2_path, filename)
            try:
                img = Image.open(img_path).convert('L')  # Gri tonlamaya çevirme
                img = img.resize(img_size)
                img_array = np.array(img).flatten()  # Görüntüyü düzleştirme
                X.append(img_array)
                y.append(1)  # Hastalık 2 etiketi
            except Exception as e:
                print(f"Görüntü yükleme hatası {img_path}: {e}")
else:
    print(f"{disease2_path} klasörü bulunamadı.")

# Veriyi numpy dizilerine çevirme
X = np.array(X)
y = np.array(y)

print(f'Görüntü sayısı: {X.shape[0]}')
print(f'Özellik sayısı (her görüntü için): {X.shape[1]}')

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'Eğitim seti boyutu: {X_train.shape[0]}')
print(f'Test seti boyutu: {X_test.shape[0]}')

# Modeli oluşturma
model = LogisticRegression(max_iter=1000, solver='lbfgs')

# Modeli eğitme
model.fit(X_train, y_train)

# Tahminler yapma
y_pred = model.predict(X_test)

# Doğruluk oranını hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Karışıklık matrisini oluşturma
cm = confusion_matrix(y_test, y_pred)
print('Karışıklık Matrisi:')
print(cm)

# Sınıflandırma raporu
report = classification_report(y_test, y_pred, target_names=['Meningotelyal Meningiom', 'Transizyonel Meningiom'])
print('Sınıflandırma Raporu:')
print(report)

# Karışıklık matrisinin görselleştirilmesi
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Meningotelyal Meningiom', 'Transizyonel Meningiom'],
            yticklabels=['Meningotelyal Meningiom', 'Transizyonel Meningiom'])
plt.ylabel('Gerçek')
plt.xlabel('Tahmin')
plt.title('Karışıklık Matrisi')
plt.show()

# Doğru ve yanlış tahminlerin görselleştirilmesi
correct = np.where(y_pred == y_test)[0]
incorrect = np.where(y_pred != y_test)[0]

num_samples = 5
num_correct = len(correct)
num_incorrect = len(incorrect)

print(f'Doğru tahmin sayısı: {num_correct}')
print(f'Yanlış tahmin sayısı: {num_incorrect}')

if num_correct >= num_samples and num_incorrect >= num_samples:
    correct_samples = np.random.choice(correct, num_samples, replace=False)
    incorrect_samples = np.random.choice(incorrect, num_samples, replace=False)

    # Doğru tahminler
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(correct_samples):
        plt.subplot(2, num_samples, i+1)
        img = X_test[idx].reshape(img_size)
        plt.imshow(img, cmap='gray')
        true_label = 'Meningotelyal Meningiom' if y_test[idx] == 0 else 'Transizyonel Meningiom'
        pred_label = 'Meningotelyal Meningiom' if y_pred[idx] == 0 else 'Transizyonel Meningiom'
        plt.title(f'True: {true_label}\nPred: {pred_label}')
        plt.axis('off')

    # Yanlış tahminler
    for i, idx in enumerate(incorrect_samples):
        plt.subplot(2, num_samples, num_samples + i + 1)
        img = X_test[idx].reshape(img_size)
        plt.imshow(img, cmap='gray')
        true_label = 'Meningotelyal Meningiom' if y_test[idx] == 0 else 'Transizyonel Meningiom'
        pred_label = 'Meningotelyal Meningiom' if y_pred[idx] == 0 else 'Transizyonel Meningiom'
        plt.title(f'True: {true_label}\nPred: {pred_label}')
        plt.axis('off')

    plt.suptitle('Doğru ve Yanlış Tahminler')
    plt.show()
else:
    print("Yeterli sayıda doğru veya yanlış tahmin bulunmuyor.")
