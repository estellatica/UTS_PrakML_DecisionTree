Nama  : Nabila Ismiyati Mubarokah

NIM   : 1227050099

Kelas : Praktikum B

Tahapan dan Langkah-Langkah Pembuatan Model Klasifikasi


1. Load Dataset (Memuat Data)

Tujuan: Membaca data dari file dan mempersiapkan label untuk klasifikasi.

Langkah: Membaca file CSV (pandas.read_csv)Melakukan encoding label dari kolom name: orange → 0, grapefruit → 1

df = pd.read_csv("citrus.csv")
df['label'] = df['name'].map({'orange': 0, 'grapefruit': 1})


2. Preprocessing Data (Pra-pemrosesan Data)

Tujuan: Membersihkan dan mempersiapkan data sebelum pelatihan model.

Langkah: Menangani nilai yang hilang (jika ada) dengan mean imputation, Memisahkan fitur (X) dan target (y), Membagi data menjadi: 80% data latih, 20% data uji (dengan stratifikasi)

df = df.fillna(df.mean())
X = df.drop(['name', 'label'], axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)


3. Pemilihan Model & Hyperparameter Tuning

Tujuan: Melatih model Decision Tree dengan parameter terbaik.

Langkah: Menggunakan model Decision Tree Classifier, Melakukan Grid Search untuk mencari parameter terbaik (max_depth, min_samples_split, min_samples_leaf), Menggunakan 5-fold cross-validation untuk evaluasi model selama tuning.

grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)


4. Pelatihan Model Akhir

Tujuan: Melatih model terbaik yang ditemukan dari GridSearchCV.

Langkah:Mengambil model terbaik (best_estimator_) dan melatih ulang.

best_model = grid.best_estimator_
best_model.fit(X_train, y_train)


5. Evaluasi Model

Tujuan: Mengukur performa model terhadap data uji.

Langkah: Menggunakan metrik (Accuracy, Precision, Recall, F1-score), Menampilkan classification report dan confusion matrix.

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


6. Visualisasi Hasil

Tujuan: Memberikan pemahaman visual terhadap model dan hasil klasifikasi.

Langkah: Menampilkan struktur pohon keputusan, Menampilkan confusion matrix dalam bentuk heatmap.

plot_tree(model, feature_names=..., class_names=..., filled=True)
plt.imshow(confusion_matrix, cmap=plt.cm.Blues)


7. Evaluasi Akhir (Opsional)

Langkah tambahan: Mengukur akurasi terhadap data latih sebagai pembanding.

print(f"Training Accuracy: {model.score(X_train, y_train):.4f}")
