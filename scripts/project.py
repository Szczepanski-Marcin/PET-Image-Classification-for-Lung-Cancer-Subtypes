import os
import pydicom
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



base_path = "/Users/Marcin_1/Desktop/PET/imbalanced_dataset/"

features = []

for label in os.listdir(base_path):
    folder_path = os.path.join(base_path, label)
    
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".dcm"):
                
                file_path = os.path.join(folder_path, file)
                ds = pydicom.dcmread(file_path)
                img = ds.pixel_array
                
                # Skip empty images
                if not np.any(img):
                    continue
                
                # Normalize (per image)
                min_val = img.min()
                max_val = img.max()
                
                if max_val > min_val:
                    img = (img - min_val) / (max_val - min_val)
                
                # ROI
                threshold = np.mean(img) + np.std(img)
                roi = img > threshold
                
                # Features
                roi_size = np.sum(roi)
                mean_intensity = np.mean(img)
                max_intensity = np.max(img)
                std_intensity = np.std(img)

                roi_mean_intensity = np.mean(img[roi]) if np.any(roi) else 0
                
                features.append([
                    roi_size,
                    mean_intensity,
                    max_intensity,
                    std_intensity,
                    roi_mean_intensity,
                    label
                ])

# Convert to DataFrame
df = pd.DataFrame(features, columns=[
    "ROI_Size", "Mean_Intensity", "Max_Intensity", "Std_Intensity", "ROI_Mean_Intensity", "Label"
])

print(df.head())

X = df[["ROI_Size", "Mean_Intensity", "Max_Intensity", "Std_Intensity", "ROI_Mean_Intensity"]]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,         
    random_state=42
)


model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))


# Plot ROI size distribution
plt.hist(df["ROI_Size"], bins=30)
plt.title("ROI Size Distribution")
plt.xlabel("ROI Size")
plt.ylabel("Frequency")
plt.show()

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()