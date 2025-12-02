# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # Constants
# IMG_HEIGHT, IMG_WIDTH = 128, 128
# BATCH_SIZE = 32
# EPOCHS = 100
# DATASET_PATH = 'dataset/'
# MODEL_SAVE_PATH = 'model/uniform_model.keras'
# os.makedirs('model', exist_ok=True)

# # Model definition with L2 regularization and higher dropout
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
#     MaxPooling2D((2, 2)),

#     Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
#     MaxPooling2D((2, 2)),

#     Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
#     MaxPooling2D((2, 2)),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')  # Binary classification output
# ])

# # Compile the model
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # Improved data augmentation
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     brightness_range=[0.8, 1.2],
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# val_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
# )

# # Training data generator
# train_gen = train_datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='training',
#     shuffle=True
# )

# # Validation data generator
# val_gen = val_datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     class_mode='binary',
#     subset='validation',
#     shuffle=False
# )

# # Early stopping and model checkpoint
# early_stop = EarlyStopping(patience=10, restore_best_weights=True)
# checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)

# # Train the model
# model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=EPOCHS,
#     callbacks=[early_stop, checkpoint]
# )

# # Show model architecture
# model.summary()

# print(f"✅ Model saved successfully at '{MODEL_SAVE_PATH}'")


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# -----------------------------
# Constants & Paths
# -----------------------------
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 100
DATASET_PATH = 'dataset/'              
MODEL_SAVE_PATH = 'model/uniform_model_final_stable.keras'

os.makedirs('model', exist_ok=True)

# -----------------------------
# Model Definition (CNN) - L2 Regularization Relaxed
# -----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu',
           kernel_regularizer=l2(0.003), 
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.003)), 
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.003)), 
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

# -----------------------------
# Compile the model - KEY FIX: Lower Learning Rate
# -----------------------------
custom_adam = Adam(learning_rate=0.0001)

model.compile(
    optimizer=custom_adam,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Data Generators
# -----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("Class indices:", train_gen.class_indices)

# -----------------------------
# Callbacks
# -----------------------------
early_stop = EarlyStopping(
    patience=15, 
    restore_best_weights=True,
    monitor='val_loss'
)

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    save_best_only=True,
    monitor='val_loss'
)

# -----------------------------
# Train the Model
# -----------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# -----------------------------
# Evaluation and Plotting
# -----------------------------
model.summary()
print(f"✅ Final Stable Model saved successfully at '{MODEL_SAVE_PATH}'")

# -----------------------------
# 1. Plot Accuracy & Loss Curves
# -----------------------------
# Accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('model/training_curves_final.png', dpi=300)
plt.show()

# -----------------------------
# 2. Confusion Matrix & Classification Report
# -----------------------------
val_gen.reset()
y_true = val_gen.classes 
y_prob = model.predict(val_gen)
y_pred = (y_prob > 0.5).astype("int32").flatten()

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

target_names = list(val_gen.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Final Model')
plt.colorbar()
tick_marks = np.arange(len(target_names))
plt.xticks(tick_marks, target_names, rotation=45)
plt.yticks(tick_marks, target_names)

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('model/confusion_matrix_final.png', dpi=300)
plt.show()

# -----------------------------
# 3. ROC Curve & AUC
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

print(f"\nROC AUC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Final Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('model/roc_curve_final.png', dpi=300)
plt.show()

# -----------------------------
# 4. Additional Metrics
# -----------------------------
if cm.shape == (2, 2):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nAdditional Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"False Positive Rate: {fp/(fp+tn):.4f}" if (fp+tn) > 0 else "False Positive Rate: 0.0000")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}" if (fn+tp) > 0 else "False Negative Rate: 0.0000")

print("✅ All performance graphs and metrics generated successfully.")