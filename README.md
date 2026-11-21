# PRAVEESH.B.M.___CNN___Streamlit__Project
# =======================================================
# AIML Minor Project
# Part A - CNN Image Classification (MNIST)
# Part B - Streamlit Student Marks Calculator
# =======================================================

# -----------------------------
# PART A - CNN IMAGE CLASSIFICATION
# -----------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training Images:", x_train.shape)
print("Test Images:", x_test.shape)

# Preprocessing
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Training
history = model.fit(
    x_train,
    y_train_cat,
    epochs=5,
    validation_split=0.2,
    batch_size=64
)

# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test_cat)

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Accuracy Plot
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")

# Loss Plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss")

plt.show()

# Sample Predictions
predictions = model.predict(x_test)

plt.figure(figsize=(12,6))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')

    actual = np.argmax(y_test_cat[i])
    predicted = np.argmax(predictions[i])

    plt.title(f"A:{actual} P:{predicted}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# PART B - STREAMLIT APP
# Save below section separately and run:
# streamlit run filename.py
# -----------------------------

try:
    import streamlit as st

    st.title("Student Marks Calculator")

    st.write("Enter marks for 5 subjects")

    m1 = st.number_input("Subject 1", 0, 100)
    m2 = st.number_input("Subject 2", 0, 100)
    m3 = st.number_input("Subject 3", 0, 100)
    m4 = st.number_input("Subject 4", 0, 100)
    m5 = st.number_input("Subject 5", 0, 100)

    if st.button("Calculate Result"):

        total = m1 + m2 + m3 + m4 + m5
        percentage = total / 5

        if percentage >= 90:
            grade = "A+"
        elif percentage >= 80:
            grade = "A"
        elif percentage >= 70:
            grade = "B"
        elif percentage >= 60:
            grade = "C"
        elif percentage >= 50:
            grade = "D"
        else:
            grade = "Fail"

        st.success(f"Total Marks: {total}")
        st.info(f"Percentage: {percentage:.2f}%")
        st.write(f"Grade: {grade}")

except:
    pass
