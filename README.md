

# 🧠 Handwritten Digit Recognition using MNIST Dataset

This repository contains a simple implementation of a neural network for recognizing handwritten digits using the MNIST dataset. It showcases two different architectures:
1. A **Single-Layer Neural Network** (No hidden layer)
2. A **Multi-Layer Neural Network** (With one hidden layer)

## 📊 Accuracy
- **Single-Layer Model**: 92%  
- **Multi-Layer Model**: 97%

---

## 📁 Dataset
The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) consists of:
- 60,000 training images
- 10,000 test images  
Each image is 28x28 pixels of grayscale handwritten digits from 0 to 9.

---

## 🧪 Models and Architecture

### 1. 📦 Model without Hidden Layer
```python
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)
````

* **Activation**: Sigmoid
* **Final Accuracy**: \~92%

---

### 2. 🔁 Model with One Hidden Layer

```python
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)
```

* **Hidden Layer**: 100 neurons, ReLU activation
* **Output Layer**: 10 neurons, Sigmoid activation
* **Final Accuracy**: \~97%

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib (optional, for visualization)

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/mnist-digit-recognition.git
cd mnist-digit-recognition
```

2. Install dependencies:

```bash
pip install tensorflow numpy matplotlib
```

3. Run the script:

```bash
python mnist_model.py
```

---

## 📌 Results

| Model              | Hidden Layers | Activation     | Accuracy |
| ------------------ | ------------- | -------------- | -------- |
| Single-Layer       | No            | Sigmoid        | 92%      |
| Multi-Layer (1 HL) | 100 Neurons   | ReLU + Sigmoid | 97%      |

---

## 📸 Sample Output (Optional)

You can visualize predictions and misclassified digits using `matplotlib`:

```python
import matplotlib.pyplot as plt

plt.imshow(X_test[0])
print("Predicted Label:", model.predict_classes([X_test_flattened[0]]))
```

---

## 🧠 Learnings

* Adding hidden layers and using ReLU improves model performance significantly.
* Simple feedforward networks can achieve high accuracy with minimal tuning on MNIST.

---

