# **Quantum Banknote Classifier on ODRA 5**

> **Official Research Project:** A hybrid quantum-classical machine learning model designed for **ODRA 5** — Poland's first superconducting quantum computer, launched at **Wrocław University of Science and Technology (PWr)**.

---

## 👥 Student Research Group
### **KN Axion**
* **Affiliation**: Wrocław University of Science and Technology (**PWr**).
* **Mission**: Our research is focused on **Quantum Information Science** and exploring its **practical IT applications**. We strive to identify real-world scenarios where quantum computing provides a competitive edge over classical IT systems.
* **Project Role**: This project serves as an implementation of a Variational Quantum Classifier (VQC) on the Odra 5 infrastructure, specifically targeting financial security and document authentication.
* **Group Members**: Iwo Wojtakajtis, Iwo Smura, Rafał Balicki, Karina Leśkiewicz, Maria Płatek, Michał Szczęsny
  
---

## 🏛️ About the ODRA 5 System
**"Odra"** is the name of the first Polish quantum computer, specifically the **Odra 5** model, launched at **PWr** in 2025. 

It is the first **superconducting quantum computer** in Poland (based on superconducting qubits). The system is intended for advanced research in **quantum informatics, telecommunications and cybersecurity**. Our project utilizes this state-of-the-art hardware to process data patterns critical for modern security systems.

---
## 📊 Dataset: Banknote Authentication
The dataset used in this project is sourced from the **UCI Machine Learning Repository**. It was specifically chosen to validate the classification capabilities of the **ODRA 5** quantum system.

### Data Characteristics:
* **Source**: UCI Banknote Authentication Dataset (ID: 267).
* **Origin**: Data were extracted from images of genuine and forged banknote-like specimens using **Wavelet Transform**.
* **Size**: 1372 instances with 4 continuous features and 1 binary target class.

### Features (Input for Quantum Encoding):
Each of the 4 features is mapped to a single qubit via **Angle Encoding** ($R_y$ gates):
1. **Variance** of Wavelet Transformed image.
2. **Skewness** of Wavelet Transformed image.
3. **Kurtosis** of Wavelet Transformed image.
4. **Entropy** of image.

### Target:
* **Class 0**: Authentic banknote.
* **Class 1**: Counterfeit banknote.

### Automatic Data Loading
The project uses the ucimlrepo library to fetch the data directly from the repository. This ensures the dataset is always available without the need for manual downloads.
```bash
from ucimlrepo import fetch_ucirepo 

# Fetch the banknote authentication dataset 
banknote_authentication = fetch_ucirepo(id=267) 

# Access features (X) and targets (y)
X = banknote_authentication.data.features
y = banknote_authentication.data.targets
```
---

## ⚛️ Project Overview: Banknote Authentication
The primary objective of this project is the **automated classification of banknotes** to determine their authenticity (Authentic vs. Counterfeit).

By utilizing a **Hybrid Quantum-Classical Neural Network**, we process wavelet-transformed data including variance, skewness, kurtosis and entropy of images. Mapping these classical features into the quantum Hilbert space on **Odra 5** allows the model to identify complex non-linear correlations essential for fraud detection.

### **Quantum Circuit Architecture**
The following diagram represents the 4-qubit **Variational Ansatz** executed on the **ODRA 5** system. It features a layer of parameterized $R_y$ gates followed by a ring-like entanglement structure using controlled rotations.

<img width="1675" height="618" alt="anzatz" src="https://github.com/user-attachments/assets/aa9a2030-c07d-4c52-a7ec-448a9826e168" />

---

## 🔬 Scientific Foundations & Papers

This project and its hybrid architecture are based on the methodologies established in the following research papers:

1. **Supervised learning with quantum-enhanced feature spaces** – *Havlíček et al. (2019)* [Quantum 3, 141](https://quantum-journal.org/papers/q-2019-12-09-214/)
   * **Contribution**: This work provided the core framework for mapping classical data into the quantum Hilbert space via feature maps, which is the basis for our 4-feature encoding.

2. **Quantum Machine Learning in Liquid** – *Havlíček et al. (2019)* [arXiv:1905.10876](https://arxiv.org/abs/1905.10876)
   * **Contribution**: This paper further explores the practical implementation of quantum algorithms in noisy environments. It supported our research in optimizing the variational circuits for the superconducting qubits of the **ODRA 5** system.

3. **Circuit-centric quantum classifiers** – *Mitarai et al. (2018)* [arXiv:1803.11173](https://arxiv.org/abs/1803.11173)
   * **Contribution**: This paper established the concept of Variational Quantum Classifiers (VQC) and the **Parameter Shift Rule** used in our hybrid backpropagation.
---

## 🌊 Benchmarks & Performance
The hybrid model showed stable convergence and high accuracy on the Banknote Authentication dataset.

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **~92.7%** |
| **F1-Score** | **0.92** |
| **Epochs** | **8** |

### **Training Dynamics**
The model demonstrated a consistent decrease in both training and test loss over 8 epochs, reaching high stability.

<img width="1234" height="492" alt="loss and accuracy" src="https://github.com/user-attachments/assets/c94bd3df-ef66-4176-9a00-283f331620f1" />

### **Confusion Matrix**
The final performance on the test set shows a high true positive rate, accurately identifying banknotes with minimal errors

<img width="649" height="545" alt="confusion metrix" src="https://github.com/user-attachments/assets/c32962b6-257f-4e52-95f3-be067dbf077c" />

## 🛠️ Technical Stack
* **Quantum Hardware**: ODRA 5 (Superconducting Qubits)
* **Software**: Qiskit, Qiskit Machine Learning, PyTorch, Scikit-learn
* **Dataset**: Banknote Authentication (UCI Machine Learning Repository)

## 🚀 Installation
```bash
pip install qiskit qiskit-machine-learning torch ucimlrepo scikit-learn matplotlib pandas numpy
```

## 🖥️ Usage
The model initializes a quantum layer optimized via the classical **Adam Optimizer** in a hybrid environment. You can initialize the model using the following Python code:

```python
# Initialization of the model for ODRA 5 execution
# n_qubits=4 reflects the 4 features of the banknote dataset
ansatz_circuit = ansatz(n_qubits=4, depth=2)
model = HybridModel(ansatz_circuit, num_qubits=4)
```
