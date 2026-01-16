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

It is the first **superconducting quantum computer** in Poland (based on superconducting qubits). The system is intended for advanced research in **quantum informatics, telecommunications, and cybersecurity**. Our project utilizes this state-of-the-art hardware to process data patterns critical for modern security systems.

---

## ⚛️ Project Overview: Banknote Authentication
The primary objective of this project is the **automated classification of banknotes** to determine their authenticity (Authentic vs. Counterfeit).

By utilizing a **Hybrid Quantum-Classical Neural Network**, we process wavelet-transformed data including variance, skewness, kurtosis, and entropy of images. Mapping these classical features into the quantum Hilbert space on **Odra 5** allows the model to identify complex non-linear correlations essential for fraud detection.

### **Quantum Circuit Architecture**
The following diagram represents the 4-qubit **Variational Ansatz** executed on the **ODRA 5** system. It features a layer of parameterized $R_y$ gates followed by a ring-like entanglement structure using controlled rotations.

<img width="1675" height="618" alt="anzatz" src="https://github.com/user-attachments/assets/aa9a2030-c07d-4c52-a7ec-448a9826e168" />

---

## 🔬 Scientific Foundations & Papers
This implementation of the **Quantum Banknote Classifier** and its hybrid architecture are based on the methodologies established in the following research papers:

1. **Circuit-centric quantum classifiers** – *Mitarai et al. (2018)* [arXiv:1803.11173](https://arxiv.org/abs/1803.11173)
   * **Contribution**: This paper provided the theoretical foundation for using parameterized quantum circuits as supervised learning models. We specifically utilized the **Parameter Shift Rule** described here for calculating gradients in our hybrid backpropagation.

2. **Supervised learning with quantum-enhanced feature spaces** – *Havlíček et al. (2019)* [Quantum 3, 141](https://quantum-journal.org/papers/q-2019-12-09-214/)
   * **Contribution**: This work was instrumental in our choice of **Data Encoding**. It describes how classical data can be mapped into the quantum Hilbert space via feature maps, allowing the ODRA 5 system to identify non-linear boundaries in the banknote dataset.
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

### 🚀 Usage
The model initializes a quantum layer optimized via the classical **Adam Optimizer** in a hybrid environment. You can initialize the model using the following Python code:

```python
# Initialization of the model for ODRA 5 execution
# n_qubits=4 reflects the 4 features of the banknote dataset
ansatz_circuit = ansatz(n_qubits=4, depth=2)
model = HybridModel(ansatz_circuit, num_qubits=4)
```
