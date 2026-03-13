# **Quantum Banknote Classifier on ODRA 5**

> **Official Research Project:** A hybrid quantum-classical machine learning model designed for **ODRA 5** — Poland's first superconducting quantum computer, launched at **Wrocław University of Science and Technology (WUST)**.

---

## Student Research Group: **KN Axion**
* **Affiliation**: Wrocław University of Science and Technology (**WUST**).
* **Mission**: Exploring practical IT applications of Quantum Information Science.
* **Project Role**: Implementation of a Variational Quantum Classifier (VQC) on the **IQM SPARK Odra 5** infrastructure for banknote authentication.
* **Members**: Iwo Wojtakajtis, Iwo Smura, Rafał Balicki, Karina Leśkiewicz, Maria Płatek, Michał Szczęsny.

---

## Repository Structure
* **`CrossValidation/`** – Scripts ensuring statistical stability and robustness of the results.
* **`evaluation_and_comparison/`** – Detailed analysis including:
    * `ansatz_comparison`: Hardware vs. Simulator performance.
    * `depth_comparison`: Comparizon of circuit depth, gate counts, and execution times across ansatz depths (2, 4, 6) with and without Qiskit transpiler optimization on the IQM ODRA 5 quantum processor.
    * `model_evaluation`: Evaluation of the model.
    * `ram`: RAM memory requirements .
    * `shot_noise`: shot-by-shot noise visualization on IQM SPARK.
* **`models/`** – Implementations of three distinct approaches:
    * `classical_ml`: Standard ML solution used as a performance reference.
    * `model_simulator`: VQC running on a noise-free simulator.
    * `model_odra`: Model specifically tuned for the **IQM Spark (ODRA 5)** hardware.
* **`weights/`** – Pre-trained weights for all models.
* **`ansatz.ipynb`** – Detailed description of IQM SPARK adapted circuit architectur.
* **`eda.ipynb`** – Exploratory Data Analysis of the UCI dataset.

---

## Project Overview
The core objective of this project is to use the **UCI Banknote Authentication** dataset as a benchmark for comparing various machine learning paradigms. 

In this project, we treat the dataset as a foundation for:
* **Ansatz Structure Comparison**: We test defferent circuits (VQC) architectures to determine which gives best results and compare them to classical model.
* **Hardware Constraint Analysis**: The banknote data provides a balanced environment to analyze how gate errors and decoherence on the **ODRA 5** processor affect classification accuracy compared to ideal simulated conditions.

---

## Scientific Foundations & Research Papers

This project and its hybrid architecture are based on methodologies established in the following research papers:

1.  **Supervised learning with quantum-enhanced feature spaces** – *Havlíček et al. (2019)* [Quantum 3, 141](https://quantum-journal.org/papers/q-2019-12-09-214/)
    * **Contribution**: Provided the core framework for mapping classical data into the quantum Hilbert space via feature maps, forming the basis for our 4-feature encoding.
2.  **Quantum Machine Learning in Liquid** – *Havlíček et al. (2019)* [arXiv:1905.10876](https://arxiv.org/abs/1905.10876)
    * **Contribution**: Explores practical implementation in noisy environments, supporting our research in optimizing variational circuits for the superconducting qubits of **ODRA 5**.
3.  **Circuit-centric quantum classifiers** – *Mitarai et al. (2018)* [arXiv:1803.11173](https://arxiv.org/abs/1803.11173)
    * **Contribution**: Established the concept of Variational Quantum Classifiers (VQC) and the **Parameter Shift Rule** used in our hybrid backpropagation.
4.  **The power of quantum neural networks** – *Abbas et al. (2021)* [Nature Communications 12, 1476](https://www.nature.com/articles/s41467-021-21728-w)
    * **Contribution**: Key research on the capacity and trainability of quantum neural networks, helping us evaluate the effective dimension and expressivity of our chosen **ansatz** structures.

---

## Dataset: Banknote Authentication

The dataset used in this project is sourced from the **UCI Machine Learning Repository**. It was specifically chosen to validate the classification capabilities of the **IQM SPARK ODRA 5** quantum system.



### Data Characteristics:

* **Source**: UCI Banknote Authentication Dataset (ID: 267).

* **Origin**: Data were extracted from images of genuine and forged banknote-like specimens using **Wavelet Transform**.

* **Size**: 1372 instances with 4 continuous features and binary target class.



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

## Benchmarks & Performance
Our research focuses on two primary analytical axes:
1.  **Simulation vs Reality**: Evaluating the impact of real-world noise and decoherence on the **IQM Spark** processor compared to noise-free simulations.
2.  **Circuit Depth Impact**: Investigating the trade-off between model expressivity (more layers) and the accumulation of physical gate errors on the actual QPU.

---

## Technical Stack
* **Hardware**: ODRA 5 (Superconducting Qubits, IQM Spark)
* **Software**: Qiskit, PyTorch, Scikit-learn, ucimlrepo
* **Optimization**: Hybrid Quantum-Classical Training (Adam Optimizer via Parameter Shift Rule)

---

## Usage & Workflow
1.  **Exploration**: Start with `eda.ipynb` to understand the input data distribution.
2.  **Architecture**: Review `ansatz.ipynb` to see the circuit tested.
3.  **Comparison**: Check the `evaluation_and_comparison` folder for scripts that contrast results from the simulator and the physical quantum computer.
4.  **Inference**: You can load pre-trained weights from the `/weights` folder to reproduce our results or run inference on your own QPU/Simulator.

---

