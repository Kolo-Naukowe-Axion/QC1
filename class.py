from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import Estimator 
from qiskit.quantum_info import SparsePauliOp
import numpy as np
class HybridModel:
    def __init__(self, ansatz_circuit, num_qubits, input_dimension):
        self.num_qubits = num_qubits
        self.input_dimension = input_dimension

        self.feature_map = ZZFeatureMap(feature_dimension=input_dimension, reps=1)

        self.qc = QuantumCircuit(num_qubits)
        self.qc.compose(self.feature_map,qubits=range(input_dimension), inplace=True)
        self.qc.compose(ansatz_circuit, inplace=True)

        final_circuit_params = self.qc.parameters

        feature_map_names = {p.name for p in self.feature_map.parameters}
        ansatz_names = {p.name for p in ansatz_circuit.parameters}
        
        # Sort the final parameters into w and input
        self.final_input_params = []
        self.final_weight_params = []
        
        for p in final_circuit_params:
            if p.name in feature_map_names:
                self.final_input_params.append(p)
            elif p.name in ansatz_names:
                self.final_weight_params.append(p)
        
        # check bo cos wczesniej nie zczytywalo ansatz
        if len(self.final_weight_params) == 0:
            print("CRITICAL ERROR: No weight parameters found in the circuit!")
            print(f"Ansatz names: {ansatz_names}")
            print(f"Circuit params: {[p.name for p in final_circuit_params]}")
        

        observable = SparsePauliOp.from_list([("I" * (num_qubits - 1) + "Z", 1)])

        estimator = Estimator()
        # okay skapnęłam się jeszcze, że robiłam to na Estimator, który zaraz wyjdzie z użycia w Qiskit
        # więc kolejny powinien być taki ale z nim nie testowałam kodu
        # from qiskit.primitives import StatevectorEstimator
        # estimator = StatevectorEstimator()
        gradient = ParamShiftEstimatorGradient(estimator)
        self.qnn = EstimatorQNN(
            circuit=self.qc,
            observables=observable,
            input_params=self.final_input_params,
            weight_params=self.final_weight_params,
            estimator=estimator,
            gradient = gradient
        )

    def forward(self, x, weights):
        return self.qnn.forward(x, weights)
    
    def backward(self, x, weights):
        _, weight_grads = self.qnn.backward(x, weights)
        if weight_grads is None:
            # If it fails, return zeros to prevent the loop from crashing
            # This was messy process
            print("Warning: Gradients were None. Returning Zeros.")
            return np.zeros((x.shape[0], len(weights)))
        return weight_grads
    

# Inicjalizacja modelu
my_ansatz = ansatz_14(4, 4)
qnn = HybridModel(
    ansatz_circuit=my_ansatz,
    num_qubits=5,
    input_dimension=4
)


# Inicjalizacja wag
num_weights = qnn.qnn.num_weights
rng = np.random.default_rng(seed=42)
weights = 2 * np.pi * rng.random(num_weights)
weights = weights.flatten()

# Just to be sure spłaszczam jeszcze do D1
print(f"Weights initialized. Shape: {weights.shape}")