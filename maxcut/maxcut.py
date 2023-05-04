import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from qiskit import Aer
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=4096)
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

edges = [(0, 1), (0, 3), (1, 2), (2, 3)]
graph = nx.Graph(edges)
nx.draw(graph, with_labels=True, node_color="yellow", font_weight="bold")

n = graph.number_of_nodes()
qubo = QuadraticProgram()
qubo.binary_var_list([f'x{i}' for i in range(n)])

linear = {v: 0 for v in range(n)}
quadratic = {(i, j): -0.5 for i, j in edges}
qubo.maximize(linear=linear, quadratic=quadratic)

eigensolver = MinimumEigenOptimizer(NumPyMinimumEigensolver())
result_classical = eigensolver.solve(qubo)
print("Classical solution:", result_classical)

optimizer = COBYLA(maxiter=100)
qaoa = QAOA(optimizer, reps=2, quantum_instance=quantum_instance)

qaoa_optimizer = MinimumEigenOptimizer(qaoa)

result_quantum = qaoa_optimizer.solve(qubo)
print("Quantum solution:", result_quantum)

# Visualize the results
colors = ['red' if result_quantum.x[i] == 1 else 'blue' for i in range(n)]
nx.draw(graph, with_labels=True, node_color=colors, font_weight="bold")
plt.show()
