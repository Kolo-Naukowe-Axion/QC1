def ansatz_14(n_qubits, depth):

    theta = ParameterVector('θ', 8*depth)

    qc = QuantumCircuit(n_qubits)

    for j in range(depth//2):
        #Warstwa 1/4(0-7)(16-23)
        for i in range(n_qubits):
            qc.ry(theta[j*n_qubits*4+i], i)

        qc.crx(theta[j*n_qubits*4+4], 3, 0)
        qc.crx(theta[j*n_qubits*4+5], 2, 3)
        qc.crx(theta[j*n_qubits*4+6], 1, 2)
        qc.crx(theta[j*n_qubits*4+7], 0, 1)


        #Warstwa 2/4(8-15)(25-31)
        for i in range(n_qubits):
            qc.ry(theta[j*n_qubits*4+8 + i], i)


        qc.crx(theta[j*n_qubits*4+12], 3, 2)
        qc.crx(theta[j*n_qubits*4+13], 0, 3)
        qc.crx(theta[j*n_qubits*4+14], 1, 0)
        qc.crx(theta[j*n_qubits*4+15], 2, 1)


    return qc


#91% na 4 warstwach
def ansatz_A1(n_qubits, depth):

    theta = ParameterVector('θ', 8*depth)

    qc = QuantumCircuit(n_qubits)

    for j in range(depth//2):
        #Warstwa 1/4(0-7)(16-23)
        for i in range(n_qubits):
            qc.ry(theta[j*n_qubits*4+i], i)
        qc.barrier()

        qc.crx(theta[j*n_qubits*4+4], 3, 0)
        qc.crx(theta[j*n_qubits*4+5], 2, 3)
        qc.crx(theta[j*n_qubits*4+6], 1, 2)
        qc.crx(theta[j*n_qubits*4+7], 0, 1)
        qc.barrier()

        #Warstwa 2/4(8-15)(25-31)
        for i in range(n_qubits):
            qc.rx(theta[j*n_qubits*4+8 + i], i)
        qc.barrier()

        qc.cry(theta[j*n_qubits*4+12], 3, 2)
        qc.cry(theta[j*n_qubits*4+13], 0, 3)
        qc.cry(theta[j*n_qubits*4+14], 1, 0)
        qc.cry(theta[j*n_qubits*4+15], 2, 1)
        qc.barrier()

    return qc
