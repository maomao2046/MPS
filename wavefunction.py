import numpy as np
import matplotlib.pyplot as plt
import matrix_product_state as mps
import copy as cp


class wavefunction:

    def __init__(self):
        self.spin_count = 0
        self.spin_dim = 0
        self.tensor = None
        self.bra = None
        self.ket = None
        self.denmat = None

    def addwf(self, tensor, spin_count, spin_dim):
        self.spin_count = spin_count
        self.spin_dim = spin_dim
        self.tensor = tensor
        self.bra = np.reshape(np.conj(cp.copy(tensor)), (1, spin_dim ** spin_count))
        self.ket = np.transpose(np.conj(self.bra))
        norm = np.sqrt(self.bra.dot(self.ket))
        self.bra /= norm
        self.ket /= norm
        self.tensor /= norm
        self.denmat = self.ket.dot(self.bra)

    def SingleSpinMeasurement(self, spin_num, operator):
        super_operator = self.SuperOp(spin_num, operator)
        expectation = super_operator.dot(self.ket)
        expectation = self.bra.dot(expectation)
        return expectation

    def SuperOp(self, spin_num, operator):
        order = range(self.spin_count)
        supop = operator
        for i in order:
            if i < spin_num:
                supop = np.kron(np.eye(self.spin_dim), supop)
            if i > spin_num:
                supop = np.kron(supop, np.eye(self.spin_dim))
        return supop
