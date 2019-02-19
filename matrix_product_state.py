import numpy as np
import copy as cp


class MPS:

    def __init__(self, boundary_condition, model_name=None):
        self.model_name = model_name
        self.mps = {}
        self.tensor_count = 0
        self.bc = boundary_condition

    def add_physical_tensor(self, tensor, physical_leg):
        if self.bc == 'PBC':
            if physical_leg == 0:
                new_shape = (1, 0, 2)
            if physical_leg == 1:
                new_shape = (0, 1, 2)
            if physical_leg == 2:
                new_shape = (0, 2, 1)
            new_tensor = np.transpose(tensor, new_shape)
        if self.bc == 'OPEN':
            if physical_leg == 0:
                new_shape == (0, 1)
            if physical_leg == 1:
                new_shape == (1, 0)
            new_tensor == np.transpose(tensor, new_shape)
        if self.tensor_count > 0:
            if new_tensor.shape[0] != self.mps[self.tensor_count - 1].shape[1]:
                raise ('tensors shape do not match')
        self.mps[self.tensor_count] = new_tensor
        self.tensor_count += 1

    def add_virtual_tensor(self, tensor):
        if self.tensor_count > 0 and self.bc == 'PBC':
            if tensor.shape[0] != self.mps[self.tensor_count - 1].shape[2]:
                raise ('tensors shape do not match')
        if self.tensor_count > 0 and self.bc == 'OPEN':
            if tensor.shape[0] != self.mps[self.tensor_count - 1].shape[1]:
                raise ('tensors shape do not match')
        self.mps[self.tensor_count] = tensor
        self.tensor_count += 1

    def OneSpinMeasurement(self, operator, spin_number):
        spin_number *= 2
        psi = cp.deepcopy(self.mps)
        psi[spin_number] = np.einsum('ijk,jn->ink', psi[spin_number], operator)
        expectation = self.braket(self.Dagger(), psi)
        return expectation

    def braket(self, phi_dagger, psi):
        n = self.tensor_count
        expectation = None
        if self.bc == 'PBC':
            element = np.einsum('ijk,njm->ikmn', psi[0], phi_dagger[0])
            for i in range(1, n, 2):
                element = np.einsum('ijkl,jn->inkl', element, psi[i])
                element = np.einsum('ijkl,kn->ijnl', element, phi_dagger[i])
                if i == n - 1:
                    break
                two_spins_contraction = np.einsum('ijk,njm->ikmn', psi[i + 1], phi_dagger[i + 1])
                element = np.einsum('ijkl,jnmk->inml', element, two_spins_contraction)
            element = np.einsum('ijkl->kl', element)
            element = np.einsum('ij->',element)
            expectation = element

        if self.bc == 'OPEN':
            element = np.einsum('ij,jk->ik', psi[0], phi_dagger[0])
            for i in range(1, n, 2):
                element = np.einsum('ij,in->nj', element, psi[i])
                element = np.einsum('ij,jn->in', element, phi_dagger[i])
                if i == n - 1:
                    two_spins_contraction = np.einsum('ij,kj->ik', psi[i + 1], phi_dagger[i + 1])
                    element = np.einsum('ij,ij->',element, two_spins_contraction)
                    expectation = element
                    break
                two_spins_contraction = np.einsum('ijk,njm->ikmn', psi[i + 1], phi_dagger[i + 1])
                element = np.einsum('ij,inmj->nm', element, two_spins_contraction)
            element = np.einsum('ijkl->kl', element)
            element = np.einsum('ij->',element)
            expectation = element
        return expectation

    def Dagger(self):
        psi_dagger = {}
        for i in range(self.tensor_count):
            psi_dagger[i] = np.conj(self.mps[i])
        return psi_dagger

    def TwoSpinsMeasurement(self, spin1, spin2, operator1, operator2):
        spin1_idx = spin1 * 2
        spin2_idx = spin2 * 2
        if spin1_idx > self.tensor_count or spin2_idx > self.tensor_count:
            raise ('There is no such spin')
        psi = cp.deepcopy(self.mps)
        psi[spin1_idx] = np.einsum('ijk,jn->ink', psi[spin1_idx], operator1)
        psi[spin2_idx] = np.einsum('ijk,jn->ink', psi[spin2_idx], operator2)
        return self.braket(self.Dagger(), psi)

    def NormalizationFactor(self):
        norm = self.braket(self.Dagger(), self.mps)
        return norm

"""
    def SingleSpinMeasure(self, spin, operator):
        ltensor = self.LeftContraction(spin)
        rtensor = self.RightContraction(spin)
        expectation = self.FinalContraction(ltensor, rtensor, operator)
        return expectation

    def LeftContraction(self, stop_spin):
        mps_stop_idx = stop_spin * 2
        mps = cp.deepcopy(self.mps)


    def RightContraction(self, stop_spin):

    def FinalContraction(self, left_contraction, right_contraction, operator):
"""



