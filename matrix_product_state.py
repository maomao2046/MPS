import numpy as np
import copy as cp
import MPS as ms


class MPS:

    def __init__(self, boundary_condition, model_name=None):
        self.model_name = model_name
        self.mps = {}
        self.reduced_mps = None
        self.tensor_count = 0
        self.bc = boundary_condition


    def wavefunction2mps(self, psi, k):
        mps = ms.canon_matrix_product_state(psi, k)
        for item in mps.keys():
            self.mps[item] = mps[item]
            self.tensor_count += 1

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
            element = np.einsum('ij,ik->jk', psi[0], phi_dagger[0])
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

    def SingleSpinMeasure(self, spin, operator):
        """
            only for an open boundary condition mps
        """
        ltensor = self.LeftContraction(spin)
        rtensor = self.RightContraction(spin)
        expectation = self.FinalContraction(ltensor, rtensor, spin, operator)
        return expectation

    def LeftContraction(self, stop_spin):
        mps_stop_idx = stop_spin * 2
        mps = cp.deepcopy(self.mps)
        mpsdag = self.Dagger()
        if mps_stop_idx == 0:
            return 1
        temp1 = np.einsum('ij,jk->ik', mps[0], mps[1])
        temp2 = np.einsum('ij,jk->ik', mpsdag[0], mpsdag[1])
        ltensor = np.einsum('ij,ik->jk', temp1, temp2)
        for i in range(2, mps_stop_idx, 2):
            spintensor = np.einsum('ijk,ljm->ilkm', mps[i], mpsdag[i])
            ltensor = np.einsum('ij,ijkl->kl', ltensor, spintensor)
            ltensor = np.einsum('ij,ik->kj', ltensor, mps[i + 1])
            ltensor = np.einsum('ij,jk->ik', ltensor, mpsdag[i + 1])
        return ltensor

    def RightContraction(self, stop_spin):
        mps = cp.deepcopy(self.mps)
        mps_stop_idx = stop_spin * 2
        mpsdag = self.Dagger()
        n = len(mps.keys()) - 1
        if mps_stop_idx == n:
            return 1
        temp1 = np.einsum('ij,jk->ik', mps[n - 1], mps[n])
        temp2 = np.einsum('ij,jk->ik', mpsdag[n - 1], mpsdag[n])
        rtensor = np.einsum('ij,kj->ik', temp1, temp2)
        for i in range(n - 2, mps_stop_idx, -2):
            spintensor = np.einsum('ijk,ljm->ilkm', mps[i], mpsdag[i])
            rtensor = np.einsum('ijkl,kl->ij', spintensor, rtensor)
            rtensor = np.einsum('ij,jk->ik', mps[i - 1], rtensor)
            rtensor = np.einsum('ij,kj->ki', mpsdag[i - 1], rtensor)
        return rtensor

    def FinalContraction(self, left_contraction, right_contraction, spin, operator):
        mps = cp.deepcopy(self.mps)
        mpsdag = cp.deepcopy(self.Dagger())
        mps_spin_idx = spin * 2
        if mps_spin_idx == 0:
            midtensor = np.einsum('ij,il->lj', mps[mps_spin_idx], operator)
            midtensor = np.einsum('ij,il->lj', midtensor, mpsdag[mps_spin_idx])
            element = np.einsum('ij,ji->', midtensor, right_contraction)

        elif (mps_spin_idx - len(mps.keys())) == -1:
            midtensor = np.einsum('ij,jk->ik', mps[mps_spin_idx], operator)
            midtensor = np.einsum('ij,kj->ik', midtensor, mpsdag[mps_spin_idx])
            element = np.einsum('ij,ij->', left_contraction, midtensor)

        else:
            midtensor = np.einsum('ijk,jl->ilk', mps[mps_spin_idx], operator)
            midtensor = np.einsum('ijk,njl->inkl', midtensor, mpsdag[mps_spin_idx])
            element = np.einsum('ijkl,kl->ij', midtensor, right_contraction)
            element = np.einsum('ij,ij->', left_contraction, element)
        return element

    def VirtualTensorContraction(self):
        self.reduced_mps = {}
        if self.bc == 'OPEN':
            k = 1
            self.reduced_mps[0] = np.einsum('ij,jk->ik', self.mps[0], self.mps[1])
            for i in range(2, self.tensor_count - 1, 2):
                self.reduced_mps[k] = np.einsum('ijk,kl->ijl', self.mps[i], self.mps[i + 1])
                k += 1
            self.reduced_mps[k] = self.mps[self.tensor_count - 1]









