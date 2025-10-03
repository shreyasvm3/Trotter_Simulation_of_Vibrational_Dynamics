Code for simulating vibrational dynamics on quantum computers using Trotter approximation.

Reference to full manuscript : https://arxiv.org/abs/2508.11865.

For the Christiansen form, the following is available:
1. Finding solvable fragments of the Hamiltonian using GFRO method (both in HO and VSCF basis, 4T and InfT Hamiltonians).
2. Code to check error in eigenenergies for small molecules to set tolerance for fragmentation.
3. Code to estimate T gate cost using perturbative estimate of Trotter error.
4. Code to calculate IR spretrum.

The last two use modifications of vibrant package as implemented in Pennylane, which can be found here:
https://github.com/PennyLaneAI/pennylane
