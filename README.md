# A computational program for interacting Schwinger bosons on the triangular lattice.

Constrained Schwinger bosons provide a faithful spin-S (2*S = n+1)
representation of SU(2) [1,2]. A crucial application of this technique
produces a hard-core bosonic description of the two-state spin-1/2 
system studied across the fields of magnetism and quantum
computations. The presence of constraints, however, poses a challenge
and when multiple spin systems couple via exchange interactions, the
combined representation for the composite system becomes strongly
interacting. In this computational project, a controlled perturbative
expansion of this very scenario is investigated in the triangular
lattice of arbitrary size. Our paper accompanying these calculations
appeared in:

S. Dey, J. Maciejko, and M. Vojta, [Field-driven transition from quantum spin liquid to magnetic order in triangular-lattice antiferromagnets](https://doi.org/10.1103/PhysRevB.109.224424), Phys. Rev. B 109, 224424 (2024).

The python scripts are categorized according to their computational
role and the computational entry-point is provided by "run.sh" for
cpu/gpu based implementations on a personal computer and by the
"run_cluster.sh" based implementation for a remote Slurm-managed
cluster.

The core computation stages progress as follows:

1. Setting up the triangular lattice graph of arbitrary sizes and
   its Brillouin zone.
2. Constructing the interacting Schwinger boson representation of
   the Heisenberg-Zeeman spin Hamiltonian for the graph.
3. Performing a saddle-point energy minimization to obtain the
   lowest semi-classical ground state for the system for an
   arbitrary external magnetic field (up to the saturation field).
4. Computing the lattice-based loop corrections over the
   semi-classical states to refine the accuracy of the phase
   diagram and calculate physical response functions.

To meet these computational challenges, we use PyTorch and/or MPI based
methods. Specifically,

1. The fast saddle-point solver uses the automatic derivative
   library from PyTorch.
2. The computational complexity of the nested loop-sums is
   tackled by cpu/gpu based parallelization.

The code base is built in modules so that many of its assets could be
reused to target different lattices. 


## References 

[1] A. Auerbach, [Interacting Electrons and Quantum Magnetism](https://doi.org/10.1007/978-1-4612-0869-3) (Springer, New York, 1994).

[2] S.-S. Zhang, E. A. Ghioldi, L. O. Manuel, A. E. Trumper, and C. D. Batista, [Schwinger boson theory of ordered magnets](https://doi.org/10.1103/PhysRevB.105.224404), Phys. Rev. B 105, 224404 (2022).
