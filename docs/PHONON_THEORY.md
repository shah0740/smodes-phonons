# Phonons and phonon symmetry

Jay Shah

## 1. Phonons and Phonon Symmetry

Phonons are quasiparticles that describe the collective vibrational
excitations of atoms in a crystal. They are important for understanding heat
transport, superconductivity, and phase transitions among other phenomena of
condensed matter systems. The mathematical framework for phonons established
using the Born-Oppenheimer (adiabatic) approximation is presented in
Section 2 [1]. DFT is used to calculate the phonon modes of a crystal, and
their frequencies can take real or imaginary values, with the latter
indicating the structure is dynamically unstable. These phonon modes can be
indexed by analyzing their symmetries.

Group theory is crucial to illustrate how phonon modes provide sample
identification, and so a brief overview is provided. The symmetry operations
of a crystal form a group [2]. Each symmetry operation can be represented by a
transformation matrix which maps an initial set of coordinates onto another
set after it has undergone the symmetry operation. Such a matrix is called a
representative. Different basis functions, while having the same symmetry,
may have different representatives. The complete set of representatives for a
group is called its representation. Each representation spans its chosen
basis. If all the representatives for a basis can be written in the same
block diagonal form, such that each of the representatives can be written as
the direct sum of two or more representatives of lower dimensionality, the
representation has been reduced. When representations cannot be reduced
anymore, they are called irreducible representations or irreps of the group.
As the representation is reduced to form irreps, the original basis function
has also been converted into a linear combination. This linear combination is
called the symmetry adapted linear combination (SALC).

There are a limited number of symmetry species of a group. In other words,
there are a limited set of ways that an arbitrary function can transform due
to the symmetry operations of the group. Each of these symmetry species are
irreps, and they are labelled according to their dimensionality and their
behavior under certain symmetry operations. Each of the SALCs only transforms
as one irrep. Also, one or more of the SALCs may transform as the same irrep.
The 'Wonderful Orthogonality Theorem' (WOT) is used to identify the irreps
spanned by a basis, and projection operators are used to identify SALCs that
transform as each of the irreps. Details can be found in the text by
Dresselhaus et al. [2]

Phonon modes are indexed in this manner, based on how they transform due to
the symmetry operations of the space group. The difference is instead of
using atoms as the basis, a 3N Cartesian basis is used, in which a set of
$x$-, $y$- and $z$-axes are centered on each atom. Using the WOT and
projection operators, which irreps are spanned by the Cartesian basis and
which SALCs transform as the irreps, can be identified. The SALCs in this
case are the atomic displacements associated with the phonon mode.
Specifically, only information on which atoms are displaced and how is
obtained. DFT is still required to compute the atomic distortions. SMODES, a
program in the ISOTROPY Software Suite that can be run online or downloaded,
gives the SALCs (also called the symmetry adapted modes) for all the phonon
modes at the desired reciprocal lattice $\mathbf{k}$-wavevectors for a
particular crystal [3]. DFT is required to obtain the frequency and
displacements associated with the phonons.

## 2. Phonons

Using the Born-Oppenheimer (adiabatic) approximation a mathematical framework
for phonons will be established [1]. The Hamiltonian of a periodic crystal as
a function of atomic displacements from their equilibrium positions is:

```math
\hat{H} = \Phi_0
+ \frac{1}{2}\sum_{i}\sum_{N\nu} M_\nu\,\dot{U}_i(N\nu)^2
+ \frac{1}{2}\sum_{ij}\sum_{MN\mu\nu} \Phi_{ij}\!\begin{pmatrix}M & N\\ \mu & \nu\end{pmatrix} U_i(M\mu)\,U_j(N\nu)
+ \frac{1}{3!}\sum_{ijk}\sum_{MNP\mu\nu\pi} \Phi_{ijk}\!\begin{pmatrix}M & N & P\\ \mu & \nu & \pi\end{pmatrix} U_i(M\mu)\,U_j(N\nu)\,U_k(P\pi)
+ \cdots
\tag{2.1}
```

The first term $\Phi_0$ is the ground-state electronic energy when there are
no atomic displacements, and the second term is the kinetic energy due to
nuclear motion. $M$, $N$ and $P$ are unit cell indices, $\mu$, $\nu$ and
$\pi$ are atomic indices, and $i$, $j$ and $k$ are Cartesian directions.
$U_i(M\mu)$ refers to the displacement of atom $\mu$ in unit cell $M$ in
direction $i$, $`\Phi_{ij}\begin{pmatrix}M & N\\ \mu & \nu\end{pmatrix}`$ is
the second order force constant that couples displacements $U_i(M\mu)$ and
$U_j(N\nu)$ to energy, and
$`\Phi_{ijk}\begin{pmatrix}M & N & P\\ \mu & \nu & \pi\end{pmatrix}`$ is the
third order coefficient. If the expansion is truncated to second order in
atomic displacements, the potential energy of the crystal is given by:

```math
F = \frac{1}{2}\sum_{ij}\sum_{MN\mu\nu} \Phi_{ij}\!\begin{pmatrix}M & N\\ \mu & \nu\end{pmatrix} U_i(M\mu)\,U_j(N\nu).
\tag{2.2}
```

Force can be found by differentiating the potential energy with respect to
displacement. Next, we assume solutions to the Hamiltonian are traveling
waves with reciprocal lattice wavevector $\mathbf{k}$ and frequency $\omega$
such that:

```math
U_i(M\mu) = \tilde{U}_i(M\mu)\,\exp\!\big(i\mathbf{k}\cdot\mathbf{r}(M\mu) - i\omega t\big)
\tag{2.3}
```

The above expression can be differentiated twice with respect to time to find
acceleration. And by using Newton's second law force = mass x acceleration,
we obtain the following expression for force on atom $(N\nu)$ in the
$j^{\text{th}}$ direction:

```math
\frac{\partial F}{\partial U_j(N\nu)}
= \sum_{Mi\mu} \Phi_{ij}\!\begin{pmatrix}M & N\\ \mu & \nu\end{pmatrix} U_i(M\mu)
= m_\nu\,\omega^2\,\tilde{U}_{j\nu}\,\exp\!\big(i\mathbf{k}\cdot\mathbf{r}(N\nu) - i\omega t\big)
\tag{2.4}
```

where $m_\nu$ is the mass of atom $(N\nu)$. Next using Bloch's theorem, and
defining a new index $L = M - N$, and performing some algebra, we get the
expression:

```math
\omega^2\,\tilde{e}_{j\nu}
= \sum_{Li\mu} \frac{\Phi_{ij}\!\begin{pmatrix}L & 0\\ \mu & \nu\end{pmatrix}}{\sqrt{m_\nu m_\mu}}
\exp\!\big(i\mathbf{k}\cdot[\mathbf{r}(L\mu) - \mathbf{r}(0\nu)]\big)\,\tilde{e}_{i\mu}
\tag{2.5}
```

where $`\tilde{e}_{j\nu} = m_\nu^{1/2}\,\tilde{U}_{j\nu}`$. We define the
dynamical matrix $\mathbf{D}(\mathbf{k})$:

```math
D(\mathbf{k})_{i\mu,j\nu}
= \sum_{L} \frac{\Phi_{ij}\!\begin{pmatrix}L & 0\\ \mu & \nu\end{pmatrix}}{\sqrt{m_\nu m_\mu}}
\exp\!\big(i\mathbf{k}\cdot[\mathbf{r}(L\mu) - \mathbf{r}(0\nu)]\big).
\tag{2.6}
```

The dynamical matrix satisfies the eigenvalue problem, where
$`\tilde{e}_{j\nu} = m_\nu^{1/2}\,\tilde{U}_{j\nu}`$:

```math
\omega^2\,\tilde{e}_{j\nu} = \sum_{i\mu} D(\mathbf{k})_{i\mu,j\nu}\,\tilde{e}_{i\mu}
\tag{2.7}
```

```math
\omega^2\,\tilde{\mathbf{e}} = \mathbf{D}(\mathbf{q})\,\tilde{\mathbf{e}}
\tag{2.8}
```

where we have collapsed the atomic indices and Cartesian directions into a
single compound index with a size 3 times the number of atoms in the unit
cell, and $\mathbf{k}$ is replaced by $\mathbf{q}$ to distinguish between
wavevectors for electronic states and phonons.

The eigenvalues of $\mathbf{D}(\mathbf{q})$ are the squared frequencies of
each phonon mode, and their displacements are defined by $\tilde{e}$. Since
the units of $\tilde{e}$ are length x mass$^{1/2}$, the true displacements
require a division by the square root of the atomic mass. The eigenvalues can
take positive or negative values, which means phonon frequencies can take
real or imaginary values. Imaginary phonon modes indicate the system is
dynamically unstable due to a displacement corresponding to that phonon mode.
To solve for the phonon modes, the dynamical matrix needs to be calculated. To
do so, a force constant matrix $\mathbf{F}(\mathbf{k})$ is defined:

```math
F(\mathbf{k})_{i\mu,j\nu} \equiv
\frac{\Phi_{ij}\!\begin{pmatrix}L & 0\\ \mu & \nu\end{pmatrix}}{\sqrt{m_\nu m_\mu}}
\exp\!\big(i\mathbf{k}\cdot[\mathbf{r}(L\mu) - \mathbf{r}(0\nu)]\big)
\tag{2.9}
```

```math
F(\mathbf{k})_{i\mu,j\nu} = m_\nu^{1/2}\,D(\mathbf{k})_{i\mu,j\nu}\,m_\mu^{1/2}
\tag{2.10}
```

```math
\mathbf{F}(\mathbf{q}) = \mathbf{M}^{1/2}\,\mathbf{D}(\mathbf{q})\,\mathbf{M}^{1/2}.
\tag{2.11}
```

Phonon modes are computed by first calculating the dynamical matrix from the
force constant matrix. The force constant matrix is calculated using the
finite differences method:

```math
\frac{\partial^2 E}{\partial U_j(N\nu)\,\partial U_i(M\mu)} = -\frac{\partial F}{\partial U_j(N\nu)}
\tag{2.12}
```

where finite differences of forces equal the second order derivatives of total
energy with respect to atomic position, calculated using DFT. There will be
$3n$ phonon modes where $n$ is the number of atoms in the unit cell. These
phonon modes can be indexed by analyzing their symmetries.

## 3. References

1. Dove, M.T., 1993. *Introduction to Lattice Dynamics* (No. 4). Cambridge
   University Press.
2. Dresselhaus, M.S., Dresselhaus, G. and Jorio, A., 2007. *Group Theory:
   Application to the Physics of Condensed Matter*. Springer Science &
   Business Media.
3. H. T. Stokes, D. M. Hatch, and B. J. Campbell, SMODES, ISOTROPY Software
   Suite, iso.byu.edu.
