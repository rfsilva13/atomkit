"""
Traditional AUTOSTRUCTURE Input for Fe XVII Complex CI Calculation
===================================================================
File: fe17_complex.das

This shows the TRADITIONAL way to set up a multi-configuration
calculation in AUTOSTRUCTURE, including:
- 25+ configurations in cryptic occupation number format
- Intermediate coupling (IC)
- Breit-Pauli Hamiltonian (includes Breit interaction)
- QED corrections (vacuum polarization)
"""

A.S. Fe XVII 
 &SALGEB CUP='ICR' RAD='E1' MXVORB=10 MXCONF=25 KUTSS=-999 KUTSO=0 KUTOO=1  &END
 1 0  2 0  2 1  3 0  3 1  3 2  4 0  4 1  4 2  4 3
  2    2    6    0    0    0    0    0    0    0     ! Ground: 1s2 2s2 2p6
  
  2    2    5    1    0    0    0    0    0    0     ! 2p5 3s1
  2    2    5    0    1    0    0    0    0    0     ! 2p5 3p1
  2    2    5    0    0    1    0    0    0    0     ! 2p5 3d1
  
  2    2    5    0    0    0    1    0    0    0     ! 2p5 4s1
  2    2    5    0    0    0    0    1    0    0     ! 2p5 4p1
  2    2    5    0    0    0    0    0    1    0     ! 2p5 4d1
  2    2    5    0    0    0    0    0    0    1     ! 2p5 4f1
  
  2    1    5    2    0    0    0    0    0    0     ! 2s1 2p5 3s2
  2    1    5    1    1    0    0    0    0    0     ! 2s1 2p5 3s1 3p1
  2    1    5    1    0    1    0    0    0    0     ! 2s1 2p5 3s1 3d1
  2    1    5    0    2    0    0    0    0    0     ! 2s1 2p5 3p2
  2    1    5    0    1    1    0    0    0    0     ! 2s1 2p5 3p1 3d1
  2    1    5    0    0    2    0    0    0    0     ! 2s1 2p5 3d2
  
  2    2    4    2    0    0    0    0    0    0     ! 2p4 3s2
  2    2    4    0    2    0    0    0    0    0     ! 2p4 3p2
  2    2    4    0    0    2    0    0    0    0     ! 2p4 3d2
  
  ...  (8+ more lines for complete CI) ...
  
 &SMINIM NZION=26 QED=-1 IBREIT=1 INUKE=1 IREL=2 &END

"""
Key AUTOSTRUCTURE parameters:
  CUP='IC'    - Intermediate coupling
  RAD='E1'    - Electric dipole transitions
  REL='BP'    - Breit-Pauli (includes Breit interaction)
  NUC='F'     - Finite nuclear size
  NQED=2      - QED corrections (vacuum polarization)
  MXVORB=10   - Number of orbitals (must count!)
  MXCONF=25   - Number of configurations (must be exact!)

❌ PROBLEMS:
  • EXTREMELY cryptic occupation numbers!
  • What does '2 1 5 1 1 0 0 0 0 0' mean?
  • Must manually count orbitals and configurations
  • One wrong number = wrong physics or crash!
  • Nearly impossible to verify by inspection
  • Comments help but don't prevent errors
"""
