
Experimental Force-Torque Dataset for Robot Learning of Multi-Shape Insertion
==========

- The dataset records the peg poses and the interaction forces for a variety of multiple shape insertion tasks.

- The face of the polyhedron in contact with the environment is n-gon regular convex polygons with n = {3; 4; 5; 6; 200}.

- The value of the force torque sensor and the end effector position for each point is recorded and stored as vectors of {Fx; Fy; Fz; Mx; My; Mz; Px; Py; Pz; Az; t; cont}. Px, Py, Pz are the peg positions with respect to the hole; Az is the peg angle with the respect to the hole angle; Fx, Fy, Fz, Mx, My, Mz are the forces and moments in the force sensor frame; t is the time and cont is a counter for the dataset points.

- To load the file you can use the following python code:
  ```python
  import numpy
  data = numpy.load('data.npy')
  ```

## Structure of the Dataset directory
```
Dataset/
│
├── 3_sides (polyhedron with n = 3)
│    └─ CAD_Models: directory for the peg and hole models
│    │   └── Hole: model of the hole
│    │        └── STL: stl model. This file format is supported by many software packages
│    │			       └── hole.stl
│    │        └── OpenSCAD: 3D model using free multi platform software OpenSCAD
│    │			       └── hole.scad
│    │   └── Peg: model of the peg
│    │        └── STL: stl model. This file format is supported by many software packages
│    │			       └── peg.stl
│    │        └── OpenSCAD: 3D model using free multi platform software OpenSCAD
│    │			       └── peg.scad
│    └─ Data: folder containing the data
│         └── data.npy: numpy file containing the data in the format explained above.
│    			   
│
├── 4_sides (polyhedron with n = 4)
│    └─ CAD_Models: directory for the peg and hole models
│    │   └── Hole: model of the hole
│    │        └── STL: stl model. This file format is supported by many software packages
│    │			       └── hole.stl
│    │        └── OpenSCAD: 3D model using free multi platform software OpenSCAD
│    │			       └── hole.scad
│    │   └── Peg: model of the peg
│    │        └── STL: stl model. This file format is supported by many software packages
│    │			       └── peg.stl
│    │        └── OpenSCAD: 3D model using free multi platform software OpenSCAD
│    │			       └── peg.scad
│    └─ Data: folder containing the data
│        └── data.npy: numpy file containing the data in the format explained above.
│    			   
│
├── 5_sides (polyhedron with n = 5)
│    └─ CAD_Models: directory for the peg and hole models
│    │   └── Hole: model of the hole
│    │        └── STL: stl model. This file format is supported by many software packages
│    │			       └── hole.stl
│    │        └── OpenSCAD: 3D model using free multi platform software OpenSCAD
│    │			       └── hole.scad
│    │   └── Peg: model of the peg
│    │        └── STL: stl model. This file format is supported by many software packages
│    │			       └── peg.stl
│    │        └── OpenSCAD: 3D model using free multi platform software OpenSCAD
│    │			       └── peg.scad
│    └─ Data: folder containing the data
│        └── data.npy: numpy file containing the data in the format explained above.
│    			   
│
├── 6_sides (polyhedron with n = 6)
│    └─ CAD_Models: directory for the peg and hole models
│    │   └── Hole: model of the hole
│    │        └── STL: stl model. This file format is supported by many software packages
│    │			       └── hole.stl
│    │        └── OpenSCAD: 3D model using free multi platform software OpenSCAD
│    │			       └── hole.scad
│    │   └── Peg: model of the peg
│    │        └── STL: stl model. This file format is supported by many software packages
│    │			       └── peg.stl
│    │        └── OpenSCAD: 3D model using free multi platform software OpenSCAD
│    │			       └── peg.scad
│    └─ Data: folder containing the data
│        └── data.npy: numpy file containing the data in the format explained above.
│    			   
│
├── 200_sides (polyhedron with n = 200)
│    └─ CAD_Models: directory for the peg and hole models
│    │   └── Hole: model of the hole
│    │        └── STL: stl model. This file format is supported by many software packages
│    │			       └── hole.stl
│    │        └── OpenSCAD: 3D model using free multi platform software OpenSCAD
│    │			       └── hole.scad
│    │   └── Peg: model of the peg
│    │        └── STL: stl model. This file format is supported by many software packages
│    │			       └── peg.stl
│    │        └── OpenSCAD: 3D model using free multi platform software OpenSCAD
│    │			       └── peg.scad
│    └─ Data: folder containing the data
│        └── data.npy: numpy file containing the data in the format explained above.
```
