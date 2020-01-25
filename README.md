# MPDA
Matlab source codes of the Manifold Partition Discriminant Analysis (MPDA) algorithm presented in the paper [Manifold Partition Discriminant Analysis](https://ieeexplore.ieee.org/document/7434038).

## Usage
Classification with PRODA on the two moon synthetic dataset:
```
Demo_MPDA.m
```

## Descriptions of the files in this repository  
 - DBpart.mat stores the indices for training (2 samples per class) /test data partition.
 - FERETC80A45.mat stores 320 faces (32x32) of 80 subjects (4 samples per class) from the FERET dataset.
 - Demo_MPDA.m provides example usage of MPDA for subspace learning and classification on 2D facial images.
 - MPDA.m implements the MPDA algorithm.
 - projPRODA.m projects 2D data into the subspace learned by PRODA.
 - sortProj.m sorts features by their Fisher scores in descending order.
 - logdet.m computes the logarithm of determinant of a matrix.

## Citation
If you find our codes helpful, please consider cite the following [paper](https://ieeexplore.ieee.org/document/7434038):
```
@article{
    zhou2017MPDA,
    title={Manifold Partition Discriminant Analysis},
    author={Yang Zhou and Shiliang Sun},
    journal={IEEE Transactions on Cybernetics},
    year={2017},
    volume={47},
    number={4},
    pages={830-840},
    doi={10.1109/TCYB.2016.2529299},
}
```
