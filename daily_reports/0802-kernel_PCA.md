# 02/08/18 Daily Report


## Kernel-PCA

### PCA vs Kernal PCA

The standard PCA always finds linear principal components to represent the data in lower dimension. Sometime, we need non-linear principal components.If we apply standard PCA for the below data, 
it will fail to find good representative direction. Kernel PCA (KPCA) rectifies this limitation.
  
  - Kernel PCA just performs PCA in a new space.
  - It uses Kernel trick to find principal components in different space (Possibly High Dimensional Space).
  - PCA finds new directions based on covariance matrix of original variables. It can extract maximum P (number of features) eigen values. KPCA finds new directions based on kernel matrix. It can extract n (number of observations) eigenvalues.
  - PCA allow us to reconstruct pre-image using few eigenvectors from total P eigenvectors. It may not be possible in KPCA.
  - The computational complexity for KPCA to extract principal components take more time compared to Standard PCA.

<img src="https://github.com/jwcse/DeepLearning/blob/master/img/pca_kpca.png" width="500" height="300">

### Reference

[quora PCA vs kernel PCA](https://www.quora.com/Whats-difference-between-pca-and-kernel-pca)
