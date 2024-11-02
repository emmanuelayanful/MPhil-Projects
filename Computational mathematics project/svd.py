import numpy as np
import os
import warnings

class SVD:
    """
    Singular Value Decomposition (SVD) class that performs matrix factorization.

    Parameters
    ----------
    matrix : ndarray
        The matrix to be decomposed.
    method : str, optional
        The method to use for decomposition. 'power' for Power Iteration or 'qr' for QR method. Default is 'power'.
    rank : int, optional
        Rank for the truncated decomposition. If None, it defaults to the minimum dimension of the matrix.
    full_matrices : bool, optional
        If True, U and V matrices will be of full rank. If False, they will be reduced to the specified rank.
    max_iteration : int, optional
        Maximum number of iterations for convergence. Default is 1000.
    tol : float, optional
        Convergence tolerance. Default is 1e-16.

    Attributes
    ----------
    eigenvalues : ndarray
        Array to store computed eigenvalues of the matrix.
    eigenvectors : ndarray
        Array to store computed eigenvectors of the matrix.
    singular_values : ndarray
        Singular values derived from the eigenvalues.
    U : ndarray
        Left singular vectors matrix.
    V : ndarray
        Right singular vectors matrix (transposed).

    Methods
    -------
    _power_iteration():
        Computes eigenvalues and eigenvectors using Power Iteration method.
    _qr_method(matrix):
        Computes QR decomposition of a matrix.
    _decompose():
        Performs the decomposition based on the chosen method and updates U, V, and singular_values.
    """
    def __init__(self, matrix, method='power', rank=None, full_matrices=True, max_iteration=1000, tol=1e-16):
        self.matrix = matrix
        self.method = method
        self.rank = rank
        self.full_matrices = full_matrices
        self.max_iteration = max_iteration
        self.tol = tol
        self.eigenvalues = None
        self.eigenvectors = None
        self.singular_values = None
        
    def _power_iteration(self):
        """
        Performs eigenvalue and eigenvector computation using the Power Iteration method.

        Returns
        -------
        eigenvalues : ndarray
            Computed eigenvalues for each rank.
        eigenvectors : ndarray
            Computed eigenvectors corresponding to each eigenvalue.
        """
        matrix = self.matrix.T @ self.matrix
        m, n = matrix.shape
        r = 0
        eigenvalues, eigenvectors = [], []
        
        if self.rank is not None and self.rank > m:
            self.rank = m
            warnings.warn('The rank of the matrix is greater than the number of rows. The rank is set to the number of rows.')
        if self.rank is None and self.full_matrices:
            self.rank = m
            warnings.warn('The rank of the matrix is not specified. The rank is set to the number of rows.')
       
        while r < self.rank:
            e_prev = 0
            v = np.ones(m)
            x = np.dot(matrix,v)
            
            for _ in range(self.max_iteration):
                v = x / np.linalg.norm(x)
                e = np.dot(x.T,v)
                x = np.dot(matrix, v)
                
                if np.linalg.norm(e - e_prev) < self.tol:
                    break
                e_prev = e
                
            eigenvalues.append(e)
            eigenvectors.append(v)
            
            matrix = matrix - e * np.outer(v, v)
            r += 1
            
        return np.array(eigenvalues), np.array(eigenvectors).T
    
    def _qr_method(self, matrix):
        """
        Computes the QR decomposition of a matrix.

        Parameters
        ----------
        matrix : ndarray
            The matrix to decompose.

        Returns
        -------
        Q : ndarray
            Orthogonal matrix Q.
        R : ndarray
            Upper triangular matrix R.
        """
        m, n = matrix.shape
        Q = np.zeros((m,n))
        R = np.zeros((n,n))

        for j in range(n):
            v = matrix[:,j]
            for i in range(j):
                R[i,j] = np.dot(Q[:,i], matrix[:,j])
                v = v - R[i,j]* Q[:,i]
            R[j,j] = np.linalg.norm(v)
            Q[:,j] = v / R[j,j]
        return Q, R
    
    def _decompose(self):
        """
        Performs the SVD decomposition on the matrix using the specified method.

        Returns
        -------
        None
        Updates the U, V (transposed), and singular_values attributes.
        """
        if self.method == 'power':
            self.eigenvalues, self.eigenvectors = self._power_iteration()
            self.singular_values = np.sqrt(np.maximum(self.eigenvalues, 0))
            U = self.matrix @ self.eigenvectors / self.singular_values
        
            nonzero_indices = np.where(self.singular_values > 1e-5)[0]
            self.singular_values = self.singular_values[nonzero_indices]
            U = U[:, nonzero_indices]
        
            if not self.full_matrices:
                VT = self.eigenvectors.T[nonzero_indices, :]
            else:
                VT = self.eigenvectors.T
            
            self.U, self.V = U, VT
        
        elif self.method == 'qr':
            B = self.matrix.T@self.matrix
            m, n = B.shape
            U = np.eye(m)
            V = np.eye(n)
            
            for _ in range(self.max_iteration):
                Q, R = self._qr_method(B)
                B= np.dot(R, Q)
                U = np.dot(U, Q)
                V = np.dot(V, Q.T)
                
                norm = np.linalg.norm(B - np.diag(np.diag(B)))
                if norm < self.tol:
                    break
                    
            self.singular_values = np.sqrt(np.diag(B))
            self.U, self.V = U, V.T