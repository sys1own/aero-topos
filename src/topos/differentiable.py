import numpy as np

class DifferentiableTensorLogic:
    """
    Implements a continuous neuro-symbolic contraction engine.
    Computes forward relation truth values via einsum and performs an analytical
    backward pass using the multi-linear chain rule for gradient descent.
    """
    def __init__(self, t_norm="product"):
        self.t_norm = t_norm
        self.cache = {}

    def soft_threshold(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def soft_threshold_derivative(self, cached_y):
        return cached_y * (1.0 - cached_y)

    def forward(self, T, X):
        # Linear projection step via multi-linear contraction
        net = np.einsum("ij,jk->ik", T, X)
        Y = self.soft_threshold(net)
        
        # Cache current configurations for backward pass
        self.cache['T'] = T
        self.cache['X'] = X
        self.cache['Y'] = Y
        return Y

    def backward(self, d_loss_d_Y):
        T = self.cache['T']
        X = self.cache['X']
        Y = self.cache['Y']
        
        # Local activation gradient computation
        G = d_loss_d_Y * self.soft_threshold_derivative(Y)
        
        # Dual contractions for exact analytical gradients
        grad_T = np.einsum("ik,jk->ij", G, X)
        grad_X = np.einsum("ij,ik->jk", T, G)
        return grad_T, grad_X
