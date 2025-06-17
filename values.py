import numpy as np

# Define constants
B1 = -4.0035
B12 = -0.9535
B23 = -0.9535
B34 = -0.9535
B45 = -0.9535
B56 = -0.9535

L = np.array([  [0.058, 0, 0, -0.058, 0, 0, 0, 0, 0],
                [0, 0.063, 0, 0, -0.063, 0, 0, 0, 0],
                [0, 0, 0.059, 0, 0, -0.059, 0, 0, 0],
                [-0.058, 0, 0, 0.235, 0, 0, -0.085, -0.092, 0],
                [0, -0.063, 0, 0, 0.296, 0, -0.161, 0, -0.072],
                [0, 0, -0.059, 0, 0, 0.330, 0, -0.170, -0.101],
                [0, 0, 0, -0.085, -0.161, 0, 0.246, 0, 0],
                [0, 0, 0, -0.092, 0, -0.170, 0, 0.262, 0],
                [0, 0, 0, 0, -0.072, -0.101, 0, 0, 0.173]])

# Inertia and damping
M = np.diag([0.125, 0.034, 0.016])
D = np.diag([0.125, 0.068, 0.048])

n = M.shape[0]

kp = -1.5
ki = -50

# Extract submatrices
Lgg = L[:n, :n]
Lgl = L[:n, n:]
Llg = L[n:, :n]
Lll = L[n:, n:]

m = Lll.shape[1]

M_inv = np.linalg.inv(M)
Lll_inv = np.linalg.inv(Lll)

# Define matrix A
A = np.vstack([np.hstack([np.zeros((n, n)), np.eye(n)]),
              np.hstack([M_inv * (-Lgg + np.dot(np.dot(Lgl, Lll_inv), Llg)), -np.dot(D, M_inv)])])

# Define matrix B
B = np.vstack([np.zeros((n, n)), M_inv])

# Define matrix P
P = np.vstack((np.zeros((n, m)), np.dot(M_inv, np.dot(Lgl, Lll_inv))))