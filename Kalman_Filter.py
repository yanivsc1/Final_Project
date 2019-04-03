from filterpy.kalman import KalmanFilter
from numpy import *
from numpy.linalg import inv
import numpy as np
from numpy import dot, sum, tile, linalg
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise

def kf_predict(X, P, A, Q, B, U):
 X = dot(A, X) + dot(B, U)
 P = dot(A, dot(P, A.T)) + Q
 return(X,P)
def kf_update(X, P, Y, H, R):
 IM = dot(H, X)
 IS = R + dot(H, dot(P, H.T))
 K = dot(P, dot(H.T, inv(IS)))
 X = X + dot(K, (Y-IM))
 P = P - dot(K, dot(IS, K.T))
 LH = gauss_pdf(Y, IM, IS)
 return (X,P,K,IM,IS,LH)
def gauss_pdf(X, M, S):
 if M.shape[1] == 1:
    DX = X - tile(M, X.shape[1])
    E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
    E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(np.linalg.det(S))
    P = exp(-E)
 elif X.shape[1] == 1:
    DX = tile(X, M.shape[1])- M
    E = 0.5 * sum(DX * (dot(inv(S), DX)), axis=0)
    E = E + 0.5 * M.shape[0] * log(2 * pi) + 0.5 * log(np.linalg.det(S))
    P = exp(-E)
 else:
    DX = X-M
    E = 0.5 * dot(DX.T, dot(inv(S), DX))
    E = E + 0.5 * M.shape()[0] * log(2 * pi) + 0.5 * log(np.linalg.det(S))
    P = exp(-E)
 return (P[0],E[0])


#time step of mobile movement
dt = 0.1
# Initialization of state matrices
X = array([[0.0], [0.0], [0.1], [0.1]])
P = diag((0.01, 0.01, 0.01, 0.01))
A = array([[1, 0, dt , 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0,1]])
Q = eye(X.shape[0])
B = eye(X.shape[0])
U = zeros((X.shape[0],1))
# Measurement matrices
Y = array([[X[0,0] + abs(np.random.randn(1)[0])], [X[1,0] +abs(np.random.randn(1)[0])]])
H = array([[1, 0, 0, 0], [0, 1, 0, 0]])
R = eye(Y.shape[0])
# Number of iterations in Kalman Filter
N_iter = 50
# Applying the Kalman Filter
for i in arange(0, N_iter):
 (X, P) = kf_predict(X, P, A, Q, B, U)
 (X, P, K, IM, IS, LH) = kf_update(X, P, Y, H, R)
 Y = array([[X[0,0] + abs(0.1 * np.random.randn(1)[0])],[X[1, 0] +abs(0.1 * np.random.randn(1)[0])]])



#f=KalmanFilter(dim_x=2,dim_z=1)
#f.x=np.array([[2.0],[0.0]])
#f.F=np.array([[1.,1.],[0.,1.]])
#f.H = np.array([[1.,0.]])
#f.P = np.array([[1000.,    0.],[   0., 1000.] ])
#f.R = 5
#f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
#z = 1
#f.predict()
#f.update(z)
#print(f.x)
#z=5
#f.predict()
#f.update(z)
#print(f.x)

f=KalmanFilter(dim_x=2,dim_z=2)
f.x=np.array([[2.0],[0.0]])
f.F=np.array([[1.,1.],[0.,1.]])
f.H = np.array([[2.,3.],[1.,0.]])
f.P = np.array([[1000.,    0.],[   0., 1000.] ])
f.R = np.array([[0.5,0.3],[1.0,1.5]])
f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)
z = np.array([[2.0],[0.0]])
f.predict()
f.update(z)
print(f.x)
z = np.array([[1.0],[3.0]])
f.predict()
f.update(z)
print(f.x)