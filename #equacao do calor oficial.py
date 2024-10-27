#equacao do calor oficial

import numpy as np

# Paramêtros
L = 1         # comprimento da barra
tmax = 0.1    # tempo final
N = 64        # discr. espacial
M = 100       # discr. temporal
dx = L/N      # delta x
dt = tmax/M   # passo de tempo
alpha = float(input()) 

# Construção da Matriz Tridiagonal
a = (-alpha)/dx**2
d = ((1/dt)+(2*alpha)/dx**2)
R1 = a*np.ones(N)
R2 = (d)*np.ones(N+1)
A = np.diag(R1,-1) + np.diag(R2,0) + np.diag(R1,1)

# Cond. de Contorno
A[0,:] = np.zeros(N+1)
A[0,0] = 1
A[-1,:] = np.zeros(N+1)
A[-1,-1] = 1

# Cond. Inicial
x = np.linspace(0,L,N+1)
T = np.sin(np.pi*x/L)

#Solução aproximada

for m in range(M):
    b = (1/dt)*T
    b[0] = 0
    b[-1] = 0
    T = np.linalg.solve(A,b)
  
# Solução Exata
t = 0.1
Texact = np.sin((np.pi*x)/L)*np.exp(-(alpha*t*np.pi**2)/(L**2))
Texact[0] = 0
Texact[-1] = 0

#printar
print('%.6f' % np.linalg.norm(T - Texact))