#Algoritmo de Graham

import numpy as np

def pseudo_angle(x,y,a,b):
    if np.array_equal(a,b):
        return 0
    else:
        return 1 - np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def convexo(A,B,C):
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    u = B-A
    v = C-B
    if (u[0]*v[1] - u[1]*v[0]) > 0:
        return True
    else:
       return False

#recebendo o input
coordenadas = []
for i in range(10):
    x,y = map(float,input().split())
    coordenadas.append(x)
    coordenadas.append(y)
pontos = np.array(coordenadas).reshape(10,2)

#obtendo o ponto com o menor y
menor = 2
for i in range(10):
   if pontos[i,1]<menor:
      menor = pontos[i,1]
      menorponto = pontos[i,:]


#ordenando os pontos
A = np.zeros((10,1))
for i in range(10):
   v = pontos[i,:] - menorponto
   A[i] = pseudo_angle([1,0],v,pontos[i,:],menorponto)

idx = np.argsort(A.T).T
coordenadas_ordenadas = []
for i in range(10):
    x = pontos[int(idx[i,0]),0]
    y = pontos[int(idx[i,0]),1]
    coordenadas_ordenadas.append(x)
    coordenadas_ordenadas.append(y)

pontos_ordenados = np.array(coordenadas_ordenadas).reshape(10,2)

#fecho convexo

pontos_no_fecho = [0,1]
k = 2 #ponto que esta sendo analisado
print(" ".join(map(str, pontos_no_fecho)))

while k < 10:

    A = pontos_ordenados[pontos_no_fecho[-2],:]
    B = pontos_ordenados[pontos_no_fecho[-1],:]
    C = pontos_ordenados[k,:]
    #print('k=',k,'A=',pontos_no_fecho[-2],'B=',pontos_no_fecho[-1],'C=',k)
    if convexo(A,B,C) == True:
        pontos_no_fecho.append(k)
        print(" ".join(map(str, pontos_no_fecho)))
        k += 1
    else:
        pontos_no_fecho.pop(-1)
    
    




