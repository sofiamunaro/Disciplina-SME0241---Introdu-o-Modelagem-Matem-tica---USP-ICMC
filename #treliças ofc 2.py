#treliças ofc

import numpy as np

# Construção da matriz de rigidez global
def rigidez(coord, conec, modu):
    nv, nb = coord.shape[0], conec.shape[0]
    Kglo = np.zeros((2*nv, 2*nv))
    for ib in range(nb):
        Kloc = np.zeros((4,4))
        na, nb = int(conec[ib,0]), int(conec[ib,1])
        Xa, Xb = coord[na,:].reshape((2,1)), coord[nb,:].reshape((2,1))
        d = Xb - Xa
        l0 = np.linalg.norm(d, 2)
        kk = modu[ib] / (l0**3)
        aux = d@d.T
        Kloc[0:2,0:2] =  kk * aux
        Kloc[0:2,2:4] = -kk * aux
        Kloc[2:4,0:2] = -kk * aux
        Kloc[2:4,2:4] =  kk * aux
        loc2glo = [2*na, 2*na + 1, 2*nb, 2*nb + 1]
        for j in range(4):
            for k in range(4):
                jglo = loc2glo[j]
                kglo = loc2glo[k]
                Kglo[jglo,kglo] = Kglo[jglo,kglo] + Kloc[j,k]
    return Kglo

# A função abaixo calcula as trações em todas as barras da estrutura
def tracoes(coord, conec, modu, uu):
  nv = coord.shape[0]
  nb = conec.shape[0]
  tracao = np.zeros(nb)
  ud = np.zeros(2)
  for ib in range(nb):
    ka = int(conec[ib, 0])
    kb = int(conec[ib, 1])
    Xa = coord[ka, :]
    Xb = coord[kb, :]
    ud[0] = uu[2*kb]   - uu[2*ka]
    ud[1] = uu[2*kb+1] - uu[2*ka+1]
    d = Xb - Xa
    l0 = np.linalg.norm(d)
    kkt = modu[ib] / (l0**2)
    tracao[ib] = kkt * np.dot(d, ud)
  return tracao

#recebendo inputs

nv, nb = map(int,input().split()) #num de vertices e de barras

aux = []
for i in range(nv):
    x,y = map(float,input().split())
    aux.append(x)
    aux.append(y)
vertices = np.array(aux).reshape(nv,2) #matriz com as coord dos vertices

aux = []
for i in range(nb):
    x,y = map(float,input().split())
    aux.append(x)
    aux.append(y)
conectividades = np.array(aux).reshape(nb,2) #matriz com as conectividades

vfix = list(map(int,input().split())) # vertices a serem fixados

vext, Fx, Fy = map(float,input().split()) #nó a ser aplicada a força externa e os componentes da força
vext = int(vext)

#estabalecendo parâmetros

E = 10**11 #módulo de Young
A = 10**(-5) #área da seção transversal
modulos = (E*A) * np.ones(nb)

#cálculo da matriz de rigidez
kglo = rigidez(vertices,conectividades,modulos)


#vetor das forças externas
ld = np.zeros(2*nv)
ld[(2*int(vext))] = Fx # Força em x
ld[(2*int(vext)+1)] = Fy # Força em y


# Adicionando as restrições na matriz global
kk = kglo
iden = np.eye(2*nv)
for n in vfix:
    kk[2*n:2*n+2,:] = iden[2*n:2*n+2,:]   # Restrição do nó


# Cálculo dos deslocamentos
uu = np.linalg.solve(kk, ld)

# Cálculo das tensões
tracao = tracoes(vertices, conectividades, modulos, uu)

indice_maximo = np.argmax(tracao)
indice_minimo = np.argmin(tracao)

print(indice_minimo,indice_maximo)