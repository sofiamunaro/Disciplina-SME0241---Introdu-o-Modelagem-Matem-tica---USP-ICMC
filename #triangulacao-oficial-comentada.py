#triangulacao oficial

import numpy as np

numero_de_triangulos = int(input())

lista_de_vertices = []
for i in range(numero_de_triangulos):
    a,b,c = map(int,input().split())
    lista_de_vertices.append(a)
    lista_de_vertices.append(b)
    lista_de_vertices.append(c)

vertices = np.array(lista_de_vertices).reshape(numero_de_triangulos,3)

cores = np.zeros(numero_de_triangulos+2) # lista de zeros,cada um representando um vértice
cores[vertices[0,0]] = 1 #atribui ao primeiro vertice do triangulo 1 a cor1
cores[vertices[0,1]] = 2 #atribui ao segundo vertice do triangulo 1 a cor2
cores[vertices[0,2]] = 3 #atribui ao terceiro vertice do triangulo 1 a cor3


while (np.count_nonzero(cores == 0)) != 0: #enquanto ainda houver zeros na lista de cores (cor ainda nao definida)
    for i in range(numero_de_triangulos):
        linhaatual = vertices[i,:] #vai indo triangulo por triangulo

        #ve se o triangulo sendo analisado tem dois vertices com cor definida (diferente de zero) e um com cor indefinida (igual a zero)
        #se for o caso, o vertice com cor indefinida devera receber a cor que falta
        if cores[linhaatual[0]] != 0.0 and cores[linhaatual[1]] != 0.0 and cores[linhaatual[2]] == 0.0:
            coresdalinha = []
            coresdalinha.append(cores[linhaatual[0]])
            coresdalinha.append(cores[linhaatual[1]])
            if 1 not in coresdalinha:
                corquefalta = 1
            if 2 not in coresdalinha:
                corquefalta = 2
            if 3 not in coresdalinha:
                corquefalta = 3
            cores[linhaatual[2]] = corquefalta

        if cores[linhaatual[0]] != 0.0 and cores[linhaatual[2]] != 0.0 and cores[linhaatual[1]] == 0.0:
            coresdalinha = []
            coresdalinha.append(cores[linhaatual[0]])
            coresdalinha.append(cores[linhaatual[2]])
            if 1 not in coresdalinha:
                corquefalta = 1
            if 2 not in coresdalinha:
                corquefalta = 2
            if 3 not in coresdalinha:
                corquefalta = 3
            cores[linhaatual[1]] = corquefalta

        if cores[linhaatual[1]] != 0.0 and cores[linhaatual[2]] != 0.0 and cores[linhaatual[0]] == 0.0:
            coresdalinha = []
            coresdalinha.append(cores[linhaatual[1]])
            coresdalinha.append(cores[linhaatual[2]])
            if 1 not in coresdalinha:
                corquefalta = 1
            if 2 not in coresdalinha:
                corquefalta = 2
            if 3 not in coresdalinha:
                corquefalta = 3
            cores[linhaatual[0]] = corquefalta

#conta quantas vezes aparece cada cor
cor1 = np.count_nonzero(cores == 1)
cor2 = np.count_nonzero(cores == 2)
cor3 = np.count_nonzero(cores == 3)

#retorna a localizacao da cor que mais aparece
if cor1 < cor2 and cor1 < cor3:
    indices = np.where(cores == 1)
if cor2 < cor1 and cor2 < cor3:
    indices = np.where(cores == 2)

if cor3 < cor2 and cor3 < cor1:
    indices = np.where(cores == 3)

#formata a saida
for i,r in enumerate(indices):
    lista = list(r)
    resultado = " ".join(map(str, lista))
    print(resultado)
