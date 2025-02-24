#interpolação de dados

import numpy as np
from scipy.spatial import distance_matrix
import math

dados = np.array([[-0.132435,  0.121322, -0.396841], 
          [ 0.022206, -0.253028,  0.064272],
          [ 0.189277,  0.246878,  0.518604],
          [-0.353720, -0.062568, -0.891925],
          [ 0.338860, -0.215555,  0.824920],
          [-0.114211,  0.424858, -0.275816],
          [-0.219019, -0.421707, -0.500739],
          [ 0.477176,  0.174264,  0.960294],
          [-0.498052,  0.205589, -0.948289],
          [ 0.240728, -0.514422,  0.474110],
          [ 0.178278,  0.568380,  0.333216],
          [-0.538309, -0.311961, -0.875938],
          [ 0.632472, -0.139047,  0.892913],
          [-0.386499,  0.549756, -0.608870],
          [-0.089393, -0.689840, -0.129753],
          [ 0.549340,  0.462984,  0.738047],
          [-0.739899,  0.030597, -0.728343],
          [ 0.540129, -0.537500,  0.658981],
          [-0.036162,  0.782046, -0.038058],
          [-0.514630, -0.616699, -0.565767],
          [ 0.815705,  0.109752,  0.539060],
          [-0.691510,  0.481135, -0.599958],
          [ 0.189052, -0.840354,  0.138870],
          [ 0.437461,  0.763428,  0.356128],
          [-0.855543, -0.272942, -0.398726],
          [ 0.831365, -0.384112,  0.416118],
          [-0.360294,  0.860904, -0.196213],
          [-0.321659, -0.894294, -0.140011],
          [ 0.856159,  0.449973,  0.332056],
          [-0.951246,  0.250745, -0.140885],
          [ 0.540838, -0.841127,  0.244945]])

pontos = np.delete(dados,2,1)
f = dados[:,2]
M = distance_matrix(pontos,pontos)
A = (math.e)**(-(M**2))
alfa = np.linalg.solve(A,f)

px,py = map(float,input().split())
ponto = np.array([px,py])

matrizponto = np.tile(ponto,(31,1))
M2 = distance_matrix(matrizponto,pontos)
M2 = M2[1,:]
phi = (math.e)**(-(M2**2))

soma = phi@alfa

print(str(soma)[:4])