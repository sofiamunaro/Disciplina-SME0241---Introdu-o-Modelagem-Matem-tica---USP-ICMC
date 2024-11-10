#Pêndulo simples
import numpy as np

# Euler-Richardson
def euler_richardson(theta, omega, dt):
    for i in range(1, len(theta)):
        theta_mid = theta[i-1] + omega[i-1] * 0.5 * dt
        omega_mid = omega[i-1] - np.sin(theta[i-1]) * 0.5 * dt
        theta[i] = theta[i-1] + omega_mid * dt
        if theta[i]*theta[i-1]<0:
            return theta[i-1],theta[i],(i*0.01-0.01),(i*0.01)
        omega[i] = omega[i-1] - np.sin(theta_mid) * dt


t_stop = 10.0
dt = 0.01
t = np.arange(0, t_stop, dt)

theta = np.zeros(len(t))
theta[0] = float(input())  
omega = np.zeros(len(t))

theta1,theta2,t1,t2 = euler_richardson(theta, omega, dt)
tempo = (((-theta1)*(t2-t1))/(theta2-theta1))+t1
print(f'{tempo*4:.4f}')