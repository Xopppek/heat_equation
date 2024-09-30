import numpy as np
from matplotlib import pyplot as plt

def mu_l(t):
    return 10 * np.sqrt(t)

def mu_r(t):
    return 0


kappa_0 = 0.5
sigma = 2

def K(u):
    return kappa_0 * u ** sigma

N = 50
x = np.linspace(0, 1, N + 1) # h = 0.02
h = 1 / N
tau = 2e-4
t = np.linspace(0.1, 0.2, 500 + 1) # tau = 2e-4

def step(u_start, h, tau, K, epsilon = 1e-3):
    '''
    Функция, делающая один шаг по времени (одномерная задача)
    Внутри нет явной зависимости от t, потому что она должна быть включена в u, переданный в функцию 
    Нулевая итерация - это значения с предыдущего шага + граничные условия на новом шаге на слева и справа
    '''

    N = u.size - 1
    u_prev = u_start.copy()
    alpha = np.zeros(N)
    beta = np.zeros(N)
    converged = False

    while not converged:
        A = tau / h ** 2 * K((u_prev[:-1] + u_prev[1:]) / 2) 
        # тут пока сделан случай  phi(u) = u, поэтому B и F имеют очень простой вид
        B = np.ones(N - 1)
        F = u_start[1:-1] # - u_prev[1:-1] + u_prev[1:-1] * B 
        alpha[0] = 0
        beta[0] = u_prev[0]
        for i in range(0, N - 1): 
            alpha[i + 1] = A[i + 1] / (A[i + 1] + A[i] * (1 - alpha[i]) + B[i])
            beta[i + 1] = (A[i] * beta[i] + F[i]) / (A[i + 1] + A[i] * (1 - alpha[i]) + B[i])
        u_new = np.zeros_like(u_prev)
        u_new[-1] = u_prev[-1]
        u_new[0] = u_prev[0]
        for i in range(N - 1, 0, -1):
            u_new[i] = alpha[i] * u_new[i + 1] + beta[i]
        if (np.max(np.abs(u_new - u_prev))) < epsilon:
            converged = True
        u_prev = u_new

    return u_prev
    
c = 5
x_1 = 0
def initial_condition(x):
    return ((sigma * c / kappa_0 * (c * 0.1 + x_1 - x)) * np.heaviside(x_1 + c*0.1 - x, 0)) ** (1 / sigma)

def analit(x, t):
    return ((sigma * c / kappa_0 * (c * t + x_1 - x)) * np.heaviside(x_1 + c*t - x, 0)) ** (1 / sigma)

u = initial_condition(x)

#plt.plot(u, label='0.1', color='green')
u_new = u.copy(t.size)
for i in range():
    u_new[0] = mu_l(t[i])
    u_new[-1] = mu_r(t[i])
    u_new = step(u_new, h, tau, K)
plt.plot(x, u_new, color = 'red')
plt.plot(x, analit(x, t[-1]), color='green')
plt.show()
