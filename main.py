import numpy as np
from numba import njit
import matplotlib.pyplot as plt

L = 100
N = L * L
J = 1.0
kB = 1.0
steps = 10000000 # MC steps

@njit
def calculate_energy(config, L, J):
    energy = 0
    for i in range(L):
        for j in range(L):
            S = config[i, j]
            neighbors = config[(i+1)%L, j] + config[i, (j+1)%L] + config[(i-1)%L, j] + config[i, (j-1)%L]
            energy += -J * S * neighbors
    return energy / 2  # Each pair counted twice

@njit
def monte_carlo_step(config, L, J, T):
    i = np.random.randint(0, L)
    j = np.random.randint(0, L)
    # trial flip
    S = config[i, j]
    neighbors = config[(i+1)%L, j] + config[i, (j+1)%L] + config[(i-1)%L, j] + config[i, (j-1)%L]
    dE = 2 * J * S * neighbors
    ksi = np.random.uniform(0, 1)
    A = min(1, np.exp(-dE / (kB * T)))
    if ksi < A:
        config[i, j] = -S
        return dE, -2 * S
    else:
        return 0, 0
    
@njit
def simulate_ising(config, L, T, steps): # generate <S>, <E>, C_v
    E = calculate_energy(config, L, J)
    S = np.sum(config)
    for _ in range(steps//2): # equilibration
        dE, dS = monte_carlo_step(config, L, J, T)
        E += dE
        S += dS
    E_list = np.zeros(steps)
    S_list = np.zeros(steps)
    for i in range(steps):
        dE, dS= monte_carlo_step(config, L, J, T)
        E += dE
        S += dS
        E_list[i] = E
        S_list[i] = S
    E_avg = np.mean(E_list) / N
    S_avg = np.mean(S_list) / N
    E_var = np.var(E_list)
    C_v = E_var / (kB * T * T * N)
    return S_avg, E_avg, C_v, config

def main():
    T_values = np.arange(0.2, 6.1, 0.2)
    S_values = []
    E_values = []
    C_v_values = []
    config = np.ones((L, L), dtype=np.int8)  # initial configuration (all spins up)
    for T in T_values:
        S_avg, E_avg, C_v, config = simulate_ising(config, L, T, steps)
        S_values.append(S_avg)
        E_values.append(E_avg)
        C_v_values.append(C_v)
        print(f"T={T:.2f}, <S>={S_avg:.4f}, <E>={E_avg:.4f}, C_v={C_v:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(T_values, S_values, 'o-', color='blue')
    plt.title('<S>-T Plot')
    plt.xlabel('Temperature (k_B*T/J)')
    plt.ylabel('<S>')
    plt.grid(True)
    plt.savefig('S_avg-T.png')

    plt.figure(figsize=(8, 6))
    plt.plot(T_values, E_values, 'o-', color='red')
    plt.title('<E>-T Plot')
    plt.xlabel('Temperature (k_B*T/J)')
    plt.ylabel('<E>')
    plt.grid(True)
    plt.savefig('E_avg-T.png')

    plt.figure(figsize=(8, 6))
    plt.plot(T_values, C_v_values, 'o-', color='green')
    plt.title('C_v-T Plot')
    plt.xlabel('Temperature (k_B*T/J)')
    plt.ylabel('C_v')
    plt.grid(True)
    plt.savefig('C_v-T.png')

if __name__ == "__main__":
    main()