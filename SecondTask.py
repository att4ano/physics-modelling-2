import matplotlib.pyplot as plt
import numpy as np

a, b = 3, 5
eps0 = 1
eps = 2
d = 0.2

def capacitance(eps, d, S):
    return eps0 * eps * S / d

C0 = capacitance(eps, d, a * b)
C1 = capacitance(eps, d, min(a, b) ** 2) + capacitance(1, d, a * b - min(a, b) ** 2)

x = np.linspace(0, 10, num=100)
plt.plot(x, C0 - np.abs(np.sin(x)) * (C0 - C1))
plt.xlabel("Время, с")
plt.ylabel("Ёмкость, Кл/В")
plt.show()
