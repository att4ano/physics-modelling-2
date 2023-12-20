import builtins
from numpy import *
from numpy import sqrt, array, cross, dot, fabs, arccos, radians, where, inf, pi, zeros_like, log10, linspace, meshgrid
from numpy.linalg import det
from scipy.integrate import ode
from scipy.interpolate import splrep, splev
from matplotlib import pyplot
from collections import deque

# Задание границы области
xmin, xmax = -200, 200
ymin, ymax = -200, 200
scale = 35  # Масштаб для построения графика

# Функция для вычисления нормы вектора
def norm(vec):
    return sqrt(sum(array(vec) ** 2, axis=-1))

# Функция для вычисления расстояния от точки до прямой заданной двумя точками
def point_line_distance(x0, x1, x2):
    return fabs(cross(x0 - x1, x0 - x2)) / norm(x2 - x1)

# Функция для вычисления угла между векторами
def angle(x0, x1, x2):
    a, b = x1 - x0, x1 - x2
    return arccos(dot(a, b) / (norm(a) * norm(b)))

# Функция для определения положения точки относительно прямой
def is_left(x0, x1, x2):
    matrix = array([x1 - x0, x2 - x0])
    if len(x0.shape) == 2:
        matrix = matrix.transpose((1, 2, 0))
    return det(matrix) > 0

# Класс, представляющий линию с зарядом
class Line:
    RADIUS = 1e-2

    def __init__(self, charge, point1, point2):
        self.charge, self.point1, self.point2 = charge, array(point1), array(point2)

    # Метод для вычисления силы электрического поля
    def electric_field_strength(self):
        return self.charge / norm(self.point2 - self.point1)

    # Метод для вычисления вектора электрического поля в заданной точке
    def electric_field(self, position):
        position = array(position)
        p1, p2, lam = self.point1, self.point2, self.electric_field_strength()
        theta1, theta2 = angle(position, p1, p2), pi - angle(position, p2, p1)
        a = point_line_distance(position, p1, p2)
        r1, r2 = norm(position - p1), norm(position - p2)
        sign = where(is_left(position, p1, p2), 1, -1)
        E_parallel = lam * (1 / r2 - 1 / r1)
        E_perpendicular = -sign * lam * (cos(theta2) - cos(theta1)) / where(a == 0, inf, a)
        dx = p2 - p1
        return E_perpendicular * (array([-dx[1], dx[0]]) / norm(dx)) + E_parallel * (dx / norm(dx))

    # Метод для определения, находится ли точка близко к линии
    def is_close(self, position):
        theta1 = angle(position, self.point1, self.point2)
        theta2 = angle(position, self.point2, self.point1)

        if theta1 < radians(90) and theta2 < radians(90):
            return point_line_distance(position, self.point1, self.point2) < self.RADIUS
        return min([norm(self.point1 - position), norm(self.point2 - position)]) < self.RADIUS

    # Метод для построения линии на графике
    def plot(self):
        if self.charge < 0:
            color = 'b'
        elif self.charge > 0:
            color = 'r'
        else:
            color = 'k'
        x, y = zip(self.point1, self.point2)
        width = 5 * (fabs(self.electric_field_strength()) + 1)
        pyplot.plot(x, y, color, linewidth=width)

# Класс, представляющий линию поля
class FieldLine:
    def __init__(self, positions):
        self.positions = positions
    # Метод для построения линии поля на графике
    def plot(self):
        linewidth = 0.5
        x, y = zip(*self.positions)
        pyplot.plot(x, y, 'red', linewidth=linewidth)

        if len(x) < 225:
            n = int(len(x) / 2)
        else:
            n = 75

        pyplot.arrow(x[n], y[n], (x[n + 1] - x[n]) / 100., (y[n + 1] - y[n]) / 100.,
                     fc="red", ec="red",
                     head_width=0.1 * linewidth, head_length=0.1 * linewidth)

        pyplot.arrow(x[-n], y[-n],
                     (x[-n + 1] - x[-n]) / 100., (y[-n + 1] - y[-n]) / 100.,
                     fc="red", ec="red",
                     head_width=0.1 * linewidth, head_length=0.1 * linewidth)

# Класс, представляющий электрическое поле
class ElectricField:
    TIME_STEP = 0.01

    def __init__(self, charges):
        self.charges = charges

    # Метод для вычисления вектора электрического поля в заданной точке
    def electric_field_vector(self, position):
        return sum([charge.electric_field(position) for charge in self.charges], axis=0)

    # Метод для вычисления магнитуды вектора электрического поля в заданной точке
    def electric_field_magnitude(self, position):
        return norm(self.electric_field_vector(position))

    # Метод для вычисления угла вектора электрического поля в заданной точке
    def electric_field_angle(self, position):
        return arctan2(*(self.electric_field_vector(position).T[::-1]))

    # Метод для вычисления направления вектора электрического поля в заданной точке
    def electric_field_direction(self, position):
        v = self.electric_field_vector(position)
        return (v.T / norm(v)).T

    # Метод для построения линии поля, начинающейся в заданной точке
    def field_line(self, initial_position):
        streamline = lambda t, y: list(self.electric_field_direction(y))
        solver = ode(streamline).set_integrator('vode')

        positions = deque()
        positions.append(initial_position)

        for sign in [1, -1]:
            solver.set_initial_value(initial_position, 0)
            while solver.successful():
                solver.integrate(solver.t + sign * 8e-3)
                if sign > 0:
                    positions.append(solver.y)
                else:
                    positions.appendleft(solver.y)
                flag = builtins.any(charge.is_close(solver.y) for charge in self.charges)
                if flag or not (xmin < solver.y[0] < xmax) or \
                        not ymin < solver.y[1] < ymax:
                    break
        return FieldLine([i for i in positions])

    # Метод для построения графика магнитуды вектора электрического поля
    def plot(self, min_level=-3.5, max_level=1.5):
        x, y = meshgrid(
            linspace(xmin / scale, xmax / scale, 200),
            linspace(ymin / scale, ymax / scale, 200))
        z = zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                z[i, j] = log10(self.electric_field_magnitude([x[i, j], y[i, j]]))
        levels = arange(min_level, max_level + 0.2, 0.2)
        cmap = pyplot.cm.get_cmap('Purples')
        pyplot.contourf(x, y, clip(z, min_level, max_level),
                        10, cmap=cmap, levels=levels, extend='both')


# Задание зарядов
electric_charges = [Line(1, [-1.5, -3], [-1.5, 3]),
                    Line(-1, [1.5, -3], [1.5, 3])]
electric_field = ElectricField(electric_charges)

# Задание начальных точек для линий поля
initial_points = []
for y in linspace(-3, 3, 10):
    initial_points.append([-1.51, y])

for y in linspace(-3, 3, 10):
    initial_points.append([-1.49, y])

# Построение графика
electric_field.plot()
for point in initial_points:
    electric_field.field_line(point).plot()
for charge in electric_charges:
    charge.plot()
axis = pyplot.gca()
axis.set_xticks([])
axis.set_yticks([])
pyplot.xlim(xmin / scale, xmax / scale)
pyplot.ylim(ymin / scale, ymax / scale)
pyplot.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
pyplot.show()

