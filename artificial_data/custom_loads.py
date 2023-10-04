import dolfin as df
import numpy as np

def load_1(theta: float):
    six_load = df.Expression(('0', 'abs(x[0] - 0.4) < 0.02? C: 0'), degree=1, C=-1.7E3*np.cos(theta))
    nine_load = df.Constant((0, 1.925E3*np.cos(theta)))
    surface_load = {9: nine_load, 6: six_load}
    return surface_load

def load_2(theta: float):
    six_load = df.Expression(('0', 'abs(x[0] - 0.4) < 0.02? C: 0'), degree=1, C=-0.6E3*np.cos(theta+0.25*np.pi))
    nine_load = df.Constant((0, 0.6E3*np.cos(theta)))
    surface_load = {9: nine_load, 6: six_load}
    return surface_load

def load_3(theta: float):
    six_load = df.Expression(('0', 'abs(x[0] - 0.5) < 0.04? C: 0'), degree=1, C=-0.2E3*np.cos(theta))
    nine_load = df.Constant((0, 1.4E3*np.cos(theta)))
    surface_load = {9: nine_load, 6: six_load}
    return surface_load

def load_4(theta: float):
    six_load = df.Expression(('0', 'abs(x[0] - 0.45) < 0.04? C: 0'), degree=1, C=-0.66E3*np.cos(theta))
    nine_load = df.Constant((0, 1.76E3*np.cos(theta)))
    surface_load = {9: nine_load, 6: six_load}
    return surface_load

def load_5(theta: float):
    six_load = df.Constant((0.0, 0.0))
    nine_load = df.Constant((0, 0.53E3*np.cos(theta)))
    surface_load = {9: nine_load, 6: six_load}
    return surface_load

def load_6(theta: float):
    six_load = df.Expression(('0', 'abs(x[0] - 0.4) < 0.02? C: 0'), degree=1, C=-1.94E3*np.cos(theta))
    nine_load = df.Constant((0, 1.99E3*np.cos(theta)))
    surface_load = {9: nine_load, 6: six_load}
    return surface_load

loads = [load_1, load_2, load_3, load_4, load_5, load_6]
