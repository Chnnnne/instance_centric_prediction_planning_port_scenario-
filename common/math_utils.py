import bezier
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad



def get_bezier_curve_length(curve):
    def speed(t):
        dx, dy = curve.evaluate_hodograph(t)
        return np.sqrt(dx**2 + dy**2)
    
    curve_length,error = quad(speed,0,1)
    # print(res, error)
    return curve_length

def get_control_points(start, direction_start, end, direction_end, scale):
    direction_start_normalized = direction_start / np.linalg.norm(direction_start)
    direction_end_normalized = direction_end / np.linalg.norm(direction_end)
    
    # 控制点基于起始点和朝向
    control1 = start + direction_start_normalized * scale
    
    # 控制点基于终点和朝向
    control2 = end - direction_end_normalized * scale
    
    return control1, control2

def get_bezier_function(start_point, start_direction, end_point, end_direction, scale = 1.0):
    # rad-> unit vec
    start_point = np.asarray(start_point)
    end_point = np.asarray(end_point)
    start_direction = np.asarray([math.cos(start_direction), math.sin(start_direction)])
    end_direction = np.asarray([math.cos(end_direction), math.sin(end_direction)])
    scale = np.linalg.norm(end_point - start_point)*4/8
    control1, control2 = get_control_points(start_point, start_direction, end_point, end_direction, scale)
    nodes = np.asfortranarray([
        start_point,
        control1,
        control2,
        end_point
    ]).T
    curve = bezier.Curve(nodes, degree = 3)
    return curve, nodes
    # t_values = np.linspace(0, 1, 15)
    # sampled_points = curve.evaluate_multi(t_values)

if __name__ == "__main__":
    get_bezier_curve_length()
    # curve, nodes = get_bezier_function([0,0],0, [8,8],math.pi/2)
    # t_values = np.linspace(0, 1, 50)
    # sampled_points = curve.evaluate_multi(t_values)
    # # 绘制曲线和控制点
    # fig, ax = plt.subplots()
    # curve.plot(100, ax=ax)
    # ax.plot(nodes[0, :], nodes[1, :], "ro--", label="Control points")
    # ax.legend()

    # fig.savefig("tmp.jpg")

