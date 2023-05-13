import numpy as np 

# We make sure to take the opposite of an angle as the y coordinate is inverted in the game window.

def get_norm(v):
    return np.sqrt(v[0] ** 2 + v[1] ** 2)

def get_direction(v):
    v_norm = get_norm(v)
    return v / v_norm 

def scal(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def prod(v1, v2):
    return v1[0] * v2[1] - v2[0] * v1[1]

def get_angle(v):
    v_norm = get_norm(v)
    if v_norm == 0:
        return 0 
    else:
        cos_alpha = v[0] / v_norm
        if v[1] >= 0:
            return np.arccos(cos_alpha)
        else:
            return - np.arccos(cos_alpha)

def get_rotation_mat(alpha):
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    return np.array([[cos_alpha, - sin_alpha], [sin_alpha, cos_alpha]])

def angle_to_vector(alpha, norm = 1):
    return np.array([np.cos(alpha), np.sin(alpha)]) * norm

def apply_rotation(alpha, v):
    rot_mat = get_rotation_mat(alpha)
    return rot_mat @ v

def projection(alpha, v):
    alpha_v = get_angle(v)
    return angle_to_vector(alpha, get_norm(v) * np.cos(alpha - alpha_v))
