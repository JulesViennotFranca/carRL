import numpy as np 
import random
import pygame

import config
import geometry

def random_points(n, scal):
    xscal = scal[0]
    yscal = scal[1]
    points = [np.array([random.random() * xscal - xscal / 2, random.random() * yscal - yscal / 2]) for _ in range(n)]
    return points 

def random_gauss_points(n, scal):
    xscal = scal[0]
    yscal = scal[1]
    points = []
    for _ in range(n):
        x = 2
        while abs(x) > 1:
            x = random.gauss(0, 1 / 4)
        y = 2
        while abs(y) > 1:
            y = random.gauss(0, 1 / 4)
        points.append(np.array([x * xscal, y * yscal]))
    return np.array(points)

def convex_hull(points):
    n = len(points)
    g = sum(points) / n 
    points = sorted(points, key=lambda p: geometry.get_angle(p - g))

    miny = 0, points[0][1]
    for i, point in enumerate(points):
        if point[1] < miny[1]:
            miny = i, point[1]
    points = points[miny[0]:] + points[:miny[0]]

    hull = []
    pred_pred_point = points[-2]
    pred_point = points[-1]
    for point in points:
        vp = pred_point - pred_pred_point
        v = point - pred_point
        a1 = geometry.get_angle(vp)
        v = geometry.apply_rotation(- a1, v)
        a2 = geometry.get_angle(v)
        if a2 >= 0:
            hull.append(pred_point)
            pred_pred_point = pred_point 
            pred_point = point 
        else:
            pred_point = point

    return hull

def smooth(points, rmin, amin):
    n = len(points)
    rpoints = []
    for i in range(n):
        if geometry.get_norm(points[(i+1)%n] - points[i]) > rmin:
            rpoints.append(points[i])

    n = len(rpoints)
    apoints = []
    for i in range(n):
        vp = rpoints[(i-1)%n] - rpoints[i]
        ap = geometry.get_angle(vp)
        vn = rpoints[(i+1)%n] - rpoints[i]
        vn = geometry.apply_rotation(- ap, vn)
        an = geometry.get_angle(vn)
        if abs(an) > amin:
            apoints.append(rpoints[i])

    n = len(apoints)
    flatter = 0 
    flatter_ind = 0
    for i in range(n):
        vp = rpoints[(i-1)%n] - rpoints[i]
        ap = geometry.get_angle(vp)
        vn = rpoints[(i+1)%n] - rpoints[i]
        vn = geometry.apply_rotation(- ap, vn)
        an = geometry.get_angle(vn)
        if abs(an) > flatter:
            flatter = abs(an)
            flatter_ind = i

    points = apoints[flatter_ind:] + apoints[:flatter_ind]
    return points

def bezier_curve_3(start, inter, end):
    T = np.linspace(0, 1, 50)
    P = []
    for t in T:
        p1 = start + t * (inter - start) 
        p2 = inter + t * (end - inter)
        P.append(p1 + t * (p2 - p1))
    return P

def bezier_curve_4(start, inter1, inter2, end, spand):
    nbr_sample = max(1, (geometry.get_norm(inter1 - start) + geometry.get_norm(inter2 - inter1) + geometry.get_norm(end - inter2)) / spand)
    T = np.linspace(0, 1, int(nbr_sample))
    P = []
    for t in T:
        p1 = start + t * (inter1 - start) 
        p2 = inter1 + t * (inter2 - inter1)
        p3 = inter2 + t * (end - inter2)
        p4 = p1 + t * (p2 - p1)
        p5 = p2 + t * (p3 - p2)
        P.append(p4 + t * (p5 - p4))
    return P

def bezier_next_middle(middle, end):
    return 2 * end - middle 

def bezier_compute_inter(p1, p2, p3):
    v1 = (p1 - p2)
    v3 = (p3 - p2) 
    v = v1 / geometry.get_norm(v1) + v3 / geometry.get_norm(v3)
    alpha = geometry.get_angle(v) + np.pi / 2
    inter1 = p2 + geometry.projection(alpha, v1) / 2
    inter2 = p2 + geometry.projection(alpha, v3) / 2
    return inter1, inter2

def bezier_path(points, spand):
    n = len(points)
    P = []
    for i in range(n):
        _, inter1 = bezier_compute_inter(points[i-1], points[i], points[(i+1)%n])
        inter2, _ = bezier_compute_inter(points[i], points[(i+1)%n], points[(i+2)%n])
        P = P + bezier_curve_4(points[i], inter1, inter2, points[(i+1)%n], spand)
    return P

def adapt(points, spand, width):
    n = len(points)
    P = []
    for i, p in enumerate(points):
        if geometry.get_norm(points[(i-1)%n] - p) > 2 * spand:
            P.append((points[(i-1)%n] + p) / 2)
        if len(P) == 0 or geometry.get_norm(P[-1] - p) > spand:
            P.append(p)
    n = len(P)
    C = []
    for i, p in enumerate(P):
        if len(C) == 0 or geometry.get_norm(C[-1][0] - p) > width:
            C.append([p, geometry.get_angle(p - P[(i-1)%n])])
    return C, np.array(P)

def track_skeleton(n, width, lim):
    points = random_gauss_points(n, lim)
    hull = convex_hull(points)
    rmin = 3 * width 
    amin = max(2, 2 * np.arcsin(0.5 * width / rmin))
    keypoints = smooth(hull, rmin, amin)
    spand = width / 5
    path = bezier_path(keypoints, spand)
    return adapt(path, spand, width)

class TrackBase():
    def __init__(self, width, size):
        self.width = width
        self.size = size
        self.nbr_point = sum(self.size) // 20

        self.reset()

    def reset(self):
        self.checkpoints, self.track = track_skeleton(self.nbr_point, self.width, self.size)
        self.start_pos = self.track[0]
        self.start_dir = geometry.get_angle(self.track[1] - self.track[0])

        self.opt_track = [self.track[0]]
        self.opt_track_min = 0
        self.opt_track_max = 0

        self.limit_track = [self.track[0]]
        self.limit_track_min = 0
        self.limit_track_max = 0

        self.update(self.start_pos)

    def fit_to_window(self, car_pos, window, min, max, lim):
        n = len(self.track)
        v = - car_pos

        p = self.track[min]
        while len(window) < len(self.track) and (- lim <= p + v).all() and (p + v <= lim).all():
            window = [p] + window
            min = (min - 1) % n
            p = self.track[min]
        
        p = self.track[max]
        while len(window) < len(self.track) and (- lim <= p + v).all() and (p + v <= lim).all():
            window.append(p)
            max = (max + 1) % n
            p = self.track[max]

        p = window[0]
        while len(window) > 1 and not ((- lim <= p + v).all() and (p + v <= lim).all()):
            _ = window.pop(0)
            min = (min + 1) % n
            p = window[0]

        p = window[-1]
        while len(window) > 1 and not ((- lim <= p + v).all() and (p + v <= lim).all()):
            _ = window.pop()
            max = (max - 1) % n
            p = window[-1]
        return window, min, max

    def update(self, car_pos):
        self.limit_track, self.limit_track_min, self.limit_track_max = self.fit_to_window(car_pos, self.limit_track, self.limit_track_min, self.limit_track_max, np.ones(2) * self.width)

        lim_vec = np.array(config.resolution) / 2 + np.ones(2) * self.width
        self.opt_track, self.opt_track_min, self.opt_track_max = self.fit_to_window(car_pos, self.opt_track, self.opt_track_min, self.opt_track_max, lim_vec)

class TrackSprite(pygame.sprite.Sprite, TrackBase):
    def __init__(self, width, size):
        pygame.sprite.Sprite.__init__(self)

        self.color = config.track_color

        TrackBase.__init__(self, width, size)

    def reset(self):
        TrackBase.reset(self)

        self.update_sprite(self.start_pos)

    def update_sprite(self, car_pos):
        self.image = pygame.Surface(config.resolution).convert_alpha()
        self.image.fill((255, 255, 255, 0))

        for p in self.opt_track:
            pygame.draw.circle(self.image, self.color, p + config.screen_mid - car_pos, self.width)

        nbr = 10
        for i in range(nbr):
            size = 2 * self.width / nbr

            start_rect_white = np.array([[np.array([j, i * j]) * size / 2 for i in range(-1, 2, 2)] for j in range(-1, 2, 2)])
            offset_white = np.array([(2 * self.width - size) / 2 - i * size, size / 2 - (i%2) * size])
            start_rect_white = np.array([[geometry.apply_rotation(self.start_dir + np.pi / 2, offset_white + start_rect_white[i][j]) for j in range(2)] for i in range(2)])
            start_rect_white = np.reshape(start_rect_white, (4, 2))
            start_rect_white += self.start_pos + config.screen_mid - car_pos 
            start_rect_white = list(start_rect_white)
            pygame.draw.polygon(self.image, (255, 255, 255), start_rect_white)

            start_rect_black = np.array([[np.array([j, i * j]) * size / 2 for i in range(-1, 2, 2)] for j in range(-1, 2, 2)])
            offset_black = np.array([(2 * self.width - size) / 2 - i * size, size / 2 - ((i+1)%2) * size])
            start_rect_black = np.array([[geometry.apply_rotation(self.start_dir + np.pi / 2, offset_black + start_rect_black[i][j]) for j in range(2)] for i in range(2)])
            start_rect_black = np.reshape(start_rect_black, (4, 2))
            start_rect_black += self.start_pos + config.screen_mid - car_pos 
            start_rect_black = list(start_rect_black)
            pygame.draw.polygon(self.image, (0, 0, 0), start_rect_black)

        self.rect = self.image.get_rect()
        self.top_left = np.zeros(2)
        self.rect.center = [config.resolution[0] / 2, config.resolution[1] / 2]
    
    def update(self, car_pos):
        TrackBase.update(self, car_pos)
        self.update_sprite(car_pos)

        