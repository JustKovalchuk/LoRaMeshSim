from dataclasses import dataclass
import random
from typing import List, Tuple

import numpy as np


Obstacle = Tuple[float, float, float, float]


class Geometry:
    @staticmethod
    def segments_intersect(a, b, c, d):
        def ccw(p_a, p_b, p_c):
            return (p_c[1] - p_a[1]) * (p_b[0] - p_a[0]) > (p_b[1] - p_a[1]) * (p_c[0] - p_a[0])

        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)

    @staticmethod
    def line_intersects_rect(p1, p2, rect: Obstacle):
        x, y, w, h = rect
        rect_lines = [
            ((x, y), (x + w, y)),
            ((x + w, y), (x + w, y + h)),
            ((x + w, y + h), (x, y + h)),
            ((x, y + h), (x, y)),
        ]
        return any(Geometry.segments_intersect(p1, p2, r_p1, r_p2) for r_p1, r_p2 in rect_lines)

    @staticmethod
    def rects_overlap(r1: Obstacle, r2: Obstacle):
        return not (
            r1[0] + r1[2] < r2[0]
            or r1[0] > r2[0] + r2[2]
            or r1[1] + r1[3] < r2[1]
            or r1[1] > r2[1] + r2[3]
        )

@dataclass
class SceneConfig:
    field_size: float
    obstacles_count: int
    nodes_count: int
    obstacle_profile: str
    node_profile: str


class SceneBuilder:
    def __init__(self, gateway_pos: np.ndarray):
        self.gateway_pos = np.array(gateway_pos, dtype=float)
        self.obstacles: List[Obstacle] = []
        self.nodes = np.array([])

    def update_gateway(self, gateway_pos):
        self.gateway_pos = np.array(gateway_pos, dtype=float)

    def is_point_in_obstacle(self, x, y):
        return any(ox <= x <= ox + ow and oy <= y <= oy + oh for ox, oy, ow, oh in self.obstacles)

    def is_blocked(self, p1, p2):
        return any(Geometry.line_intersects_rect(p1, p2, obs) for obs in self.obstacles)

    def _sample_obstacle(self, field_size, obstacle_profile):
        if obstacle_profile == "buildings":
            w = random.uniform(field_size * 0.06, field_size * 0.12)
            h = random.uniform(field_size * 0.06, field_size * 0.12)
        elif obstacle_profile == "walls":
            if random.random() > 0.5:
                w = random.uniform(field_size * 0.15, field_size * 0.45)
                h = 0.05
            else:
                w = 0.05
                h = random.uniform(field_size * 0.15, field_size * 0.45)
        else:
            obs_type = random.randint(0, 1)
            if obs_type == 0:
                w = random.uniform(field_size * 0.05, field_size * 0.1)
                h = random.uniform(field_size * 0.05, field_size * 0.1)
            elif random.random() > 0.5:
                w = random.uniform(field_size * 0.1, field_size * 0.4)
                h = 0.05
            else:
                w = 0.05
                h = random.uniform(field_size * 0.1, field_size * 0.4)

        x = random.uniform(-field_size / 2, field_size / 2 - w)
        y = random.uniform(-field_size / 2, field_size / 2 - h)
        return (x, y, w, h)

    def generate_obstacles(self, cfg: SceneConfig):
        self.obstacles = []
        attempts = 0
        gw_rect = (self.gateway_pos[0] - 0.3, self.gateway_pos[1] - 0.3, 0.6, 0.6)

        while len(self.obstacles) < cfg.obstacles_count and attempts < 1000:
            attempts += 1
            new_rect = self._sample_obstacle(cfg.field_size, cfg.obstacle_profile)
            if Geometry.rects_overlap(new_rect, gw_rect):
                continue

            x, y, w, h = new_rect
            padded = (x - 0.1, y - 0.1, w + 0.2, h + 0.2)
            if any(Geometry.rects_overlap(padded, obs) for obs in self.obstacles):
                continue
            self.obstacles.append(new_rect)
        return self.obstacles

    def _generate_nodes_uniform(self, cfg: SceneConfig):
        valid_nodes = []
        attempts = 0
        while len(valid_nodes) < cfg.nodes_count and attempts < cfg.nodes_count * 100:
            attempts += 1
            x = random.uniform(-cfg.field_size / 2, cfg.field_size / 2)
            y = random.uniform(-cfg.field_size / 2, cfg.field_size / 2)
            if not self.is_point_in_obstacle(x, y):
                valid_nodes.append([x, y])
        return np.array(valid_nodes, dtype=float)

    def _generate_nodes_clustered(self, cfg: SceneConfig):
        clusters = max(2, cfg.nodes_count // 10)
        centers = []
        for _ in range(clusters):
            centers.append(
                (
                    random.uniform(-cfg.field_size / 2, cfg.field_size / 2),
                    random.uniform(-cfg.field_size / 2, cfg.field_size / 2),
                )
            )

        nodes = []
        attempts = 0
        while len(nodes) < cfg.nodes_count and attempts < cfg.nodes_count * 200:
            attempts += 1
            cx, cy = random.choice(centers)
            x = np.clip(random.gauss(cx, cfg.field_size * 0.08), -cfg.field_size / 2, cfg.field_size / 2)
            y = np.clip(random.gauss(cy, cfg.field_size * 0.08), -cfg.field_size / 2, cfg.field_size / 2)
            if not self.is_point_in_obstacle(x, y):
                nodes.append([x, y])
        return np.array(nodes, dtype=float)

    def generate_nodes(self, cfg: SceneConfig):
        if cfg.node_profile == "clustered":
            self.nodes = self._generate_nodes_clustered(cfg)
        else:
            self.nodes = self._generate_nodes_uniform(cfg)
        return self.nodes
