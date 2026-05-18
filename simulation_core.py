from dataclasses import dataclass
import math
import random
from typing import Callable

import numpy as np
import simpy


@dataclass
class SimulationConfig:
    v_supply: float
    tx_currents: dict
    current_tx_power: int
    i_rx_lna_on: float
    sf: int
    bw: int
    cr: int
    payload_len: int
    preamble_len: int
    receiver_sensitivity_dbm: float
    snr_threshold_db: float
    capture_threshold_db: float
    l0_db: float
    path_loss_exp: float
    distance_radius_km: float
    link_model: str
    routing_model: str
    packet_interval_s: float
    packets_per_node: int
    max_retries: int
    mobility_speed: float
    mobility_step_s: float
    routing_update_s: float
    field_size: float
    unit_mode: str


class LoRaSimulationEngine:
    def __init__(
        self,
        config: SimulationConfig,
        gateway_pos: np.ndarray,
        nodes: np.ndarray,
        is_blocked_cb: Callable[[np.ndarray, np.ndarray], bool],
        is_point_in_obstacle_cb: Callable[[float, float], bool],
    ):
        self.cfg = config
        self.gateway_pos = np.array(gateway_pos, dtype=float)
        self.nodes = np.array(nodes, dtype=float)
        self.is_blocked = is_blocked_cb
        self.is_point_in_obstacle = is_point_in_obstacle_cb
        self.base_packet_interval_s = max(0.1, self.cfg.packet_interval_s)
        self.trace = []
        self.trace_limit = 0

    def _capture_snapshot(self, event_name, details=None):
        if self.trace_limit <= 0 or len(self.trace) >= self.trace_limit:
            return
        self.trace.append(
            {
                "time": float(self.env.now),
                "event": event_name,
                "details": details or {},
                "positions": self.positions.copy(),
                "parent": self.parent.copy(),
                "success_packets": int(self.metrics["success_packets"]),
                "total_packets": int(self.metrics["total_packets"]),
            }
        )

    def calculate_toa(self, crc_enabled=True, implicit_header=False):
        t_symbol = (2 ** self.cfg.sf) / self.cfg.bw
        de = 1 if t_symbol > 0.016 else 0
        ih = 1 if implicit_header else 0
        crc = 1 if crc_enabled else 0

        payload_symb_nb = 8 + max(
            math.ceil(
                (
                    8 * self.cfg.payload_len
                    - 4 * self.cfg.sf
                    + 28
                    + 16 * crc
                    - 20 * ih
                )
                / (4 * (self.cfg.sf - 2 * de))
            )
            * (self.cfg.cr + 4),
            0,
        )
        t_preamble = (self.cfg.preamble_len + 4.25) * t_symbol
        t_payload = payload_symb_nb * t_symbol
        return t_preamble + t_payload

    def apply_scenario(self, scenario_name):
        self.packet_interval_s = self.base_packet_interval_s
        if scenario_name == "ideal":
            self.noise_floor_dbm = -121.0
            self.shadow_sigma_db = 2.0
            self.obstacle_extra_loss_db = 18.0
        elif scenario_name == "noisy":
            self.noise_floor_dbm = -114.0
            self.shadow_sigma_db = 4.0
            self.obstacle_extra_loss_db = 24.0
        elif scenario_name == "dense":
        else:
            self.noise_floor_dbm = -118.0
            self.shadow_sigma_db = 3.0
            self.obstacle_extra_loss_db = 22.0
            self.packet_interval_s = max(0.8, self.packet_interval_s * 0.6)

    def compute_rssi_snr(self, p_tx, p_rx):
        d_km = max(np.linalg.norm(p_tx - p_rx), 0.001)
        shadow_db = random.gauss(0.0, self.shadow_sigma_db)
        loss_db = self.cfg.l0_db + 10 * self.cfg.path_loss_exp * math.log10(d_km * 1000) + shadow_db
        if self.is_blocked(p_tx, p_rx):
            loss_db += self.obstacle_extra_loss_db
        rssi_dbm = self.cfg.current_tx_power - loss_db
        snr_db = rssi_dbm - self.noise_floor_dbm
        return rssi_dbm, snr_db

    def link_ok(self, p_tx, p_rx):
        if self.cfg.link_model == "distance":
            dist_km = np.linalg.norm(p_tx - p_rx)
            ok = dist_km <= self.cfg.distance_radius_km and not self.is_blocked(p_tx, p_rx)
            return ok, -100.0

        rssi_dbm, snr_db = self.compute_rssi_snr(p_tx, p_rx)
        ok = (
            rssi_dbm >= self.cfg.receiver_sensitivity_dbm
            and snr_db >= self.cfg.snr_threshold_db
        )
        return ok, rssi_dbm

    def build_mesh_parent(self, positions):
        n = len(positions)
        costs = np.full(n, np.inf)
        costs[0] = 0.0
        parent = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)

        for _ in range(n):
            candidates = np.where(~visited)[0]
            if len(candidates) == 0:
                break
            u = candidates[np.argmin(costs[candidates])]
            if costs[u] == np.inf:
                break
            visited[u] = True

            for v in range(n):
                if v == u or visited[v]:
                    continue
                ok, rssi = self.link_ok(positions[u], positions[v])
                if not ok:
                    continue
                hop_penalty = 1.0
                distance_penalty = np.linalg.norm(positions[u] - positions[v]) / max(self.cfg.field_size, 1e-6)
                if self.cfg.routing_model == "min_hops":
                    edge_cost = hop_penalty
                elif self.cfg.routing_model == "distance_first":
                    edge_cost = hop_penalty + (2.0 * distance_penalty)
                else:
                    quality_penalty = 0.0 if self.cfg.link_model == "distance" else max(0.0, (self.cfg.receiver_sensitivity_dbm + 20.0 - rssi) / 20.0)
                    edge_cost = hop_penalty + distance_penalty + quality_penalty
                if costs[u] + edge_cost < costs[v]:
                    costs[v] = costs[u] + edge_cost
                    parent[v] = u
        return parent

    def register_signal(self, receiver_id, signal):
        active = self.active_signals.setdefault(receiver_id, [])
        active = [s for s in active if s["end"] > self.env.now]
        for other in active:
            if signal["rssi"] - other["rssi"] >= self.cfg.capture_threshold_db:
                other["captured"] = True
            elif other["rssi"] - signal["rssi"] >= self.cfg.capture_threshold_db:
                signal["captured"] = True
            else:
                signal["collided"] = True
                other["collided"] = True
        active.append(signal)
        self.active_signals[receiver_id] = active

    def energy_units(self):
        t_packet = self.calculate_toa()
        i_tx_active = self.cfg.tx_currents.get(self.cfg.current_tx_power, 0.087)
        i_rx_active = self.cfg.i_rx_lna_on
        if self.cfg.unit_mode == "Joules":
            return self.cfg.v_supply * i_tx_active * t_packet, self.cfg.v_supply * i_rx_active * t_packet
        return (i_tx_active * t_packet * 1000) / 3600, (i_rx_active * t_packet * 1000) / 3600

    def get_route(self, src):
        if self.mode == "star":
            return [src, 0] if self.link_ok(self.positions[src], self.positions[0])[0] else None

        if self.parent[src] == -1:
            return None
        route = [src]
        cur = src
        visited = {src}
        while cur != 0:
            cur = int(self.parent[cur])
            if cur in visited or cur == -1:
                return None
            visited.add(cur)
            route.append(cur)
        return route

    def transmit_route(self, route, src, packet_id):
        tx_e, rx_e = self.energy_units()
        toa = self.calculate_toa()
        for hop_idx in range(len(route) - 1):
            sender = route[hop_idx]
            receiver = route[hop_idx + 1]
            ok, rssi = self.link_ok(self.positions[sender], self.positions[receiver])
            self.metrics["energy"][sender] += tx_e
            if sender != src:
                self.metrics["relay_load"][sender] += 1
            if not ok:
                return False

            if self.cfg.link_model == "distance":
                yield self.env.timeout(toa)
                self.metrics["energy"][receiver] += rx_e
                continue

            signal = {
                "id": (src, packet_id, hop_idx, self.env.now),
                "end": self.env.now + toa,
                "rssi": rssi,
                "collided": False,
                "captured": False,
            }
            self.register_signal(receiver, signal)
            yield self.env.timeout(toa)
            self.metrics["energy"][receiver] += rx_e
            if signal["collided"] or signal["captured"]:
                return False
        return True

    def node_process(self, node_id):
        for packet_id in range(self.cfg.packets_per_node):
            yield self.env.timeout(random.uniform(0.0, self.packet_interval_s))
            delivered = False
            start = self.env.now
            attempts = 0
            while attempts < self.cfg.max_retries + 1 and not delivered:
                attempts += 1
                route = self.get_route(node_id)
                if route is None:
                    break
                delivered = yield self.env.process(self.transmit_route(route, node_id, packet_id))
                if not delivered:
                    yield self.env.timeout(0.1)

            self.metrics["total_packets"] += 1
            self.metrics["retry_hist"].append(attempts - 1)
            if delivered:
                self.metrics["success_packets"] += 1
                self.metrics["delays"].append(self.env.now - start)
            self._capture_snapshot(
                "packet",
                {
                    "node_id": int(node_id),
                    "packet_id": int(packet_id),
                    "delivered": bool(delivered),
                    "attempts": int(attempts),
                    "delay_s": float(self.env.now - start),
                    "mode": self.mode,
                },
            )

    def mobility_process(self):
        bound = self.cfg.field_size / 2.0
        while True:
            dt = self.cfg.mobility_step_s
            yield self.env.timeout(dt)
            for i in range(1, len(self.positions)):
                old_pos = self.positions[i].copy()
                self.positions[i] = self.positions[i] + self.velocities[i] * dt
                for axis in (0, 1):
                    if self.positions[i][axis] > bound or self.positions[i][axis] < -bound:
                        self.velocities[i][axis] *= -1
                        self.positions[i][axis] = np.clip(self.positions[i][axis], -bound, bound)
                if self.is_point_in_obstacle(self.positions[i][0], self.positions[i][1]):
                    self.positions[i] = old_pos
                    self.velocities[i] = -self.velocities[i]
            self._capture_snapshot("mobility")

    def routing_process(self):
        while True:
            self.parent = self.build_mesh_parent(self.positions)
            self._capture_snapshot("routing")
            yield self.env.timeout(max(0.5, self.cfg.routing_update_s))

    def simulate(self, mode, scenario_name, collect_trace=False, max_trace_steps=500):
        self.apply_scenario(scenario_name)
        self.mode = mode
        self.env = simpy.Environment()
        self.active_signals = {}
        self.trace = []
        self.trace_limit = max_trace_steps if collect_trace else 0

        self.positions = np.vstack([self.gateway_pos, self.nodes]).astype(float)
        self.velocities = np.zeros_like(self.positions)
        for i in range(1, len(self.positions)):
            ang = random.uniform(0.0, 2 * math.pi)
            speed = self.cfg.mobility_speed
            self.velocities[i] = np.array([math.cos(ang) * speed, math.sin(ang) * speed])

        self.parent = self.build_mesh_parent(self.positions)
        self.metrics = {
            "total_packets": 0,
            "success_packets": 0,
            "delays": [],
            "retry_hist": [],
            "energy": np.zeros(len(self.positions)),
            "relay_load": np.zeros(len(self.positions)),
        }
        self._capture_snapshot("start")

        self.env.process(self.mobility_process())
        if mode == "mesh":
            self.env.process(self.routing_process())
        for node_id in range(1, len(self.positions)):
            self.env.process(self.node_process(node_id))

        sim_horizon = self.cfg.packets_per_node * self.packet_interval_s + 15
        self.env.run(until=max(10.0, sim_horizon))
        self._capture_snapshot("end")

        connected = 0
        direct_links = np.zeros(len(self.positions), dtype=bool)
        for i in range(1, len(self.positions)):
            direct_links[i] = self.link_ok(self.positions[i], self.positions[0])[0]
            if mode == "star" and direct_links[i]:
                connected += 1
            elif mode == "mesh" and self.parent[i] != -1:
                connected += 1

        pdr = (self.metrics["success_packets"] / max(1, self.metrics["total_packets"])) * 100.0
        avg_delay = float(np.mean(self.metrics["delays"])) if self.metrics["delays"] else 0.0
        avg_retry = float(np.mean(self.metrics["retry_hist"])) if self.metrics["retry_hist"] else 0.0

        return {
            "pdr": pdr,
            "avg_delay": avg_delay,
            "avg_retry": avg_retry,
            "energy": self.metrics["energy"],
            "relay_load": self.metrics["relay_load"],
            "connected_nodes": connected,
            "parent": self.parent.copy(),
            "positions": self.positions.copy(),
            "direct_links": direct_links,
            "trace": self.trace,
        }
