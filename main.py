import math
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

from scene_builder import SceneBuilder, SceneConfig
from simulation_core import LoRaSimulationEngine, SimulationConfig


def node_color(direct, mesh_connected):
    if direct and mesh_connected:
        return "#3498db"
    if mesh_connected:
        return "#2ecc71"
    return "#e74c3c"


def draw_network_topology(ax, positions, parent, direct_links):
    for i in range(1, len(positions)):
        p = int(parent[i])
        if p != -1:
            ax.annotate(
                "",
                xy=(positions[p, 0], positions[p, 1]),
                xytext=(positions[i, 0], positions[i, 1]),
                arrowprops=dict(arrowstyle="->", color="green", alpha=0.3, lw=1.5, connectionstyle="arc3"),
                zorder=2,
            )
    for i in range(1, len(positions)):
        if direct_links[i]:
            ax.plot(
                [positions[i, 0], positions[0, 0]],
                [positions[i, 1], positions[0, 1]],
                color="#3498db",
                alpha=0.25,
                linewidth=1.0,
                linestyle="--",
                zorder=1,
            )


class LoRaMeshSim:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRa Star vs Mesh Network Simulator")

        # --- Фізичні параметри LoRa (SX127x) ---
        self.v_supply = 3.3       # Напруга живлення (В)
        
        self.tx_currents = {
            20: 0.120,   # +20 dBm (PA_BOOST)
            17: 0.087,   # +17 dBm (PA_BOOST)
            13: 0.029,   # +13 dBm (RFO)
            7:  0.020    # +7 dBm (RFO)
        }
        self.current_tx_power = 17
        self.i_rx_lna_on = 0.0115

        # LoRa PHY
        self.sf = 9
        self.bw = 125000
        self.cr = 1
        self.payload_len = 20
        self.preamble_len = 8
        self.receiver_sensitivity_dbm = -137.0
        self.snr_threshold_db = -7.5
        self.capture_threshold_db = 6.0

        # Радіоканал
        self.l0_db = 40.0
        self.path_loss_exp = 2.8

        # Параметри симуляції
        self.unit_mode = tk.StringVar(value="Joules")
        self.nodes_count = tk.IntVar(value=30)
        self.obstacles_count = tk.IntVar(value=4)
        self.field_size = tk.DoubleVar(value=6.0)
        self.r_max = tk.DoubleVar(value=2.0)
        self.mobility_mode = tk.StringVar(value="static")
        self.mobility_speed = tk.DoubleVar(value=0.004)
        self.mobility_step_s = tk.DoubleVar(value=3.0)
        self.routing_update_s = tk.DoubleVar(value=3.0)
        self.packet_interval_s = tk.DoubleVar(value=60.0)
        self.max_retries = tk.IntVar(value=2)
        self.packets_per_node = tk.IntVar(value=30)
        self.scenario = tk.StringVar(value="ideal")
        self.link_model = tk.StringVar(value="realistic")
        self.routing_model = tk.StringVar(value="balanced")
        self.obstacle_profile = tk.StringVar(value="mixed")
        self.node_profile = tk.StringVar(value="uniform")

        self.gateway_pos = np.array([0.0, 0.0])
        self.nodes = None
        self.obstacles = []
        self.last_metrics = None
        self._last_sim_velocities = None
        self._last_scenario = None
        self.scene_builder = SceneBuilder(self.gateway_pos)

        self.setup_ui()
        self.generate_obstacles()
        self.generate_nodes()

    def calculate_toa(self, crc_enabled=True, implicit_header=False):
        t_symbol = (2 ** self.sf) / self.bw
        de = 1 if t_symbol > 0.016 else 0
        ih = 1 if implicit_header else 0
        crc = 1 if crc_enabled else 0
        payload_symb_nb = 8 + max(
            math.ceil((8 * self.payload_len - 4 * self.sf + 28 + 16 * crc - 20 * ih) / (4 * (self.sf - 2 * de))) * (self.cr + 4),
            0,
        )
        return (self.preamble_len + 4.25) * t_symbol + payload_symb_nb * t_symbol

    def build_config(self):
        return SimulationConfig(
            v_supply=self.v_supply,
            tx_currents=self.tx_currents,
            current_tx_power=self.current_tx_power,
            i_rx_lna_on=self.i_rx_lna_on,
            sf=self.sf,
            bw=self.bw,
            cr=self.cr,
            payload_len=self.payload_len,
            preamble_len=self.preamble_len,
            receiver_sensitivity_dbm=self.receiver_sensitivity_dbm,
            snr_threshold_db=self.snr_threshold_db,
            capture_threshold_db=self.capture_threshold_db,
            l0_db=self.l0_db,
            path_loss_exp=self.path_loss_exp,
            distance_radius_km=self.r_max.get(),
            link_model=self.link_model.get(),
            routing_model=self.routing_model.get(),
            packet_interval_s=max(0.1, self.packet_interval_s.get()),
            packets_per_node=self.packets_per_node.get(),
            max_retries=self.max_retries.get(),
            mobility_speed=self.mobility_speed.get(),
            mobility_step_s=self.mobility_step_s.get(),
            routing_update_s=self.routing_update_s.get(),
            field_size=self.field_size.get(),
            unit_mode=self.unit_mode.get(),
            mobility_enabled=self.mobility_mode.get() == "dynamic",
        )

    def create_engine(self):
        return LoRaSimulationEngine(
            config=self.build_config(),
            gateway_pos=self.gateway_pos,
            nodes=self.nodes,
            is_blocked_cb=self.scene_builder.is_blocked,
            is_point_in_obstacle_cb=self.scene_builder.is_point_in_obstacle,
        )

    def build_scene_config(self):
        return SceneConfig(
            field_size=self.field_size.get(),
            obstacles_count=self.obstacles_count.get(),
            nodes_count=self.nodes_count.get(),
            obstacle_profile=self.obstacle_profile.get(),
            node_profile=self.node_profile.get(),
        )

    def generate_obstacles(self):
        self.scene_builder.update_gateway(self.gateway_pos)
        self.obstacles = self.scene_builder.generate_obstacles(self.build_scene_config())

        if self.nodes is not None and len(self.nodes) > 0:
            self.draw_network()

    def clear_obstacles(self):
        self.obstacles = []
        self.scene_builder.obstacles = []
        self.obstacles_count.set(0)
        if self.nodes is not None and len(self.nodes) > 0:
            self.draw_network()

    def setup_ui(self):
        sidebar = ttk.Frame(self.root)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)

        self.ctrl_canvas = tk.Canvas(sidebar, width=360, highlightthickness=0)
        self.ctrl_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        scrollbar = ttk.Scrollbar(sidebar, orient=tk.VERTICAL, command=self.ctrl_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.ctrl_canvas.configure(yscrollcommand=scrollbar.set)

        scrollable = ttk.Frame(self.ctrl_canvas)
        canvas_window = self.ctrl_canvas.create_window((0, 0), window=scrollable, anchor="nw")

        def on_scrollable_configure(_event):
            self.ctrl_canvas.configure(scrollregion=self.ctrl_canvas.bbox("all"))

        def on_canvas_configure(event):
            self.ctrl_canvas.itemconfigure(canvas_window, width=event.width)

        def on_mousewheel(event):
            self.ctrl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_wheel(_event):
            self.ctrl_canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_wheel(_event):
            self.ctrl_canvas.unbind_all("<MouseWheel>")

        scrollable.bind("<Configure>", on_scrollable_configure)
        self.ctrl_canvas.bind("<Configure>", on_canvas_configure)
        self.ctrl_canvas.bind("<Enter>", bind_wheel)
        self.ctrl_canvas.bind("<Leave>", unbind_wheel)

        ctrl_frame = ttk.LabelFrame(scrollable, text="Параметри симуляції (LoRa SX1276 + SimPy)")
        ctrl_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(ctrl_frame, text="Кількість перешкод:").pack()
        ttk.Entry(ctrl_frame, textvariable=self.obstacles_count).pack(pady=2)

        ttk.Label(ctrl_frame, text="Кількість вузлів:").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.nodes_count).pack()

        ttk.Label(ctrl_frame, text="Розмір поля (км):").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.field_size).pack()

        ttk.Label(ctrl_frame, text="R_max для режиму 'distance' (км):").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.r_max).pack()

        ttk.Label(ctrl_frame, text="Режим мобільності:").pack(pady=5)
        mobility_combo = ttk.Combobox(
            ctrl_frame,
            textvariable=self.mobility_mode,
            values=("static", "dynamic"),
            state="readonly",
        )
        mobility_combo.pack(fill="x")

        ttk.Label(ctrl_frame, text="Пакетів на вузол:").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.packets_per_node).pack()

        ttk.Label(ctrl_frame, text="Інтервал трафіку (с):").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.packet_interval_s).pack()

        ttk.Label(ctrl_frame, text="Швидкість руху (км/с):").pack(pady=5)
        self.mobility_speed_entry = ttk.Entry(ctrl_frame, textvariable=self.mobility_speed)
        self.mobility_speed_entry.pack(pady=2)

        ttk.Label(ctrl_frame, text="Крок руху (с):").pack(pady=5)
        self.mobility_step_entry = ttk.Entry(ctrl_frame, textvariable=self.mobility_step_s)
        self.mobility_step_entry.pack(pady=2)

        def on_mobility_mode_change(*_args):
            dynamic = self.mobility_mode.get() == "dynamic"
            state = "normal" if dynamic else "disabled"
            self.mobility_speed_entry.configure(state=state)
            self.mobility_step_entry.configure(state=state)

        self.mobility_mode.trace_add("write", on_mobility_mode_change)
        on_mobility_mode_change()

        ttk.Label(ctrl_frame, text="Ретраї (без ACK):").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.max_retries).pack()

        ttk.Label(ctrl_frame, text="Сценарій середовища:").pack(pady=5)
        ttk.Combobox(
            ctrl_frame,
            textvariable=self.scenario,
            values=("ideal", "noisy", "dense", "blocked"),
            state="readonly",
        ).pack(fill="x")

        ttk.Label(ctrl_frame, text="Модель доставки:").pack(pady=5)
        ttk.Combobox(
            ctrl_frame,
            textvariable=self.link_model,
            values=("realistic", "distance"),
            state="readonly",
        ).pack(fill="x")

        ttk.Label(ctrl_frame, text="Модель маршрутизації (Mesh):").pack(pady=5)
        ttk.Combobox(
            ctrl_frame,
            textvariable=self.routing_model,
            values=("balanced", "min_hops", "distance_first"),
            state="readonly",
        ).pack(fill="x")

        ttk.Label(ctrl_frame, text="Тип перешкод:").pack(pady=5)
        ttk.Combobox(
            ctrl_frame,
            textvariable=self.obstacle_profile,
            values=("mixed", "buildings", "walls"),
            state="readonly",
        ).pack(fill="x")

        ttk.Label(ctrl_frame, text="Тип розміщення нод:").pack(pady=5)
        ttk.Combobox(
            ctrl_frame,
            textvariable=self.node_profile,
            values=("uniform", "clustered"),
            state="readonly",
        ).pack(fill="x")

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(ctrl_frame, text="Одиниці енергії:").pack(pady=5)
        ttk.Radiobutton(ctrl_frame, text="Джоулі (J/mJ)", variable=self.unit_mode, value="Joules").pack(anchor=tk.W)
        ttk.Radiobutton(ctrl_frame, text="Ампер-години (mAh)", variable=self.unit_mode, value="mAh").pack(anchor=tk.W)

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", pady=10)
        toa_ms = self.calculate_toa() * 1000
        ttk.Label(ctrl_frame, text=f"SF: {self.sf} | BW: {self.bw / 1000:.0f} kHz", foreground="darkgreen").pack()
        ttk.Label(ctrl_frame, text=f"ToA: {toa_ms:.2f} ms | Capture: {self.capture_threshold_db:.1f} dB").pack()
        ttk.Label(ctrl_frame, text=f"Sensitivity: {self.receiver_sensitivity_dbm:.1f} dBm").pack()

        ttk.Button(ctrl_frame, text="Генерувати перешкоди", command=self.generate_obstacles).pack(fill="x", pady=5)
        ttk.Button(ctrl_frame, text="Очистити перешкоди", command=self.clear_obstacles).pack(fill="x", pady=2)
        ttk.Button(ctrl_frame, text="Генерувати вузли", command=self.generate_nodes).pack(fill="x", pady=5)
        ttk.Button(ctrl_frame, text="Запустити симуляцію", command=self.run_simulation).pack(fill="x", pady=5)
        ttk.Button(ctrl_frame, text="Покрокова візуалізація (Mesh)", command=self.run_step_visualization).pack(fill="x", pady=2)

        ttk.Separator(ctrl_frame, orient="horizontal").pack(fill="x", pady=5)
        ttk.Button(
            ctrl_frame,
            text="Запустити серію (3 сценарії × 30)",
            command=lambda: self.run_batch_simulation(30),
        ).pack(fill="x", pady=5)

        ttk.Label(ctrl_frame, text="* Клік по мапі змінює Gateway", foreground="blue").pack(pady=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.gateway_pos = np.array([event.xdata, event.ydata])
            self.scene_builder.update_gateway(self.gateway_pos)
            self.draw_network()

    def generate_nodes(self):
        target_count = self.nodes_count.get()
        self.nodes = self.scene_builder.generate_nodes(self.build_scene_config())
        if len(self.nodes) < target_count:
            print(f"Попередження: вдалося розмістити лише {len(self.nodes)} вузлів через щільність перешкод.")
        self.draw_network()

    def draw_network(self):
        self.ax.clear()
        for x, y, w, h in self.obstacles:
            self.ax.add_patch(Rectangle((x, y), w, h, color="red", alpha=0.3))
        if self.nodes is not None and len(self.nodes) > 0:
            self.ax.scatter(self.nodes[:, 0], self.nodes[:, 1], c="gray", alpha=0.5, label="Вузли")
        self.ax.scatter(self.gateway_pos[0], self.gateway_pos[1], c="gold", s=200, marker="*", edgecolors="black", label="Gateway")
        self.ax.set_title("Попередній перегляд структури мережі з перешкодами")
        self.ax.grid(True, linestyle=":")
        self.canvas.draw()

    def run_simulation(self):
        if self.nodes is None or len(self.nodes) == 0:
            messagebox.showwarning("Помилка", "Спочатку згенеруйте вузли.")
            return
        engine = self.create_engine()
        scenario_name = self.scenario.get()
        n_nodes = len(self.nodes) + 1
        velocities = engine.sample_velocities(n_nodes)
        star = engine.simulate("star", scenario_name, velocities=velocities.copy())
        mesh = engine.simulate("mesh", scenario_name, velocities=velocities.copy())
        self._last_sim_velocities = velocities.copy()
        self._last_scenario = scenario_name
        self.last_metrics = (star, mesh, scenario_name)
        self.show_results(star, mesh, scenario_name)

    def run_step_visualization(self):
        if self.nodes is None or len(self.nodes) == 0:
            messagebox.showwarning("Помилка", "Спочатку згенеруйте вузли.")
            return
        engine = self.create_engine()
        scenario_name = self.scenario.get()
        n_nodes = len(self.nodes) + 1
        if (
            self._last_sim_velocities is not None
            and self._last_scenario == scenario_name
            and len(self._last_sim_velocities) == n_nodes
        ):
            velocities = self._last_sim_velocities.copy()
        else:
            velocities = engine.sample_velocities(n_nodes)
        mesh = engine.simulate("mesh", scenario_name, collect_trace=True, max_trace_steps=700, velocities=velocities)
        trace = mesh.get("trace", [])
        if not trace:
            messagebox.showwarning("Немає даних", "Не вдалося зібрати кроки для візуалізації.")
            return
        self.show_step_visualizer(trace, scenario_name)

    def show_step_visualizer(self, trace, scenario_name):
        viewer = tk.Toplevel(self.root)
        viewer.title("Покрокова візуалізація руху та обрахунку (Mesh)")

        control = ttk.Frame(viewer)
        control.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        frame_idx = tk.IntVar(value=0)
        playing = {"value": False}
        status_var = tk.StringVar(value="")

        ttk.Label(control, text=f"Сценарій: {scenario_name.upper()}").pack(side=tk.LEFT, padx=4)
        ttk.Label(control, text="Крок:").pack(side=tk.LEFT, padx=(12, 4))
        scale = ttk.Scale(control, from_=0, to=len(trace) - 1, orient=tk.HORIZONTAL)
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        fig, ax = plt.subplots(figsize=(7, 6))
        canvas = FigureCanvasTkAgg(fig, master=viewer)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        ttk.Label(viewer, textvariable=status_var, justify=tk.LEFT).pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)

        def render_frame(index):
            idx = max(0, min(len(trace) - 1, int(index)))
            frame_idx.set(idx)
            frame = trace[idx]
            ax.clear()

            for x, y, w, h in self.obstacles:
                ax.add_patch(Rectangle((x, y), w, h, color="red", alpha=0.35))

            positions = frame["positions"]
            parent = frame["parent"]
            direct_links = frame.get("direct_links", np.zeros(len(positions), dtype=bool))

            draw_network_topology(ax, positions, parent, direct_links)

            colors = [
                node_color(bool(direct_links[i]), int(parent[i]) != -1)
                for i in range(1, len(positions))
            ]
            ax.scatter(
                positions[1:, 0],
                positions[1:, 1],
                c=colors,
                s=45,
                edgecolors="black",
                alpha=0.8,
                zorder=3,
            )
            ax.scatter(positions[0, 0], positions[0, 1], c="gold", s=300, marker="*", edgecolors="black", zorder=5)

            details = frame.get("details", {})
            if frame["event"] == "packet" and details:
                note = (
                    f"t={frame['time']:.2f}s | packet node={details.get('node_id')} "
                    f"id={details.get('packet_id')} delivered={details.get('delivered')} "
                    f"attempts={details.get('attempts')}"
                )
            else:
                note = f"t={frame['time']:.2f}s | event={frame['event']}"

            pdr = 100.0 * frame["success_packets"] / max(1, frame["total_packets"])
            status_var.set(
                f"{note}\nУспішно: {frame['success_packets']} / {frame['total_packets']}  (PDR {pdr:.1f}%)"
            )

            ax.set_title("Mesh: покроковий стан мережі")
            ax.set_xlabel("Відстань (км)")
            ax.set_ylabel("Відстань (км)")
            ax.grid(True, linestyle=":", alpha=0.5)
            canvas.draw()

        def on_scale_change(value):
            render_frame(float(value))

        scale.configure(command=on_scale_change)

        def step_prev():
            playing["value"] = False
            render_frame(frame_idx.get() - 1)
            scale.set(frame_idx.get())

        def step_next():
            playing["value"] = False
            render_frame(frame_idx.get() + 1)
            scale.set(frame_idx.get())

        def play_loop():
            if not playing["value"]:
                return
            next_idx = frame_idx.get() + 1
            if next_idx >= len(trace):
                playing["value"] = False
                return
            render_frame(next_idx)
            scale.set(frame_idx.get())
            viewer.after(250, play_loop)

        def toggle_play():
            playing["value"] = not playing["value"]
            play_btn.configure(text="Пауза" if playing["value"] else "Play")
            if playing["value"]:
                play_loop()

        ttk.Button(control, text="Prev", command=step_prev).pack(side=tk.RIGHT, padx=3)
        play_btn = ttk.Button(control, text="Play", command=toggle_play)
        play_btn.pack(side=tk.RIGHT, padx=3)
        ttk.Button(control, text="Next", command=step_next).pack(side=tk.RIGHT, padx=3)

        end_indices = [i for i, f in enumerate(trace) if f.get("event") == "end"]
        start_idx = end_indices[-1] if end_indices else len(trace) - 1
        scale.set(start_idx)
        render_frame(start_idx)

    def run_batch_simulation(self, iterations=30):
        scenarios = ("ideal", "noisy", "dense", "blocked")
        lines = ["РЕЗУЛЬТАТИ ПОРІВНЯННЯ (Star vs Mesh)"]
        for scenario_name in scenarios:
            star_pdr, mesh_pdr = [], []
            star_delay, mesh_delay = [], []
            for _ in range(iterations):
                self.generate_obstacles()
                self.generate_nodes()
                engine = self.create_engine()
                n_nodes = len(self.nodes) + 1
                velocities = engine.sample_velocities(n_nodes)
                star = engine.simulate("star", scenario_name, velocities=velocities.copy())
                mesh = engine.simulate("mesh", scenario_name, velocities=velocities.copy())
                star_pdr.append(star["pdr"])
                mesh_pdr.append(mesh["pdr"])
                star_delay.append(star["avg_delay"])
                mesh_delay.append(mesh["avg_delay"])

            lines.append(
                f"\n[{scenario_name.upper()}] PDR Star: {np.mean(star_pdr):.1f}% | PDR Mesh: {np.mean(mesh_pdr):.1f}% | ΔPDR: {np.mean(mesh_pdr) - np.mean(star_pdr):.1f}%"
            )
            lines.append(f"Delay Star: {np.mean(star_delay):.2f}s | Delay Mesh: {np.mean(mesh_delay):.2f}s")
        messagebox.showinfo("Результати серії тестів", "\n".join(lines))

    def show_results(self, star, mesh, scenario_name):
        res_win = tk.Toplevel(self.root)
        res_win.title("Результати LoRa Star vs Mesh (динамічна SimPy-модель)")

        relay_loads = mesh["relay_load"][1:].copy()
        sorted_relays = (np.argsort(relay_loads)[::-1] + 1).tolist()
        while len(sorted_relays) < 3:
            sorted_relays.append(1)
        top_relay_indices = np.array(sorted_relays[:3])

        avg_mesh_energy = np.mean(mesh["energy"][1:]) if len(mesh["energy"]) > 1 else 0.0
        total_mesh_energy = float(np.sum(mesh["energy"][1:]))
        avg_star_energy = np.mean(star["energy"][1:]) if len(star["energy"]) > 1 else 0.0
        total_star_energy = float(np.sum(star["energy"][1:]))

        info_frame = ttk.Frame(res_win)
        info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
        unit_label = "Дж" if self.unit_mode.get() == "Joules" else "mAh"
        sub_unit_label = "мДж" if self.unit_mode.get() == "Joules" else "mAh"
        mult = 1000 if self.unit_mode.get() == "Joules" else 1

        results_text = f"""
=== МЕТРИКИ ЕФЕКТИВНОСТІ ===
Сценарій: {scenario_name.upper()}
Режим: {"Реалістичний (RSSI/SNR + колізії)" if self.link_model.get() == "realistic" else "Спрощений (по відстані)"}
Мобільність: {"Динамічний" if self.mobility_mode.get() == "dynamic" else "Статичний"}
Маршрутизація Mesh: {self.routing_model.get()}
Розміщення нод: {self.node_profile.get()} | Перешкоди: {self.obstacle_profile.get()}

PDR (Доставка пакетів):
• Star (LoRaWAN): {star['pdr']:.1f}%
• Mesh (LoRa):    {mesh['pdr']:.1f}%

ПЕРЕВАГА MESH: {mesh['pdr'] - star['pdr']:.1f}%
Сер. затримка (Star/Mesh): {star['avg_delay']:.2f}s / {mesh['avg_delay']:.2f}s
Сер. ретраї (Star/Mesh): {star['avg_retry']:.2f} / {mesh['avg_retry']:.2f}

ЕНЕРГОСПОЖИВАННЯ SX1276 ({unit_label})
Загальне:
• Star (LoRaWAN): {total_star_energy:.4f} {unit_label}
• Mesh (LoRa):    {total_mesh_energy:.4f} {unit_label}
Середнє на вузол:
• Star (LoRaWAN): {avg_star_energy * mult:.3f} {sub_unit_label}
• Mesh (LoRa):    {avg_mesh_energy * mult:.3f} {sub_unit_label}

Покриття вузлів:
• Star (LoRaWAN): {star['connected_nodes']}/{len(self.nodes)} вузлів
• Mesh (LoRa):    {mesh['connected_nodes']}/{len(self.nodes)} вузлів

Критичні вузли (для Mesh мережі):
1. Вузол #{top_relay_indices[0]}: {int(mesh['relay_load'][top_relay_indices[0]])} пак.
2. Вузол #{top_relay_indices[1]}: {int(mesh['relay_load'][top_relay_indices[1]])} пак.
3. Вузол #{top_relay_indices[2]}: {int(mesh['relay_load'][top_relay_indices[2]])} пак.
"""
        tk.Label(info_frame, text=results_text, justify=tk.LEFT, font=("Courier", 10), background="#f8f9fa", relief="solid", padx=15, pady=15).pack(pady=10)
        ttk.Button(info_frame, text="Закрити", command=res_win.destroy).pack(pady=10)

        fig_res, ax_res = plt.subplots(figsize=(6, 6))
        all_pts = mesh["positions"]
        direct_links = mesh["direct_links"]
        parent = mesh["parent"]

        for obs in self.obstacles:
            ax_res.add_patch(Rectangle((obs[0], obs[1]), obs[2], obs[3], color="red", alpha=0.4))

        draw_network_topology(ax_res, all_pts, parent, direct_links)

        max_energy = np.max(mesh["energy"][1:]) if len(mesh["energy"]) > 1 else 1
        for i in range(1, len(all_pts)):
            has_direct_link = bool(direct_links[i])
            is_connected_mesh = parent[i] != -1
            color = node_color(has_direct_link, is_connected_mesh)

            relative_energy = mesh["energy"][i] / max_energy
            node_size = 60 + (relative_energy * 400)
            ax_res.scatter(all_pts[i, 0], all_pts[i, 1], c=color, s=node_size, edgecolors="black", alpha=0.7, zorder=3)

            if mesh["energy"][i] > avg_mesh_energy * 1.5:
                ax_res.text(
                    all_pts[i, 0],
                    all_pts[i, 1] - 0.5,
                    f"Node #{i+1} \n({mesh['energy'][i]*mult:.2f} {sub_unit_label})",
                    fontsize=8,
                    ha="center",
                    color="#2c3e50",
                    weight="bold",
                )

            if i in top_relay_indices:
                ax_res.text(all_pts[i, 0], all_pts[i, 1] + 0.2, "CRITICAL", color="darkred", weight="bold", fontsize=8, ha="center")

        ax_res.scatter(self.gateway_pos[0], self.gateway_pos[1], c="gold", s=450, marker="*", edgecolors="black", label="Gateway", zorder=10)
        mobility_label = "динамічний" if self.mobility_mode.get() == "dynamic" else "статичний"
        ax_res.set_title(f"Маршрути Mesh ({mobility_label}) + енергоспоживання")
        ax_res.set_xlabel("Відстань (км)")
        ax_res.set_ylabel("Відстань (км)")
        ax_res.grid(True, linestyle=":", alpha=0.5)

        canvas_res = FigureCanvasTkAgg(fig_res, master=res_win)
        canvas_res.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        canvas_res.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = LoRaMeshSim(root)
    root.mainloop()