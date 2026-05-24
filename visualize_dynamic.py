# -*- coding: utf-8 -*-
"""
visualize_dynamic.py — Інтерактивна кастомна візуалізація динамічного LoRa-середовища.

Запуск:
  python visualize_dynamic.py

Можливості (вмикаються галочками):
  • Карта мережі — фінальні позиції, маршрути, стан зв'язку
  • Траєкторії руху вузлів — згасаючі шляхи, старт/фініш
  • PDR у часі — ковзне вікно + наростаюче значення
  • Зв'язність мережі у часі — кількість досяжних вузлів
  • Затримки / Ретраї — розподіл гістограмами
"""

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle
import numpy as np

from scene_builder import SceneBuilder, SceneConfig
from simulation_core import LoRaSimulationEngine, SimulationConfig

# ── Константи ─────────────────────────────────────────────────────────────────

GATEWAY_POS = np.array([0.0, 0.0])

STAR_COLOR  = "#1565C0"
MESH_COLOR  = "#2E7D32"

SCENARIO_DEFS = {
    "ideal": ("ideal",  0, "mixed"),
    "noisy": ("noisy",  4, "mixed"),
    "dense": ("dense", 10, "buildings"),
}

BASE_PHY = dict(
    v_supply=3.3,
    tx_currents={20: 0.120, 17: 0.087, 13: 0.029, 7: 0.020},
    current_tx_power=17,
    i_rx_lna_on=0.0115,
    sf=9,
    bw=125000,
    cr=1,
    payload_len=20,
    preamble_len=8,
    receiver_sensitivity_dbm=-137.0,
    snr_threshold_db=-7.5,
    capture_threshold_db=6.0,
    l0_db=40.0,
    path_loss_exp=2.8,
    distance_radius_km=2.0,
    link_model="realistic",
    routing_model="balanced",
    packet_interval_s=60.0,
    max_retries=2,
    mobility_speed=0.004,
    mobility_step_s=3.0,
    routing_update_s=3.0,
    field_size=6.0,
    unit_mode="mAh",
    mobility_enabled=True,
)

# ── Аналіз трасування ─────────────────────────────────────────────────────────

def extract_trajectories(trace: list) -> dict:
    """
    Повертає {node_id: [(t, x, y), ...]} з мобільних подій трасування.
    """
    mob = [f for f in trace if f["event"] in ("start", "mobility", "end")]
    if not mob:
        return {}
    n = len(mob[0]["positions"])
    paths = {i: [] for i in range(1, n)}
    for frame in mob:
        t = frame["time"]
        for i in range(1, n):
            x, y = frame["positions"][i]
            paths[i].append((t, x, y))
    return paths


def extract_pdr_series(trace: list, window: int = 10):
    """
    Повертає (times, rolling_pdr, cumulative_pdr) з подій пакетів.
    rolling_pdr — PDR за останні `window` спроб.
    """
    events = [f for f in trace if f["event"] == "packet"]
    if not events:
        return np.array([]), np.array([]), np.array([])

    times     = np.array([f["time"] for f in events])
    delivered = np.array([f["details"].get("delivered", False) for f in events], dtype=float)

    rolling = np.array([
        delivered[max(0, i - window + 1): i + 1].mean() * 100
        for i in range(len(delivered))
    ])
    cumul = np.array([
        delivered[: i + 1].mean() * 100
        for i in range(len(delivered))
    ])
    return times, rolling, cumul


def extract_connectivity_series(trace: list, mode: str):
    """
    Повертає (times, n_connected) з мобільних та стартової/фінальної подій.
    """
    frames = [f for f in trace if f["event"] in ("start", "mobility", "end")]
    if not frames:
        return np.array([]), np.array([])

    times, counts = [], []
    for f in frames:
        times.append(f["time"])
        if mode == "star":
            counts.append(int(np.sum(f["direct_links"][1:])))
        else:
            counts.append(int(np.sum(np.array(f["parent"][1:]) != -1)))
    return np.array(times), np.array(counts)


def extract_delays_retries(trace: list):
    """
    Повертає (delays_s, retries) з подій пакетів.
    delays_s  — лише для успішно доставлених.
    retries   — для всіх (attempts - 1).
    """
    events = [f for f in trace if f["event"] == "packet"]
    delays  = [f["details"]["delay_s"]      for f in events if f["details"].get("delivered")]
    retries = [f["details"]["attempts"] - 1 for f in events]
    return np.array(delays), np.array(retries, dtype=int)


# ── Головне вікно ─────────────────────────────────────────────────────────────

class DynamicVisualizer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LoRa Dynamic Visualizer")
        self.root.minsize(1280, 720)

        self.fig        = None
        self.mpl_canvas = None
        self.obstacles  = []

        self._build_ui()

    # ── Побудова інтерфейсу ───────────────────────────────────────────────────

    def _build_ui(self):
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)

        # ── Ліва панель: налаштування ─────────────────────────────────────────
        left = ttk.Frame(pane, width=310)
        left.pack_propagate(False)
        pane.add(left, weight=0)

        ctrl_canvas = tk.Canvas(left, width=300, highlightthickness=0)
        ctrl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=ctrl_canvas.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_canvas.configure(yscrollcommand=sb.set)

        scrollable = ttk.Frame(ctrl_canvas)
        win_id = ctrl_canvas.create_window((0, 0), window=scrollable, anchor="nw")
        scrollable.bind("<Configure>",
                        lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all")))
        ctrl_canvas.bind("<Configure>",
                         lambda e: ctrl_canvas.itemconfigure(win_id, width=e.width))
        ctrl_canvas.bind("<Enter>",
                         lambda e: ctrl_canvas.bind_all(
                             "<MouseWheel>",
                             lambda ev: ctrl_canvas.yview_scroll(int(-ev.delta / 120), "units")))
        ctrl_canvas.bind("<Leave>",
                         lambda e: ctrl_canvas.unbind_all("<MouseWheel>"))

        self._build_controls(scrollable)

        # ── Права панель: графіки ─────────────────────────────────────────────
        self.plot_frame = ttk.Frame(pane)
        pane.add(self.plot_frame, weight=1)

        # Підказка до запуску
        ttk.Label(
            self.plot_frame,
            text="Налаштуйте параметри та натисніть «Запустити».",
            foreground="gray", font=("", 11),
        ).pack(expand=True)

    def _build_controls(self, parent: ttk.Frame):
        pad = {"padx": 8, "pady": 3}

        # ── Симуляція ─────────────────────────────────────────────────────────
        sim = ttk.LabelFrame(parent, text="Параметри симуляції")
        sim.pack(fill="x", **pad)

        def row(label, widget_factory):
            ttk.Label(sim, text=label).pack(anchor="w", padx=4, pady=(6, 0))
            w = widget_factory(sim)
            w.pack(fill="x", padx=4, pady=2)
            return w

        self.scenario_var = tk.StringVar(value="ideal")
        row("Сценарій середовища:",
            lambda f: ttk.Combobox(f, textvariable=self.scenario_var,
                                   values=("ideal", "noisy", "dense"), state="readonly"))

        self.topology_var = tk.StringVar(value="mesh")
        row("Топологія:",
            lambda f: ttk.Combobox(f, textvariable=self.topology_var,
                                   values=("star", "mesh", "обидві"), state="readonly"))

        self.nodes_var = tk.IntVar(value=20)
        row("Кількість вузлів:",
            lambda f: ttk.Spinbox(f, textvariable=self.nodes_var,
                                  from_=5, to=100, increment=5))

        self.packets_var = tk.IntVar(value=20)
        row("Пакетів на вузол:",
            lambda f: ttk.Spinbox(f, textvariable=self.packets_var,
                                  from_=5, to=200, increment=5))

        self.obstacles_var = tk.IntVar(value=4)
        row("Кількість перешкод:",
            lambda f: ttk.Spinbox(f, textvariable=self.obstacles_var,
                                  from_=0, to=30, increment=1))

        self.routing_var = tk.StringVar(value="balanced")
        row("Маршрутизація Mesh:",
            lambda f: ttk.Combobox(f, textvariable=self.routing_var,
                                   values=("balanced", "min_hops", "distance_first"),
                                   state="readonly"))

        self.pdr_window_var = tk.IntVar(value=10)
        row("Вікно PDR (пакетів):",
            lambda f: ttk.Spinbox(f, textvariable=self.pdr_window_var,
                                  from_=3, to=50, increment=1))

        # ── Мобільність ───────────────────────────────────────────────────────
        mob = ttk.LabelFrame(parent, text="Мобільність вузлів")
        mob.pack(fill="x", **pad)

        self.mobility_var = tk.StringVar(value="dynamic")
        mob_row = ttk.Frame(mob)
        mob_row.pack(fill="x", padx=4, pady=4)
        ttk.Radiobutton(mob_row, text="Статична", variable=self.mobility_var,
                        value="static",  command=self._on_mob_change).pack(side=tk.LEFT)
        ttk.Radiobutton(mob_row, text="Динамічна", variable=self.mobility_var,
                        value="dynamic", command=self._on_mob_change).pack(side=tk.LEFT, padx=8)

        ttk.Label(mob, text="Швидкість (км/с):").pack(anchor="w", padx=4, pady=(4, 0))
        self.speed_var = tk.DoubleVar(value=0.004)
        self.speed_entry = ttk.Entry(mob, textvariable=self.speed_var)
        self.speed_entry.pack(fill="x", padx=4, pady=2)

        ttk.Label(mob, text="Крок оновлення (с):").pack(anchor="w", padx=4, pady=(4, 0))
        self.step_var = tk.DoubleVar(value=3.0)
        self.step_entry = ttk.Entry(mob, textvariable=self.step_var)
        self.step_entry.pack(fill="x", padx=4, pady=(2, 6))

        # ── Що відображати ────────────────────────────────────────────────────
        vis = ttk.LabelFrame(parent, text="Що відображати")
        vis.pack(fill="x", **pad)

        self.show_traj  = tk.BooleanVar(value=True)
        self.show_pdr   = tk.BooleanVar(value=True)
        self.show_conn  = tk.BooleanVar(value=True)
        self.show_hist  = tk.BooleanVar(value=True)

        def chk(text, var, hint=""):
            f = ttk.Frame(vis)
            f.pack(fill="x", padx=4, pady=2)
            ttk.Checkbutton(f, text=text, variable=var).pack(side=tk.LEFT)
            if hint:
                ttk.Label(f, text=hint, foreground="#888", font=("", 8)).pack(side=tk.LEFT)

        chk("Карта мережі",               tk.BooleanVar(value=True), "(завжди)")
        chk("Траєкторії руху вузлів",      self.show_traj,  "")
        chk("PDR у часі",                  self.show_pdr,   "")
        chk("Зв'язність мережі у часі",    self.show_conn,  "")
        chk("Гістограми затримок / ретраїв", self.show_hist, "")

        # ── Запуск ────────────────────────────────────────────────────────────
        btn = ttk.LabelFrame(parent, text="Управління")
        btn.pack(fill="x", **pad)

        ttk.Button(btn, text="▶  Запустити симуляцію",
                   command=self.run).pack(fill="x", padx=4, pady=6)

        self.status_var = tk.StringVar(value="Готово до запуску.")
        ttk.Label(btn, textvariable=self.status_var,
                  wraplength=270, foreground="gray",
                  justify="left").pack(anchor="w", padx=4, pady=(0, 6))

        self._on_mob_change()

    def _on_mob_change(self):
        dynamic = self.mobility_var.get() == "dynamic"
        st = "normal" if dynamic else "disabled"
        self.speed_entry.configure(state=st)
        self.step_entry.configure(state=st)

    # ── Запуск симуляції ──────────────────────────────────────────────────────

    def run(self):
        self._status("Симуляція запускається...", "blue")
        try:
            self._do_run()
        except Exception as exc:
            messagebox.showerror("Помилка", str(exc))
            self._status(f"Помилка: {exc}", "red")

    def _status(self, msg: str, color: str = "gray"):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _do_run(self):
        scenario_key = self.scenario_var.get()
        topology     = self.topology_var.get()
        n_nodes      = self.nodes_var.get()
        n_packets    = self.packets_var.get()
        n_obs        = self.obstacles_var.get()
        routing      = self.routing_var.get()
        is_dynamic   = self.mobility_var.get() == "dynamic"
        speed        = self.speed_var.get()
        step_s       = self.step_var.get()
        pdr_win      = self.pdr_window_var.get()

        scenario_name, _, obs_profile = SCENARIO_DEFS[scenario_key]

        scene = SceneBuilder(GATEWAY_POS)
        scene_cfg = SceneConfig(
            field_size=BASE_PHY["field_size"],
            obstacles_count=n_obs,
            nodes_count=n_nodes,
            obstacle_profile=obs_profile,
            node_profile="uniform",
        )
        self.obstacles = scene.generate_obstacles(scene_cfg)
        nodes = scene.generate_nodes(scene_cfg)
        if len(nodes) < 2:
            raise ValueError(
                "Не вдалось розмістити вузли — зменшіть кількість перешкод."
            )

        cfg = SimulationConfig(**{
            **BASE_PHY,
            "packets_per_node":  n_packets,
            "routing_model":     routing,
            "mobility_enabled":  is_dynamic,
            "mobility_speed":    speed,
            "mobility_step_s":   step_s,
        })

        engine = LoRaSimulationEngine(
            config=cfg,
            gateway_pos=GATEWAY_POS,
            nodes=nodes,
            is_blocked_cb=scene.is_blocked,
            is_point_in_obstacle_cb=scene.is_point_in_obstacle,
        )

        n_total    = len(nodes) + 1
        velocities = engine.sample_velocities(n_total)

        modes = []
        if topology in ("star", "обидві"):
            modes.append("star")
        if topology in ("mesh", "обидві"):
            modes.append("mesh")

        results = {}
        for idx, mode in enumerate(modes):
            self._status(f"Запуск {mode.upper()} ({idx + 1}/{len(modes)})...")
            res = engine.simulate(
                mode, scenario_name,
                collect_trace=True,
                max_trace_steps=3000,
                velocities=velocities.copy(),
            )
            results[mode] = res

        self._status("Побудова графіків...")
        self._render(results, scenario_key, topology, pdr_win, is_dynamic)

        summary = "  |  ".join(
            f"{m.upper()} PDR={r['pdr']:.1f}%  "
            f"затр.={r['avg_delay']:.2f}с  "
            f"ретр.={r['avg_retry']:.1f}"
            for m, r in results.items()
        )
        self._status(summary, "darkgreen")

    # ── Рендер ───────────────────────────────────────────────────────────────

    def _render(self, results: dict, scenario_key: str, topology: str,
                pdr_win: int, is_dynamic: bool):

        for w in self.plot_frame.winfo_children():
            w.destroy()
        if self.fig:
            plt.close(self.fig)

        # Визначаємо активні панелі (крім карти, яка завжди є)
        extras = []
        if self.show_pdr.get():
            extras.append("pdr")
        if self.show_conn.get():
            extras.append("conn")
        if self.show_hist.get():
            extras.append("hist")

        show_traj = self.show_traj.get() and is_dynamic

        # ── Сітка підграфіків ─────────────────────────────────────────────────
        # Карта завжди займає першу колонку (всю висоту).
        # Решта панелей — у правій колонці, вертикально.
        n_extra = len(extras)
        n_right_rows = max(n_extra, 1)

        pw = max(self.plot_frame.winfo_width(),  900)
        ph = max(self.plot_frame.winfo_height(), 640)
        self.fig = plt.figure(figsize=(pw / 96, ph / 96), dpi=96)

        if n_extra == 0:
            gs = gridspec.GridSpec(1, 1, figure=self.fig)
            ax_map = self.fig.add_subplot(gs[0, 0])
            axes_extra = {}
        else:
            gs = gridspec.GridSpec(
                n_right_rows, 2,
                figure=self.fig,
                wspace=0.38,
                hspace=0.42,
                left=0.06, right=0.97,
                top=0.93,  bottom=0.08,
            )
            ax_map = self.fig.add_subplot(gs[:, 0])
            axes_extra = {
                key: self.fig.add_subplot(gs[i, 1])
                for i, key in enumerate(extras)
            }

        # ── Карта ─────────────────────────────────────────────────────────────
        self._draw_map(ax_map, results, show_traj)

        mode_colors = {"star": STAR_COLOR, "mesh": MESH_COLOR}
        mode_labels = {
            "star": "Зірка (LoRaWAN)",
            "mesh": "Сітка (LoRa Mesh)",
        }

        # ── PDR у часі ────────────────────────────────────────────────────────
        if "pdr" in axes_extra:
            ax = axes_extra["pdr"]
            for mode, res in results.items():
                t, roll, cum = extract_pdr_series(res["trace"], pdr_win)
                if len(t) == 0:
                    continue
                ax.plot(t, roll, color=mode_colors[mode], alpha=0.45,
                        linewidth=1.2, label=f"{mode_labels[mode]} (ковзне)")
                ax.plot(t, cum,  color=mode_colors[mode], linewidth=2.0,
                        linestyle="--", label=f"{mode_labels[mode]} (нарост.)")
            ax.set_title("PDR у часі")
            ax.set_xlabel("Час (с)")
            ax.set_ylabel("PDR (%)")
            ax.set_ylim(-2, 107)
            ax.legend(fontsize=8, ncol=1)
            ax.grid(True, linestyle="--", alpha=0.4)

        # ── Зв'язність у часі ─────────────────────────────────────────────────
        if "conn" in axes_extra:
            ax = axes_extra["conn"]
            n_total_nodes = len(list(results.values())[0]["positions"]) - 1
            for mode, res in results.items():
                t, cnt = extract_connectivity_series(res["trace"], mode)
                if len(t) == 0:
                    continue
                ax.plot(t, cnt, color=mode_colors[mode], linewidth=2.0,
                        label=mode_labels[mode])
            ax.axhline(n_total_nodes, color="gray", linestyle=":",
                       linewidth=1.2, label=f"Всього вузлів ({n_total_nodes})")
            ax.set_title("Зв'язність мережі у часі")
            ax.set_xlabel("Час (с)")
            ax.set_ylabel("Підключених вузлів")
            ax.set_ylim(-0.5, n_total_nodes + 2)
            ax.legend(fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)

        # ── Гістограми: затримки та ретраї ────────────────────────────────────
        if "hist" in axes_extra:
            ax = axes_extra["hist"]
            # Ділимо вісь X на дві частини: зліва — затримки, справа — ретраї
            ax.set_visible(False)  # ховаємо основний axes
            # Вставляємо два підграфіки через inset
            bbox = ax.get_position()
            left_ax  = self.fig.add_axes([bbox.x0,
                                          bbox.y0,
                                          bbox.width * 0.47,
                                          bbox.height])
            right_ax = self.fig.add_axes([bbox.x0 + bbox.width * 0.53,
                                          bbox.y0,
                                          bbox.width * 0.47,
                                          bbox.height])

            for mode, res in results.items():
                delays, retries = extract_delays_retries(res["trace"])
                c = mode_colors[mode]
                lbl = mode_labels[mode]

                if len(delays):
                    left_ax.hist(delays, bins=14, alpha=0.6, color=c,
                                 edgecolor="white", linewidth=0.4, label=lbl)
                if len(retries):
                    max_r = int(retries.max()) + 1
                    bins  = np.arange(-0.5, max_r + 0.5)
                    right_ax.hist(retries, bins=bins, alpha=0.6, color=c,
                                  edgecolor="white", linewidth=0.4, label=lbl)

            left_ax.set_title("Затримки доставки")
            left_ax.set_xlabel("Затримка (с)")
            left_ax.set_ylabel("Кількість пакетів")
            left_ax.legend(fontsize=8)
            left_ax.grid(True, linestyle="--", alpha=0.4)

            right_ax.set_title("Кількість ретраїв")
            right_ax.set_xlabel("Ретраїв (0 = з першої спроби)")
            right_ax.set_ylabel("")
            right_ax.legend(fontsize=8)
            right_ax.grid(True, linestyle="--", alpha=0.4)

        # ── Заголовок і відображення ──────────────────────────────────────────
        scenario_ua = {"ideal": "Ideal", "noisy": "Noisy", "dense": "Dense"}
        mob_ua = "динамічна" if is_dynamic else "статична"
        self.fig.suptitle(
            f"LoRa симуляція | Сценарій: {scenario_ua.get(scenario_key)}  "
            f"| Мобільність: {mob_ua}  | Топологія: {topology}",
            fontsize=11,
        )

        toolbar_frame = ttk.Frame(self.plot_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)

        mpl_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        mpl_canvas.draw()
        mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        NavigationToolbar2Tk(mpl_canvas, toolbar_frame).update()
        self.mpl_canvas = mpl_canvas

    # ── Малювання карти ───────────────────────────────────────────────────────

    def _draw_map(self, ax, results: dict, show_traj: bool):
        """Просторова карта: перешкоди, траєкторії, фінальний стан топології."""
        fs = BASE_PHY["field_size"] / 2.0
        ax.set_aspect("equal")
        ax.set_xlim(-fs - 0.4, fs + 0.4)
        ax.set_ylim(-fs - 0.4, fs + 0.4)

        # Перешкоди
        for x, y, w, h in self.obstacles:
            ax.add_patch(Rectangle((x, y), w, h,
                                   color="#c0392b", alpha=0.28, zorder=1,
                                   linewidth=0.8, edgecolor="#922b21"))

        mode_colors = {"star": STAR_COLOR, "mesh": MESH_COLOR}
        mode_labels = {"star": "Зірка", "mesh": "Сітка"}
        # якщо обидві топології — трохи зменшуємо непрозорість щоб не зливались
        alpha_node = 0.80 if len(results) == 1 else 0.65

        for mode, res in results.items():
            color     = mode_colors[mode]
            positions = res["positions"]
            n         = len(positions)
            parent    = res["parent"]
            direct    = res["direct_links"]
            trace     = res["trace"]

            # ── Траєкторії руху ───────────────────────────────────────────────
            if show_traj:
                paths = extract_trajectories(trace)
                for node_id, pts in paths.items():
                    if len(pts) < 2:
                        continue
                    xs = [p[1] for p in pts]
                    ys = [p[2] for p in pts]
                    k  = len(xs)
                    # Від блідого (старт) до насиченого (фініш)
                    for j in range(k - 1):
                        t_ratio = j / max(k - 2, 1)
                        alpha   = 0.05 + 0.40 * t_ratio
                        lw      = 0.6 + 0.7 * t_ratio
                        ax.plot([xs[j], xs[j + 1]], [ys[j], ys[j + 1]],
                                color=color, alpha=alpha, linewidth=lw, zorder=2)
                    # Точка старту
                    ax.scatter(xs[0], ys[0], s=12, color=color, alpha=0.3,
                               marker="o", zorder=3, linewidths=0)

            # ── Ребра топології (фінальний стан) ──────────────────────────────
            if mode == "mesh":
                for i in range(1, n):
                    p = int(parent[i])
                    if p != -1:
                        ax.plot(
                            [positions[i, 0], positions[p, 0]],
                            [positions[i, 1], positions[p, 1]],
                            color=color, alpha=0.35, linewidth=1.3, zorder=3,
                        )
            else:  # star
                for i in range(1, n):
                    if direct[i]:
                        ax.plot(
                            [positions[i, 0], positions[0, 0]],
                            [positions[i, 1], positions[0, 1]],
                            color=color, alpha=0.20, linewidth=0.9,
                            linestyle="--", zorder=3,
                        )

            # ── Фінальні позиції вузлів ───────────────────────────────────────
            if mode == "mesh":
                node_clr = [
                    "#27ae60" if parent[i] != -1 else "#e74c3c"
                    for i in range(1, n)
                ]
            else:
                node_clr = [
                    "#2980b9" if direct[i] else "#e74c3c"
                    for i in range(1, n)
                ]

            ax.scatter(
                positions[1:, 0], positions[1:, 1],
                c=node_clr, s=40, edgecolors="white", linewidths=0.6,
                alpha=alpha_node, zorder=5,
                label=f"{mode_labels[mode]}  PDR={res['pdr']:.1f}%",
            )

        # Gateway
        ax.scatter(
            GATEWAY_POS[0], GATEWAY_POS[1],
            s=280, c="gold", marker="*",
            edgecolors="black", linewidths=0.8, zorder=10, label="Gateway",
        )

        ax.set_title("Карта мережі" + (" + Траєкторії" if show_traj else ""),
                     fontsize=11)
        ax.set_xlabel("Відстань (км)")
        ax.set_ylabel("Відстань (км)")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.9)


# ── Точка входу ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app  = DynamicVisualizer(root)
    root.mainloop()
