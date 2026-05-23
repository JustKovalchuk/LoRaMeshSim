"""
generate_report.py — Автоматичний запуск симуляцій та побудова графіків
для практичної частини курсової роботи.

Генерує 4 файли у папці report_figures/:
  fig3_6_ideal_pdr.png  — PDR для сценарію Ideal  (Рис. 3.6)
  fig3_7_noisy_pdr.png  — PDR для сценарію Noisy  (Рис. 3.7)
  fig3_8_dense_pdr.png  — PDR для сценарію Dense  (Рис. 3.8)
  fig3_9_energy.png     — Енергоспоживання        (Рис. 3.9)

Використання:
  python generate_report.py
  python generate_report.py --runs 10      # більше ітерацій (точніше, повільніше)
  python generate_report.py --fast         # швидкий тест (2 ітерації, 5 пакетів)
  python generate_report.py --nodes "10,20,30,40,50"
"""

import argparse
import io
import os
import sys
import time

# ── Примусове UTF-8 для консолі Windows ──────────────────────────────────────
# Без цього PowerShell (cp1251) не може вивести кирилицю.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")  # без GUI — рендеримо лише у файли

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from scene_builder import SceneBuilder, SceneConfig
from simulation_core import LoRaSimulationEngine, SimulationConfig

# ── Параметри за замовчуванням ────────────────────────────────────────────────

OUTPUT_DIR = "report_figures"

# Кількість вузлів для дослідження масштабування (ТЗ: 10–50)
NODE_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

DEFAULT_RUNS    = 5   # незалежних запусків для усереднення на точку
DEFAULT_PACKETS = 20  # пакетів на вузол за симуляцію

GATEWAY_POS = np.array([0.0, 0.0])

# Базові фізичні параметри LoRa SX1276 (відповідають main.py)
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
    field_size=6.0,
    link_model="realistic",
    routing_model="balanced",
    packet_interval_s=60.0,
    max_retries=2,
    mobility_speed=0.004,
    mobility_step_s=3.0,
    routing_update_s=3.0,
    unit_mode="mAh",
    mobility_enabled=False,
)

# Визначення сценаріїв: (назва_сценарію, к-ть_перешкод, профіль_перешкод)
#   ideal — відкритий простір, жодних перешкод
#   noisy — кілька перешкод + деградація каналу
#   dense — щільна міська забудова (NLoS), багато будівель
SCENARIO_DEFS = {
    "ideal": ("ideal",  20, "mixed"),
    "noisy": ("noisy",  20, "mixed"),
    "dense": ("dense", 20, "mixed"),
}

# Підписи та кольори для легенди графіків
STAR_COLOR = "#1565C0"        # синій — топологія Зірка
MESH_COLOR = "#2E7D32"        # зелений — топологія Сітка
STAR_LABEL = "Зірка (LoRaWAN)"
MESH_LABEL = "Сітка (LoRa Mesh)"

# Стиль для академічних публікаційних графіків
PLOT_RC = {
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "legend.fontsize":   11,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "lines.linewidth":   2.2,
    "lines.markersize":  7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


# ── Допоміжні функції ─────────────────────────────────────────────────────────

def make_config(packets_per_node: int) -> SimulationConfig:
    return SimulationConfig(**{**BASE_PHY, "packets_per_node": packets_per_node})


def run_single(scenario_key: str, n_nodes: int, packets_per_node: int):
    """
    Один запуск симуляції (star + mesh) для заданого сценарію і кількості вузлів.

    Повертає (star_pdr, mesh_pdr, star_energy_mah, mesh_energy_mah)
    або None при помилці (не вдалося розмістити вузли).
    """
    scenario_name, obstacles_count, obs_profile = SCENARIO_DEFS[scenario_key]

    scene = SceneBuilder(GATEWAY_POS)
    scene_cfg = SceneConfig(
        field_size=BASE_PHY["field_size"],
        obstacles_count=obstacles_count,
        nodes_count=n_nodes,
        obstacle_profile=obs_profile,
        node_profile="uniform",
    )
    scene.generate_obstacles(scene_cfg)
    nodes = scene.generate_nodes(scene_cfg)

    if len(nodes) < 2:
        return None

    cfg    = make_config(packets_per_node)
    engine = LoRaSimulationEngine(
        config=cfg,
        gateway_pos=GATEWAY_POS,
        nodes=nodes,
        is_blocked_cb=scene.is_blocked,
        is_point_in_obstacle_cb=scene.is_point_in_obstacle,
    )

    n_total    = len(nodes) + 1
    velocities = engine.sample_velocities(n_total)

    star = engine.simulate("star", scenario_name, velocities=velocities.copy())
    mesh = engine.simulate("mesh", scenario_name, velocities=velocities.copy())

    # Сумарна енергія вузлів (gateway — індекс 0 — не враховується)
    star_e = float(np.sum(star["energy"][1:]))
    mesh_e = float(np.sum(mesh["energy"][1:]))

    return star["pdr"], mesh["pdr"], star_e, mesh_e


def collect_scenario_data(scenario_key: str, runs: int, packets: int) -> dict:
    """
    Збирає усереднені дані для всіх кількостей вузлів заданого сценарію.
    Повертає dict із масивами mean/std для PDR та енергії.
    """
    records: dict = {
        "star_pdr": [], "mesh_pdr": [],
        "star_pdr_std": [], "mesh_pdr_std": [],
        "star_energy": [], "mesh_energy": [],
        "star_energy_std": [], "mesh_energy_std": [],
    }

    total_runs = len(NODE_COUNTS) * runs

    for ni, n in enumerate(NODE_COUNTS):
        s_pdrs, m_pdrs, s_es, m_es = [], [], [], []

        for r in range(runs):
            done = ni * runs + r + 1
            print(
                f"  [{scenario_key.upper()}] вузлів={n:2d}  "
                f"запуск {r + 1}/{runs}  (загалом {done}/{total_runs})",
                end="",
                flush=True,
            )
            t0      = time.monotonic()
            result  = run_single(scenario_key, n, packets)
            elapsed = time.monotonic() - t0

            if result is None:
                print(f"  [!] пропущено ({elapsed:.1f}с)")
                continue

            s_pdr, m_pdr, s_e, m_e = result
            s_pdrs.append(s_pdr)
            m_pdrs.append(m_pdr)
            s_es.append(s_e)
            m_es.append(m_e)
            print(f"  Зірка={s_pdr:.1f}%  Сітка={m_pdr:.1f}%  ({elapsed:.1f}с)")

        records["star_pdr"].append(np.mean(s_pdrs)     if s_pdrs else 0.0)
        records["mesh_pdr"].append(np.mean(m_pdrs)     if m_pdrs else 0.0)
        records["star_pdr_std"].append(np.std(s_pdrs)  if s_pdrs else 0.0)
        records["mesh_pdr_std"].append(np.std(m_pdrs)  if m_pdrs else 0.0)
        records["star_energy"].append(np.mean(s_es)    if s_es else 0.0)
        records["mesh_energy"].append(np.mean(m_es)    if m_es else 0.0)
        records["star_energy_std"].append(np.std(s_es) if s_es else 0.0)
        records["mesh_energy_std"].append(np.std(m_es) if m_es else 0.0)

    return {k: np.array(v) for k, v in records.items()}


# ── Побудова графіків ─────────────────────────────────────────────────────────

def _apply_pdr_axes(ax, title: str):
    """Спільне форматування осей для PDR-графіків."""
    ax.set_title(title, pad=10)
    ax.set_xlabel("Кількість вузлів")
    ax.set_ylabel("PDR (%)")
    ax.set_xlim(NODE_COUNTS[0] - 2, NODE_COUNTS[-1] + 2)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.legend(loc="lower left")


def plot_pdr(data: dict, scenario_key: str, fig_label: str, out_path: str):
    """Рис. 3.6 / 3.7 / 3.8 — PDR залежно від кількості вузлів."""
    scenario_ua = {
        "ideal": "Ideal (ідеальні умови)",
        "noisy": "Noisy (зашумлене середовище)",
        "dense": "Dense (щільна забудова, NLoS)",
    }
    title = (
        f"Рис. {fig_label}. Залежність PDR від кількості вузлів\n"
        f"Сценарій: {scenario_ua.get(scenario_key, scenario_key)}"
    )

    x      = np.array(NODE_COUNTS)
    s_mean = data["star_pdr"]
    m_mean = data["mesh_pdr"]
    s_std  = data["star_pdr_std"]
    m_std  = data["mesh_pdr_std"]

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(x, s_mean, "o-", color=STAR_COLOR, label=STAR_LABEL, zorder=3)
        ax.plot(x, m_mean, "s-", color=MESH_COLOR, label=MESH_LABEL, zorder=3)

        # Смуга ±1σ (розкид між запусками)
        ax.fill_between(x, s_mean - s_std, s_mean + s_std,
                        color=STAR_COLOR, alpha=0.12)
        ax.fill_between(x, m_mean - m_std, m_mean + m_std,
                        color=MESH_COLOR, alpha=0.12)

        _apply_pdr_axes(ax, title)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"  [OK] Збережено: {out_path}")


def plot_energy(all_data: dict, out_path: str):
    """
    Рис. 3.9 — Загальне енергоспоживання мережі залежно від кількості вузлів.
    Показує всі три сценарії на одному графіку:
      суцільні лінії = Сітка, пунктирні = Зірка; колір = сценарій.
    """
    title = (
        "Рис. 3.9. Порівняння загального енергоспоживання мережі\n"
        "залежно від кількості вузлів"
    )

    # Кольори та підписи для кожного сценарію
    scenario_styles = {
        "ideal": ("#1565C0", "Ideal"),   # синій
        "noisy": ("#E65100", "Noisy"),   # помаранчевий
        "dense": ("#6A1B9A", "Dense"),   # фіолетовий
    }

    x = np.array(NODE_COUNTS)

    # Авто-масштаб за максимальним значенням серед усіх сценаріїв
    all_max = max(
        float(np.max(d["mesh_energy"])) for d in all_data.values() if len(d["mesh_energy"])
    )
    scale, unit = (1000.0, "мкАг") if all_max * 1000.0 <= 500 else (1.0, "мАг")

    with plt.rc_context(PLOT_RC):
        fig, ax = plt.subplots(figsize=(9, 5))

        for scenario_key, (color, label_ua) in scenario_styles.items():
            if scenario_key not in all_data:
                continue
            data = all_data[scenario_key]

            sm = data["star_energy"] * scale
            mm = data["mesh_energy"] * scale
            ss = data["star_energy_std"] * scale
            ms = data["mesh_energy_std"] * scale

            # Зірка — пунктир, Сітка — суцільна
            ax.plot(x, sm, "o--", color=color, alpha=0.75,
                    label=f"Зірка — {label_ua}", zorder=3)
            ax.plot(x, mm, "s-",  color=color,
                    label=f"Сітка — {label_ua}", zorder=3)

            ax.fill_between(x, sm - ss, sm + ss, color=color, alpha=0.07)
            ax.fill_between(x, mm - ms, mm + ms, color=color, alpha=0.07)

        ax.set_title(title, pad=10)
        ax.set_xlabel("Кількість вузлів")
        ax.set_ylabel(f"Загальне споживання ({unit})")
        ax.set_xlim(NODE_COUNTS[0] - 2, NODE_COUNTS[-1] + 2)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.legend(loc="upper left", ncol=2, fontsize=9)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    print(f"  [OK] Збережено: {out_path}")


# ── Таблиця статистики ────────────────────────────────────────────────────────

def print_summary(all_data: dict):
    """Виводить зведену таблицю результатів у консоль."""
    sep = "-" * 80
    print()
    print(sep)
    print("ЗВЕДЕНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ СИМУЛЯЦІЇ")
    print(sep)
    print(
        f"{'Сценарій':<8} {'Вузлів':>6}  "
        f"{'PDR Зірка':>10} {'ств':>5}  "
        f"{'PDR Сітка':>10} {'ств':>5}  "
        f"{'Дельта':>8}"
    )
    print(sep)

    for scenario_key, data in all_data.items():
        for i, n in enumerate(NODE_COUNTS):
            s     = data["star_pdr"][i]
            m     = data["mesh_pdr"][i]
            ss    = data["star_pdr_std"][i]
            ms    = data["mesh_pdr_std"][i]
            delta = m - s
            sign  = "+" if delta >= 0 else ""
            print(
                f"{scenario_key.upper():<8} {n:>6}  "
                f"{s:>9.1f}% {ss:>4.1f}  "
                f"{m:>9.1f}% {ms:>4.1f}  "
                f"{sign}{delta:>7.1f}%"
            )
        print()

    print(sep)
    print()
    print("Спостереження:")
    for scenario_key, data in all_data.items():
        avg_star = np.mean(data["star_pdr"])
        avg_mesh = np.mean(data["mesh_pdr"])
        delta    = avg_mesh - avg_star
        sign     = "+" if delta >= 0 else ""
        adv      = "Сітка краща" if delta > 0 else "Зірка краща"
        print(
            f"  {scenario_key.upper():<6}: сер. Зірка={avg_star:.1f}%  "
            f"сер. Сітка={avg_mesh:.1f}%  "
            f"D={sign}{delta:.1f}%  -> {adv}"
        )

    idx30 = NODE_COUNTS.index(30) if 30 in NODE_COUNTS else len(NODE_COUNTS) // 2
    print()
    print(f"Енергоспоживання (при {NODE_COUNTS[idx30]} вузлах):")
    print(f"  {'Сценарій':<8}  {'Зірка (мкАг)':>14}  {'Сітка (мкАг)':>14}  {'Множник':>9}")
    for scenario_key, data in all_data.items():
        se    = data["star_energy"][idx30] * 1000
        me    = data["mesh_energy"][idx30] * 1000
        ratio = me / se if se > 0 else float("inf")
        print(f"  {scenario_key.upper():<8}  {se:>14.3f}  {me:>14.3f}  {ratio:>8.1f}x")
    print(sep)


# ── Головна функція ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Генератор графіків для курсової: LoRa Зірка vs Сітка"
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"Кількість ітерацій для усереднення (за замовч. {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--packets", type=int, default=DEFAULT_PACKETS,
        help=f"Пакетів на вузол за симуляцію (за замовч. {DEFAULT_PACKETS})",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Швидкий тест: 2 ітерації, 5 пакетів",
    )
    parser.add_argument(
        "--nodes", type=str, default=None,
        help="Список кількостей вузлів через кому, напр. '10,20,30,40,50'",
    )
    args = parser.parse_args()

    runs    = 2 if args.fast else args.runs
    packets = 5 if args.fast else args.packets

    global NODE_COUNTS
    if args.nodes:
        NODE_COUNTS = [int(x.strip()) for x in args.nodes.split(",")]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  LoRa Зірка vs Сітка — Генератор звіту")
    print("=" * 60)
    print(f"  Вузлів:          {NODE_COUNTS}")
    print(f"  Ітерацій:        {runs}")
    print(f"  Пакетів/вузол:   {packets}")
    print(f"  Вихідна папка:   {OUTPUT_DIR}/")
    print("=" * 60)

    t_start  = time.monotonic()
    all_data = {}

    scenarios_info = [
        ("ideal", "3.6", "fig3_6_ideal_pdr.png"),
        ("noisy", "3.7", "fig3_7_noisy_pdr.png"),
        ("dense", "3.8", "fig3_8_dense_pdr.png"),
    ]

    for scenario_key, fig_label, fname in scenarios_info:
        print()
        print("-" * 60)
        print(f"  Збір даних: сценарій {scenario_key.upper()}")
        print("-" * 60)
        data = collect_scenario_data(scenario_key, runs, packets)
        all_data[scenario_key] = data

        out_path = os.path.join(OUTPUT_DIR, fname)
        plot_pdr(data, scenario_key, fig_label, out_path)

    # Рис. 3.9 — Енергоспоживання (всі три сценарії)
    print()
    print("-" * 60)
    print("  Побудова Рис. 3.9 — Енергоспоживання")
    print("-" * 60)
    plot_energy(all_data, os.path.join(OUTPUT_DIR, "fig3_9_energy.png"))

    print_summary(all_data)

    elapsed = time.monotonic() - t_start
    print(f"\n  Загальний час: {elapsed:.0f}с ({elapsed / 60:.1f} хв)")
    print(f"  Файли збережено у: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
