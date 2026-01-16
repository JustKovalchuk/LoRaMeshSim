import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle
from collections import deque
import math
import random

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
        self.i_rx_lna_on = 0.0115  # LnaBoost On (типово для стабільного зв'язку)
        self.i_rx_lna_off = 0.0108 # LnaBoost Off

        # Параметри протоколу
        self.sf = 7               # Spreading Factor
        self.bw = 125000          # Bandwidth (Гц)
        self.cr = 1               # Coding Rate (1=4/5)
        self.payload_len = 20     # Розмір повідомлення (байтів)
        self.preamble_len = 8     # Довжина преамбули

        # Налаштування поля та вузлів
        self.unit_mode = tk.StringVar(value="Joules")
        self.nodes_count = tk.IntVar(value=30)
        self.obstacles_count = tk.IntVar(value=4)
        self.r_max = tk.DoubleVar(value=2.0)
        self.field_size = tk.DoubleVar(value=6.0)
        self.packets_per_node = 100
        self.gateway_pos = np.array([0.0, 0.0])
        self.nodes = None
        self.obstacles = []
        
        self.setup_ui()
        self.generate_obstacles()
        self.generate_nodes()

    def calculate_toa(self, crc_enabled=True, implicit_header=False):
        t_symbol = (2**self.sf) / self.bw
        de = 1 if (t_symbol > 0.016) else 0 # Low Data Rate Optimization
        ih = 1 if implicit_header else 0
        crc = 1 if crc_enabled else 0

        payload_symb_nb = 8 + max(
            math.ceil(
                (8 * self.payload_len - 4 * self.sf + 28 + 16 * crc - 20 * ih) / 
                (4 * (self.sf - 2 * de))
            ) * (self.cr + 4), 0
        )
        t_preamble = (self.preamble_len + 4.25) * t_symbol
        t_payload = payload_symb_nb * t_symbol
        return t_preamble + t_payload
    
    def line_intersects_rect(self, p1, p2, rect):
        """Перевірка чи перетинає відрізок p1-p2 прямокутник rect (x, y, w, h)"""
        x, y, w, h = rect
        # Грані прямокутника
        rect_lines = [
            ((x, y), (x + w, y)),
            ((x + w, y), (x + w, y + h)),
            ((x + w, y + h), (x, y + h)),
            ((x, y + h), (x, y))
        ]
        
        for r_p1, r_p2 in rect_lines:
            if self.segments_intersect(p1, p2, r_p1, r_p2):
                return True
        return False
    
    def generate_obstacles(self):
        """Генерує випадкові прямокутні перешкоди"""
        self.obstacles = []
        size = self.field_size.get()
        target_count = self.obstacles_count.get()

        # Функція для перевірки перетину двох прямокутників
        def rects_overlap(r1, r2):
            # r = (x, y, w, h)
            return not (r1[0] + r1[2] < r2[0] or 
                        r1[0] > r2[0] + r2[2] or 
                        r1[1] + r1[3] < r2[1] or 
                        r1[1] > r2[1] + r2[3])
        
        max_attempts = 1000
        attempts = 0

        while len(self.obstacles) < target_count and attempts < max_attempts:
            attempts += 1
            # Випадково обираємо тип: 0 - будинок (квадратний), 1 - стіна (довга)
            obs_type = random.randint(0, 1)
            
            if obs_type == 0:
                w = random.uniform(size * 0.05, size * 0.1)
                h = random.uniform(size * 0.05, size * 0.1)
            else:
                if random.random() > 0.5:
                    w = random.uniform(size * 0.1, size * 0.4)
                    h = random.uniform(0.05, 0.05)
                else:
                    w = random.uniform(0.05, 0.05)
                    h = random.uniform(size * 0.1, size * 0.4)

            x = random.uniform(-size/2, size/2 - w)
            y = random.uniform(-size/2, size/2 - h)
            new_rect = (x, y, w, h)
            
            margin = 0.3
            gw_rect = (self.gateway_pos[0] - margin, self.gateway_pos[1] - margin, margin*2, margin*2)
            if rects_overlap(new_rect, gw_rect):
                continue
            
            gap = 0.1
            collision = False
            for obs in self.obstacles:
                if rects_overlap((x-gap, y-gap, w+gap*2, h+gap*2), obs):
                    collision = True
                    break
            
            if not collision:
                self.obstacles.append(new_rect)
        
        if self.nodes is not None:
            self.draw_network()

    def clear_obstacles(self):
        self.obstacles = []
        
        self.obstacles_count.set(0)
        self.draw_network()
    
    def segments_intersect(self, a, b, c, d):
        """Перевірка перетину двох відрізків AB та CD"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(a,c,d) != ccw(b,c,d) and ccw(a,b,c) != ccw(a,b,d)

    def is_blocked(self, p1, p2):
        """Перевірка чи заблокований шлях будь-якою перешкодою"""
        for obs in self.obstacles:
            if self.line_intersects_rect(p1, p2, obs):
                return True
        return False

    def setup_ui(self):
        ctrl_frame = ttk.LabelFrame(self.root, text="Параметри симуляції (LoRa SX1276)")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(ctrl_frame, text="Кількість перешкод:").pack()
        ttk.Entry(ctrl_frame, textvariable=self.obstacles_count).pack(pady=2)

        ttk.Label(ctrl_frame, text="Кількість вузлів:").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.nodes_count).pack()

        ttk.Label(ctrl_frame, text="Радіус зв'язку R_max (км):").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.r_max).pack()

        ttk.Label(ctrl_frame, text="Розмір поля (км):").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.field_size).pack()

        ttk.Separator(ctrl_frame, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(ctrl_frame, text="Одиниці вимірювання енергії:").pack(pady=5)
        ttk.Radiobutton(ctrl_frame, text="Джоулі (J/mJ)", variable=self.unit_mode, value="Joules").pack(anchor=tk.W)
        ttk.Radiobutton(ctrl_frame, text="Ампер-години (mAh)", variable=self.unit_mode, value="mAh").pack(anchor=tk.W)
        
        ttk.Separator(ctrl_frame, orient='horizontal').pack(fill='x', pady=10)

        toa = self.calculate_toa() * 1000 # в мс
        ttk.Label(ctrl_frame, text=f"SF: {self.sf} | BW: {self.bw/1000}kHz", foreground="darkgreen").pack()
        ttk.Label(ctrl_frame, text=f"Packet ToA: {toa:.2f} ms").pack()
        
        ttk.Button(ctrl_frame, text="Генерувати перешкоди", command=self.generate_obstacles).pack(fill='x', pady=5)
        ttk.Button(ctrl_frame, text="Очистити перешкоди", command=self.clear_obstacles).pack(fill='x', pady=2)
        ttk.Button(ctrl_frame, text="Генерувати вузли", command=self.generate_nodes).pack(fill='x', pady=5)
        ttk.Button(ctrl_frame, text="Запустити симуляцію", command=self.run_simulation).pack(fill='x', pady=5)

        ttk.Label(ctrl_frame, text="* Клікніть на мапі,\nщоб переставити Gateway", foreground="blue").pack(pady=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.gateway_pos = np.array([event.xdata, event.ydata])
            self.draw_network()
    
    def is_point_in_obstacle(self, x, y):
        for ox, oy, ow, oh in self.obstacles:
            if ox <= x <= ox + ow and oy <= y <= oy + oh:
                return True
        return False
    
    def generate_nodes(self):
        size = self.field_size.get()
        target_count = self.nodes_count.get()
        valid_nodes = []

        max_attempts = target_count * 100
        attempts = 0
        
        while len(valid_nodes) < target_count and attempts < max_attempts:
            attempts += 1
            x = random.uniform(-size/2, size/2)
            y = random.uniform(-size/2, size/2)
            
            if not self.is_point_in_obstacle(x, y):
                valid_nodes.append([x, y])
        
        self.nodes = np.array(valid_nodes)
        
        if len(self.nodes) < target_count:
            print(f"Попередження: вдалося розмістити лише {len(self.nodes)} вузлів через щільність перешкод.")
            
        self.draw_network()

    def draw_network(self):
        self.ax.clear()
        for x, y, w, h in self.obstacles:
            self.ax.add_patch(Rectangle((x, y), w, h, color='red', alpha=0.3))
        
        self.ax.scatter(self.nodes[:,0], self.nodes[:,1], c='gray', alpha=0.5, label='Вузли')
        self.ax.scatter(self.gateway_pos[0], self.gateway_pos[1], c='gold', s=200, marker='*', edgecolors='black', label='Gateway')
        self.ax.set_title("Попередній перегляд структури мережі з перешкодами")
        self.ax.grid(True, linestyle=':')
        self.canvas.draw()

    def run_simulation(self):
        r = self.r_max.get()
        all_pts = np.vstack([self.gateway_pos, self.nodes])
        n = len(all_pts)
        
        # Матриця відстаней та суміжності (Unit Disk Graph)
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(all_pts[i] - all_pts[j])
                if dist <= r:
                    if not self.is_blocked(all_pts[i], all_pts[j]):
                        adj[i,j] = adj[j,i] = 1
        
        # print("Матриця суміжності:\n", adj)

        # Розрахунок мінімальної кількості хопів через BFS
        hops = np.full(n, -1)
        parent = np.full(n, -1)
        hops[0] = 0
        q = deque([0])
        while q:
            u = q.popleft()
            for v, conn in enumerate(adj[u]):
                if conn and hops[v] == -1:
                    hops[v] = hops[u] + 1
                    parent[v] = u
                    q.append(v)
        
        # Розрахунок PDR для Star (тільки пряма видимість до GW)
        star_success_packets = 0
        for i in range(1, n):
            dist = np.linalg.norm(all_pts[i] - self.gateway_pos)
            if dist <= r and not self.is_blocked(all_pts[i], self.gateway_pos):
                star_success_packets += self.packets_per_node
        
        # Розрахунок енергії
        t_packet = self.calculate_toa()

        i_tx_active = self.tx_currents.get(self.current_tx_power, 0.087)
        i_rx_active = self.i_rx_lna_on

        if self.unit_mode.get() == "Joules":
            # Енергія в Джоулях: E = U * I * t
            unit_tx = self.v_supply * i_tx_active * t_packet
            unit_rx = self.v_supply * i_rx_active * t_packet
            unit_name = "Дж"
            display_mult = 1000 # для мДж
            display_name = "мДж"
        else:
            # Заряд в Ампер-годинах: Q = (I * t) / 3600
            # множимо на 1000 для переведення в mAh
            unit_tx = (i_tx_active * t_packet * 1000) / 3600
            unit_rx = (i_rx_active * t_packet * 1000) / 3600
            unit_name = "mAh"
            display_mult = 1 # залишаємо в mAh
            display_name = "mAh"
        
        # Симуляція трафіку, навантаження та енергоспоживання
        mesh_energy = np.zeros(n) # Енергоспоживання на кожному вузлі MESH
        star_energy = np.zeros(n) # Енергоспоживання на кожному вузлі STAR
        load = np.zeros(n) # Кількість оброблених пакетів на кожному вузлі
        total_packets = (n - 1) * self.packets_per_node # Сумарна к-ть згенерованих пакетів
        mesh_success_packets = 0
        
        for node_idx in range(1, n):
            star_energy[node_idx] = self.packets_per_node * unit_tx
            mesh_energy[node_idx] += self.packets_per_node * unit_tx
            
            can_reach_gateway = hops[node_idx] != -1
            if can_reach_gateway:
                mesh_success_packets += self.packets_per_node
                curr = node_idx
                load[curr] += self.packets_per_node
                p = parent[curr]
                while p != 0: # Поки не дійшли до Gateway
                    # Ретранслятор спочатку приймає, потім передає
                    mesh_energy[p] += self.packets_per_node * (unit_rx + unit_tx)
                    load[p] += self.packets_per_node
                    curr = p
                    p = parent[curr]
            else:
                pass
        
        self.show_results(hops, r, star_success_packets, mesh_success_packets, total_packets, star_energy, mesh_energy, load, parent, adj)

    def show_results(self, hops, r, star_success_packets, mesh_success_packets, total_packets, star_energy, mesh_energy, load, parent, adj):
        res_win = tk.Toplevel(self.root)
        res_win.title("Результати роботи мережі LoRa Star vs LoRa Mesh (SX1276)")
        
        star_pdr = (star_success_packets / total_packets) * 100 if total_packets > 0 else 0
        mesh_pdr = (mesh_success_packets / total_packets) * 100 if total_packets > 0 else 0
        
        # Пошук найбільш завантажених вузлів
        relay_loads = load[1:].copy()
        top_relay_indices = np.argsort(relay_loads)[-3:][::-1] + 1 
        
        # Розрахунок середньої енергії
        avg_mesh_energy = np.mean(mesh_energy[1:]) if len(mesh_energy) > 1 else 0
        total_mesh_energy = np.sum(mesh_energy[1:])

        avg_star_energy = np.mean(star_energy[1:]) if len(star_energy) > 1 else 0
        total_star_energy = np.sum(star_energy[1:])

        # 1. Інтерфейс: Ліва панель
        info_frame = ttk.Frame(res_win)
        info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
        
        unit_label = "Дж" if self.unit_mode.get() == "Joules" else "mAh"
        sub_unit_label = "мДж" if self.unit_mode.get() == "Joules" else "mAh"
        mult = 1000 if self.unit_mode.get() == "Joules" else 1
        
        results_text = f"""
=== МЕТРИКИ ЕФЕКТИВНОСТІ ===

PDR (Доставка пакетів):
• Star (LoRaWAN): {star_pdr:.1f}%
• Mesh (LoRa):    {mesh_pdr:.1f}%

ПЕРЕВАГА MESH: {mesh_pdr - star_pdr:.1f}%

=== ЕНЕРГОСПОЖИВАННЯ SX1276 ({unit_label}) ===
Загальне:
• Star (LoRaWAN): {total_star_energy:.4f} {unit_label}
• Mesh (LoRa):    {total_mesh_energy:.4f} {unit_label}
Середнє на вузол:
• Star (LoRaWAN): {avg_star_energy * mult:.3f} {sub_unit_label}
• Mesh (LoRa):    {avg_mesh_energy * mult:.3f} {sub_unit_label}

Критичні вузли (для Mesh мережі):
1. Вузол #{top_relay_indices[0]}: {int(load[top_relay_indices[0]])} пак.
2. Вузол #{top_relay_indices[1]}: {int(load[top_relay_indices[1]])} пак.
3. Вузол #{top_relay_indices[2]}: {int(load[top_relay_indices[2]])} пак.
"""
        tk.Label(info_frame, text=results_text, justify=tk.LEFT, font=("Courier", 10), 
                background="#f8f9fa", relief="solid", padx=15, pady=15).pack(pady=10)
        
        ttk.Button(info_frame, text="Закрити", command=res_win.destroy).pack(pady=10)

        # Візуалізація шляхів
        fig_res, ax_res = plt.subplots(figsize=(6, 6))
        all_pts = np.vstack([self.gateway_pos, self.nodes])
        
        # Малюємо перешкоди
        for obs in self.obstacles:
            ax_res.add_patch(Rectangle((obs[0], obs[1]), obs[2], obs[3], color='red', alpha=0.4))

        # Малюємо лінії зв'язку
        for i in range(1, len(all_pts)):
            p = parent[i]
            if p != -1:
                ax_res.annotate(
                    "", 
                    xy=(all_pts[p,0], all_pts[p,1]),
                    xytext=(all_pts[i,0], all_pts[i,1]),
                    arrowprops=dict(
                        arrowstyle="->", 
                        color='green', 
                        alpha=0.3, 
                        lw=1.5,
                        connectionstyle="arc3"
                    ),
                    zorder=2)
                # ax_res.plot([all_pts[i,0], all_pts[j,0]], 
                #             [all_pts[i,1], all_pts[j,1]], 
                #             color='gray', linestyle='--', alpha=0.3, lw=1, zorder=2)
        
        max_energy = np.max(mesh_energy[1:]) if len(mesh_energy) > 1 else 1
        # Малюємо вузли: Синій - Star ok, Зелений - Mesh ok, Червоний - Dead
        for i in range(1, len(all_pts)):
            has_direct_link = (adj[0, i] == 1)
            is_connected_mesh = (hops[i] != -1)

            if has_direct_link:
                color = '#3498db'
                label_type = "Star"
            elif is_connected_mesh:
                color = '#2ecc71'
                label_type = "Mesh Only"
            else:
                color = '#e74c3c'
                label_type = "No Signal"
            
            relative_energy = mesh_energy[i] / max_energy
            node_size = 60 + (relative_energy * 400)

            ax_res.scatter(all_pts[i,0], all_pts[i,1], c=color, s=node_size, edgecolors='black', alpha=0.7, zorder=3)
            
            # Додаємо підпис енергії для найбільш навантажених
            if mesh_energy[i] > avg_mesh_energy * 1.5:
                ax_res.text(all_pts[i,0], all_pts[i,1]-0.3, f"Node #{i+1} \n({mesh_energy[i]*mult:.2f} {sub_unit_label})", 
                            fontsize=8, ha='center', color='#2c3e50', weight='bold')

            if i in top_relay_indices:
                ax_res.text(all_pts[i,0], all_pts[i,1]+0.15, "CRITICAL", 
                            color='darkred', weight='bold', fontsize=8, ha='center')

        # Gateway
        ax_res.scatter(self.gateway_pos[0], self.gateway_pos[1], c='gold', s=450, 
                       marker='*', edgecolors='black', label='Gateway', zorder=10)
        
        ax_res.set_title("Аналіз енергоспоживання та навантаження")
        ax_res.set_xlabel("Відстань (км)")
        ax_res.set_ylabel("Відстань (км)")
        ax_res.grid(True, linestyle=':', alpha=0.5)
        
        canvas_res = FigureCanvasTkAgg(fig_res, master=res_win)
        canvas_res.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        canvas_res.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = LoRaMeshSim(root)
    root.mainloop()