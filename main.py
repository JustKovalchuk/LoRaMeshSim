import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import math

class LoRaMeshSim:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRa Mesh Network Simulator")

        # --- Фізичні параметри LoRa (SX127x) ---
        self.v_supply = 3.3       # Напруга живлення (В)
        self.i_tx = 0.028         # Струм передачі (28 мА при +13 дБм)
        self.i_rx = 0.0103        # Струм прийому (10.3 мА при BW 125 кГц)
        
        # Параметри протоколу
        self.sf = 7               # Spreading Factor
        self.bw = 125000          # Bandwidth (Гц)
        self.cr = 1               # Coding Rate (1=4/5)
        self.payload_len = 20     # Розмір повідомлення (байтів)
        self.preamble_len = 8     # Довжина преамбули

        # Початкові параметри 
        self.nodes_count = tk.IntVar(value=30)
        self.r_max = tk.DoubleVar(value=2.0)
        self.field_size = tk.DoubleVar(value=6.0)
        self.packets_per_node = 100
        self.gateway_pos = np.array([0.0, 0.0])
        self.nodes = None
        
        self.setup_ui()
    
    def calculate_toa(self):
        t_symbol = (2**self.sf) / self.bw
        t_preamble = (self.preamble_len + 4.25) * t_symbol
        
        # Коефіцієнт оптимізації низької швидкості (LDRO)
        de = 1 if (t_symbol > 0.016) else 0
        
        # Формула кількості символів корисного навантаження
        payload_symb_nb = 8 + max(
            math.ceil(
                (8 * self.payload_len - 4 * self.sf + 28 + 16 - 0) / 
                (4 * (self.sf - 2 * de))
            ) * (self.cr + 4), 0
        )
        t_payload = payload_symb_nb * t_symbol
        return t_preamble + t_payload

    def setup_ui(self):
        ctrl_frame = ttk.LabelFrame(self.root, text="Параметри симуляції (LoRa SX1276)")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(ctrl_frame, text="Кількість вузлів:").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.nodes_count).pack()

        ttk.Label(ctrl_frame, text="Радіус зв'язку R_max (км):").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.r_max).pack()

        ttk.Label(ctrl_frame, text="Розмір поля (км):").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.field_size).pack()

        ttk.Separator(ctrl_frame, orient='horizontal').pack(fill='x', pady=10)

        toa = self.calculate_toa() * 1000 # в мс
        ttk.Label(ctrl_frame, text=f"SF: {self.sf} | BW: {self.bw/1000}kHz", foreground="darkgreen").pack()
        ttk.Label(ctrl_frame, text=f"Packet ToA: {toa:.2f} ms", font=('Arial', 9, 'bold')).pack()
        
        ttk.Button(ctrl_frame, text="Генерувати вузли", command=self.generate_nodes).pack(fill='x', pady=5)
        ttk.Button(ctrl_frame, text="Запустити симуляцію", command=self.run_simulation).pack(fill='x', pady=5)
        
        ttk.Label(ctrl_frame, text="* Клікніть на мапі,\nщоб переставити Gateway", foreground="blue").pack(pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.generate_nodes()

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.gateway_pos = np.array([event.xdata, event.ydata])
            self.draw_network()

    def generate_nodes(self):
        size = self.field_size.get()
        count = self.nodes_count.get()

        # Генерація випадкових координат
        self.nodes = np.random.uniform(-size/2, size/2, (count, 2))
        self.draw_network()

    def draw_network(self):
        self.ax.clear()
        self.ax.scatter(self.nodes[:,0], self.nodes[:,1], c='gray', alpha=0.5, label='Вузли')
        self.ax.scatter(self.gateway_pos[0], self.gateway_pos[1], c='gold', s=200, marker='*', edgecolors='black', label='Gateway')
        self.ax.set_title("Попередній перегляд структури мережі (Клікніть для переміщення GW)")
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
                if np.linalg.norm(all_pts[i] - all_pts[j]) <= r:
                    adj[i,j] = adj[j,i] = 1

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
        # Розрахунок енергії
        t_packet = self.calculate_toa()
        e_tx_packet = self.v_supply * self.i_tx * t_packet  # Енергія на 1 передачу (Дж)
        e_rx_packet = self.v_supply * self.i_rx * t_packet  # Енергія на 1 прийом (Дж)
        
        # Симуляція трафіку, навантаження та енергоспоживання
        energy_spent = np.zeros(n)
        node_load = np.zeros(n)
        successful_packets = 0
        total_packets = (n - 1) * self.packets_per_node

        for node_idx in range(1, n):
            if hops[node_idx] != -1:
                successful_packets += self.packets_per_node
                curr = node_idx
                
                energy_spent[curr] += self.packets_per_node * e_tx_packet
                node_load[curr] += self.packets_per_node

                p = parent[curr]
                while p != 0: # Поки не дійшли до Gateway
                    # Ретранслятор спочатку приймає, потім передає
                    energy_spent[p] += self.packets_per_node * (e_rx_packet + e_tx_packet)
                    node_load[p] += self.packets_per_node
                    curr = p
                    p = parent[curr]

        self.show_results(hops, r, node_load, energy_spent, successful_packets, total_packets, t_packet)

    def show_results(self, hops, r, load, energy, success, total, t_packet):
        res_win = tk.Toplevel(self.root)
        res_win.title("Результати аналізу LoRa Star vs LoRa Mesh (SX1276)")
        res_win.geometry("1100x750")

        avg_e = np.mean(energy[1:])
        total_e = np.sum(energy[1:])
        max_node = np.argmax(energy[1:]) + 1

        # Розрахунок метрик
        star_mask = np.linalg.norm(self.nodes - self.gateway_pos, axis=1) <= r
        star_pdr = star_mask.mean() * 100
        mesh_pdr = (success / total) * 100 if total > 0 else 0
        
        # Пошук найбільш завантажених вузлів
        relay_loads = load[1:].copy()
        top_relay_indices = np.argsort(relay_loads)[-3:][::-1] + 1 
        
        # Розрахунок середньої енергії
        avg_energy = np.mean(energy[1:]) if len(energy) > 1 else 0
        max_energy = np.max(energy)

        # 1. Інтерфейс: Ліва панель
        info_frame = ttk.Frame(res_win)
        info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        results_text = f"""
=== МЕТРИКИ ЕФЕКТИВНОСТІ ===

PDR (Доставка даних):
• Star (LoRaWAN): {star_pdr:.1f}%
• Mesh (LoRa):    {mesh_pdr:.1f}%

=== ЕНЕРГЕТИЧНІ ЗАТРАТИ SX1276 (3.3V) ===

Час передачі пакета: {t_packet*1000:.2f} мс
Витрати на 1 TX: {self.v_supply*self.i_tx*t_packet*1000:.3f} мДж
Витрати на 1 RX: {self.v_supply*self.i_rx*t_packet*1000:.3f} мДж

--- ПІДСУМКИ ---
Загальна енергія: {total_e:.3f} Дж
Середня на вузол: {avg_e*1000:.2f} мДж

Критичні вузли (Top Load):
1. Вузол #{top_relay_indices[0]}: {int(load[top_relay_indices[0]])} пак.
2. Вузол #{top_relay_indices[1]}: {int(load[top_relay_indices[1]])} пак.
3. Вузол #{top_relay_indices[2]}: {int(load[top_relay_indices[2]])} пак.
"""
        tk.Label(info_frame, text=results_text, justify=tk.LEFT, font=("Courier", 10), 
                 background="#f8f9fa", relief="solid", padx=15, pady=15).pack(pady=10)
        
        ttk.Button(info_frame, text="Закрити", command=res_win.destroy).pack(pady=10)

        # 2. Інтерфейс: Права панель
        fig_res, ax_res = plt.subplots(figsize=(7, 7))
        all_pts = np.vstack([self.gateway_pos, self.nodes])
        
        # Малюємо лінії зв'язку
        for i in range(1, len(hops)):
            if hops[i] > 0:
                for j in range(len(all_pts)):
                    dist = np.linalg.norm(all_pts[i] - all_pts[j])
                    if dist <= r and hops[j] == hops[i] - 1:
                        ax_res.annotate("", 
                                        xy=(all_pts[j,0], all_pts[j,1]),
                                        xytext=(all_pts[i,0], all_pts[i,1]),
                                        arrowprops=dict(
                                            arrowstyle="->", 
                                            color='green', 
                                            alpha=0.6, 
                                            lw=1.5,
                                            connectionstyle="arc3"
                                        ),
                                        zorder=2)
                        break
        
        max_e = np.max(energy[1:]) if len(energy) > 1 else 1
        if max_e == 0: 
            max_e = 1 # Запобігання діленню на нуль
        
        # Візуалізація вузлів
        for i in range(1, len(all_pts)):
            # Колір за станом зв'язку
            if np.linalg.norm(all_pts[i] - self.gateway_pos) <= r:
                color = '#3498db' # Blue (Star)
            elif hops[i] != -1:
                color = '#2ecc71' # Green (Mesh)
            else:
                color = '#e74c3c' # Red (Disconnected)
            
            # Розмір вузла тепер відображає витрачену енергію
            # Базовий розмір 50 + енергія/2 для наочності
            relative_energy = energy[i] / max_e
            e_size = 60 + (relative_energy * 400) 

            scatter = ax_res.scatter(all_pts[i,0], all_pts[i,1], c=color, s=e_size, 
                                    edgecolors='black', alpha=0.7, zorder=3)
                        
            # Додаємо підпис енергії для найбільш навантажених
            if energy[i] > avg_energy * 1.5:
                ax_res.text(all_pts[i,0], all_pts[i,1]-0.3, f"E:{int(energy[i])}", 
                            fontsize=8, ha='center', color='#2c3e50', weight='bold')

            if i in top_relay_indices:
                ax_res.text(all_pts[i,0], all_pts[i,1]+0.3, "CRITICAL", 
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