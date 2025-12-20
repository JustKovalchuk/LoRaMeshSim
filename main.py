import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

class LoRaMeshSim:
    def __init__(self, root):
        self.root = root
        self.root.title("LoRa Mesh Network Simulator")
        
        # Початкові параметри 
        self.nodes_count = tk.IntVar(value=30)
        self.r_max = tk.DoubleVar(value=2.0)
        self.field_size = tk.DoubleVar(value=6.0)
        self.gateway_pos = np.array([0.0, 0.0])
        self.nodes = None
        
        self.setup_ui()

    def setup_ui(self):
        # Панель керування (ліворуч)
        ctrl_frame = ttk.LabelFrame(self.root, text="Параметри симуляції")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(ctrl_frame, text="Кількість вузлів:").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.nodes_count).pack()

        ttk.Label(ctrl_frame, text="Радіус зв'язку R_max (км):").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.r_max).pack()

        ttk.Label(ctrl_frame, text="Розмір поля (км):").pack(pady=5)
        ttk.Entry(ctrl_frame, textvariable=self.field_size).pack()

        ttk.Separator(ctrl_frame, orient='horizontal').pack(fill='x', pady=10)
        
        ttk.Button(ctrl_frame, text="Генерувати вузли", command=self.generate_nodes).pack(fill='x', pady=5)
        ttk.Button(ctrl_frame, text="Запустити симуляцію", command=self.run_simulation).pack(fill='x', pady=5)
        
        ttk.Label(ctrl_frame, text="* Клікніть на мапі,\nщоб змінити місце Gateway", foreground="blue").pack(pady=10)

        # Область графіку (праворуч)
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
        self.nodes = np.random.uniform(-size/2, size/2, (count, 2))
        self.draw_network()

    def draw_network(self):
        self.ax.clear()
        self.ax.scatter(self.nodes[:,0], self.nodes[:,1], c='gray', alpha=0.5, label='Вузли')
        self.ax.scatter(self.gateway_pos[0], self.gateway_pos[1], c='gold', s=200, marker='*', edgecolors='black', label='Gateway')
        self.ax.set_title("Мапа мережі (Налаштування)")
        self.ax.grid(True, linestyle=':')
        self.canvas.draw()

    def run_simulation(self):
        # Логіка розрахунку Mesh
        r = self.r_max.get()
        all_pts = np.vstack([self.gateway_pos, self.nodes])
        n = len(all_pts)
        
        # Побудова сусіда
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(all_pts[i] - all_pts[j]) <= r:
                    adj[i,j] = adj[j,i] = 1

        # BFS
        hops = np.full(n, -1)
        hops[0] = 0
        q = deque([0])
        while q:
            u = q.popleft()
            for v, conn in enumerate(adj[u]):
                if conn and hops[v] == -1:
                    hops[v] = hops[u] + 1
                    q.append(v)

        self.show_results(hops, r)

    def show_results(self, hops, r):
        res_win = tk.Toplevel(self.root)
        res_win.title("Результати та Візуалізація")
        res_win.geometry("900x600") # Збільшуємо вікно для графіку

        # Розрахунок метрик
        star_mask = np.linalg.norm(self.nodes - self.gateway_pos, axis=1) <= r
        star_pdr = star_mask.mean() * 100
        mesh_pdr = (hops[1:] != -1).mean() * 100
        avg_hops = np.mean(hops[hops > 0]) if any(hops > 0) else 0

        # Ліва частина: Текстовий звіт
        info_frame = ttk.Frame(res_win)
        info_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)

        text = f"""
=== АНАЛІЗ ЕФЕКТИВНОСТІ ===

1. Доставка даних (PDR):
   - LoRaWAN (Зірка): {star_pdr:.1f}%
   - LoRa Mesh: {mesh_pdr:.1f}%

2. Покриття:
   - Збільшення зони: {mesh_pdr - star_pdr:.1f}%

3. Характеристики шляху:
   - Сер. к-ть хопів: {avg_hops:.2f}

ЛЕГЕНДА МАПИ:
● Синій: Прямий зв'язок (Star)
● Зелений: Через ретрансляцію
● Червоний: Поза зоною
        """
        tk.Label(info_frame, text=text, justify=tk.LEFT, font=("Courier", 10)).pack()
        ttk.Button(info_frame, text="Закрити", command=res_win.destroy).pack(pady=20)

        # Права частина: Графік результатів
        fig_res, ax_res = plt.subplots(figsize=(6, 6))
        
        # Визначаємо зв'язки (матриця суміжності для малювання ліній)
        all_pts = np.vstack([self.gateway_pos, self.nodes])
        
        # Малюємо лінії зв'язку Mesh
        for i in range(1, len(hops)):
            if hops[i] > 0:
                # Шукаємо сусіда, який ближче до Gateway на 1 хоп
                for j in range(len(all_pts)):
                    dist = np.linalg.norm(all_pts[i] - all_pts[j])
                    if dist <= r and hops[j] == hops[i] - 1:
                        ax_res.plot([all_pts[i,0], all_pts[j,0]], 
                                    [all_pts[i,1], all_pts[j,1]], 
                                    'g-', alpha=0.3, lw=1)
                        break

        # Малюємо вузли з різними кольорами
        for i in range(len(self.nodes)):
            idx = i + 1 # +1 бо 0 - це Gateway
            if star_mask[i]:
                color = 'blue'  # Прямий зв'язок
            elif hops[idx] != -1:
                color = 'green' # Mesh зв'язок
            else:
                color = 'red'   # Немає зв'язку
            
            ax_res.scatter(self.nodes[i,0], self.nodes[i,1], c=color, s=40, zorder=3)
            # Додаємо номер хопа біля вузла
            if hops[idx] != -1:
                ax_res.text(self.nodes[i,0], self.nodes[i,1], f" {hops[idx]}", fontsize=8)

        # Gateway
        ax_res.scatter(self.gateway_pos[0], self.gateway_pos[1], c='gold', s=250, marker='*', edgecolors='black', zorder=5)
        
        ax_res.set_title("Топологія маршрутів LoRa Mesh")
        ax_res.grid(True, linestyle=':')
        
        canvas_res = FigureCanvasTkAgg(fig_res, master=res_win)
        canvas_res.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        canvas_res.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = LoRaMeshSim(root)
    root.mainloop()