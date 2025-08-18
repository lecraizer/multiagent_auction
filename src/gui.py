import tkinter as tk
import threading

def show_auction_episode(values, bids, rewards):
    def run_window():
        root = tk.Tk()
        root.title("Auction Episode")

        canvas_height = 700
        canvas = tk.Canvas(root, width=800, height=canvas_height, bg='white')
        canvas.pack()

        agent_width = 200
        spacing = 50

        #### ---- LEILOEIRO (topo central) ---- ####
        lx = 400  # centro da tela
        ly = 70   # posição vertical

        # Cabeça
        canvas.create_oval(lx - 15, ly - 60, lx + 15, ly - 30, fill="orange", outline="black")
        # Corpo
        canvas.create_line(lx, ly - 30, lx, ly + 10, width=2)
        # Braços
        canvas.create_line(lx - 20, ly - 10, lx + 20, ly - 10, width=2)
        # Pernas
        canvas.create_line(lx, ly + 10, lx - 15, ly + 40, width=2)
        canvas.create_line(lx, ly + 10, lx + 15, ly + 40, width=2)

        # Mesa
        canvas.create_rectangle(lx - 60, ly + 45, lx + 60, ly + 60, fill="#8B4513", outline="black")

        #### ---- AGENTES (distribuição dinâmica) ---- ####
        cols = 3
        rows = (len(values) + cols - 1) // cols

        x_padding = 50
        y_start = 240
        x_spacing = (800 - x_padding * 2) // cols
        y_spacing = 220  # AUMENTADO!

        for i, (v, b, r) in enumerate(zip(values, bids, rewards)):
            col = i % cols
            row = i // cols

            base_x = x_padding + col * x_spacing + x_spacing // 2
            base_y = y_start + row * y_spacing

            # Cabeça
            canvas.create_oval(base_x - 15, base_y - 60, base_x + 15, base_y - 30, fill="lightblue", outline="black")
            # Corpo
            canvas.create_line(base_x, base_y - 30, base_x, base_y + 10, width=2)
            # Braços
            canvas.create_line(base_x - 20, base_y - 10, base_x + 20, base_y - 10, width=2)
            # Pernas
            canvas.create_line(base_x, base_y + 10, base_x - 15, base_y + 40, width=2)
            canvas.create_line(base_x, base_y + 10, base_x + 15, base_y + 40, width=2)

            # Nome do agente
            canvas.create_text(base_x, base_y + 60, text=f"Agent {i+1}", font=("Helvetica", 12, "bold"))

            # Informações
            canvas.create_text(base_x, base_y + 80, text=f"Value: {v:.2f}", font=("Helvetica", 11))
            canvas.create_text(base_x, base_y + 100, text=f"Bid: {b:.2f}", font=("Helvetica", 11))
            canvas.create_text(base_x, base_y + 120, text=f"Reward: {r:.2f}", font=("Helvetica", 11), fill="green")

        # root.after(4000, root.destroy)
        root.mainloop()

    thread = threading.Thread(target=run_window)
    thread.daemon = True
    thread.start()
