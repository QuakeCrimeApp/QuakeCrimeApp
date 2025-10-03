import sys, os

# --- Logs para depuración ---
sys.stdout = open(os.path.join(os.path.dirname(__file__), "stdout.log"), "w")
sys.stderr = open(os.path.join(os.path.dirname(__file__), "stderr.log"), "w")

if hasattr(sys, "_MEIPASS"):
    os.environ["GDAL_DATA"] = os.path.join(sys._MEIPASS, "gdal_data")
    os.environ["GDAL_DRIVER_PATH"] = os.path.join(sys._MEIPASS, "gdalplugins")

sys.setrecursionlimit(5000)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import threading
from datetime import datetime
import numpyro.distributions as dist
from bstpp.main import Hawkes_Model

# --- Global variables ---
paths = {"events": None, "roads": None}
trained_model = None
gdf_roads_global = None
gdf_train_global = None

# --- Helper functions ---
def select_file(label, filetype):
    filepath = filedialog.askopenfilename(
        title=f"Select {filetype} shapefile",
        filetypes=[("Shapefiles", "*.shp")]
    )
    if filepath:
        label.config(text=os.path.basename(filepath))
        paths[filetype] = filepath
        if filetype == "events":
            update_date_range()

def update_date_range():
    try:
        gdf_events = gpd.read_file(paths["events"])
        gdf_events["Date"] = pd.to_datetime(gdf_events["Fecha"], format="%d/%m/%Y", errors="coerce")
        min_date = gdf_events["Date"].min()
        max_date = gdf_events["Date"].max()
        if min_date is not pd.NaT and max_date is not pd.NaT:
            entry_train_start.delete(0, tk.END)
            entry_train_start.insert(0, min_date.strftime("%d/%m/%Y"))
            entry_train_end.delete(0, tk.END)
            entry_train_end.insert(0, (min_date + pd.Timedelta(days=365)).strftime("%d/%m/%Y"))
            entry_sim_start.delete(0, tk.END)
            entry_sim_start.insert(0, (min_date + pd.Timedelta(days=366)).strftime("%d/%m/%Y"))
            entry_sim_end.delete(0, tk.END)
            entry_sim_end.insert(0, max_date.strftime("%d/%m/%Y"))
    except Exception:
        pass

def validate_date(text):
    try:
        return datetime.strptime(text, "%d/%m/%Y")
    except:
        return None

# --- Model execution ---
def run_model_thread():
    btn_run.config(state=tk.DISABLED)
    progress_bar.start()
    thread = threading.Thread(target=run_model)
    thread.start()

def run_model():
    global trained_model, gdf_roads_global, gdf_train_global
    try:
        if paths["events"] is None or paths["roads"] is None:
            messagebox.showerror("Error", "Both shapefiles must be loaded before running the model.")
            return

        gdf_events = gpd.read_file(paths["events"]).to_crs("EPSG:4326")
        gdf_roads = gpd.read_file(paths["roads"]).to_crs("EPSG:4326")
        gdf_roads_global = gdf_roads.copy()

        for col in ["Fecha", "Long", "Lat"]:
            if col not in gdf_events.columns:
                messagebox.showerror("Error", f"The events shapefile must contain column '{col}'.")
                return

        gdf_events["Fecha"] = pd.to_datetime(gdf_events["Fecha"], format="%d/%m/%Y", errors="coerce")
        if gdf_events["Fecha"].isnull().all():
            messagebox.showerror("Error", "Could not parse 'Fecha' column to datetime.")
            return

        train_start = validate_date(entry_train_start.get())
        train_end = validate_date(entry_train_end.get())
        sim_start = validate_date(entry_sim_start.get())
        sim_end = validate_date(entry_sim_end.get())

        if None in [train_start, train_end, sim_start, sim_end]:
            messagebox.showerror("Error", "Please enter valid dates in dd/mm/yyyy format.")
            return

        if not (train_start < train_end < sim_start < sim_end):
            messagebox.showerror("Error", "Dates must follow: train start < train end < simulation start < simulation end.")
            return

        try:
            lr = float(entry_lr.get())
            num_steps = int(entry_num_steps.get())
            if lr <= 0 or num_steps <= 0:
                raise ValueError
        except:
            messagebox.showerror("Error", "Learning rate must be positive and num_steps must be a positive integer.")
            return

        gdf_train = gdf_events[(gdf_events["Fecha"] >= train_start) & (gdf_events["Fecha"] <= train_end)]
        gdf_test = gdf_events[(gdf_events["Fecha"] >= sim_start) & (gdf_events["Fecha"] <= sim_end)]

        if gdf_train.empty:
            messagebox.showerror("Error", "No training data found for the selected period.")
            return

        t0 = gdf_train["Fecha"].min()
        gdf_train["t"] = (gdf_train["Fecha"] - t0).dt.total_seconds() / 86400
        gdf_train = gdf_train.sort_values("t").reset_index(drop=True)

        gdf_test["t"] = (gdf_test["Fecha"] - t0).dt.total_seconds() / 86400
        gdf_test = gdf_test.sort_values("t").reset_index(drop=True)

        gdf_buffered = gdf_roads.copy()
        gdf_buffered["geometry"] = gdf_buffered.buffer(0.00015)

        data_model = gdf_train[["t", "Long", "Lat"]].rename(columns={"t": "T", "Long": "X", "Lat": "Y"})

        # --- Hawkes model ---
        model = Hawkes_Model(
            data=data_model,
            A=gdf_buffered,
            T=gdf_train["t"].max(),
            cox_background=True,
            a_0=dist.Normal(1, 10),
            alpha=dist.Beta(20, 60),
            beta=dist.HalfNormal(2.0),
            sigmax_2=dist.HalfNormal(0.25)
        )

        model.run_svi(lr=lr, num_steps=num_steps)

        data_test = gdf_test[["t", "Long", "Lat"]].rename(columns={"t": "T", "Long": "X", "Lat": "Y"})
        results = f"\nLog Expected Likelihood: {model.log_expected_likelihood(data_test):.2f}\n"
        results += f"Expected AIC: {model.expected_AIC():.2f}\n"
        messagebox.showinfo("Model Trained", results)

        trained_model = model
        gdf_train_global = gdf_train

        # --- Diagnostic plots ---
        model.plot_spatial(include_cov=False)
        plt.title("🛣️ Spatial intensity on road network (Cox background)")
        plt.show()

        model.plot_prop_excitation()
        plt.title("🔥 Proportion of events explained by self-excitation")
        plt.show()

        model.plot_temporal()
        plt.title("⏳ Temporal intensity")
        plt.show()

        # --- Posterior distributions of beta and sigmax_2 ---
        model.plot_trigger_posterior(trace=True)
        plt.title("📍 Posterior of trigger parameters (alpha, beta, sigmax_2)")
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

    btn_run.config(state=tk.NORMAL)
    progress_bar.stop()

# --- GUI ---
root = tk.Tk()
root.title("QUAKECRIME: Hawkes Model on Road Network")
root.geometry("720x600")

style = ttk.Style()
style.configure("TLabel", font=("Segoe UI", 10))
style.configure("TButton", font=("Segoe UI", 10), padding=6)

frame = ttk.Frame(root)
frame.pack(pady=15)

# File inputs
ttk.Label(frame, text="Events (shapefile)").grid(row=0, column=0, sticky="w")
label_events = ttk.Label(frame, text="Not selected", width=40)
label_events.grid(row=0, column=1)
ttk.Button(frame, text="Load", command=lambda: select_file(label_events, "events")).grid(row=0, column=2)

ttk.Label(frame, text="Road network (shapefile)").grid(row=1, column=0, sticky="w")
label_roads = ttk.Label(frame, text="Not selected", width=40)
label_roads.grid(row=1, column=1)
ttk.Button(frame, text="Load", command=lambda: select_file(label_roads, "roads")).grid(row=1, column=2)

# Dates
ttk.Label(frame, text="Training start (dd/mm/yyyy)").grid(row=2, column=0, sticky="w")
entry_train_start = ttk.Entry(frame); entry_train_start.grid(row=2, column=1)

ttk.Label(frame, text="Training end").grid(row=3, column=0, sticky="w")
entry_train_end = ttk.Entry(frame); entry_train_end.grid(row=3, column=1)

ttk.Label(frame, text="Simulation start").grid(row=4, column=0, sticky="w")
entry_sim_start = ttk.Entry(frame); entry_sim_start.grid(row=4, column=1)

ttk.Label(frame, text="Simulation end").grid(row=5, column=0, sticky="w")
entry_sim_end = ttk.Entry(frame); entry_sim_end.grid(row=5, column=1)

# Parameters
ttk.Label(frame, text="Learning rate").grid(row=6, column=0, sticky="w")
entry_lr = ttk.Entry(frame); entry_lr.insert(0, "0.001"); entry_lr.grid(row=6, column=1)

ttk.Label(frame, text="Num steps").grid(row=7, column=0, sticky="w")
entry_num_steps = ttk.Entry(frame); entry_num_steps.insert(0, "500"); entry_num_steps.grid(row=7, column=1)

# Buttons
btn_run = ttk.Button(frame, text="Run model", command=run_model_thread)
btn_run.grid(row=8, column=0, pady=15)

# Progress bar
progress_bar = ttk.Progressbar(root, mode="indeterminate")
progress_bar.pack(fill="x", padx=20, pady=10)

root.mainloop()
