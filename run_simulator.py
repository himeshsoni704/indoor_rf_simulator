"""
Interactive viewer for the Indoor RF + Crowd Digital Twin.

Run:
    python3 run_simulator.py

Controls:
    - Left-click on the floorplan or RF panel : add a router there
    - Right-click on the floorplan or RF panel : remove the nearest router
    - Sliders: coverage threshold, number of agents, number of routers to
      auto-place
    - Radio buttons: switch between LoRa / Wi-Fi frequencies
    - Buttons: auto-optimize router placement, clear routers, reset agents

This file only handles drawing and UI events. All the physics, pathfinding,
and statistics live in indoor_rf_digital_twin.py — see that file for the
scientific explanation of each model.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, RadioButtons, Slider

from indoor_rf_digital_twin import (
    DEFAULT_COVERAGE_THRESHOLD_DBM,
    GRID_H,
    GRID_W,
    NETWORKS,
    NOISE_FLOOR_DBM,
    WALL,
    DOOR,
    Agent,
    Floorplan,
    compute_coverage_stats,
    compute_density_grid,
    compute_rf_grid,
    greedy_router_placement,
)


# ----------------------------------------------------------------------------
# Color maps that mirror the original "cold -> hot" RF look and the
# "green -> yellow -> red" crowd-density look.
# ----------------------------------------------------------------------------
RF_CMAP = LinearSegmentedColormap.from_list(
    "rf_signal", ["#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff0000"]
)
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "crowd_density", ["#ffffff00", "#c8c800", "#ff0000"]
)


class DigitalTwinApp:
    def __init__(self):
        self.floor = Floorplan()
        self.radio_name = "WiFi 2.4 GHz"
        self.coverage_threshold = DEFAULT_COVERAGE_THRESHOLD_DBM
        self.n_agents = 12
        self.n_auto_routers = 3

        self.routers: list[tuple[int, int]] = []
        self.rf_grid: np.ndarray | None = None
        self.agents: list[Agent] = [Agent.spawn(self.floor) for _ in range(self.n_agents)]

        self._build_figure()
        self._auto_optimize(None)  # place an initial good set of routers
        self.timer = self.fig.canvas.new_timer(interval=120)
        self.timer.add_callback(self._on_tick)
        self.timer.start()

    # ------------------------------------------------------------------
    # Figure / widget layout
    # ------------------------------------------------------------------
    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.suptitle("Indoor RF + Crowd Digital Twin", fontsize=14, fontweight="bold")

        gs = self.fig.add_gridspec(
            2, 2,
            left=0.04, right=0.66, top=0.90, bottom=0.06, hspace=0.30, wspace=0.18,
        )

        self.ax_floor = self.fig.add_subplot(gs[0, 0])
        self.ax_rf = self.fig.add_subplot(gs[0, 1])
        self.ax_density = self.fig.add_subplot(gs[1, 0])
        self.ax_stats = self.fig.add_subplot(gs[1, 1])

        for ax, title in [
            (self.ax_floor, "Floorplan + agents (click=add router, right-click=remove)"),
            (self.ax_rf, "RF signal heatmap"),
            (self.ax_density, "Crowd density heatmap"),
        ]:
            ax.set_title(title, fontsize=9)
            ax.set_xlim(0, GRID_W)
            ax.set_ylim(GRID_H, 0)  # flip y so row 0 is at the top
            ax.set_xticks([])
            ax.set_yticks([])

        self.ax_stats.axis("off")
        self.ax_stats.set_title("Live statistics", fontsize=9)

        # --- background wall image (static, recomputed only if needed) ---
        self.wall_rgb = self._build_wall_image()
        self.im_floor_bg = self.ax_floor.imshow(self.wall_rgb, extent=(0, GRID_W, GRID_H, 0))
        self.im_rf_bg = self.ax_rf.imshow(self.wall_rgb, extent=(0, GRID_W, GRID_H, 0))
        self.im_density_bg = self.ax_density.imshow(self.wall_rgb, extent=(0, GRID_W, GRID_H, 0))

        for (lbl, lx, ly) in self.floor.labels:
            for ax in (self.ax_floor, self.ax_rf, self.ax_density):
                ax.text(lx, ly, lbl, fontsize=7, ha="center", va="center",
                        color="0.4", fontweight="bold")

        # --- dynamic layers (re-drawn every tick) ---
        self.im_rf = self.ax_rf.imshow(
            np.full((GRID_H, GRID_W), np.nan), extent=(0, GRID_W, GRID_H, 0),
            cmap=RF_CMAP, vmin=NOISE_FLOOR_DBM, vmax=-30, alpha=0.75,
        )
        self.im_density = self.ax_density.imshow(
            np.zeros((GRID_H, GRID_W)), extent=(0, GRID_W, GRID_H, 0),
            cmap=DENSITY_CMAP, vmin=0, vmax=2.0, alpha=0.85,
        )
        self.agent_scatter = self.ax_floor.scatter([], [], s=28, c="#00b8d4",
                                                    edgecolors="white", linewidths=0.8, zorder=5)
        self.router_scatter_floor = self.ax_floor.scatter([], [], marker="^", s=90,
                                                            c="#ff00ff", edgecolors="white", zorder=6)
        self.router_scatter_rf = self.ax_rf.scatter([], [], marker="^", s=90,
                                                      c="#ff00ff", edgecolors="white", zorder=6)
        self.router_scatter_den = self.ax_density.scatter([], [], marker="^", s=90,
                                                            c="#ff00ff", edgecolors="white", zorder=6)
        self.threshold_text = self.ax_rf.text(
            1, 1.5, "", fontsize=8, color="0.3", va="top"
        )

        self.stat_texts = {}
        labels = ["Coverage", "Dead zones", "Avg RSSI", "Active agents"]
        for i, label in enumerate(labels):
            row, col = divmod(i, 2)
            x, y = 0.05 + col * 0.5, 0.75 - row * 0.35
            self.ax_stats.text(x, y, label, fontsize=9, color="0.4", transform=self.ax_stats.transAxes)
            t = self.ax_stats.text(x, y - 0.1, "—", fontsize=16, fontweight="bold",
                                    transform=self.ax_stats.transAxes)
            self.stat_texts[label] = t

        self.status_text = self.ax_stats.text(
            0.05, 0.05, "", fontsize=9, color="0.3", transform=self.ax_stats.transAxes
        )

        self._build_controls(gs)
        self._connect_events()

    def _build_wall_image(self) -> np.ndarray:
        """Render the static wall/door/floor layer once as an RGB image."""
        rgb = np.zeros((GRID_H, GRID_W, 3))
        wall_color = np.array([0.13, 0.13, 0.13])
        door_color = np.array([0.13, 0.78, 0.36])
        floor_color = np.array([0.97, 0.97, 0.96])
        for y in range(GRID_H):
            for x in range(GRID_W):
                v = self.floor.occ[y, x]
                rgb[y, x] = wall_color if v == WALL else (door_color if v == DOOR else floor_color)
        return rgb

    def _build_controls(self, gs):
        # Network radio buttons
        ax_radio = self.fig.add_axes([0.70, 0.80, 0.27, 0.13])
        ax_radio.set_title("Network", fontsize=9, loc="left")
        self.radio_net = RadioButtons(ax_radio, list(NETWORKS.keys()),
                                       active=list(NETWORKS.keys()).index(self.radio_name))
        self.radio_net.on_clicked(self._on_network_change)

        # Sliders
        ax_thr = self.fig.add_axes([0.72, 0.71, 0.23, 0.03])
        self.slider_thr = Slider(ax_thr, "Threshold (dBm)", -120, -60,
                                  valinit=self.coverage_threshold, valstep=1)
        self.slider_thr.on_changed(self._on_threshold_change)

        ax_ppl = self.fig.add_axes([0.72, 0.64, 0.23, 0.03])
        self.slider_ppl = Slider(ax_ppl, "Agents", 0, 40, valinit=self.n_agents, valstep=1)
        self.slider_ppl.on_changed(self._on_agents_change)

        ax_rtr = self.fig.add_axes([0.72, 0.57, 0.23, 0.03])
        self.slider_rtr = Slider(ax_rtr, "Routers to place", 1, 6,
                                  valinit=self.n_auto_routers, valstep=1)
        self.slider_rtr.on_changed(self._on_n_routers_change)

        # Buttons
        ax_opt = self.fig.add_axes([0.70, 0.47, 0.13, 0.045])
        self.btn_opt = Button(ax_opt, "Auto-optimize")
        self.btn_opt.on_clicked(self._auto_optimize)

        ax_clr = self.fig.add_axes([0.84, 0.47, 0.13, 0.045])
        self.btn_clr = Button(ax_clr, "Clear routers")
        self.btn_clr.on_clicked(self._on_clear_routers)

        ax_rst = self.fig.add_axes([0.70, 0.41, 0.27, 0.045])
        self.btn_rst = Button(ax_rst, "Reset agents")
        self.btn_rst.on_clicked(self._on_reset_agents)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_network_change(self, label):
        self.radio_name = label
        self._recompute_rf()

    def _on_threshold_change(self, val):
        self.coverage_threshold = val

    def _on_agents_change(self, val):
        target = int(val)
        if target > len(self.agents):
            self.agents += [Agent.spawn(self.floor) for _ in range(target - len(self.agents))]
        else:
            self.agents = self.agents[:target]
        self.n_agents = target

    def _on_n_routers_change(self, val):
        self.n_auto_routers = int(val)

    def _auto_optimize(self, _event):
        radio = NETWORKS[self.radio_name]
        self.routers, self.rf_grid = greedy_router_placement(
            self.floor, radio, self.n_auto_routers, self.coverage_threshold
        )

    def _on_clear_routers(self, _event):
        self.routers = []
        self.rf_grid = None

    def _on_reset_agents(self, _event):
        self.agents = [Agent.spawn(self.floor) for _ in range(self.n_agents)]

    def _on_click(self, event):
        if event.inaxes not in (self.ax_floor, self.ax_rf, self.ax_density):
            return
        if event.xdata is None or event.ydata is None:
            return
        gx, gy = int(event.xdata), int(event.ydata)
        if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
            return
        if self.floor.is_wall(gx, gy):
            return

        if event.button == 3:  # right-click: remove nearest router
            if not self.routers:
                return
            dists = [((rx - gx) ** 2 + (ry - gy) ** 2) for rx, ry in self.routers]
            self.routers.pop(int(np.argmin(dists)))
        elif event.button == 1:  # left-click: add a router
            self.routers.append((gx, gy))
        self._recompute_rf()

    def _recompute_rf(self):
        radio = NETWORKS[self.radio_name]
        self.rf_grid = compute_rf_grid(self.floor, radio, self.routers)

    # ------------------------------------------------------------------
    # Animation tick
    # ------------------------------------------------------------------
    def _on_tick(self):
        for agent in self.agents:
            agent.step()
        self._redraw()

    def _redraw(self):
        # Agents
        if self.agents:
            xs = [a.x for a in self.agents]
            ys = [a.y for a in self.agents]
            self.agent_scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            self.agent_scatter.set_offsets(np.empty((0, 2)))

        # Routers (shown on all three panels)
        if self.routers:
            rxy = np.array([(rx + 0.5, ry + 0.5) for rx, ry in self.routers])
        else:
            rxy = np.empty((0, 2))
        self.router_scatter_floor.set_offsets(rxy)
        self.router_scatter_rf.set_offsets(rxy)
        self.router_scatter_den.set_offsets(rxy)

        # RF heatmap
        if self.rf_grid is not None:
            display_grid = np.where(self.floor.occ > WALL, self.rf_grid, np.nan)
            self.im_rf.set_data(display_grid)
        else:
            self.im_rf.set_data(np.full((GRID_H, GRID_W), np.nan))
        self.threshold_text.set_text(f"Threshold: {self.coverage_threshold:.0f} dBm")

        # Density heatmap
        density = compute_density_grid(self.agents)
        display_density = np.where((self.floor.occ > WALL) & (density > 0.05), density, np.nan)
        self.im_density.set_data(display_density)

        # Stats
        stats = compute_coverage_stats(self.floor, self.rf_grid, self.coverage_threshold,
                                        len(self.agents))
        self.stat_texts["Coverage"].set_text(f"{stats.coverage_pct:.1f}%")
        self.stat_texts["Dead zones"].set_text(f"{stats.dead_zone_pct:.1f}%")
        self.stat_texts["Avg RSSI"].set_text(f"{stats.avg_rssi_dbm:.0f} dBm")
        self.stat_texts["Active agents"].set_text(f"{stats.n_agents}")
        n = len(self.routers)
        self.status_text.set_text(f"{self.radio_name} · {n} router{'s' if n != 1 else ''}")

        self.fig.canvas.draw_idle()

    def show(self):
        self._redraw()
        plt.show()


def main():
    app = DigitalTwinApp()
    app.show()


if __name__ == "__main__":
    main()
