"""
Indoor RF + Crowd Digital Twin Simulator
==========================================

A physically-grounded simulator of wireless signal coverage inside a house,
combined with a simple agent-based model of people moving room to room.

WHAT IT MODELS (and why it's scientifically reasonable):

1. RF PROPAGATION — "Log-Distance Path Loss with Multi-Wall Attenuation"
   This is a standard, widely-used indoor propagation model (see e.g. the
   ITU-R P.1238 indoor propagation recommendation, and the classic
   "Motley-Keenan" multi-wall model). Two physical effects are combined:

   a) Free-Space Path Loss (FSPL) — how much a radio wave weakens simply
      from spreading out over distance. The standard formula (distance in
      km, frequency in MHz) is:

          FSPL(dB) = 32.44 + 20*log10(d_km) + 20*log10(f_MHz)

      This comes directly from the Friis transmission equation. Signal
      power falls off with the *square* of distance in free space, and the
      log10 turns that into a clean dB number. Higher frequencies also lose
      more energy to free-space spreading for a fixed antenna size, which
      is why 5 GHz Wi-Fi covers less area than 2.4 GHz Wi-Fi for the same
      transmit power.

   b) Wall attenuation — every wall a signal passes through removes a
      roughly constant chunk of power (in dB), which is an empirically
      well-supported simplification (real walls vary, but ~3-15 dB per
      wall is typical depending on material and frequency). We also add an
      extra penalty for "around the corner" / non-line-of-sight paths,
      since multipath and diffraction losses stack up there.

2. RECEIVED SIGNAL: RSSI(dBm) = TX_POWER(dBm) - FSPL(dB) - wall_losses(dB)
   This is just conservation of energy in decibel form: every loss term
   subtracts from the transmitted power.

3. CROWD DENSITY — Kernel Density Estimation (Gaussian kernel)
   Each person contributes a 2D Gaussian "bump" centered on their position.
   Summing these bumps gives a smooth density field — the exact same
   technique used in real crowd-analytics and epidemiological heatmaps.

4. AGENT MOVEMENT — A* pathfinding on a grid
   People don't walk through walls. Each agent picks a random destination
   room and uses the A* search algorithm (the standard optimal pathfinding
   algorithm in robotics/games) to walk the shortest legal route, with
   diagonal moves costing sqrt(2) instead of 1, which is the correct
   Euclidean step cost on a grid.

5. ROUTER PLACEMENT OPTIMIZATION — Greedy maximum-coverage algorithm
   Choosing the best K router locations to maximize coverage is the
   "maximum coverage problem", which is NP-hard in general. The standard,
   provably-good (within ~63% of optimal, per Nemhauser et al. 1978)
   practical approach is a greedy algorithm: repeatedly place the next
   router wherever it adds the most newly-covered area. That's what we do.

Run this file directly to launch an interactive Matplotlib window:
    python3 indoor_rf_digital_twin.py
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ============================================================================
# 1. PHYSICAL CONSTANTS AND RADIO CONFIGURATIONS
# ============================================================================

# Each cell of our grid represents 1 meter x 1 meter of floor space.
METERS_PER_CELL = 1.0

# Transmit power of the router, in dBm (decibels relative to 1 milliwatt).
# +14 dBm is roughly 25 mW, a realistic low-power indoor Wi-Fi/IoT setting.
TX_POWER_DBM = 14.0

# Thermal/receiver noise floor, in dBm. Below this, there is effectively no
# usable signal. Real Wi-Fi noise floors are commonly around -95 to -100 dBm;
# we use -120 dBm as a clean "no signal" baseline for the grid background.
NOISE_FLOOR_DBM = -120.0

# Extra loss (dB) applied when a path has to "turn a corner" (cross 2+
# walls) to represent additional diffraction/multipath loss beyond what
# wall attenuation alone captures.
NON_LINE_OF_SIGHT_PENALTY_DB = -8.0
CORNER_PENALTY_DB = -5.0

# Default coverage threshold: any cell with RSSI at or above this is
# considered "covered" (i.e. usable signal). -90 dBm is a commonly cited
# rule of thumb for "still works, but weak" Wi-Fi.
DEFAULT_COVERAGE_THRESHOLD_DBM = -90.0

# Radio configurations: each network standard has a carrier frequency and an
# empirical "loss per wall" figure in dB. Real numbers vary by wall
# material, but these are realistic, commonly cited ballpark values:
#   - Lower frequencies (LoRa @ 868/915 MHz) penetrate walls much better.
#   - Higher frequencies (Wi-Fi 5 GHz) attenuate faster, both in free space
#     and per wall, which is exactly why 5 GHz networks have shorter range.
@dataclass(frozen=True)
class RadioConfig:
    name: str
    freq_mhz: float          # carrier frequency in MHz
    wall_loss_db: float      # average attenuation per wall crossed, in dB


NETWORKS: dict[str, RadioConfig] = {
    "LoRa 868 MHz": RadioConfig("LoRa 868 MHz", freq_mhz=868, wall_loss_db=3.0),
    "LoRa 915 MHz": RadioConfig("LoRa 915 MHz", freq_mhz=915, wall_loss_db=3.2),
    "WiFi 2.4 GHz": RadioConfig("WiFi 2.4 GHz", freq_mhz=2400, wall_loss_db=8.0),
    "WiFi 5 GHz":   RadioConfig("WiFi 5 GHz",   freq_mhz=5200, wall_loss_db=12.0),
}


# ============================================================================
# 2. FLOORPLAN: an occupancy grid (0 = wall, 1 = free floor, 2 = doorway)
# ============================================================================

GRID_W, GRID_H = 60, 30   # grid cells -> a 60m x 30m floor (1 cell = 1 meter)

WALL, FREE, DOOR = 0, 1, 2


class Floorplan:
    """
    Holds the occupancy grid for a small house with bedrooms, bathroom,
    living room, kitchen, dining room, office, and a connecting corridor.

    The layout mirrors a believable real floorplan: rooms are separated by
    walls (value 0) with discrete doorway gaps (value 2) that agents and RF
    rays can pass through. Doorways are not radio-special; they just don't
    block movement or signal the way a solid wall does.
    """

    def __init__(self):
        self.occ = np.ones((GRID_H, GRID_W), dtype=np.uint8)  # start all-free
        self._build_house()
        # Cache the list of all walkable (non-wall) cells for quick random sampling.
        ys, xs = np.where(self.occ > WALL)
        self.free_cells = list(zip(xs.tolist(), ys.tolist()))
        self.labels = [
            ("Bedroom 1", 9, 7), ("Bathroom", 23, 7),
            ("Living room", 39, 9), ("Bedroom 2", 54, 9),
            ("Corridor", 25, 20), ("Kitchen", 14, 26),
            ("Dining", 39, 26), ("Office", 54, 26), ("Entrance", 5, 26),
        ]

    # -- helpers for drawing straight wall segments with optional door gaps --
    def _hwall(self, x0, x1, y, door0=None, door1=None):
        for x in range(x0, x1 + 1):
            self.occ[y, x] = DOOR if (door0 is not None and door0 <= x <= door1) else WALL

    def _vwall(self, y0, y1, x, door0=None, door1=None):
        for y in range(y0, y1 + 1):
            self.occ[y, x] = DOOR if (door0 is not None and door0 <= y <= door1) else WALL

    def _build_house(self):
        W, H = GRID_W, GRID_H
        # Outer perimeter walls.
        self._hwall(0, W - 1, 0)
        self._hwall(0, W - 1, H - 1)
        self._vwall(0, H - 1, 0)
        self._vwall(0, H - 1, W - 1)

        # Bedroom 1 (top-left): wall below it with a door, wall to its right.
        self._hwall(0, 18, 14, door0=8, door1=10)
        self._vwall(0, 14, 18, door0=14, door1=14)

        # Bathroom (top-middle): door on its left wall, solid wall below.
        self._vwall(0, 14, 28, door0=5, door1=7)
        self._hwall(18, 28, 14)

        # Living room (top-middle-right): wall to its left, door in its
        # bottom wall leading to the corridor.
        self._vwall(0, 18, 50)
        self._hwall(28, 50, 18, door0=38, door1=42)

        # Bedroom 2 (top-right): wall below it.
        self._hwall(50, 59, 18)

        # Corridor (runs the width of the house): a door down into the kitchen.
        self._hwall(0, 59, 22, door0=12, door1=16)

        # Kitchen (bottom-left): door in its right wall.
        self._vwall(22, H - 1, 28, door0=25, door1=27)

        # Office (bottom-right): wall on its left (dining room stays open
        # to the corridor, matching a typical open-plan dining area).
        self._vwall(22, H - 1, 50)

    def is_wall(self, x: int, y: int) -> bool:
        return self.occ[y, x] == WALL

    def random_free_cell(self) -> tuple[int, int]:
        return random.choice(self.free_cells)


# ============================================================================
# 3. RF PROPAGATION MODEL
# ============================================================================

def free_space_path_loss_db(distance_m: float, freq_mhz: float) -> float:
    """
    Free-Space Path Loss (Friis equation, in decibel form).

        FSPL(dB) = 32.44 + 20*log10(d_km) + 20*log10(f_MHz)

    This says signal power drops with the square of distance, and with the
    square of frequency, both physically due to how a wave's energy spreads
    over an ever-larger sphere as it propagates. We floor the distance at
    0.5 m to avoid a (non-physical) singularity exactly at the transmitter.
    """
    d_km = max(0.5, distance_m) / 1000.0
    return 32.44 + 20 * math.log10(d_km) + 20 * math.log10(freq_mhz)


def count_walls_crossed(floor: Floorplan, x0: float, y0: float,
                         x1: float, y1: float) -> int:
    """
    Walk a straight ray from (x0,y0) to (x1,y1) in small steps and count how
    many distinct wall cells it passes through. This is a simple but
    effective approximation of real ray-tracing used in radio planning
    tools, sampling finely enough (2 samples per meter) not to skip a wall.
    """
    dist = math.hypot(x1 - x0, y1 - y0)
    steps = max(1, math.ceil(dist * 2))
    walls = 0
    prev_cell = (-1, -1)
    for i in range(steps + 1):
        t = i / steps
        cx = round(x0 + (x1 - x0) * t)
        cy = round(y0 + (y1 - y0) * t)
        if 0 <= cx < GRID_W and 0 <= cy < GRID_H and floor.is_wall(cx, cy):
            if (cx, cy) != prev_cell:
                walls += 1
                prev_cell = (cx, cy)
    return walls


def rssi_at_point(floor: Floorplan, radio: RadioConfig,
                   tx_x: float, tx_y: float, rx_x: float, rx_y: float) -> float:
    """
    Predicted received signal strength (RSSI, in dBm) at (rx_x, rx_y) from a
    transmitter at (tx_x, tx_y), combining:

        RSSI = TX_POWER - FreeSpacePathLoss - (walls_crossed * wall_loss)
               [- extra penalty if the path is non-line-of-sight]

    This is the standard "multi-wall model" used in indoor radio planning.
    """
    dx, dy = rx_x - tx_x, rx_y - tx_y
    distance_m = max(0.5, math.hypot(dx, dy)) * METERS_PER_CELL
    fspl = free_space_path_loss_db(distance_m, radio.freq_mhz)
    walls = count_walls_crossed(floor, tx_x, tx_y, rx_x, rx_y)

    rssi = TX_POWER_DBM - fspl - walls * radio.wall_loss_db
    if walls >= 2:
        # Crossing 2+ walls implies an indirect / obstructed path: add the
        # corner-diffraction and non-line-of-sight penalties.
        rssi += CORNER_PENALTY_DB + NON_LINE_OF_SIGHT_PENALTY_DB
    return rssi


def compute_rf_grid(floor: Floorplan, radio: RadioConfig,
                     router_positions: list[tuple[int, int]]) -> np.ndarray:
    """
    Compute the full RSSI heatmap (one value per grid cell) given a list of
    router (x, y) positions. Where multiple routers cover the same cell, we
    keep the *strongest* signal — a receiver naturally locks onto whichever
    access point it hears best.
    """
    grid = np.full((GRID_H, GRID_W), NOISE_FLOOR_DBM, dtype=np.float32)
    if not router_positions:
        return grid

    for (rx, ry) in router_positions:
        for y in range(GRID_H):
            for x in range(GRID_W):
                if floor.is_wall(x, y):
                    continue
                rssi = rssi_at_point(floor, radio, rx + 0.5, ry + 0.5, x + 0.5, y + 0.5)
                if rssi > grid[y, x]:
                    grid[y, x] = rssi
    return grid


def greedy_router_placement(floor: Floorplan, radio: RadioConfig, n_routers: int,
                             coverage_threshold_dbm: float,
                             candidate_step: int = 2,
                             min_router_spacing_m: float = 5.0
                             ) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    Greedy maximum-coverage router placement.

    The "choose K points to maximize covered area" problem is NP-hard, but
    the greedy heuristic — repeatedly adding whichever new router location
    covers the most *previously uncovered* floor area — is the textbook
    approximation algorithm and is guaranteed to reach at least
    (1 - 1/e) ≈ 63% of the truly optimal coverage (Nemhauser, Wolsey &
    Fisher, 1978, on submodular set-function maximization).

    We restrict candidate router sites to a coarse grid (every
    `candidate_step` cells) purely for computational speed, and keep newly
    placed routers at least `min_router_spacing_m` apart so they don't
    cluster uselessly on top of each other.
    """
    candidates = [
        (x, y)
        for y in range(candidate_step, GRID_H - candidate_step, candidate_step)
        for x in range(candidate_step, GRID_W - candidate_step, candidate_step)
        if not floor.is_wall(x, y)
    ]

    placed: list[tuple[int, int]] = []
    best_grid = np.full((GRID_H, GRID_W), NOISE_FLOOR_DBM, dtype=np.float32)

    for _ in range(n_routers):
        best_candidate = None
        best_covered_count = -1
        best_candidate_grid = None

        for (cx, cy) in candidates:
            if any(math.hypot(cx - px, cy - py) < min_router_spacing_m for px, py in placed):
                continue  # too close to an already-placed router

            trial_grid = np.full((GRID_H, GRID_W), NOISE_FLOOR_DBM, dtype=np.float32)
            for y in range(GRID_H):
                for x in range(GRID_W):
                    if floor.is_wall(x, y):
                        continue
                    trial_grid[y, x] = rssi_at_point(
                        floor, radio, cx + 0.5, cy + 0.5, x + 0.5, y + 0.5
                    )

            combined = np.maximum(best_grid, trial_grid)
            covered_count = int(np.sum(combined >= coverage_threshold_dbm))

            if covered_count > best_covered_count:
                best_covered_count = covered_count
                best_candidate = (cx, cy)
                best_candidate_grid = combined

        if best_candidate is None:
            break  # no more useful spot to add (e.g. ran out of candidates)

        placed.append(best_candidate)
        best_grid = best_candidate_grid

    return placed, best_grid


# ============================================================================
# 4. AGENT MOVEMENT: A* PATHFINDING ON THE FLOORPLAN GRID
# ============================================================================

def astar_path(floor: Floorplan, start: tuple[int, int], goal: tuple[int, int]
               ) -> Optional[list[tuple[int, int]]]:
    """
    Classic A* shortest-path search on the occupancy grid.

    A* explores outward from `start`, always expanding the most promising
    node first, where "promising" = (cost so far) + (heuristic estimate of
    remaining cost). With an admissible heuristic (one that never
    overestimates the true remaining distance), A* is guaranteed to find
    the shortest path while typically visiting far fewer nodes than a plain
    breadth-first search.

    We use the Euclidean distance as the heuristic (admissible here, since
    diagonal steps are allowed and cost sqrt(2)), and allow the 8
    neighboring cells (4 straight + 4 diagonal), matching how a person
    actually walks through a room.
    """
    if floor.is_wall(*goal):
        return None

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0  # tie-breaker so heap comparisons never compare tuples of coords
    heapq.heappush(open_heap, (heuristic(start, goal), counter, start))

    g_score = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    visited = {start}

    max_iterations = 3000  # safety cap, matches the spirit of the original
    iterations = 0

    while open_heap and iterations < max_iterations:
        iterations += 1
        _, _, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruct the path by walking back through came_from.
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
                continue
            if floor.is_wall(nx, ny):
                continue
            neighbor = (nx, ny)
            if neighbor in visited:
                continue
            visited.add(neighbor)

            step_cost = math.sqrt(2) if (dx and dy) else 1.0
            tentative_g = g_score[current] + step_cost
            g_score[neighbor] = tentative_g
            came_from[neighbor] = current
            counter += 1
            f_score = tentative_g + heuristic(neighbor, goal)
            heapq.heappush(open_heap, (f_score, counter, neighbor))

    return None  # no path found within the search budget


@dataclass
class Agent:
    """A simulated person wandering the house, room to room."""
    x: float
    y: float
    floor: Floorplan = field(repr=False)
    path: list[tuple[int, int]] = field(default_factory=list)
    path_index: int = 0

    @classmethod
    def spawn(cls, floor: Floorplan) -> "Agent":
        gx, gy = floor.random_free_cell()
        agent = cls(x=gx + 0.5, y=gy + 0.5, floor=floor)
        agent._pick_new_destination()
        return agent

    def _pick_new_destination(self):
        start = (round(self.x - 0.5), round(self.y - 0.5))
        goal = self.floor.random_free_cell()
        path = astar_path(self.floor, start, goal)
        if path and len(path) > 1:
            self.path, self.path_index = path, 0
        else:
            self.path, self.path_index = [], 0

    def step(self):
        """Advance one cell along the current path; pick a new one if needed."""
        if not self.path or self.path_index >= len(self.path):
            self._pick_new_destination()
            return
        nx, ny = self.path[self.path_index]
        self.x, self.y = nx + 0.5, ny + 0.5
        self.path_index += 1
        if self.path_index >= len(self.path):
            self._pick_new_destination()


# ============================================================================
# 5. CROWD DENSITY: GAUSSIAN KERNEL DENSITY ESTIMATION
# ============================================================================

def compute_density_grid(agents: list[Agent], sigma_m: float = 2.5) -> np.ndarray:
    """
    Estimate a smooth "crowdedness" field from discrete agent positions
    using Gaussian Kernel Density Estimation (KDE) — the standard
    statistical technique for turning point samples into a continuous
    density surface (used widely in crowd analytics, geostatistics, and
    epidemiology heatmaps).

    Each agent contributes a 2D Gaussian "bump" of the form
        exp(-distance^2 / (2 * sigma^2))
    centered on their location; summing these bumps over all agents gives
    the final density grid. `sigma_m` controls how far each person's
    "presence" spreads out, in meters (= grid cells here).
    """
    density = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    two_sigma_sq = 2 * sigma_m * sigma_m
    radius = math.ceil(sigma_m * 3)  # ~3 sigma covers >99% of the Gaussian's mass

    for agent in agents:
        ax, ay = agent.x, agent.y
        cell_x, cell_y = round(ax), round(ay)
        for dy in range(-radius, radius + 1):
            ny = cell_y + dy
            if not (0 <= ny < GRID_H):
                continue
            for dx in range(-radius, radius + 1):
                nx = cell_x + dx
                if not (0 <= nx < GRID_W):
                    continue
                dist_sq = (ax - nx) ** 2 + (ay - ny) ** 2
                density[ny, nx] += math.exp(-dist_sq / two_sigma_sq)

    return density


# ============================================================================
# 6. COVERAGE STATISTICS
# ============================================================================

@dataclass
class CoverageStats:
    coverage_pct: float
    dead_zone_pct: float
    avg_rssi_dbm: float
    n_agents: int


def compute_coverage_stats(floor: Floorplan, rf_grid: Optional[np.ndarray],
                            coverage_threshold_dbm: float, n_agents: int
                            ) -> CoverageStats:
    if rf_grid is None:
        return CoverageStats(0.0, 100.0, NOISE_FLOOR_DBM, n_agents)

    walkable = floor.occ > WALL
    total_cells = int(np.sum(walkable))
    if total_cells == 0:
        return CoverageStats(0.0, 100.0, NOISE_FLOOR_DBM, n_agents)

    covered = int(np.sum((rf_grid >= coverage_threshold_dbm) & walkable))
    coverage_pct = 100.0 * covered / total_cells

    signal_present = (rf_grid > NOISE_FLOOR_DBM) & walkable
    if np.any(signal_present):
        avg_rssi = float(np.mean(rf_grid[signal_present]))
    else:
        avg_rssi = NOISE_FLOOR_DBM

    return CoverageStats(
        coverage_pct=coverage_pct,
        dead_zone_pct=100.0 - coverage_pct,
        avg_rssi_dbm=avg_rssi,
        n_agents=n_agents,
    )
