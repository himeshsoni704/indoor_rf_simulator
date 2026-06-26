# Indoor RF + Crowd Digital Twin (Python version)

A Python port of the JS digital-twin simulator: a small house floorplan with
realistic RF (Wi-Fi/LoRa) coverage modeling and simple crowd simulation.

## Files

- `indoor_rf_digital_twin.py` — all the science: floorplan grid, RF
  propagation physics, A* pathfinding, crowd density estimation, and the
  router-placement optimizer. Fully commented to explain *why* each formula
  is used (Friis free-space path loss, multi-wall attenuation, Gaussian KDE,
  A* search, greedy max-coverage). No UI code here — you can import and use
  these functions on their own (e.g. in a Jupyter notebook or batch script).
- `run_simulator.py` — the interactive Matplotlib viewer that wires the
  physics up to clickable sliders, buttons, and live-updating heatmaps.

## Setup

```bash
pip install numpy scipy matplotlib
```

(`scipy` isn't strictly required by the current code but is a natural
companion if you later want to swap the hand-rolled KDE/A* for `scipy`'s
built-in versions.)

## Run

```bash
python3 run_simulator.py
```

A window opens with four panels:

1. **Floorplan + agents** — click to drop a new router, right-click to
   remove the nearest one.
2. **RF signal heatmap** — color-coded received signal strength (RSSI) in
   dBm, recomputed live as you move routers or change the radio.
3. **Crowd density heatmap** — a smooth Gaussian KDE field showing where
   people are clustering.
4. **Live statistics** — % of the home with usable signal ("coverage"),
   % without ("dead zones"), average RSSI, and active agent count.

Sliders let you change the coverage threshold, number of simulated people,
and number of routers the auto-optimizer should place. Radio buttons switch
between LoRa 868/915 MHz and Wi-Fi 2.4/5 GHz — notice how the lower-frequency
LoRa bands penetrate walls much better, while 5 GHz Wi-Fi has the shortest
range, which is physically correct behavior, not just a stylistic choice.

## The science, in one paragraph

Each grid cell is 1 square meter. For any router-to-point pair, we compute
free-space path loss from the Friis transmission equation (`32.44 +
20*log10(distance_km) + 20*log10(freq_MHz)`), then subtract a fixed
per-wall attenuation for every wall the straight-line path crosses, plus an
extra penalty if the path has to bend around a corner. Received signal
strength is just `transmit power - all those losses`, in dBm. Crowd density
is estimated with Gaussian kernel density estimation (every person
contributes a soft "bump" of presence), and people move using A* search so
they walk realistic, wall-respecting routes between rooms. Router placement
is solved with the standard greedy approximation algorithm for the
maximum-coverage problem.
