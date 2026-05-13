# 🦌 Stochastic Resonance Wildlife Detection System

A real-time wildlife roadkill prevention and alert system leveraging **Non-linear Bistable Double-Well Potential Integration** and **Adaptive SDFT Spectral Flattening**.

## 📌 Project Overview
This system is designed to detect the subtle, transient footsteps of four-legged wildlife approaching roads, even when their signals are completely submerged beneath heavy, continuous environmental noise (e.g., vehicle engine vibrations). By utilizing the physics of **Stochastic Resonance (SR)**, the system constructively utilizes background noise to amplify weak biological signals into a recognizable binary state.

## ⚙️ Core Architecture & Modules

The pipeline is strictly mapped to the mathematical chapters of the research report and fully modularized:

- **`main.py`** 
  The master orchestrator. Coordinates serial intake, data buffering (1.28s window), and the synchronized execution of all sub-modules.
  
- **`sdft_filter.py` (Spectral Flattening Filter)**
  Monitors the frequency domain via Sliding DFT. Calibrates to ambient noise for the first 5 seconds, then applies targeted **Spectral Flattening (Whitening)**. It dynamically calculates independent attenuation gains for noise-heavy frequency bins to perfectly flatten them to the surrounding background baseline, while preserving transient footstep impulses.
  
- **`msr_engine.py` (Bistable Double-Well Potential Engine)**
  The mathematical core. Solves the Non-linear Langevin Equation ($dx/dt = ax - bx^3 + S + N$) in real-time using Euler-Maruyama numerical integration. Outputs a quantized Telegraph Binary Signal ($sgn(x(t))$) indicating the physical state-hopping of a particle between two potential wells.
  
- **`step_analyzer.py` (Engineering Step Tracker)**
  Uses derivative-based Schmitt Trigger logic to track the exact physical duration of the footsteps.

- **`ui_manager.py` (Real-Time Visualization)**
  A 3-stage Matplotlib dashboard providing zero-latency visual feedback of the filtered signal, the binary double-well state, and the AI noise-tracking spectrum.

## 🚀 How to Run

1. Connect your Arduino vibration sensor.
2. Verify the `PORT` (e.g., `COM4`) in `main.py`.
3. Install dependencies:
   ```bash
   pip install numpy matplotlib pyserial
   ```
4. Run the orchestrator:
   ```bash
   python main.py
   ```

## 📊 Detection Logic
The system completely abandons traditional amplitude-based thresholds. An alert is only triggered when the **Net Transition Frequency ($N(t) - K(t) \ge 3$)** criteria is met within the 1.28s observation window, ensuring a mathematical necessity and sufficiency for continuous biological walking rhythms.
