# 2D Navier-Stokes Solver using Physics-Informed Neural Networks (PINNs)

## Overview
This project implements a **Physics-Informed Neural Network (PINN)** to solve the steady-state incompressible Navier-Stokes equations for laminar flow past a cylinder. Unlike traditional CFD methods (FVM/FEM) that rely on mesh generation, this solver approximates the solution using a deep neural network trained directly on the governing partial differential equations (PDEs).

**Key Tech Stack:** PyTorch, Python, NumPy, Matplotlib

## Problem Description
* **Domain:** 2D Channel Flow with a cylindrical obstacle.
* **Physics:** Steady-state incompressible Navier-Stokes.
* **Reynolds Number:** Re ≈ 10 (Laminar regime).
* **Boundary Conditions:**
    * Inlet: Parabolic velocity profile.
    * Outlet: Zero pressure gradient.
    * Walls/Cylinder: No-slip condition ($u=0, v=0$).

## Methodology
The network minimizes a composite loss function consisting of:
1.  **Physics Loss:** Residuals of the Continuity and Momentum equations computed via Automatic Differentiation (`torch.autograd`).
2.  **Boundary Loss:** MSE error at the inlet, outlet, walls, and cylinder surface.

## Architecture
* **Input:** $(x, y)$ spatial coordinates.
* **Output:** $(u, v, p)$ velocity fields and pressure.
* **Network:** Fully connected (MLP) with 3 hidden layers (30 neurons each) and Tanh activation.

## Results

<img width="1000" height="600" alt="result_plot" src="https://github.com/user-attachments/assets/e2d7cbdd-fef4-4d92-afeb-c2900d1ede36" />

## Structure
* `src/model.py`: PyTorch Neural Network architecture.
* `src/physics.py`: Navier-Stokes residuals and loss formulation.
* `src/train.py`: Training loop and visualization pipeline.

## Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/SaiKiran4099/NavierStokes-PINN-2D.git](https://github.com/SaiKiran4099/NavierStokes-PINN-2D.git)
   cd NavierStokes-PINN-2D
   ```
   
2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

## Usage:
```bash
 python3 src/train.py
```
