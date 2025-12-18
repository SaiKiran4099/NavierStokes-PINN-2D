import torch
import torch.optim as op
import matplotlib.pyplot as plt
import numpy as np
import os
from model import NavierStokes2D
from physics import physics_loss

# Initialize Model
model = NavierStokes2D()

optimizer = op.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoches in range(1, 10001):
    optimizer.zero_grad()
    x = torch.rand(100, 1)
    y = torch.rand(100, 1)
   
    loss = physics_loss(model, x, y)
    
    loss.backward()
    optimizer.step()

    if epoches % 1000 == 0:
        print(f"Epoch {epoches}: Loss = {loss.item()}")

# Plotting Function
def plot_steady_state_flow(model):
    c_x, c_y, r = 0.25, 0.5, 0.1 
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    output_dir = 'steady_flow_results'
    os.makedirs(output_dir, exist_ok=True)

    x_points = torch.linspace(x_min, x_max, 100)
    y_points = torch.linspace(y_min, y_max, 100)
    xg, yg = torch.meshgrid(x_points, y_points, indexing='ij')

    x_grid = xg.reshape(-1, 1)
    y_grid = yg.reshape(-1, 1)

    grid_inputs = torch.cat([x_grid, y_grid], dim=1)

    model.eval()
    with torch.no_grad():
        predictions = model(grid_inputs)

    u_vel = predictions[:, 0].reshape(100, 100)
    u_plot = u_vel.detach().numpy()

    mask_plot = (xg.numpy() - c_x)**2 + (yg.numpy() - c_y)**2 < r**2
    u_plot[mask_plot] = np.nan

    plt.figure(figsize=(10, 6))

    finite_u = u_plot[np.isfinite(u_plot)]
    if len(finite_u) > 0:
        vmin_val = np.percentile(finite_u, 1)
        if vmin_val >= 0: 
            vmin_val = 0
    else:
        vmin_val = 0
    vmax_val = 1.05

    plt.contourf(xg.numpy(), yg.numpy(), u_plot, levels=50, cmap='jet', vmin=vmin_val, vmax=vmax_val) 
    plt.colorbar(label='Horizontal Velocity u (m/s)')
    
    cylinder_circle = plt.Circle((c_x, c_y), r, color='black', fill=True)
    plt.gca().add_patch(cylinder_circle)

    plt.title('Steady-State Flow Past Cylinder (Re ≈ 10)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')

    plot_filename = os.path.join(output_dir, 'steady_state_flow_final.png')
    plt.savefig(plot_filename)
    plt.close()

    print(f"\n--- Steady-State Visualization Complete ---")
    print(f"Saved final plot to '{plot_filename}'")

# Execute Plotting
plot_steady_state_flow(model)