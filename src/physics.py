import torch
import torch.nn as nn
import torch.autograd as grad

def physics_loss(model, x, y):
    c_x = 0.25
    c_y = 0.5
    r = 0.1
    mask = (x - c_x)**2 + (y - c_y)**2
    final_mask = mask >= r**2
    x_fluid = x[final_mask].reshape(-1, 1)
    y_fluid = y[final_mask].reshape(-1, 1)
    x_inputs = x_fluid.requires_grad_(True)
    y_inputs = y_fluid.requires_grad_(True)
    Inputs = torch.cat([x_inputs, y_inputs], dim=1)
    
    predictions = model(Inputs) 

    u = predictions[:, 0].view(len(x_fluid), 1)
    v = predictions[:, 1].view(len(x_fluid), 1)
    p = predictions[:, 2].view(len(x_fluid), 1)

    u_x = grad.grad(u, x_inputs, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = grad.grad(v, x_inputs, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    p_x = grad.grad(p, x_inputs, grad_outputs=torch.ones_like(p), create_graph=True)[0] 

    u_xx = grad.grad(u_x, x_inputs, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    v_xx = grad.grad(v_x, x_inputs, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]

    u_y = grad.grad(u, y_inputs, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_y = grad.grad(v, y_inputs, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    p_y = grad.grad(p, y_inputs, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    u_yy = grad.grad(u_y, y_inputs, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_yy = grad.grad(v_y, y_inputs, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    # Physics Loss
    ### Continuity
    Continuity = u_x + v_y
    Continuity_Target = torch.zeros_like(Continuity)

    ### Momentum
    # X-Momentum
    nu = 0.01 / torch.pi
    X_Momentum = u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    X_Momentum_Target = torch.zeros_like(X_Momentum)

    # Y-Momentum
    Y_Momentum = u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    Y_Momentum_Target = torch.zeros_like(Y_Momentum)

    # MSE Losses
    mse = nn.MSELoss()

    C_loss = mse(Continuity_Target, Continuity)
    X_M_loss = mse(X_Momentum_Target, X_Momentum)
    Y_M_loss = mse(Y_Momentum_Target, Y_Momentum)

    Total_Loss = C_loss + X_M_loss + Y_M_loss

    # Boundary Conditions
    ### Bottom Wall
    x_bottom = torch.rand(100, 1)
    y_bottom = torch.zeros_like(x_bottom)

    Bottom_wall = torch.cat([x_bottom, y_bottom], dim=1)

    no_slip_bottom = model(Bottom_wall)

    bottom_vel = no_slip_bottom[:, 0:2]

    bottom_vel_target = torch.zeros_like(Bottom_wall)

    Bottom_loss = mse(bottom_vel_target, bottom_vel)

    # Top Wall
    x_top = torch.rand(100, 1)
    y_top = torch.ones_like(x_top) 

    Top_wall = torch.cat([x_top, y_top], dim=1)

    no_slip_top = model(Top_wall)

    top_u_vel = no_slip_top[:, 0].view(len(x_top), 1)
    top_v_vel = no_slip_top[:, 1].view(len(y_top), 1)
    top_vel = torch.cat([top_u_vel, top_v_vel], dim=1)

    top_vel_target = torch.zeros_like(Top_wall)

    Top_loss = mse(top_vel_target, top_vel)

    ### Left Wall
    x_left = torch.zeros(100, 1)
    y_left = torch.rand(100, 1)

    Left_wall = torch.cat([x_left, y_left], dim=1)

    constant_left_wall = model(Left_wall)

    constant_vel = constant_left_wall[:, 0:2]

    constant_u_vel_target = torch.ones_like(x_left)
    constant_v_vel_target = torch.zeros_like(x_left)
    constant_vel_target = torch.cat([constant_u_vel_target, constant_v_vel_target], dim=1)

    Left_loss = mse(constant_vel_target, constant_vel)

    ### Right Wall
    x_right = torch.ones(100, 1)
    y_right = torch.rand(100, 1)

    Right_wall = torch.cat([x_right, y_right], dim=1)

    Zero_Pressure_right = model(Right_wall)

    Right_pressure = Zero_Pressure_right[:, 2:3]

    Right_target = torch.zeros_like(Right_pressure)

    Right_loss = mse(Right_target, Right_pressure)

    ### Cylinder
    theta = torch.linspace(0, 2*torch.pi, 100)

    x_cylinder = c_x + r*torch.cos(theta).reshape(-1, 1)
    y_cylinder = c_y + r*torch.sin(theta).reshape(-1, 1)

    cylinder_wall = torch.cat([x_cylinder, y_cylinder], dim=1) 

    cyl_predictions = model(cylinder_wall)

    cyl_x_y = cyl_predictions[:, 0:2]

    cylinder_targets = torch.zeros_like(cyl_x_y)

    Cylinder_loss = mse(cylinder_targets, cyl_x_y)

    Final_loss = 10*Total_Loss + Bottom_loss + Top_loss + 10*Left_loss + Right_loss + Cylinder_loss

    return Final_loss