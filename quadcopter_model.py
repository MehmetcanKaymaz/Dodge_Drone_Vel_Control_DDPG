import math

def dynamic_model(states, U, Nsolver, dt):
    x, y, z, u, v, w, phi, theta, psi, p, q, r = states
    tau_roll, tau_pitch, tau_yaw, f = U

    Ixx = 0.0075
    Iyy = 0.0075
    Izz = 0.013
    m = 0.65
    g = 9.81

    for i in range(Nsolver):
        state_dot = (u,
                     v,
                     w,
                     f * (math.cos(phi) * math.sin(theta) * math.cos(psi) + math.sin(phi) * math.sin(psi)) / m,
                     f * (math.cos(phi) * math.sin(theta) * math.sin(psi) - math.sin(phi) * math.cos(psi)) / m,
                     f * (math.cos(phi) * math.cos(theta)) / m - g,
                     p,
                     q,
                     r,
                     q * r * ((Iyy - Izz) / Ixx) + tau_roll / Ixx,
                     p * r * ((Izz - Ixx) / Iyy) + tau_pitch / Iyy,
                     q * p * ((Ixx - Iyy) / Izz) + tau_yaw / Izz
                     )
        for j in range(len(states)):
            states[j]=states[j]+state_dot[j]*dt

    return states