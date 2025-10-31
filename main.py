import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def torque_on_dipole(p, E):
    """
    Calculate the torque on an electric dipole in a 2D electric field.

    Parameters:
    p (numpy array): 2-element array representing the dipole moment vector [px, py].
    E (numpy array): 2-element array representing the electric field vector [Ex, Ey].

    Returns:
    float: The torque (scalar, since 2D rotation around z-axis).
    """
    return p[0] * E[1] - p[1] * E[0]

def electric_field_due_to_dipole(p, r):
    """
    Calculate the electric field at a point due to an electric dipole in 2D approximation.

    Parameters:
    p (numpy array): 2-element array representing the dipole moment vector [px, py].
    r (numpy array): 2-element array representing the position vector from the dipole to the point [rx, ry].

    Returns:
    numpy array: 2-element array representing the electric field vector [Ex, Ey] at the point.
    """
    r_magnitude = np.linalg.norm(r)
    if r_magnitude == 0:
        return np.zeros(2)
    r_magnitude_cubed = r_magnitude ** 3  # Approximation using 3D formula for simplicity
    p_dot_r = np.dot(p, r)
    Ex = 3 * r[0] * p_dot_r / r_magnitude_cubed - p[0] / r_magnitude_cubed
    Ey = 3 * r[1] * p_dot_r / r_magnitude_cubed - p[1] / r_magnitude_cubed
    return np.array([Ex, Ey])

class Dipole:
    def __init__(self, position, dipole_moment, moment_of_inertia=1.0):
        self.position = np.array(position)  # [x, y]
        self.dipole_moment = np.array(dipole_moment)  # [px, py]
        self.magnitude = np.linalg.norm(dipole_moment)
        self.angle = math.atan2(dipole_moment[1], dipole_moment[0])  # Initial angle
        self.angular_velocity = 0.0
        self.I = moment_of_inertia

def simulate_dipoles(dipole1, dipole2, dt=0.01, total_time=10.0):
    num_steps = int(total_time / dt)
    angles1 = []
    angles2 = []
    
    for _ in range(num_steps):
        # Calculate electric field at dipole1 due to dipole2
        r12 = dipole1.position - dipole2.position
        E_at_1 = electric_field_due_to_dipole(dipole2.dipole_moment, r12)
        
        # Torque on dipole1
        torque1 = torque_on_dipole(dipole1.dipole_moment, E_at_1)
        
        # Calculate electric field at dipole2 due to dipole1
        r21 = dipole2.position - dipole1.position
        E_at_2 = electric_field_due_to_dipole(dipole1.dipole_moment, r21)
        
        # Torque on dipole2
        torque2 = torque_on_dipole(dipole2.dipole_moment, E_at_2)
        
        # Update angular velocities
        dipole1.angular_velocity += torque1 / dipole1.I * dt
        dipole2.angular_velocity += torque2 / dipole2.I * dt
        
        # Update angles
        dipole1.angle += dipole1.angular_velocity * dt
        dipole2.angle += dipole2.angular_velocity * dt
        
        # Update dipole moments
        dipole1.dipole_moment = dipole1.magnitude * np.array([math.cos(dipole1.angle), math.sin(dipole1.angle)])
        dipole2.dipole_moment = dipole2.magnitude * np.array([math.cos(dipole2.angle), math.sin(dipole2.angle)])
        
        # Store angles for animation
        angles1.append(dipole1.angle)
        angles2.append(dipole2.angle)
    
    return angles1, angles2

def animate_dipoles(dipole1, dipole2, angles1, angles2, dt=0.01):
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        
        # Plot positions
        ax.scatter(dipole1.position[0], dipole1.position[1], color='red', s=100)
        ax.scatter(dipole2.position[0], dipole2.position[1], color='blue', s=100)
        
        angle1 = angles1[frame]
        angle2 = angles2[frame]
        mom1 = dipole1.magnitude * np.array([math.cos(angle1), math.sin(angle1)])
        mom2 = dipole2.magnitude * np.array([math.cos(angle2), math.sin(angle2)])
        
        # Plot arrows
        ax.arrow(dipole1.position[0], dipole1.position[1], mom1[0]*0.5, mom1[1]*0.5, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red')
        ax.arrow(dipole2.position[0], dipole2.position[1], mom2[0]*0.5, mom2[1]*0.5, 
                 head_width=0.1, head_length=0.1, fc='blue', ec='blue')
        
        ax.set_title(f'Time: {frame * dt:.2f}s')
    
    ani = FuncAnimation(fig, update, frames=len(angles1), interval=50)
    plt.show()

# Example usage
if __name__ == "__main__":
    dipole1 = Dipole([0, 0], [1.0, 0.0])  # At origin, pointing right
    dipole2 = Dipole([1.0, 0.0], [0.0, 1.0])  # At (1,0), pointing up
    
    angles1, angles2 = simulate_dipoles(dipole1, dipole2)
    animate_dipoles(dipole1, dipole2, angles1, angles2)