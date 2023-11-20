import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Transfer function coefficients
#num = [7794.29437875918, 165.979218425375]
#den = [1, 534.867143038177, 9686.97075576594, 165.979218425375]

num = [6023]
den = [1, 477.4, 6023]

# Create the system
system = signal.TransferFunction(num, den)

# Time vector (assuming you want to simulate for 1 second)
time_step = 100*0.025
sampling_time = 0.002
t = np.arange(0, time_step, sampling_time)

# Define the input signal X here
# For example, a step input:
desired_rad_s = 25.3222
X = desired_rad_s*np.ones_like(t)
Kp = 0.4
Ki = 0.001

# Compute the response
t_out, y_out, _ = signal.lsim(system, X, t, X0=0)

# Calculate the total displacement using trapezoidal integration
total_displacement = np.trapz(y_out, t_out)

# Plot the response
plt.plot(t_out, y_out)
plt.title('Response of the Brushless DC Motor')
plt.xlabel('Time (seconds)')
plt.ylabel('Response')
plt.grid(True)
plt.show()
print(total_displacement)

######

def pi_controller(Kp, Ki, setpoint, input_signal, dt):
    """
    Simple PI controller.
    
    Kp: Proportional gain
    Ki: Integral gain
    setpoint: Desired setpoint
    input_signal: Input signal (e.g., system output)
    dt: Time step
    """
    error = setpoint - input_signal
    integral = 0
    output = np.zeros_like(input_signal)
    
    for i in range(1, len(input_signal)):
        integral += error[i] * dt
        output[i] = Kp * error[i] + Ki * integral

    return output

# Parameters
Kp = 0.5  # Proportional gain, adjust as needed
Ki = 0.1  # Integral gain, adjust as needed
setpoint = 1.0  # Desired setpoint
dt = 0.002  # Time step
time = np.arange(0, 0.3, dt)  # Total time for simulation (0.3 s)

# Example input signal (replace this with your actual input)
input_signal = np.zeros_like(time)

# Apply the PI controller
output_signal = pi_controller(Kp, Ki, setpoint, input_signal, dt)

# Plot
plt.plot(time, output_signal)
plt.title('PI Controller Output')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.grid(True)
plt.show()
