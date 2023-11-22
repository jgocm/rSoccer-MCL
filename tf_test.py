import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def simulate_motor(system, control_inputs=[0], time_step=0.025, sampling_time=0.005):
    """
    Simulate output of a continuous-time linear system for a sequence of inputs.

    Parameters
    ----------
    system : an instance of the LTI class from a scipy transfer function.
    control_inputs : list or numpy array
        An input array containing the sequence of inputs that will be sent
        to the system updated at each time step.
    time_step : value (float)
        The time steps at which the inputs to the system are updated.
    sampling_time : value (float)
        The sampling time the system will be simulated and its feedbacks will be generated.

    Returns
    -------
    t_out : 1D ndarray
        Time values for the output.
    y_out : 1D ndarray
        System response.
    x_out : ndarray
        Time evolution of the state vector.    
    """
    n_steps = len(control_inputs)
    t = np.arange(0, n_steps*time_step, sampling_time)
    X = []
    for t_k in t:
        i = int(t_k/time_step)
        x_k = control_inputs[i]
        X.append(x_k)

    X = np.array(X)
    t_out, y_out, x_out = signal.lsim(system, X, t)

    return t_out, y_out, x_out

# Transfer function coefficients
num = [6023]
den = [1, 477.4, 6023]

# Create the system
system = signal.TransferFunction(num, den)

# Sequence of control inputs
control_inputs = [0, 25, 50, 0]

# Simulate system for sequential inputs
t_out, y_out, x_out = simulate_motor(system=system, 
                                     control_inputs=control_inputs,
                                     time_step=0.025,
                                     sampling_time=0.002)

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
