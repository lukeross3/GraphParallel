import numpy as np

# Windowed average for smoother plotting
def window_avg(times, N):
    y = np.convolve(times, np.ones((N,))/N, mode='valid')
    x = np.arange(len(y)) + N/2
    return (x, y)

# Compute linear best fit line for plotting
def fit(times):
    x = np.arange(len(times))
    y = np.array(times)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    x_r = [x[0], x[-1]]
    y_r = [m*x[0] + c, m*x[-1] + c]
    return (x_r, y_r)

# String formatting for printing times
def time_str(time_len):
    unit = ' seconds'
    if time_len > 60:
        time_len /= 60
        unit = ' minutes'
    if time_len > 60:
        time_len /= 60
        unit = ' hours'
    return "{:.2f}".format(time_len) + unit