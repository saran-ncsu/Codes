import numpy as np

def get_response_spectrum(EQ, xi, dt, T):
    """
    Calculate response spectrum for earthquake data

    Parameters:
    ----------
    EQ : ndarray
        Earthquake acceleration data
    xi : float
        Damping ratio
    dt : float
        Time step of earthquake data
    T : ndarray
        Array of periods to calculate response for

    Returns:
    -------
    RS : ndarray
        Response spectrum values for each period
    """
    # Convert inputs to numpy arrays if they aren't already
    EQ = np.asarray(EQ)
    T = np.asarray(T)

    nT = len(T)
    RS = np.zeros(nT)

    # Precompute constants used in the time-stepping loop
    DDT = 0.001  # Integration time step
    NNG = len(EQ)
    TIA = NNG * dt
    NTIM = int(TIA / DDT)

    # Prepare linear interpolation of earthquake data for efficiency
    times = np.arange(0, NNG) * dt
    interp_times = np.arange(0, TIA, DDT)
    interp_EQ = np.interp(interp_times, times, EQ)

    for i in range(nT):
        Tn = T[i]
        omega = 2 * np.pi / Tn
        omega_damped = omega * np.sqrt(1 - xi**2)

        # Precompute all coefficients outside the time-stepping loop
        eterm = np.exp(-xi * omega * DDT)
        sinterm = np.sin(omega_damped * DDT)
        costerm = np.cos(omega_damped * DDT)

        # Precompute matrix coefficients
        a11 = eterm * (costerm + xi / np.sqrt(1 - xi**2) * sinterm)
        a12 = eterm / omega_damped * sinterm
        a21 = -omega / np.sqrt(1 - xi**2) * eterm * sinterm
        a22 = eterm * (costerm - xi / np.sqrt(1 - xi**2) * sinterm)

        # Terms for computing b coefficients
        term1 = (2 * xi**2 - 1) / (omega**2 * DDT)
        term2 = xi / omega
        term3 = 2 * xi / (omega**3 * DDT)
        term4 = 1 / omega**2

        b11 = eterm * (((term1 + term2) * sinterm / omega_damped +
                       (term3 + term4) * costerm)) - term3
        b12 = -eterm * (term1 * sinterm / omega_damped + term3 * costerm) - term4 + term3
        b21 = -(a11 - 1) / (omega**2 * DDT) - a12
        b22 = -b21 - a12

        # Initial conditions
        ui = 0.0
        udoti = 0.0
        umax = 0.0

        # Loop through the interpolated earthquake record
        for j in range(1, len(interp_times)):
            A1 = interp_EQ[j-1]
            A2 = interp_EQ[j]

            # State update equations
            ui1 = ui * a11 + udoti * a12 + A1 * b11 + A2 * b12
            udoti1 = ui * a21 + udoti * a22 + A1 * b21 + A2 * b22

            # Track maximum displacement
            umax = max(umax, abs(ui1))

            # Update state variables
            ui = ui1
            udoti = udoti1

        # Calculate spectral acceleration
        RS[i] = omega**2 * umax

    return RS

# Optional: JIT compilation for even better performance
# Uncomment if numba is available
# from numba import jit
# get_response_spectrum = jit(nopython=True)(get_response_spectrum)