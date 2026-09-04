import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import allantools
import os

CSV_FILE = "imu_extracted_raw2.csv"
OUT_DIR = "allan_results_raw"

os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(CSV_FILE)

t = df["timestamp_sec"].to_numpy()
dt = np.mean(np.diff(t))
fs = 1.0 / dt

print(f"Estimated sampling rate: {fs:.2f} Hz")


# ----------------------------
# Signals
# ----------------------------
gyro = {
    "x": df["gyro_x"].to_numpy()/32768.0 * 2000 * np.pi / 180,
    "y": df["gyro_y"].to_numpy()/32768.0 * 2000 * np.pi / 180,
    "z": df["gyro_z"].to_numpy()/32768.0 * 2000 * np.pi / 180
}

accel = {
    "x": df["accel_x"].to_numpy()/ 32768.0 * 16 * 9.8,
    "y": df["accel_y"].to_numpy()/ 32768.0 * 16 * 9.8,
    "z": df["accel_z"].to_numpy()/ 32768.0 * 16 * 9.8
}


# ----------------------------
# Allan variance function
# ----------------------------
def allan(signal, fs):
    taus, adev, _, _ = allantools.oadev(
        signal,
        rate=fs,
        data_type="freq",
        taus="octave"
    )
    return taus, adev


# ----------------------------
# Group analysis + plotting
# ----------------------------
def analyze_group(name, signals, unit):
    plt.figure(figsize=(8, 6))

    results = {}

    for axis, sig in signals.items():
        taus, adev = allan(sig, fs)
        results[axis] = (taus, adev)

        plt.loglog(taus, adev, label=f"{axis}")

    plt.xlabel("Averaging time τ (s)")
    plt.ylabel(f"Allan deviation ({unit})")
    plt.title(f"Allan Variance - {name}")
    plt.grid(True, which="both")
    plt.legend()

    save_path = os.path.join(OUT_DIR, f"allan_{name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")

    return results


gyro_res = analyze_group("gyro", gyro, "rad/s")
accel_res = analyze_group("accel", accel, "m/s²")


# Noise parameter estimation
def metrics(taus, adev):

    log_tau = np.log10(taus)
    log_adev = np.log10(adev)

    # slope between consecutive points
    slope = np.diff(log_adev) / np.diff(log_tau)

    # --------------------------------------------------
    # White noise coefficient N (slope ≈ -0.5)
    # --------------------------------------------------
    mask_N = np.abs(slope + 0.5) < 0.1

    x_N = log_tau[:-1][mask_N]
    y_N = log_adev[:-1][mask_N]

    if len(x_N) >= 2:
        m_N, c_N = np.polyfit(x_N, y_N, 1)
        # Evaluate fitted line at log10(tau)=0 (tau = 1 s)
        log_N = c_N
        N = 10**log_N
    else:
        # Fallback if insufficient points
        idx = np.argmin(np.abs(slope + 0.5))
        N = adev[idx] * np.sqrt(taus[idx])

    # --------------------------------------------------
    # Bias instability B
    # --------------------------------------------------
    idx_B = np.argmin(adev)
    tau_B = taus[idx_B]
    B = adev[idx_B] / 0.664

    # --------------------------------------------------
    # Rate random walk K (slope ≈ +0.5)
    # --------------------------------------------------
    mask_K = np.abs(slope - 0.5) < 0.1

    x_K = log_tau[:-1][mask_K]
    y_K = log_adev[:-1][mask_K]

    if len(x_K) >= 2:
        m_K, c_K = np.polyfit(x_K, y_K, 1)

        # Evaluate fitted line at tau = 3 s
        log_tau3 = np.log10(3.0)
        log_sigma3 = m_K * log_tau3 + c_K
        sigma3 = 10**log_sigma3

        K = sigma3
    else:
        idx = np.argmin(np.abs(slope - 0.5))
        K = adev[idx] * np.sqrt(3.0 / taus[idx])

    return N, B, K, tau_B

def print_group(name, results):
    print(f"\n===== {name.upper()} =====")

    for axis, (taus, adev) in results.items():

        N, B, K, tau_B = metrics(taus, adev)

        print(f"\n{axis}-axis:")
        print(f"  White noise coefficient (N): {N:.6e}")
        print(f"  Bias instability (B):        {B:.6e}")
        print(f"  Rate random walk (K):        {K:.6e}")
        print(f"  Bias minimum at τ = {tau_B:.3f} s")


print_group("gyro", gyro_res)
print_group("accel", accel_res)


