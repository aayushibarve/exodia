import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import allantools
import os

CSV_FILE = "imu_extracted_fixed.csv"
OUT_DIR = "allan_results"

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
    "x": df["gyro_x"].to_numpy(),
    "y": df["gyro_y"].to_numpy(),
    "z": df["gyro_z"].to_numpy()
}

accel = {
    "x": df["accel_x"].to_numpy(),
    "y": df["accel_y"].to_numpy(),
    "z": df["accel_z"].to_numpy()
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
    # ARW / VRW approximation (−0.5 slope region)
    slope = np.diff(np.log(adev)) / np.diff(np.log(taus))
    idx_arw = np.argmin(np.abs(slope + 0.5))

    noise_density = adev[idx_arw] * np.sqrt(taus[idx_arw])

    # bias instability (minimum Allan deviation)
    idx_bias = np.argmin(adev)
    bias = adev[idx_bias]
    tau_bias = taus[idx_bias]

    return noise_density, bias, tau_bias


def print_group(name, results):
    print(f"\n===== {name.upper()} =====")

    for axis, (taus, adev) in results.items():
        noise, bias, tau_bias = metrics(taus, adev)

        print(f"\n{axis}-axis:")
        print(f"  Noise density: {noise:.6e}")
        print(f"  Bias instability: {bias:.6e} at τ = {tau_bias:.3f} s")


print_group("gyro", gyro_res)
print_group("accel", accel_res)


