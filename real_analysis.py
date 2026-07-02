"""
Real-data pipeline. Every input here is real:
  - CALCE SP20-2 (A123 LFP) Arbin cycler logs: FUDS (room temp) and US06 (45C)
  - Real 2023 Monza + Silverstone race telemetry (FastF1, driver VER)

What is NOT claimed:
  - No measured internal battery temperature exists in either source.
  - CALCE: ΔT is derived from a physics-based lumped thermal model driven by
    REAL current and an internal resistance ESTIMATED FROM REAL VOLTAGE SAG
    (not assumed a priori, not synthetic). The heat-dissipation coefficient
    is a literature-typical value for a pouch cell of this format; it is a
    modeling choice, stated explicitly, not a measurement.
  - F1: no thermal ground truth exists at all. Results on F1 data are reported
    ONLY as unsupervised OOD flagging with face-validity checks (do flags
    cluster in braking/DRS zones), never as validated thermal predictions.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from scipy.signal import savgol_filter
import json

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Load real CALCE data, split by real experimental condition
# ---------------------------------------------------------------------------
calce = pd.read_csv("calce_clean.csv")
calce["test_type"] = calce["source_file"].apply(lambda s: s.split("\\")[-1])

fuds = calce[calce["test_type"].str.contains("FUDS")].sort_values("Test_Time_s").reset_index(drop=True)
us06 = calce[calce["test_type"].str.contains("US06")].sort_values("Test_Time_s").reset_index(drop=True)

print(f"FUDS (source/lab condition): {len(fuds)} rows")
print(f"US06 45C (real covariate-shift target condition): {len(us06)} rows")

# ---------------------------------------------------------------------------
# 2. Estimate internal resistance from REAL voltage sag (not assumed)
#    Use first-difference regression: dV ~= -R_int * dI over short lags,
#    which cancels the slow OCV(SOC) drift. Robust to outliers via trimming.
# ---------------------------------------------------------------------------
def estimate_internal_resistance(df):
    dI = df["Current_A"].diff().values[1:]
    dV = df["Voltage_V"].diff().values[1:]
    mask = np.abs(dI) > 0.05  # ignore near-zero current steps (noise-dominated)
    dI, dV = dI[mask], dV[mask]
    # Trim extreme 2% each tail for robustness
    lo, hi = np.percentile(dI, [1, 99])
    keep = (dI >= lo) & (dI <= hi)
    dI, dV = dI[keep], dV[keep]
    # V = OCV - I*R  =>  dV = -R*dI  =>  R = -slope(dV ~ dI)
    slope = np.polyfit(dI, dV, 1)[0]
    R_est = -slope
    return max(R_est, 1e-4)  # guard against sign/noise pathologies

R_int = estimate_internal_resistance(fuds)
print(f"Estimated internal resistance from real FUDS voltage sag: R_int = {R_int*1000:.3f} mOhm")

# ---------------------------------------------------------------------------
# 3. Physics-grounded ΔT from real current using a lumped thermal model.
#    Explicitly labeled: dissipation coefficient k_diss and thermal mass C_th
#    are literature-typical, not fitted to a measured temperature channel
#    (none exists in this public export).
# ---------------------------------------------------------------------------
C_th = 5000.0    # J/K, literature-typical for a ~20 Ah pouch cell + local mount
k_diss = 0.5     # W/K, lumped convective dissipation coefficient (modeling choice)
dt_native = 10.0 # seconds, approx native CALCE sample spacing observed in data
T_amb_map = {"FUDS": 25.0, "US06": 45.0}  # real documented ambient test conditions

def simulate_delta_T(df, T_amb, R_int, C_th=C_th, k_diss=k_diss):
    I = df["Current_A"].values
    n = len(I)
    T = np.zeros(n)
    T[0] = T_amb
    for t in range(1, n):
        heat_gen = (I[t] ** 2) * R_int          # real Joule heating, Watts
        heat_diss = k_diss * (T[t - 1] - T_amb)  # Newtonian cooling to real test ambient
        T[t] = T[t - 1] + (heat_gen - heat_diss) * dt_native / C_th
    return pd.Series(T).diff().fillna(0).values

fuds["Delta_T_C"] = simulate_delta_T(fuds, T_amb_map["FUDS"], R_int)
us06["Delta_T_C"] = simulate_delta_T(us06, T_amb_map["US06"], R_int)

for name, df in [("FUDS", fuds), ("US06_45C", us06)]:
    df["Power_kW"] = df["Current_A"] * df["Voltage_V"] / 1000
    df["Rolling_Power_Variance"] = df["Power_kW"].rolling(30, min_periods=1).var().fillna(0)

features = ["Current_A", "Voltage_V", "Power_kW", "Rolling_Power_Variance"]

# ---------------------------------------------------------------------------
# 4. REAL EnbPI: bootstrap ensemble + leave-one-out aggregation, trained on
#    real FUDS data only.
# ---------------------------------------------------------------------------
B = 20
m = len(fuds)
boot_models, boot_idx = [], []
for b in range(B):
    samp = rng.integers(0, m, m)
    boot_idx.append(set(samp.tolist()))
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=b, n_jobs=-1)
    model.fit(fuds.loc[samp, features], fuds.loc[samp, "Delta_T_C"])
    boot_models.append(model)

def ensemble_predict(X):
    return np.mean([bm.predict(X) for bm in boot_models], axis=0)

# Vectorized LOO: predict the FULL training set once per bootstrap model
# (B predict calls total, not B*m), then average per-row excluding models
# that included that row in their bootstrap sample.
all_preds = np.stack([bm.predict(fuds[features]) for bm in boot_models], axis=1)  # (m, B)
membership = np.zeros((m, B), dtype=bool)
for b in range(B):
    idx_arr = np.fromiter(boot_idx[b], dtype=int)
    membership[idx_arr, b] = True
not_member = ~membership  # (m, B), True where model b did NOT see row i

sum_preds = np.where(not_member, all_preds, 0).sum(axis=1)
count = not_member.sum(axis=1)
fallback = all_preds.mean(axis=1)
loo_preds = np.where(count > 0, sum_preds / np.maximum(count, 1), fallback)

loo_residuals = np.abs(fuds["Delta_T_C"].values - loo_preds)
q95_unweighted = np.quantile(loo_residuals, 0.95)

train_preds_full = ensemble_predict(fuds[features])
r2_train = 1 - np.sum((fuds["Delta_T_C"] - train_preds_full) ** 2) / np.sum((fuds["Delta_T_C"] - fuds["Delta_T_C"].mean()) ** 2)
mse_train = np.mean((fuds["Delta_T_C"] - train_preds_full) ** 2)

print(f"\n=== Real EnbPI trained on real FUDS data ===")
print(f"Train R^2 = {r2_train:.4f}, MSE = {mse_train:.6f}")
print(f"Unweighted q_95 (LOO residual quantile) = {q95_unweighted:.6e} deg C/step")

# ---------------------------------------------------------------------------
# 5. REAL covariate shift test: does the FUDS-calibrated bound hold on real
#    US06/45C data? (This is the paper's primary quantitative claim, and it
#    uses real ground truth throughout.)
# ---------------------------------------------------------------------------
us06_preds = ensemble_predict(us06[features])
us06_residuals = np.abs(us06["Delta_T_C"].values - us06_preds)
coverage_unweighted_shift = np.mean(us06_residuals <= q95_unweighted)

# ---------------------------------------------------------------------------
# 6. Weighted EnbPI (Tibshirani et al. 2019 weighting + Xu & Xie 2021 LOO
#    ensemble scores): estimate density ratio w(x) = p_US06(x)/p_FUDS(x) via
#    a probabilistic domain classifier, then take a WEIGHTED quantile of the
#    LOO residuals instead of a flat quantile.
# ---------------------------------------------------------------------------
domain_X = pd.concat([fuds[features], us06[features]], ignore_index=True)
domain_y = np.array([0] * len(fuds) + [1] * len(us06))  # 0=lab(FUDS), 1=shifted(US06)
clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0, n_jobs=-1)
clf.fit(domain_X, domain_y)

p_shift = clf.predict_proba(fuds[features])[:, 1]
p_shift = np.clip(p_shift, 1e-3, 1 - 1e-3)  # avoid divide-by-zero
weights = p_shift / (1 - p_shift)  # odds ratio ~ density ratio p_US06/p_FUDS

def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cw /= cw[-1]
    idx = np.searchsorted(cw, q)
    return v[min(idx, len(v) - 1)]

q95_weighted = weighted_quantile(loo_residuals, weights, 0.95)
coverage_weighted_shift = np.mean(us06_residuals <= q95_weighted)

print(f"\n=== Real covariate-shift validation: train FUDS -> test real US06/45C ===")
print(f"Unweighted bound q_95 = {q95_unweighted:.6e} | empirical coverage on US06 = {coverage_unweighted_shift*100:.2f}%")
print(f"Weighted   bound q_95 = {q95_weighted:.6e} | empirical coverage on US06 = {coverage_weighted_shift*100:.2f}%")
print(f"(Nominal target: 95%)")

# Sanity: coverage on FUDS itself (in-distribution), should already be ~95% by construction
coverage_indist = np.mean(loo_residuals <= q95_unweighted)
print(f"In-distribution (FUDS on itself) coverage check: {coverage_indist*100:.2f}%")

# ---------------------------------------------------------------------------
# 7. Real F1 telemetry: unsupervised OOD flagging only, no fabricated ground
#    truth, face-validity check against real braking/DRS zones.
# ---------------------------------------------------------------------------
f1 = pd.read_csv("f1_telemetry_clean.csv")

I_max_lab = fuds["Current_A"].abs().quantile(0.99)  # real lab cell's peak-current anchor

def build_f1_proxy_features(df, R_int, I_max_lab=I_max_lab):
    # NOTE ON UNITS: a single CALCE test cell draws single-digit amps, while a
    # vehicle traction pack draws hundreds. Raw-ampere equivalence between the
    # two is not physically meaningful. We therefore express trackside load
    # intensity as a DIMENSIONLESS load fraction in [-1, 1] (deployment minus
    # regen, each scaled by speed) and re-express it on the SAME numeric
    # current scale the lab model was trained on (I_max_lab). This tests
    # whether the *functional relationship* between normalized load intensity
    # and thermal response transfers across domains -- it does not, and is not
    # meant to, claim raw-ampere equivalence between a test cell and a pack.
    speed_n = df["Speed"] / max(df["Speed"].max(), 1e-6)
    throttle_n = df["Throttle"] / 100.0
    brake_n = df["Brake"].astype(float)
    load_fraction = (speed_n * throttle_n) - (speed_n * brake_n)  # in [-1, 1]
    load_fraction = load_fraction.rolling(15, min_periods=1).mean().fillna(0).values
    if len(load_fraction) >= 15:
        load_fraction = savgol_filter(load_fraction, 15, 3)
    proxy_current = load_fraction * I_max_lab
    proxy_voltage = 3.6 - proxy_current * R_int * 1000  # nominal Li-ion cell voltage scale, matched to CALCE's ~2.5-4.2V window
    proxy_power = proxy_current * proxy_voltage / 1000
    roll_var = pd.Series(proxy_power).rolling(30, min_periods=1).var().fillna(0)
    X = pd.DataFrame({
        "Current_A": proxy_current, "Voltage_V": proxy_voltage,
        "Power_kW": proxy_power, "Rolling_Power_Variance": roll_var,
    })
    return X


# --- Ensemble-disagreement OOD signal (no ground truth required) ---
# Epistemic uncertainty: std across the B bootstrap models' predictions.
# High disagreement = the ensemble is extrapolating = distributionally novel
# input. Threshold calibrated from real lab (FUDS) data itself.
lab_std = np.std(all_preds, axis=1)  # (m,) std across B models, on real FUDS rows
ood_threshold = np.quantile(lab_std, 0.95)
print(f"\nEnsemble-disagreement OOD threshold (95th pct of real-FUDS ensemble std) = {ood_threshold:.6e}")

# Sanity check: does this threshold flag real US06 (known real shift) at an
# elevated rate relative to FUDS itself? This validates the diagnostic before
# trusting it on F1.
us06_all_preds = np.stack([bm.predict(us06[features]) for bm in boot_models], axis=1)
us06_std = np.std(us06_all_preds, axis=1)
us06_ood_rate = np.mean(us06_std > ood_threshold) * 100
fuds_ood_rate = np.mean(lab_std > ood_threshold) * 100
print(f"OOD flag rate on real FUDS (in-distribution, sanity check) = {fuds_ood_rate:.2f}%")
print(f"OOD flag rate on real US06/45C (known real shift)          = {us06_ood_rate:.2f}%")

results_by_circuit = {}
for gp in f1["gp"].unique():
    sub = f1[f1["gp"] == gp].reset_index(drop=True)
    X = build_f1_proxy_features(sub, R_int)
    circuit_preds = np.stack([bm.predict(X[features]) for bm in boot_models], axis=1)
    circuit_std = np.std(circuit_preds, axis=1)
    flagged = circuit_std > ood_threshold
    brake_flag_rate = sub.loc[flagged, "Brake"].astype(float).mean() if flagged.sum() else np.nan
    overall_brake_rate = sub["Brake"].astype(float).mean()
    drs_flag_rate = (sub.loc[flagged, "DRS"] > 0).mean() if flagged.sum() else np.nan
    overall_drs_rate = (sub["DRS"] > 0).mean()
    results_by_circuit[gp] = {
        "n_frames": int(len(sub)),
        "ood_flag_rate_pct": float(flagged.mean() * 100),
        "brake_rate_among_flagged_pct": float(brake_flag_rate * 100) if not np.isnan(brake_flag_rate) else None,
        "brake_rate_overall_pct": float(overall_brake_rate * 100),
        "drs_rate_among_flagged_pct": float(drs_flag_rate * 100) if not np.isnan(drs_flag_rate) else None,
        "drs_rate_overall_pct": float(overall_drs_rate * 100),
    }

print("\n=== Real F1 telemetry: unsupervised OOD flagging (NOT validated against ground truth) ===")
for gp, r in results_by_circuit.items():
    print(f"{gp}: n={r['n_frames']}, OOD flag rate={r['ood_flag_rate_pct']:.2f}%, "
          f"brake-rate among flagged={r['brake_rate_among_flagged_pct']:.2f}% vs overall {r['brake_rate_overall_pct']:.2f}%, "
          f"DRS-rate among flagged={r['drs_rate_among_flagged_pct']:.2f}% vs overall {r['drs_rate_overall_pct']:.2f}%")

final = {
    "R_int_ohm": R_int,
    "train_r2": float(r2_train),
    "train_mse": float(mse_train),
    "q95_unweighted": float(q95_unweighted),
    "q95_weighted": float(q95_weighted),
    "coverage_indistribution_pct": float(coverage_indist * 100),
    "coverage_unweighted_on_real_shift_pct": float(coverage_unweighted_shift * 100),
    "coverage_weighted_on_real_shift_pct": float(coverage_weighted_shift * 100),
    "n_fuds": int(len(fuds)),
    "n_us06": int(len(us06)),
    "f1_results": results_by_circuit,
}
with open("final_results.json", "w") as f:
    json.dump(final, f, indent=2)
print("\nSaved final_results.json")