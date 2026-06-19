# ============================================================
# Section 4: Wasserstein Manifold Analysis
# ============================================================
# W₂ distance between Tracking and Playback embedding
# distributions.  Physical interpretation: the minimum
# "kinetic energy" (control cost) to maintain the closed-loop
# manifold against entropic drift.
#
# Unlike the Lie algebra fitting, this analysis:
#   - Does NOT require dR/dt to be predictable
#   - Does NOT assume rotational dynamics
#   - Only requires the embedding to preserve topological
#     relationships (which CEBRA is designed to do)

try:
    import ot
    HAS_POT = True
    print("POT (optimal transport): OK")
except ImportError:
    HAS_POT = True  # will install below
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "POT"])
    import ot
    print("POT installed and imported")

# ---- Wasserstein helpers ------------------------------------------------
def compute_wasserstein_between_conditions(
    emb_tracking, emb_playback,
    n_shuffles=50, reg=0.01, random_state=42
):
    """
    Compute W₂² between Tracking and Playback embedding distributions.

    Parameters
    ----------
    emb_tracking : (T₁, D)  CEBRA/E2E embeddings for Tracking
    emb_playback : (T₂, D)  CEBRA/E2E embeddings for Playback
    n_shuffles   : int      Number of shuffle null realisations
    reg          : float    Sinkhorn regularisation (small = exact, large = blurry)

    Returns
    -------
    W2_sq       : float    W₂² distance (true labels)
    W2_shuffle  : float    Mean W₂² under shuffled labels
    W2_sh_sem   : float    SEM of shuffled W₂²
    """
    import numpy as np

    n1, n2 = len(emb_tracking), len(emb_playback)
    if n1 < 10 or n2 < 10:
        return np.nan, np.nan, np.nan

    # Uniform weights (equal mass per timepoint)
    a = np.ones(n1) / n1
    b = np.ones(n2) / n2

    # Cost matrix: squared Euclidean distances
    M = ot.dist(emb_tracking, emb_playback, metric='sqeuclidean')

    # Sinkhorn regularised OT (fast, stable for 3D point clouds)
    W2_sq = ot.sinkhorn2(a, b, M, reg=reg)

    # ---- Shuffle null ----
    z_all = np.vstack([emb_tracking, emb_playback])
    n_total = n1 + n2
    rng = np.random.default_rng(random_state)
    null_vals = []

    for _ in range(n_shuffles):
        idx = rng.permutation(n_total)
        z_shuf_tr = z_all[idx[:n1]]
        z_shuf_pb = z_all[idx[n1:]]
        M_shuf = ot.dist(z_shuf_tr, z_shuf_pb, metric='sqeuclidean')
        null_vals.append(ot.sinkhorn2(a, b, M_shuf, reg=reg))

    W2_shuffle = np.mean(null_vals)
    W2_sh_sem  = np.std(null_vals, ddof=1) / np.sqrt(n_shuffles)

    return W2_sq, W2_shuffle, W2_sh_sem


# ---- Main analysis -------------------------------------------------------
print("\n" + "=" * 60)
print("  Wasserstein Manifold Analysis")
print("=" * 60)

wass_results = []

for idx in range(len(n_data_all)):
    n_data_session = n_data_all[idx]
    f_df = f_data_all[idx]
    hs_label = "hs0" if idx < n_hs0 else "hs1"

    # ---- Train per-session CEBRA (lightweight: 3000 iter, same as baseline) ----
    if not HAS_CEBRA:
        print(f"  Session {idx}: CEBRA not available, skipping")
        continue

    # Extract macro-epochs for this session
    all_epochs = []
    all_labels = []
    n_track_ep = 0

    for val, label in [(0.0, "Tracking"), (1.0, "Playback")]:
        epochs_n, epochs_l, _ = extract_epochs(
            n_data_session, f_df, val, dt,
            label_col=CEBRA_LABEL,
            step=1)

        if val == 0.0:
            n_track_ep = len(epochs_n)

        for ep_n, ep_l in zip(epochs_n, epochs_l):
            all_epochs.append(ep_n)
            all_labels.append(ep_l)

    if len(all_epochs) < 2:
        print(f"  Session {idx}: too few epochs, skipping")
        continue

    # Train CEBRA (same as baseline config)
    try:
        cebra_model = CEBRA(
            model_architecture=CEBRA_ARCH,
            output_dimension=CEBRA_EMBEDDING_DIM,
            max_iterations=3000,
            batch_size=512,
            learning_rate=3e-4,
            temperature=TEMPERATURE,
            distance=CEBRA_DISTANCE,
            conditional="time_delta",
            device=CEBRA_DEVICE,
            verbose=False
        )
        cebra_model.fit(all_epochs, all_labels)
    except Exception as e:
        print(f"  Session {idx} CEBRA training failed: {e}")
        continue

    # ---- Extract embeddings per condition ----
    emb_tracking = []
    emb_playback = []

    for ei, (ep, lab) in enumerate(zip(all_epochs, all_labels)):
        try:
            emb = cebra_model.transform(ep, session_id=ei)
        except Exception:
            continue

        if ei < n_track_ep:
            emb_tracking.append(emb.astype(np.float32))
        else:
            emb_playback.append(emb.astype(np.float32))

    # Pool condition-specific timepoints
    if emb_tracking and emb_playback:
        z_tr = np.concatenate(emb_tracking, axis=0)   # (T_TR, D)
        z_pb = np.concatenate(emb_playback, axis=0)   # (T_PB, D)

        # Compute Wasserstein
        W2, W2_sh, W2_sh_sem = compute_wasserstein_between_conditions(
            z_tr, z_pb, n_shuffles=N_SHUFFLES)

        # Also compute: within-condition self-Wasserstein (measure of spread)
        # Split TR into two halves, compute W₂ — smaller = tighter manifold
        n_tr_half = len(z_tr) // 2
        W2_tr_self, _, _ = compute_wasserstein_between_conditions(
            z_tr[:n_tr_half], z_tr[n_tr_half:2*n_tr_half],
            n_shuffles=N_SHUFFLES)

        n_pb_half = len(z_pb) // 2
        W2_pb_self, _, _ = compute_wasserstein_between_conditions(
            z_pb[:n_pb_half], z_pb[n_pb_half:2*n_pb_half],
            n_shuffles=N_SHUFFLES)

        wass_results.append({
            "Subject": "SKIEUR",
            "Session_Idx": idx,
            "Headstage": hs_label,
            "N_Tracking": len(z_tr),
            "N_Playback": len(z_pb),
            "W2_TR_vs_PB": W2,
            "W2_TR_vs_PB_shuffle": W2_sh,
            "W2_TR_vs_PB_shuffle_sem": W2_sh_sem,
            "W2_ratio": W2 / (W2_sh + 1e-9),      # > 1 = TR/PB distributions are
                                                      # more different than chance
            "W2_TR_self": W2_tr_self,               # smaller = tighter Tracking
            "W2_PB_self": W2_pb_self,               # smaller = tighter Playback
        })

        print(f"  S{idx:<2d} {hs_label}  "
              f"W₂(TR,PB)={W2:.4f}  "
              f"W₂(shuf)={W2_sh:.4f}  "
              f"ratio={W2/(W2_sh+1e-9):.2f}  "
              f"TR_self={W2_tr_self:.4f}  "
              f"PB_self={W2_pb_self:.4f}")

    del cebra_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---- Aggregate results ----
if wass_results:
    wass_df = pd.DataFrame(wass_results)

    # Per-condition self-Wasserstein (tightness comparison)
    tr_self = wass_df["W2_TR_self"].values
    pb_self = wass_df["W2_PB_self"].values

    print(f"\n{'='*60}")
    print(f"  Wasserstein Summary (N = {len(wass_df)} sessions)")
    print(f"{'='*60}")
    print(f"  W₂(TR vs PB):           {wass_df['W2_TR_vs_PB'].mean():.4f}  "
          f"± {wass_df['W2_TR_vs_PB'].sem():.4f}")
    print(f"  W₂(shuffle null):       {wass_df['W2_TR_vs_PB_shuffle'].mean():.4f}  "
          f"± {wass_df['W2_TR_vs_PB_shuffle_sem'].mean():.4f}")
    print(f"  W₂ ratio (true/shuf):   {wass_df['W2_ratio'].mean():.2f}  "
          f"± {wass_df['W2_ratio'].sem():.2f}")
    print(f"  ---")
    print(f"  TR self-W₂ (half-split):  {np.mean(tr_self):.4f}  "
          f"± {np.std(tr_self, ddof=1)/np.sqrt(len(tr_self)):.4f}")
    print(f"  PB self-W₂ (half-split):  {np.mean(pb_self):.4f}  "
          f"± {np.std(pb_self, ddof=1)/np.sqrt(len(pb_self)):.4f}")

    if len(wass_df) > 1:
        t_self, p_self = ttest_rel(tr_self, pb_self)
        print(f"  TR vs PB self-W₂ paired t: t={t_self:.3f}, p={p_self:.4f}")
        print(f"  → {'TR tighter' if np.mean(tr_self) < np.mean(pb_self) else 'PB tighter'}")

    print(f"\n  Per-session W₂ ratios (true/shuf):")
    for _, row in wass_df.iterrows():
        marker = " ★" if row["W2_ratio"] > 1.5 else ""
        print(f"    S{int(row['Session_Idx']):<2d} {row['Headstage']}  "
              f"ratio={row['W2_ratio']:.2f}{marker}")

    print(f"\n  Interpretation:")
    print(f"    W₂ ratio > 1.0  : TR and PB occupy different regions of state space")
    print(f"    W₂ ratio ≈ 1.0  : TR/PB difference is explained by sampling noise")
    print(f"    TR_self < PB_self: Tracking manifold is tighter (lower entropy)")
    print(f"    TR_self > PB_self: Tracking manifold is more dispersed")

else:
    wass_df = None
    print("  No Wasserstein results computed.")

print("Wasserstein analysis complete.")
