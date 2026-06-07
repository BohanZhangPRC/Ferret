# ============================================================
# Phase 0 -- Baseline Two-Stage + Dummy-CEBRA Control
# ============================================================
# For each session:
#   1. Train pooled CEBRA on true labels
#   2. Train dummy CEBRA on permuted labels (key negative control)
#   3. Per epoch: transform -> fit Lie -> compute SR/R2/R2_drive
#   4. Per epoch: shuffle label -> fit Lie -> compute null metrics
#   5. Per epoch: dummy embed -> fit Lie -> compute dummy metrics

CEBRA_LIE_ITERS = 3000
LIE_OUTPUT_DIR = f"Skieur_LieE2E_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(LIE_OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {LIE_OUTPUT_DIR}")

baseline_results = []  # per session-condition aggregate
dummy_results = []     # per session-condition for dummy CEBRA

if HAS_CEBRA:
    n_skipped_epochs, n_skipped_cond, n_sessions_used = 0, 0, 0

    for idx, (n_data_session, f_df) in enumerate(
            zip(tqdm(n_data_all, desc="Baseline+Dummy"), f_data_all)):
        hs_label = "hs0" if idx < n_hs0 else "hs1"

        # ---- Extract + filter epochs for BOTH conditions ----
        cond_epochs = {}
        cond_labels = {}
        skip_session = False
        for val, label in [(0.0, "Tracking"), (1.0, "Playback")]:
            epochs_n, epochs_l, n_ep = extract_epochs(
                n_data_session, f_df, val, dt, label_col="Velocity_x")
            # Filter short epochs
            valid_idx = [i for i in range(len(epochs_n))
                         if epochs_n[i].shape[0] >= MIN_EPOCH_TIMEPOINTS]
            n_skipped_epochs += len(epochs_n) - len(valid_idx)
            if len(valid_idx) < MIN_EPOCHS_PER_COND:
                skip_session = True
                n_skipped_cond += 1
                break
            cond_epochs[label] = [epochs_n[i] for i in valid_idx]
            cond_labels[label] = [epochs_l[i] for i in valid_idx]

        if skip_session:
            continue
        n_sessions_used += 1

        # ---- Pool epochs across conditions ----
        all_epochs = cond_epochs["Tracking"] + cond_epochs["Playback"]
        all_labels = cond_labels["Tracking"] + cond_labels["Playback"]
        n_track_ep = len(cond_epochs["Tracking"])

        # ---- Train ONE pooled CEBRA (true labels) ----
        try:
            cebra_true = CEBRA(
                model_architecture=CEBRA_ARCH,
                output_dimension=CEBRA_EMBEDDING_DIM,
                max_iterations=CEBRA_LIE_ITERS, batch_size=2048,
                learning_rate=3e-4, temperature=1.5,
                distance=CEBRA_DISTANCE,
                conditional="time_delta", device="cuda", verbose=False)
            cebra_true.fit(all_epochs, all_labels)
            loss_true = float(cebra_true.state_dict_['loss'][-1])
        except Exception as e:
            print(f"  Session {idx} CEBRA true failed: {e}")
            continue

        # ---- Train dummy CEBRA (shuffled labels) ----
        # NOTE: uses full permutation (destroys autocorrelation). This is a
        # "strong negative control" — destroys ALL label structure. For an
        # autocorrelation-preserving dummy, use circular-shift labels instead.
        all_labels_shuf = [np.random.permutation(l) for l in all_labels]
        cebra_dummy = None
        try:
            cebra_dummy = CEBRA(
                model_architecture=CEBRA_ARCH,
                output_dimension=CEBRA_EMBEDDING_DIM,
                max_iterations=CEBRA_LIE_ITERS, batch_size=2048,
                learning_rate=3e-4, temperature=1.5,
                distance=CEBRA_DISTANCE,
                conditional="time_delta", device="cuda", verbose=False)
            cebra_dummy.fit(all_epochs, all_labels_shuf)
        except Exception as e:
            print(f"  Session {idx} Dummy CEBRA failed: {e}")

        # ---- Per-condition evaluation ----
        for cond_name, start_idx in [("Tracking", 0),
                                      ("Playback", n_track_ep)]:
            n_ep = (n_track_ep if cond_name == "Tracking"
                    else len(all_epochs) - n_track_ep)
            ep_list = all_epochs[start_idx:start_idx + n_ep]
            lab_list = all_labels[start_idx:start_idx + n_ep]

            sr_vals, r2_vals, r2d_vals = [], [], []
            sr_sh_vals, r2_sh_vals, r2d_sh_vals = [], [], []
            eig_real_vals, eig_imag_vals = [], []
            sr_dummy_vals, r2d_dummy_vals = [], []

            for ei, (ep, el) in enumerate(zip(ep_list, lab_list)):
                # --- True embedding ---
                emb = cebra_true.transform(ep, session_id=start_idx + ei)
                J_s, sr, r2, J_ols, r2d = fit_lie_algebra_with_leak(emb, el)
                sr_vals.append(sr)
                r2_vals.append(r2)
                r2d_vals.append(r2d)
                re, im = compute_eigenvalue_metrics(J_ols)
                eig_real_vals.append(re)
                eig_imag_vals.append(im)

                # --- Shuffle control (same embedding, circular-shifted labels) ---
                # Uses np.roll to preserve autocorrelation structure (cf. E2E null
                # and lie_algebra_method_description.md section 12.3).
                T_lab = len(el)
                min_shift = max(1, MINI_TRAJ_LEN)
                s_sr, s_r2, s_r2d = [], [], []
                for _ in range(N_SHUFFLES):
                    shift = np.random.randint(min_shift, max(min_shift + 1, T_lab - min_shift))
                    el_sh = np.roll(el, shift)
                    _, sr_sh, r2_sh, _, r2d_sh = fit_lie_algebra_with_leak(
                        emb, el_sh)
                    s_sr.append(sr_sh)
                    s_r2.append(r2_sh)
                    s_r2d.append(r2d_sh)
                sr_sh_vals.append(np.mean(s_sr))
                r2_sh_vals.append(np.mean(s_r2))
                r2d_sh_vals.append(np.mean(s_r2d))

                # --- Dummy embedding (shuffled-label CEBRA) ---
                if cebra_dummy is not None:
                    emb_dummy = cebra_dummy.transform(
                        ep, session_id=start_idx + ei)
                    _, sr_d, _, _, r2d_d = fit_lie_algebra_with_leak(
                        emb_dummy, el)
                    sr_dummy_vals.append(sr_d)
                    r2d_dummy_vals.append(r2d_d)

            n_ep_valid = len(sr_vals)
            baseline_results.append({
                "Subject": "SKIEUR", "Session_Idx": idx,
                "Headstage": hs_label, "Condition": cond_name,
                "Space": "CEBRA", "N_Epochs": n_ep_valid,
                "N_Neurons": n_data_session.shape[0],
                "SR": np.mean(sr_vals), "SR_shuffle": np.mean(sr_sh_vals),
                "R2": np.mean(r2_vals), "R2_shuffle": np.mean(r2_sh_vals),
                "R2_drive": np.mean(r2d_vals),
                "R2_drive_shuffle": np.mean(r2d_sh_vals),
                "Eig_Real_Mean": np.mean(eig_real_vals),
                "Eig_Imag_Mean": np.mean(eig_imag_vals),
            })
            if sr_dummy_vals:
                dummy_results.append({
                    "Subject": "SKIEUR", "Session_Idx": idx,
                    "Headstage": hs_label, "Condition": cond_name,
                    "N_Epochs": n_ep_valid,
                    "SR_dummy": np.mean(sr_dummy_vals),
                    "R2_drive_dummy": np.mean(r2d_dummy_vals),
                })

    baseline_df = pd.DataFrame(baseline_results)
    dummy_df = pd.DataFrame(dummy_results) if dummy_results else None
    print(f"Baseline: {len(baseline_df)} session-conditions "
          f"({n_sessions_used} sessions used, {n_skipped_cond} skipped)")
    if dummy_df is not None:
        print(f"Dummy CEBRA: {len(dummy_df)} entries")

else:
    print("CEBRA not installed. Skipping baseline + dummy control.")
    baseline_df, dummy_df = None, None
