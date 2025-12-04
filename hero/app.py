# SentinelNet (robust DDoS fallback + UI contrast + Conn # label black)

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import random
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score

ROOT = Path.cwd()

# Helpers: KDD artifacts auto-generation (safe)

def ensure_kdd_artifacts(train_csv: Path, test_csv: Path,
                         imputer_p: Path, scaler_p: Path, model_p: Path, encoders_p: Path):
    if imputer_p.exists() and scaler_p.exists() and model_p.exists() and encoders_p.exists():
        return

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Cannot auto-generate KDD artifacts: kdd_train.csv or kdd_test.csv missing.")

    print("Auto-generating KDD artifacts (this may take a short while)...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    label_col = "labels" if "labels" in train_df.columns else train_df.columns[-1]
    categorical_cols = [c for c in ["protocol_type", "service", "flag"] if c in train_df.columns]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
        le.fit(combined.unique())
        encoders[col] = le
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    X_train = train_df.drop(columns=[label_col], errors='ignore')
    y_train = train_df[label_col].apply(lambda x: 0 if str(x).strip().lower() == "normal" else 1).values

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)

    from sklearn.ensemble import RandomForestClassifier
    model = None
    for params in ({"n_estimators":200, "max_depth":16, "n_jobs":1},
                   {"n_estimators":120, "max_depth":14, "n_jobs":1},
                   {"n_estimators":80,  "max_depth":12, "n_jobs":1},
                   {"n_estimators":50,  "max_depth":10, "n_jobs":1}):
        try:
            clf = RandomForestClassifier(random_state=42, **params)
            clf.fit(X_train_scaled, y_train)
            model = clf
            print("KDD model trained with params:", params)
            break
        except Exception as e:
            print("KDD training failed for params", params, "—", type(e).__name__, str(e)[:200])
            model = None
            continue

    if model is None:
        raise RuntimeError("Failed to train a KDD model in the available memory.")

    joblib.dump(imputer, imputer_p)
    joblib.dump(scaler, scaler_p)
    joblib.dump(model, model_p)
    joblib.dump({c: encoders[c].classes_.tolist() for c in encoders}, encoders_p)
    print("Saved KDD artifacts.")


# Robust DDoS artifact ensure (replacement)

def ensure_ddos_artifacts(ddos_csv: Path, scaler_p: Path, model_p: Path,
                          label_enc_p: Path, encoders_p: Path,
                          sample_target: int = 8000):
    """
    Ensure ddos_scaler.pkl, ddos_best_model.pkl, ddos_label_encoder.joblib, ddos_encoders.joblib exist.
    If an existing model file is present but cannot be loaded due to version incompatibility,
    train a small fallback model using a reservoir-sampled subset and overwrite model_p.
    """
    if scaler_p.exists() and model_p.exists() and label_enc_p.exists() and encoders_p.exists():
        try:
            _ = joblib.load(model_p)
            return
        except Exception as e:
            print("Existing DDoS model load failed (will attempt fallback retrain):", repr(e))

    if not ddos_csv.exists():
        raise FileNotFoundError("Cannot auto-generate DDoS artifacts: CIC-DDoS CSV missing.")

    reservoir = []
    total = 0
    with open(ddos_csv, "r", encoding="utf-8", errors="ignore") as fh:
        header = fh.readline()
        for line in fh:
            total += 1
            if len(reservoir) < sample_target:
                reservoir.append(line)
            else:
                j = random.randint(0, total - 1)
                if j < sample_target:
                    reservoir[j] = line

    if len(reservoir) == 0:
        raise RuntimeError("No rows sampled from the CIC-DDoS CSV (check the file).")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
    tmp.write(header)
    tmp.writelines(reservoir)
    tmp.flush(); tmp.close()
    sampled_path = Path(tmp.name)

    df = pd.read_csv(sampled_path)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        sampled_path.unlink(missing_ok=True)
        raise FileNotFoundError("Expected 'Label' column in CIC-DDoS CSV sample.")

    df = df.replace([np.inf, -np.inf], np.nan)
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=constant_cols, errors='ignore')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    skew_vals = df[num_cols].skew().fillna(0)
    for col in num_cols:
        if abs(skew_vals.get(col, 0.0)) <= 0.5:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")

    label_encoder = LabelEncoder()
    df["Label_enc"] = label_encoder.fit_transform(df["Label"].astype(str))
    y = df["Label_enc"].values
    X = df.drop(columns=["Label", "Label_enc"], errors='ignore')

    cat_cols_in_X = X.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in cat_cols_in_X:
        le = LabelEncoder()
        le.fit(X[col].astype(str).unique())
        encoders[col] = le
        X[col] = le.transform(X[col].astype(str))

    scaler = StandardScaler()
    X_vals = X.values.astype(float)
    scaler.fit(X_vals)
    X_scaled = scaler.transform(X_vals)

    model = None
    try:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=80, max_depth=12, n_jobs=1, random_state=42)
        clf.fit(X_scaled, y)
        model = clf
    except Exception as e:
        print("RandomForest training failed, trying LogisticRegression:", e)
        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=500, solver="saga")
            lr.fit(X_scaled, y)
            model = lr
        except Exception as e2:
            sampled_path.unlink(missing_ok=True)
            raise RuntimeError("Failed to train fallback DDoS model: " + str(e2))

    joblib.dump(scaler, scaler_p)
    joblib.dump(model, model_p)
    joblib.dump(label_encoder, label_enc_p)
    joblib.dump({c: encoders[c].classes_.tolist() for c in encoders}, encoders_p)

    sampled_path.unlink(missing_ok=True)
    print("Saved DDoS artifacts (fallback trained).")


# Loaders (KDD & DDoS) - DDoS loader robust replacement

def load_kdd_assets() -> Dict[str, Any]:
    train_path = ROOT / "kdd_train.csv"
    test_path = ROOT / "kdd_test.csv"
    imputer_path = ROOT / "kdd_imputer.pkl"
    scaler_path = ROOT / "kdd_scaler.pkl"
    encoders_path = ROOT / "kdd_label_encoders.joblib"

    try:
        ensure_kdd_artifacts(train_path, test_path, imputer_path, scaler_path, ROOT/"kdd_best_model.pkl", encoders_path)
    except Exception as e:
        print("KDD artifact auto-gen step:", e)

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected kdd_train.csv and kdd_test.csv in project root.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df = train_df.drop_duplicates().reset_index(drop=True)
    test_df = test_df.drop_duplicates().reset_index(drop=True)

    categorical_cols = [c for c in ["protocol_type", "service", "flag"] if c in train_df.columns]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)], axis=0)
        le.fit(combined.unique())
        encoders[col] = le
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))

    label_col = "labels" if "labels" in train_df.columns else train_df.columns[-1]
    train_df["attack_binary"] = train_df[label_col].apply(lambda x: 0 if str(x).strip().lower() == "normal" else 1)
    test_df["attack_binary"] = test_df[label_col].apply(lambda x: 0 if str(x).strip().lower() == "normal" else 1)

    X_test = test_df.drop(columns=[label_col, "attack_binary"], errors='ignore')
    y_test = test_df["attack_binary"].values

    if not imputer_path.exists() or not scaler_path.exists():
        raise FileNotFoundError("Missing kdd_imputer.pkl or kdd_scaler.pkl in project root.")

    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)

    X_test_imp = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)

    model = None
    for fname in ["kdd_best_model.pkl", "kdd_decision_tree_model.pkl", "kdd_model.pkl", "model.joblib"]:
        p = ROOT / fname
        if p.exists():
            model = joblib.load(p)
            break
    if model is None:
        raise FileNotFoundError("No KDD model file found. Provide kdd_best_model.pkl or similar in project root.")

    y_pred = model.predict(X_test_scaled)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))

    test_display = test_df.copy()
    test_display["prediction"] = y_pred

    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "encoders": encoders,
        "categorical_cols": categorical_cols,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "test_display": test_display,
        "accuracy": acc,
        "precision": prec,
        "label_col": label_col,
    }


def load_ddos_assets() -> Dict[str, Any]:
    csv_path = ROOT / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    scaler_path = ROOT / "ddos_scaler.pkl"
    model_path = ROOT / "ddos_best_model.pkl"
    label_enc_path = ROOT / "ddos_label_encoder.joblib"
    encoders_path = ROOT / "ddos_encoders.joblib"

    try:
        ensure_ddos_artifacts(csv_path, scaler_path, model_path, label_enc_path, encoders_path, sample_target=8000)
    except Exception as e:
        raise RuntimeError(f"DDoS artifact generation failed: {e}")

    if not csv_path.exists():
        raise FileNotFoundError("Expected Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv in project root.")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.replace([np.inf, -np.inf], np.nan)
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=constant_cols, errors='ignore')

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    skew_vals = df[num_cols].skew().fillna(0)
    for col in num_cols:
        if abs(skew_vals.get(col, 0.0)) <= 0.5:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")

    if "Label" not in df.columns:
        raise FileNotFoundError("Expected 'Label' column in CIC-DDoS CSV.")

    if Path(label_enc_path).exists():
        try:
            label_encoder = joblib.load(label_enc_path)
        except Exception:
            label_encoder = LabelEncoder().fit(df["Label"].astype(str))
            joblib.dump(label_encoder, label_enc_path)
    else:
        label_encoder = LabelEncoder().fit(df["Label"].astype(str))
        joblib.dump(label_encoder, label_enc_path)

    df["Label_enc"] = label_encoder.transform(df["Label"].astype(str))
    y = df["Label_enc"].values
    X = df.drop(columns=["Label", "Label_enc"], errors='ignore')

    cat_cols_in_X = X.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in cat_cols_in_X:
        le = LabelEncoder()
        le.fit(X[col].astype(str).unique())
        encoders[col] = le
        X[col] = le.transform(X[col].astype(str))

    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ddos_scaler.pkl: {e}")

    model = None
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print("Failed to load ddos model (will retrain fallback):", repr(e))
        sample_rows = min(8000, len(df))
        df_sample = df.sample(sample_rows, random_state=42).reset_index(drop=True)
        Xs = df_sample.drop(columns=["Label", "Label_enc"], errors='ignore')
        ys = df_sample["Label_enc"].values

        for col in Xs.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            Xs[col] = le.fit_transform(Xs[col].astype(str))

        Xvals = Xs.values.astype(float)
        try:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42, n_jobs=1)
            clf.fit(Xvals, ys)
            model = clf
            joblib.dump(model, model_path)
            print("Trained and saved fallback ddos model to", model_path)
        except Exception as e2:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(max_iter=500, solver="saga")
            lr.fit(Xvals, ys)
            model = lr
            joblib.dump(model, model_path)
            print("Trained and saved fallback logistic ddos model to", model_path)

    X_vals = X.values.astype(float)
    X_scaled = scaler.transform(X_vals)
    y_pred = model.predict(X_scaled)

    true_labels = label_encoder.inverse_transform(y)
    pred_labels = label_encoder.inverse_transform(y_pred)

    true_binary = np.array([0 if "BENIGN" in lab.upper() else 1 for lab in true_labels])
    pred_binary = np.array([0 if "BENIGN" in lab.upper() else 1 for lab in pred_labels])

    acc = float(accuracy_score(true_binary, pred_binary))
    prec = float(precision_score(true_binary, pred_binary, zero_division=0))

    df_display = df.copy()
    df_display["pred_label"] = pred_labels

    return {
        "model": model,
        "scaler": scaler,
        "encoders": encoders,
        "label_encoder": label_encoder,
        "feature_cols": list(X.columns),
        "cat_cols": cat_cols_in_X,
        "X_scaled": X_scaled,
        "y_true": true_binary,
        "df_display": df_display,
        "accuracy": acc,
        "precision": prec,
    }


# UI helpers (Conn # label black)

def metric_card_html(title: str, value: str, subtitle: str) -> str:
    return f"""
    <div class="sn-metric-card" style="background: linear-gradient(135deg,#ffffff,#f1f6ff); border-radius:12px; padding:14px; text-align:center;">
      <div class="title" style="font-size:0.9rem; opacity:0.85; margin-bottom:6px; color: #071127;">{title}</div>
      <div class="value" style="font-size:1.4rem; font-weight:700; color: #071127;">{value}</div>
      <div class="subtitle" style="font-size:0.8rem; opacity:0.75; margin-top:6px; color: #334155;">{subtitle}</div>
    </div>
    """

def alert_card_html(conn_label: str, alert_msg: str) -> str:
    return f"""
    <div class="sn-alert-card" style="background:#fff3f2; border-radius:10px; padding:10px; margin-bottom:8px; border:1px solid #fecaca; display:flex; align-items:center;">
      <div class="label" style="font-weight:700; margin-right:10px; color:#000000 !important;">{conn_label}</div>
      <div class="msg" style="font-weight:500; color:#2b2b2b;">{alert_msg}</div>
    </div>
    """


# Global cache

GLOBAL: Dict[str, Any] = {"kdd": None, "ddos": None}
def ensure_assets(dataset_key: str):
    if GLOBAL.get(dataset_key) is not None:
        return GLOBAL[dataset_key]
    if dataset_key == "kdd":
        assets = load_kdd_assets()
    else:
        assets = load_ddos_assets()
    GLOBAL[dataset_key] = assets
    return assets


# Refresh view and CSV export helpers

def get_summary_csv(dataset_label: str, mode: str, algo: str, total_packets:int, intrusions:int, normals:int, intrusion_rate:float, acc:float, prec:float) -> Tuple[str, bytes]:
    df = pd.DataFrame([{
        "Dataset": dataset_label,
        "Mode": mode,
        "Algorithm": algo,
        "Total_Packets": total_packets,
        "Intrusions": intrusions,
        "Normal": normals,
        "Intrusion_Rate_percent": intrusion_rate,
        "Accuracy_percent": acc * 100,
        "Precision_percent": prec * 100
    }])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(csv_bytes); tmp.flush(); tmp.close()
    return tmp.name, csv_bytes

def refresh_view(dataset_label: str, dataset_key: str, algo: str, monitoring_active: bool, cleared: bool, classification_mode: str = "Binary"):
    try:
        assets = ensure_assets(dataset_key)
    except Exception as e:
        return f"<div style='color:#f87171'>Error loading assets: {str(e)}</div>", "", "<div style='color:#f87171'>Failed to load assets.</div>", pd.DataFrame(), "Accuracy: --", "Precision: --", None

    sample_frac = 0.30

    def compute_metrics_on_random_subset(assets, frac: float):
        acc = assets.get("accuracy", 0.0)
        prec = assets.get("precision", 0.0)
        try:
            if "X_test_scaled" in assets and "y_test" in assets:
                Xs = assets["X_test_scaled"]
                ys = np.array(assets["y_test"])
                n = len(ys)
                if n > 0:
                    mask = np.random.rand(n) < frac
                    if mask.sum() == 0:
                        mask = np.ones(n, dtype=bool)
                    model = assets["model"]
                    preds = model.predict(Xs[mask])
                    prec = float(precision_score(ys[mask], preds, zero_division=0))
                    acc = float(accuracy_score(ys[mask], preds))
            elif "y_true" in assets and "X_scaled" in assets:
                ys = np.array(assets["y_true"])
                Xs = assets["X_scaled"]
                n = len(ys)
                if n > 0:
                    mask = np.random.rand(n) < frac
                    if mask.sum() == 0:
                        mask = np.ones(n, dtype=bool)
                    preds = assets["model"].predict(Xs[mask])
                    acc = float(accuracy_score(ys[mask], preds))
                    prec = float(precision_score(ys[mask], preds, zero_division=0))
        except Exception as e:
            print("Random-subset metric computation failed:", e)
        return acc, prec

    acc, prec = compute_metrics_on_random_subset(assets, sample_frac)

    if dataset_key == "kdd":
        total_packets = len(assets["test_display"])
        if classification_mode == "Binary":
            intrusions = int((assets["test_display"]["prediction"] == 1).sum())
        else:
            label_col = assets.get("label_col", None)
            if label_col and label_col in assets["test_display"].columns:
                intrusions = int((~assets["test_display"][label_col].astype(str).str.lower().str.contains("normal")).sum())
            else:
                intrusions = int((assets["test_display"]["prediction"] != 0).sum())
        normals = total_packets - intrusions

        sample_n = min(10, max(1, total_packets))
        sample_df = assets["test_display"].sample(sample_n, random_state=None).reset_index(drop=True)

        df_alerts = assets["test_display"]["prediction"] == 1 if classification_mode == "Binary" else None
        if classification_mode == "Binary":
            df_alerts = assets["test_display"][assets["test_display"]["prediction"] == 1]
        else:
            if assets.get("label_col", None) in assets["test_display"].columns:
                df_alerts = assets["test_display"][~assets["test_display"][assets.get("label_col","")].astype(str).str.lower().str.contains("normal")]
            else:
                df_alerts = assets["test_display"][assets["test_display"]["prediction"] != 0]

        if df_alerts.empty or cleared:
            alerts = ["<div style='opacity:0.9;color:#cbd5e1'>No intrusions detected in the sampled KDD test data.</div>"]
        else:
            alert_n = min(8, len(df_alerts))
            df_alerts = df_alerts.sample(alert_n, random_state=None).reset_index(drop=True)
            alerts = [alert_card_html(f"Conn #{idx}", f"Attack detected – protocol {row.get('protocol_type','')}, service {row.get('service','')}") for idx, row in df_alerts.iterrows()]

    else:
        total_packets = len(assets["df_display"])
        pred_labels = assets["df_display"]["pred_label"].astype(str).values
        benign_mask = np.array([("BENIGN" in lab.upper()) for lab in pred_labels])
        intrusions = int((~benign_mask).sum())
        normals = int(benign_mask.sum())

        sample_n = min(10, max(1, total_packets))
        sample_df = assets["df_display"].sample(sample_n, random_state=None).reset_index(drop=True)

        df_attack = assets["df_display"][~assets["df_display"]["pred_label"].str.upper().str.contains("BENIGN")]
        if df_attack.empty or cleared:
            alerts = ["<div style='opacity:0.9;color:#cbd5e1'>No intrusions detected in the sampled CIC-DDoS data.</div>"]
        else:
            alert_n = min(8, len(df_attack))
            df_attack = df_attack.sample(alert_n, random_state=None).reset_index(drop=True)
            alerts = [alert_card_html(f"Conn #{idx}", f"{row.get('pred_label','UNKNOWN')} traffic detected.") for idx, row in df_attack.iterrows()]

    intrusion_rate = (intrusions / total_packets * 100) if total_packets > 0 else 0.0

    metrics_html = "<div style='display:flex;gap:12px;'>" \
                   + metric_card_html("Total Packets", str(total_packets), "Analyzed in current session") \
                   + metric_card_html("Intrusions", str(intrusions), "Predicted malicious connections") \
                   + metric_card_html("Normal", str(normals), "Predicted safe connections") \
                   + metric_card_html("Intrusion Rate", f"{intrusion_rate:.1f}%", "Intrusions / Total packets") \
                   + "</div>"

    header_html = f"""
    <div class="sn-header" style="background: linear-gradient(90deg,#0052D4,#4364F7,#6FB1FC); color:white; padding:12px; border-radius:12px; margin-bottom:12px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <div style="font-size:1.6rem; font-weight:800;">SentinelNet</div>
          <div style="font-size:0.9rem; opacity:0.95;">AI-Powered Network Intrusion Detection System</div>
        </div>
        <div style="text-align:right;">
          Dataset: <b style="color:#e6eef8">{dataset_label}</b><br/>
          Mode: <b style="color:#e6eef8">{"Live Monitoring" if monitoring_active else "File"}</b><br/>
          Classification: <b style="color:#e6eef8">{classification_mode}</b>
        </div>
      </div>
    </div>
    """

    alerts_html = "<div>" + "".join(alerts) + "</div>"
    fname, csv_bytes = get_summary_csv(dataset_label, "Live" if monitoring_active else "File", algo, total_packets, intrusions, normals, intrusion_rate, acc, prec)
    acc_text = f"Accuracy: {acc*100:.1f}%"
    prec_text = f"Precision: {prec*100:.1f}%"
    return header_html, metrics_html, alerts_html, sample_df, acc_text, prec_text, fname


# Build Gradio UI with dark-theme fixes

with gr.Blocks(title="SentinelNet (Gradio)") as demo:
    gr.HTML("""
    <style>
      :root{
        --sn-light-text: #e6eef8;
        --sn-muted: #b6c2d6;
        --sn-accent: #77a7ff;
        --sn-card-dark: #071127;
      }
      body, .gradio-container { background-color: #0b0b0c !important; color: var(--sn-light-text) !important; }
      .gradio-container, .gradio-container * { color: var(--sn-light-text) !important; }
      .sn-header { color: #ffffff !important; text-shadow: 0 1px 2px rgba(0,0,0,0.45); }
      .sn-metric-card { color: var(--sn-card-dark) !important; }
      .sn-metric-card .title, .sn-metric-card .value, .sn-metric-card .subtitle { color: var(--sn-card-dark) !important; }
      .sn-alert-card, .sn-alert-card * { color: #3b0b0b !important; }

      /* FORCE only the Conn # label to black — specific selector with !important overrides the global rule */
      .sn-alert-card .label { color: #000000 !important; }

      .gr-markdown, .gr-markdown * , .gr-html, .gr-html * { color: var(--sn-light-text) !important; }
      .gradio-container .label, .gr-label, label, .gradio-container .gr-input, .gradio-container .gr-dropdown { color: var(--sn-light-text) !important; }
      .gr-button, .gr-button * { color: #ffffff !important; background-color: #0b62d6 !important; border-color: #084aa8 !important; }
      .gr-dataframe, .gr-dataframe table, .dataframe, .dataframe table, table.dataframe { color: var(--sn-light-text) !important; background: transparent !important; }
      .gr-dataframe table th, .dataframe table th, table.dataframe thead th { color: var(--sn-light-text) !important; background: rgba(255,255,255,0.03) !important; border-color: rgba(255,255,255,0.06) !important; }
      .gr-dataframe table td, .dataframe table td, table.dataframe td { color: var(--sn-light-text) !important; border-color: rgba(255,255,255,0.03) !important; }
      ::-webkit-scrollbar { height:10px; width:10px; background: #0b0b0c; }
      ::-webkit-scrollbar-thumb { background: #2f3a45; border-radius:6px; }
      [style*="color: #fff"], [style*="color: white"], [style*="color: rgba(255, 255, 255"] { color: var(--sn-light-text) !important; }
      @media (max-width:640px) { .sn-metric-card { padding:10px; } }
    </style>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙ Configuration")
            mode_input = gr.Radio(choices=["Live Monitoring", "File Analysis"], value="Live Monitoring", label="Detection Mode")
            gr.Markdown("### 📊 Dataset")
            dataset_choice = gr.Dropdown(choices=["KDD (NSL-KDD)", "CIC-DDoS (CIC-IDS2017)"], value="KDD (NSL-KDD)", label="Select Dataset")
            gr.Markdown("### 🧠 Algorithm")
            algo_select = gr.Dropdown(choices=[
                "Decision Tree (Recommended)",
                "Random Forest",
                "Logistic Regression",
                "Gradient Boosting",
                "SVM",
                "KNN",
                "Gaussian NB"
            ], value="Decision Tree (Recommended)", label="Select Algorithm")

            classification_mode = gr.Radio(choices=["Binary", "Multiclass"], value="Binary", label="Classification Mode")

            gr.Markdown("### Controls")
            startstop_btn = gr.Button("⏹ Stop Monitoring")
            clear_btn = gr.Button("🧹 Clear Data")
            export_file = gr.File(label="📤 Export Summary (click after Refresh)", interactive=True)

            metric_acc = gr.Textbox(value="Accuracy: --", label="Accuracy", interactive=False)
            metric_prec = gr.Textbox(value="Precision: --", label="Precision", interactive=False)
            gr.Markdown("---")
            gr.Markdown("Notes: App will auto-generate artifacts when CSVs are present. For production, prebuild artifacts and place them in the project root.")

        with gr.Column(scale=3):
            header_out = gr.HTML("<div></div>")
            metrics_out = gr.HTML("<div></div>")
            gr.Markdown("")
            gr.Markdown("### ⚠ Recent Alerts")
            alerts_out = gr.HTML("<div></div>")
            gr.Markdown("---")
            gr.Markdown("### Sample test rows")
            sample_table = gr.Dataframe(value=pd.DataFrame(), label="Sample rows")

    dummy_state = gr.State()

    def on_refresh(dataset_label, mode_val, algo_val, _unused, classification_mode_val):
        DATASET_MAP = {"KDD (NSL-KDD)": "kdd", "CIC-DDoS (CIC-IDS2017)": "ddos"}
        dataset_key = DATASET_MAP.get(dataset_label, "kdd")

        GLOBAL[dataset_key] = None
        header_html, metrics_html, alerts_html, sample_df, acc_text, prec_text, csv_path = refresh_view(
            dataset_label, dataset_key, algo_val, mode_val == "Live Monitoring", False, classification_mode_val
        )
        export_obj = None
        if csv_path is not None and os.path.exists(csv_path):
            export_obj = csv_path
        return header_html, metrics_html, alerts_html, sample_df, acc_text, prec_text, export_obj

    refresh_btn = gr.Button("Refresh View")

    refresh_btn.click(on_refresh, inputs=[dataset_choice, mode_input, algo_select, dummy_state, classification_mode],
                      outputs=[header_out, metrics_out, alerts_out, sample_table, metric_acc, metric_prec, export_file])

    dataset_choice.change(on_refresh, inputs=[dataset_choice, mode_input, algo_select, dummy_state, classification_mode],
                         outputs=[header_out, metrics_out, alerts_out, sample_table, metric_acc, metric_prec, export_file])
    mode_input.change(on_refresh, inputs=[dataset_choice, mode_input, algo_select, dummy_state, classification_mode],
                     outputs=[header_out, metrics_out, alerts_out, sample_table, metric_acc, metric_prec, export_file])
    algo_select.change(on_refresh, inputs=[dataset_choice, mode_input, algo_select, dummy_state, classification_mode],
                      outputs=[header_out, metrics_out, alerts_out, sample_table, metric_acc, metric_prec, export_file])
    classification_mode.change(on_refresh, inputs=[dataset_choice, mode_input, algo_select, dummy_state, classification_mode],
                               outputs=[header_out, metrics_out, alerts_out, sample_table, metric_acc, metric_prec, export_file])

    def toggle_monitor_label(current_label):
        return "▶ Start Monitoring" if current_label == "⏹ Stop Monitoring" else "⏹ Stop Monitoring"
    startstop_btn.click(toggle_monitor_label, inputs=[startstop_btn], outputs=[startstop_btn])

    def clear_action():
        return "Data cleared!","Data cleared!"
    clear_btn.click(clear_action, outputs=[metric_acc,metric_prec])

    demo.load(on_refresh, inputs=[dataset_choice, mode_input, algo_select, dummy_state, classification_mode],
              outputs=[header_out, metrics_out, alerts_out, sample_table, metric_acc, metric_prec, export_file])

if __name__ == "__main__":
    demo.launch()
