import os, sys, json, glob, io, zipfile, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_learning_curve_columns(df):
    """Normalize learning-curve DataFrame to ensure a 'fraction' column exists."""
    if df is None:
        return df
    cols = set(df.columns)
    if "fraction" not in cols and "train_size_fraction" in cols:
        df = df.rename(columns={"train_size_fraction": "fraction"})
    return df



# ----- Matplotlib color helpers (since user asked for colorful plots) -----
def cmap_colors(n, cmap_name="tab20"):
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i % cmap.N) for i in range(n)]

# ------------------------ path & zip helpers ------------------------
def parse_root(argv):
    for a in argv[1:]:
        if a and not a.startswith('-'):
            return os.path.normpath(a)
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

def list_zip_files(root):
    return [os.path.join(root, f) for f in os.listdir(root)
            if f.lower().endswith('.zip') and os.path.isfile(os.path.join(root, f))]

def safe_extract(zip_path, out_dir):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)

# ------------------------ recursive readers ------------------------
def read_first_json_under(base_dir):
    for p in glob.glob(os.path.join(base_dir, '**', '*.json'), recursive=True):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    return data[0]
                return data
        except Exception:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    lines = [json.loads(line) for line in f if line.strip()]
                    if lines:
                        return lines[0]
            except Exception:
                continue
    return {}

def read_first_csv_under(base_dir, prefer=None):
    # If prefer is provided (e.g., 'cv_results_csv'), try those first
    if prefer:
        for p in glob.glob(os.path.join(base_dir, '**', '*.csv'), recursive=True):
            if prefer in p.replace('\\','/').lower():
                try:
                    return pd.read_csv(p)
                except Exception:
                    pass
    # Fallback: any csv
    for p in glob.glob(os.path.join(base_dir, '**', '*.csv'), recursive=True):
        try:
            return pd.read_csv(p)
        except Exception:
            continue
    return pd.DataFrame()

def find_predictions_df(base_dir):
    """
    Search for any CSV with columns resembling predictions & labels.
    """
    # ----- UPDATED: Added your specific column names to the search sets -----
    cand_cols_pred = {'prediction', 'pred', 'predicted', 'y_pred', 'predicted_temperature_c'}
    cand_cols_true = {'label', 'target', 'y_true', 'actual', 'obs', 'temperature', 'temp', 'y', 'actual_temperature_c'}
    # ------------------------------------------------------------------------

    for p in glob.glob(os.path.join(base_dir, '**', '*.csv'), recursive=True):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        cols = set(c.lower() for c in df.columns)
        pred_col = next((c for c in df.columns if c.lower() in cand_cols_pred), None)
        true_col = next((c for c in df.columns if c.lower() in cand_cols_true), None)

        # Special handling for Spark default names (in case the aliases failed)
        if pred_col is None and 'prediction' in cols:
            pred_col = 'prediction'
        if true_col is None and 'label' in cols:
            true_col = 'label'

        if pred_col and true_col:
            # Filter out nulls
            sub = df[[true_col, pred_col]].dropna()
            if len(sub) >= 10:
                # Rename to standard 'label' and 'prediction' for plotting function
                return sub.rename(columns={true_col: 'label', pred_col: 'prediction'})

    return pd.DataFrame()

# ------------------------ plotting helpers ------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def savefig(path):
    ensure_dir(os.path.dirname(path))
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot_validation_curves(cv_df, model_name, plots_dir):
    made = []
    if cv_df.empty or 'avg_rmse' not in cv_df.columns:
        return made
    metric = 'avg_rmse'
    for col in cv_df.columns:
        if col == metric:
            continue
        if pd.api.types.is_numeric_dtype(cv_df[col]):
            # average across other params
            agg = cv_df.groupby(col, as_index=False)[metric].mean().sort_values(col)
            if agg.empty:
                continue
            plt.figure()
            plt.plot(agg[col], agg[metric], marker='o')
            plt.xlabel(col); plt.ylabel('avg_rmse'); plt.title(f'{model_name} Validation Curve: {col}')
            out = os.path.join(plots_dir, f'{model_name}_valcurve_{col}.png')
            savefig(out)
            made.append(('Validation Curve: ' + col, os.path.join('plots', f'{model_name}_valcurve_{col}.png')))
    return made

def plot_learning_curve(lc_df, model_name, plots_dir):
    """Plots both Train RMSE and Test RMSE against training fraction."""
    if lc_df.empty or ('fraction' not in lc_df.columns and 'train_size_fraction' not in lc_df.columns):
        return None

    # Standardize column names
    lc_df = lc_df.rename(columns={'rmse': 'test_rmse'})

    # Check for the required columns
    has_test = 'test_rmse' in lc_df.columns
    has_train = 'train_rmse' in lc_df.columns

    if not has_test and not has_train:
        return None # No RMSE data to plot

    plt.figure()

    # Plot Test RMSE (required)
    if has_test:
        plt.plot(lc_df['fraction'], lc_df['test_rmse'], marker='o', label='Test RMSE (or Validation)')

    # Plot Train RMSE (if available)
    if has_train:
        plt.plot(lc_df['fraction'], lc_df['train_rmse'], marker='x', label='Train RMSE')

    plt.xlabel('Train Fraction'); plt.ylabel('RMSE')
    plt.title(f'{model_name} Learning Curve')

    if has_test and has_train:
        plt.legend()
        out = os.path.join(plots_dir, f'{model_name}_learning_curve_train_test.png')
        plot_label = 'Learning Curve (Train/Test RMSE)'
    elif has_test: # Fallback to existing logic if only one column is present
        out = os.path.join(plots_dir, f'{model_name}_learning_curve.png')
        plot_label = 'Learning Curve (Test/Validation RMSE)'
    else:
        plt.close()
        return None

    savefig(out)
    return (plot_label, os.path.join('plots', os.path.basename(out)))

def plot_feature_importance(fi_df, model_name, plots_dir):
    if fi_df.empty or not {'feature','importance'}.issubset(fi_df.columns):
        return None
    top = fi_df.sort_values('importance', ascending=False).head(20)
    colors = cmap_colors(len(top))
    plt.figure(figsize=(8, max(3, len(top)*0.25)))
    plt.barh(top['feature'], top['importance'], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel('Importance'); plt.title(f'{model_name} Feature Importance')
    out = os.path.join(plots_dir, f'{model_name}_feature_importance.png')
    savefig(out)
    return ('Feature Importance', os.path.join('plots', f'{model_name}_feature_importance.png'))

def plot_coefficients(coef_df, model_name, plots_dir):
    if coef_df.empty or not {'feature','coefficient'}.issubset(coef_df.columns):
        return None
    cc = coef_df.copy()
    if 'abs_coeff' not in cc.columns:
        try:
            cc['abs_coeff'] = cc['coefficient'].astype(float).abs()
        except Exception:
            return None
    top = cc.sort_values('abs_coeff', ascending=False).head(20)
    colors = cmap_colors(len(top), 'tab10')
    plt.figure(figsize=(8, max(3, len(top)*0.25)))
    plt.barh(top['feature'], top['abs_coeff'], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel('|Coefficient|'); plt.title(f'{model_name} Coefficients')
    out = os.path.join(plots_dir, f'{model_name}_coefficients.png')
    savefig(out)
    return ('Coefficients', os.path.join('plots', f'{model_name}_coefficients.png'))

def plot_predictions_scatter(pred_df, model_name, plots_dir):
    if pred_df.empty or not {'label','prediction'}.issubset(pred_df.columns):
        return None, {}
    # Compute stats
    y = pred_df['label'].astype(float).values
    yhat = pred_df['prediction'].astype(float).values
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    # R2
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan

    # Scatter
    plt.figure(figsize=(6,6))
    plt.scatter(y, yhat, s=8, alpha=0.5)
    mn = float(np.min([y.min(), yhat.min()])); mx = float(np.max([y.max(), yhat.max()]))
    plt.plot([mn, mx], [mn, mx], linestyle='--')
    plt.xlabel('Actual'); plt.ylabel('Predicted')
    plt.title(f'{model_name} Predicted vs Actual\nRMSE={rmse:.3f}, R²={r2:.3f}')
    out = os.path.join(plots_dir, f'{model_name}_pred_vs_actual.png')
    savefig(out)
    return ('Predicted vs Actual', os.path.join('plots', f'{model_name}_pred_vs_actual.png')), {'rmse_from_scatter': rmse, 'r2_from_scatter': r2}

def write_html(report_dir, leaderboard_df, sections):
    html = io.StringIO()
    html.write('<html><head><meta charset="utf-8"><title>Assignment Report</title>')
    html.write('<style>body{font-family:system-ui,Segoe UI,Roboto,Arial} img{max-width:900px}</style></head><body>')
    html.write('<h1>Model Comparison Report</h1>')

    # Leaderboard table
    if not leaderboard_df.empty:
        html.write('<h2>Leaderboard</h2>')
        html.write(leaderboard_df.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))

    # Leaderboard comparison plots
    plots_dir = os.path.join(report_dir, 'plots')
    rmse_bar = os.path.join(plots_dir, 'leaderboard_rmse.png')
    r2_bar   = os.path.join(plots_dir, 'leaderboard_r2.png')
    if os.path.exists(rmse_bar):
        html.write('<h3>RMSE Comparison</h3>')
        html.write("<img src='plots/leaderboard_rmse.png'>")
    if os.path.exists(r2_bar):
        html.write('<h3>R² Comparison</h3>')
        html.write("<img src='plots/leaderboard_r2.png'>")

    # Per-model sections
    for sec in sections:
        m = sec['model']; metrics = sec.get('metrics', {})
        html.write(f'<hr><h2>{m}</h2>')
        if metrics:
            md = pd.DataFrame([metrics])
            html.write('<h3>Metrics</h3>')
            html.write(md.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x))
        if sec.get('plots'):
            html.write('<h3>Plots</h3><ul>')
            for label, rel in sec['plots']:
                html.write(f"<li>{label}<br><img src='{rel}'></li>")
            html.write('</ul>')
    html.write('</body></html>')

    with open(os.path.join(report_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html.getvalue())

def main():
    ROOT = parse_root(sys.argv)
    report_dir = os.path.join(ROOT, 'report')
    plots_dir  = os.path.join(report_dir, 'plots')
    tables_dir = os.path.join(report_dir, 'tables')
    for d in (report_dir, plots_dir, tables_dir):
        os.makedirs(d, exist_ok=True)

    zips = list_zip_files(ROOT)
    if not zips:
        print('[INFO] No *.zip files found beside the script.')
        sys.exit(0)

    leaderboard_rows = []
    sections = []

    for z in zips:
        base = os.path.basename(z)
        # Model name = zip base without extension (preserves RF.zip → RF)
        model_name = os.path.splitext(base)[0]
        out_dir = os.path.join(ROOT, model_name)
        safe_extract(z, out_dir)

        # Read artifacts (recursive)
        metrics = read_first_json_under(out_dir) or {}
        cv_df   = read_first_csv_under(out_dir, prefer='cv_results_csv')
        # --- Changed the preferred name to capture the learning curve CSV ---
        lc_df   = read_first_csv_under(out_dir, prefer='learning_curve')
        # -------------------------------------------------------------------
        fi_df   = read_first_csv_under(out_dir, prefer='feature_importances_csv')
        coef_df = read_first_csv_under(out_dir, prefer='coefficients_csv')
        pred_df = find_predictions_df(out_dir) # <-- This will now find your CSV

        # Skip totally unrecognized zips
        if not metrics and cv_df.empty and lc_df.empty and fi_df.empty and coef_df.empty and pred_df.empty:
            print(f'[WARN] Skipping {base} (no recognizable outputs).')
            continue

        # Build per-model section with plots
        model_plots = []

        # Learning curve (UPDATED)
        lc_plot = plot_learning_curve(lc_df, model_name, plots_dir)
        if lc_plot: model_plots.append(lc_plot)

        # CV curves
        model_plots.extend(plot_validation_curves(cv_df, model_name, plots_dir))

        # Feature importance / coefficients
        fi_plot = plot_feature_importance(fi_df, model_name, plots_dir)
        if fi_plot: model_plots.append(fi_plot)
        coef_plot = plot_coefficients(coef_df, model_name, plots_dir)
        if coef_plot: model_plots.append(coef_plot)

        # Predictions scatter
        scatter_plot, scatter_stats = plot_predictions_scatter(pred_df, model_name, plots_dir)
        if scatter_plot: model_plots.append(scatter_plot)

        # --- FIX: Prioritize Scatter Metrics ---
        if scatter_stats:
            # Overwrite the original test_rmse/test_r2 from the JSON file
            # with the more accurate metrics calculated from the prediction CSV.
            if 'rmse_from_scatter' in scatter_stats:
                metrics['test_rmse'] = scatter_stats['rmse_from_scatter']
            if 'r2_from_scatter' in scatter_stats:
                metrics['test_r2'] = scatter_stats['r2_from_scatter']

            # Also add the scatter keys for display in the per-model table
            metrics['scatter_rmse'] = scatter_stats.get('rmse_from_scatter')
            metrics['scatter_r2'] = scatter_stats.get('r2_from_scatter')
        # --- END FIX ---

        # Leaderboard row (now uses the potentially overwritten metrics)
        leaderboard_rows.append({
            'model': model_name,
            'test_rmse': metrics.get('test_rmse', np.nan),
            'test_r2': metrics.get('test_r2', np.nan),
        })

        sections.append({'model': model_name, 'metrics': metrics, 'plots': model_plots})

        # Persist tidy tables for appendix
        if not cv_df.empty:
            cv_df.to_csv(os.path.join(tables_dir, f'{model_name}_cv_results.csv'), index=False)
        if not lc_df.empty:
            lc_df.to_csv(os.path.join(tables_dir, f'{model_name}_learning_curve.csv'), index=False)
        if not fi_df.empty:
            fi_df.to_csv(os.path.join(tables_dir, f'{model_name}_feature_importances.csv'), index=False)
        if not coef_df.empty:
            coef_df.to_csv(os.path.join(tables_dir, f'{model_name}_coefficients.csv'), index=False)
        if not pred_df.empty:
            # We rename the columns back for the saved CSV
            pred_df.rename(columns={'label': 'Actual_Temperature_C', 'prediction': 'Predicted_Temperature_C'}) \
                   .head(50000).to_csv(os.path.join(tables_dir, f'{model_name}_predictions_sample.csv'), index=False)

    # Leaderboard table
    lb = pd.DataFrame(leaderboard_rows)
    lb_path = os.path.join(report_dir, 'leaderboard.csv')
    lb.to_csv(lb_path, index=False)

    # RMSE / R2 comparison bars
    if not lb.empty:
        # RMSE
        if 'test_rmse' in lb.columns and lb['test_rmse'].notna().any():
            plt.figure()
            colors = cmap_colors(len(lb), 'tab10')
            plt.bar(lb['model'], lb['test_rmse'], color=colors)
            plt.xticks(rotation=30, ha='right'); plt.ylabel('Test RMSE'); plt.title('Leaderboard: Test RMSE by Model')
            savefig(os.path.join(plots_dir, 'leaderboard_rmse.png'))
        # R^2
        if 'test_r2' in lb.columns and lb['test_r2'].notna().any():
            plt.figure()
            colors = cmap_colors(len(lb), 'tab10')
            plt.bar(lb['model'], lb['test_r2'], color=colors)
            plt.xticks(rotation=30, ha='right'); plt.ylabel('Test R²'); plt.title('Leaderboard: Test R² by Model')
            savefig(os.path.join(plots_dir, 'leaderboard_r2.png'))

    # HTML report
    write_html(report_dir, lb, sections)

    # Zip the report
    zip_out = os.path.join(ROOT, 'assignment_report.zip')
    if os.path.exists(zip_out):
        os.remove(zip_out)
    with zipfile.ZipFile(zip_out, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for folder, _, files in os.walk(report_dir):
            for fn in files:
                full = os.path.join(folder, fn)
                rel = os.path.relpath(full, ROOT)
                zf.write(full, arcname=rel)

    # Console summary
    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    if not lb.empty:
        try:
            print(lb[['model','test_rmse','test_r2']].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        except Exception:
            print(lb[['model','test_rmse','test_r2']])
    print(f"\n[DONE] Report folder: {report_dir}")
    print(f"[DONE] Zipped report: {zip_out}")

if __name__ == '__main__':
    main()
