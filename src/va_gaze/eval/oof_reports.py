import csv
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from va_gaze.eval.metrics import (
    VA_DIMENSIONS,
    calculate_va_metrics,
    concordance_correlation_coefficient,
    effective_logvars,
    safe_pearson_correlation,
)


DEFAULT_LOGVAR_MIN = -5.0
DEFAULT_LOGVAR_MAX = 3.0
UNCERTAINTY_CALIBRATION_BINS = 10
RISK_COVERAGE_LEVELS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)

preds_dir = None


def set_preds_dir(path):
    global preds_dir
    preds_dir = path


# This function handles CTRL-L C interrupt, erasing unused folders and terminating the program
def handle_signal(signum, stackframe):
    ''' signal handler '''
    # Best-effort cleanup of the output folder if it is still empty.
    if preds_dir and os.path.isdir(preds_dir):
        try:
            os.rmdir(preds_dir)
        except OSError:
            pass
    print('\n')
    sys.exit(1)


def _safe_pearson_corr(y_true, y_pred):
    return safe_pearson_correlation(y_true, y_pred)


def _calculate_va_metrics(
    df,
    logvar_min=DEFAULT_LOGVAR_MIN,
    logvar_max=DEFAULT_LOGVAR_MAX,
):
    """Calculate shared point and optional uncertainty metrics for one OOF slice."""

    uncertainty_columns = [
        "valence_logvar_pred",
        "arousal_logvar_pred",
    ]
    present_uncertainty_columns = [
        column for column in uncertainty_columns if column in df.columns
    ]
    if present_uncertainty_columns and len(present_uncertainty_columns) != 2:
        raise ValueError(
            "OOF predictions must contain both valence_logvar_pred and "
            "arousal_logvar_pred, or neither."
        )

    labels = df[["valence_true", "arousal_true"]].to_numpy(dtype=np.float64)
    prediction_columns = ["valence_pred", "arousal_pred"]
    prediction_columns.extend(present_uncertainty_columns)
    predictions = df[prediction_columns].to_numpy(dtype=np.float64)
    metrics = {"num_samples": int(len(df))}
    metrics.update(
        calculate_va_metrics(
            labels,
            predictions,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
    )
    return metrics


def _json_safe_metrics(metrics):
    """Convert NumPy scalars and non-finite floats to strict JSON values."""

    safe = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        safe[key] = value
    return safe


def _load_logvar_bounds(path):
    """Load and validate the training-time log-variance bounds for OOF reports."""

    parameters_path = os.path.join(path, "training_parameters.json")
    if not os.path.isfile(parameters_path):
        return DEFAULT_LOGVAR_MIN, DEFAULT_LOGVAR_MAX
    with open(parameters_path) as input_file:
        parameters = json.load(input_file)
    try:
        logvar_min = float(parameters.get("hetero_logvar_min", DEFAULT_LOGVAR_MIN))
        logvar_max = float(parameters.get("hetero_logvar_max", DEFAULT_LOGVAR_MAX))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "training_parameters.json must contain numeric hetero_logvar_min and "
            "hetero_logvar_max values."
        ) from error
    if not math.isfinite(logvar_min) or not math.isfinite(logvar_max):
        raise ValueError("Heteroscedastic log-variance bounds must be finite.")
    if logvar_min >= logvar_max:
        raise ValueError(
            "hetero_logvar_min must be smaller than hetero_logvar_max in "
            "training_parameters.json."
        )
    return logvar_min, logvar_max


def _has_heteroscedastic_predictions(df, prediction_filename=None):
    """Validate the raw log-variance pair and report whether it is present."""

    uncertainty_columns = {
        "valence_logvar_pred",
        "arousal_logvar_pred",
    }
    present_columns = uncertainty_columns.intersection(df.columns)
    if present_columns and present_columns != uncertainty_columns:
        source = f" in {prediction_filename}" if prediction_filename else ""
        raise ValueError(
            "Heteroscedastic predictions"
            f"{source} must contain both raw log-variance columns."
        )
    return present_columns == uncertainty_columns


def _add_effective_uncertainty_columns(df, logvar_min, logvar_max):
    """Add bounded log-variance and variance columns without replacing raw outputs."""

    if not _has_heteroscedastic_predictions(df):
        return df
    result = df.copy()
    raw_logvars = result[
        ["valence_logvar_pred", "arousal_logvar_pred"]
    ].to_numpy(dtype=np.float64)
    bounded_logvars = effective_logvars(
        raw_logvars,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )
    for index, dimension in enumerate(VA_DIMENSIONS):
        result[f"{dimension}_effective_logvar_pred"] = bounded_logvars[:, index]
        result[f"{dimension}_variance_pred"] = np.exp(bounded_logvars[:, index])
    return result


def _write_out_of_fold_metrics(
    path,
    df_join,
    logvar_min=DEFAULT_LOGVAR_MIN,
    logvar_max=DEFAULT_LOGVAR_MAX,
):
    """Write aggregate and per-dataset OOF metrics using shared metric definitions."""

    overall_metrics = _calculate_va_metrics(
        df_join,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )
    if _has_heteroscedastic_predictions(df_join):
        overall_metrics["hetero_logvar_min"] = float(logvar_min)
        overall_metrics["hetero_logvar_max"] = float(logvar_max)
    pd.DataFrame([overall_metrics]).to_csv(path + "/overall_metrics.csv", index=False)
    with open(path + "/overall_metrics.json", "w") as output_file:
        json.dump(
            _json_safe_metrics(overall_metrics),
            output_file,
            indent=2,
            allow_nan=False,
        )

    dataset_rows = []
    for dataset_name, df_dataset in df_join.groupby("dataset_of_origin"):
        row = {"dataset_of_origin": dataset_name}
        row.update(
            _calculate_va_metrics(
                df_dataset,
                logvar_min=logvar_min,
                logvar_max=logvar_max,
            )
        )
        dataset_rows.append(row)
    dataset_metrics = pd.DataFrame(dataset_rows)
    if not dataset_metrics.empty:
        dataset_metrics = dataset_metrics.sort_values("dataset_of_origin")
    dataset_metrics.to_csv(path + "/dataset_metrics.csv", index=False)


def _equal_count_bin_ids(values, max_bins=UNCERTAINTY_CALIBRATION_BINS):
    """Assign approximate equal-count bins without splitting tied uncertainties."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Equal-count binning requires a non-empty one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("Equal-count binning values must be finite.")
    number_of_bins = min(
        int(max_bins),
        int(values.size),
        int(np.unique(values).size),
    )
    if number_of_bins < 1:
        raise ValueError("max_bins must be at least one.")
    _, inverse_indices, tie_counts = np.unique(
        values,
        return_inverse=True,
        return_counts=True,
    )
    tie_midpoints = np.cumsum(tie_counts) - 0.5 * tie_counts
    tie_bin_ids = np.floor(
        tie_midpoints * number_of_bins / values.size
    ).astype(np.int64)
    tie_bin_ids = np.clip(tie_bin_ids, 0, number_of_bins - 1)
    _, contiguous_ids = np.unique(tie_bin_ids, return_inverse=True)
    return contiguous_ids[inverse_indices]


def _build_uncertainty_calibration_rows(df_join):
    """Build equal-count uncertainty-bin calibration diagnostics for both VA axes."""

    rows = []
    total_samples = len(df_join)
    for dimension in VA_DIMENSIONS:
        y_true = df_join[f"{dimension}_true"].to_numpy(dtype=np.float64)
        y_pred = df_join[f"{dimension}_pred"].to_numpy(dtype=np.float64)
        effective_logvar = df_join[
            f"{dimension}_effective_logvar_pred"
        ].to_numpy(dtype=np.float64)
        variance = df_join[f"{dimension}_variance_pred"].to_numpy(dtype=np.float64)
        bin_ids = _equal_count_bin_ids(variance)
        for bin_id in np.unique(bin_ids):
            selected = bin_ids == bin_id
            selected_error = y_true[selected] - y_pred[selected]
            selected_squared_error = np.square(selected_error)
            selected_variance = variance[selected]
            selected_logvar = effective_logvar[selected]
            selected_stddev = np.sqrt(selected_variance)
            mse = float(np.mean(selected_squared_error))
            gaussian_nll = 0.5 * (
                math.log(2.0 * math.pi)
                + selected_logvar
                + selected_squared_error / selected_variance
            )
            rows.append(
                {
                    "dimension": dimension,
                    "uncertainty_bin": int(bin_id) + 1,
                    "num_samples": int(np.sum(selected)),
                    "sample_fraction": float(np.mean(selected)),
                    "min_variance": float(np.min(selected_variance)),
                    "mean_variance": float(np.mean(selected_variance)),
                    "max_variance": float(np.max(selected_variance)),
                    "mean_effective_logvar": float(np.mean(selected_logvar)),
                    "mse": mse,
                    "rmse": float(np.sqrt(mse)),
                    "mae": float(np.mean(np.abs(selected_error))),
                    "gaussian_nll": float(np.mean(gaussian_nll)),
                    "coverage_1sigma": float(
                        np.mean(np.abs(selected_error) <= selected_stddev)
                    ),
                    "coverage_2sigma": float(
                        np.mean(np.abs(selected_error) <= 2.0 * selected_stddev)
                    ),
                    "mse_to_mean_variance_ratio": float(
                        mse / np.mean(selected_variance)
                    ),
                    "total_samples": int(total_samples),
                }
            )
    return rows


def _finite_mean(values):
    """Return the mean of finite scalars or NaN when no scalar is defined."""

    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite_values)) if finite_values else np.nan


def _risk_metrics(y_true, y_pred):
    """Calculate scalar point risks and safe correlations for a selected subset."""

    errors = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean(np.square(errors)))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(errors))),
        "pearson_corr": safe_pearson_correlation(y_true, y_pred),
        "ccc": concordance_correlation_coefficient(y_true, y_pred),
    }


def _coverage_counts(total_samples):
    """Return unique retained counts for the configured descending coverage levels."""

    counts = []
    for target_coverage in RISK_COVERAGE_LEVELS:
        retained = max(1, int(math.ceil(float(target_coverage) * total_samples)))
        if retained not in {count for _, count in counts}:
            counts.append((float(target_coverage), retained))
    return counts


def _least_uncertain_selections(uncertainty):
    """Select complete uncertainty tie groups at each requested coverage level."""

    uncertainty = np.asarray(uncertainty, dtype=np.float64)
    if uncertainty.ndim != 1 or uncertainty.size == 0:
        raise ValueError("Risk-coverage uncertainty must be a non-empty 1D array.")
    if not np.isfinite(uncertainty).all():
        raise ValueError("Risk-coverage uncertainty must contain only finite values.")
    sorted_uncertainty = np.sort(uncertainty)
    seen_selected_counts = set()
    selections = []
    for target_coverage, desired_count in _coverage_counts(uncertainty.size):
        cutoff = sorted_uncertainty[desired_count - 1]
        selected = np.flatnonzero(uncertainty <= cutoff)
        if selected.size in seen_selected_counts:
            continue
        seen_selected_counts.add(int(selected.size))
        selections.append((target_coverage, selected))
    return selections


def _build_uncertainty_risk_coverage_rows(df_join):
    """Build selective-prediction curves by retaining the least uncertain samples."""

    rows = []
    total_samples = len(df_join)
    for dimension in VA_DIMENSIONS:
        y_true = df_join[f"{dimension}_true"].to_numpy(dtype=np.float64)
        y_pred = df_join[f"{dimension}_pred"].to_numpy(dtype=np.float64)
        uncertainty = df_join[f"{dimension}_variance_pred"].to_numpy(dtype=np.float64)
        for target_coverage, selected in _least_uncertain_selections(uncertainty):
            retained = selected.size
            row = {
                "dimension": dimension,
                "target_coverage": target_coverage,
                "actual_coverage": float(retained / total_samples),
                "num_selected": int(retained),
                "total_samples": int(total_samples),
                "mean_uncertainty_score": float(np.mean(uncertainty[selected])),
                "max_uncertainty_score": float(np.max(uncertainty[selected])),
            }
            row.update(_risk_metrics(y_true[selected], y_pred[selected]))
            rows.append(row)

    joint_uncertainty = np.mean(
        df_join[["valence_variance_pred", "arousal_variance_pred"]].to_numpy(
            dtype=np.float64
        ),
        axis=1,
    )
    labels = df_join[["valence_true", "arousal_true"]].to_numpy(dtype=np.float64)
    predictions = df_join[["valence_pred", "arousal_pred"]].to_numpy(dtype=np.float64)
    for target_coverage, selected in _least_uncertain_selections(joint_uncertainty):
        retained = selected.size
        point_metrics = calculate_va_metrics(labels[selected], predictions[selected])
        row = {
            "dimension": "joint",
            "target_coverage": target_coverage,
            "actual_coverage": float(retained / total_samples),
            "num_selected": int(retained),
            "total_samples": int(total_samples),
            "mean_uncertainty_score": float(np.mean(joint_uncertainty[selected])),
            "max_uncertainty_score": float(np.max(joint_uncertainty[selected])),
            "mse": float(
                np.mean(
                    [
                        point_metrics["mse_valence"],
                        point_metrics["mse_arousal"],
                    ]
                )
            ),
            "mae": float(
                np.mean(
                    [
                        point_metrics["mae_valence"],
                        point_metrics["mae_arousal"],
                    ]
                )
            ),
            "pearson_corr": _finite_mean(
                [
                    point_metrics["pearson_corr_valence"],
                    point_metrics["pearson_corr_arousal"],
                ]
            ),
            "ccc": point_metrics["ccc_mean"],
        }
        row["rmse"] = float(np.sqrt(row["mse"]))
        rows.append(row)
    return rows


def _write_uncertainty_reports(path, df_join):
    """Write calibration and risk-coverage reports for heteroscedastic outputs."""

    calibration_rows = _build_uncertainty_calibration_rows(df_join)
    pd.DataFrame(calibration_rows).to_csv(
        os.path.join(path, "uncertainty_calibration.csv"),
        index=False,
    )
    risk_coverage_rows = _build_uncertainty_risk_coverage_rows(df_join)
    pd.DataFrame(risk_coverage_rows).to_csv(
        os.path.join(path, "uncertainty_risk_coverage.csv"),
        index=False,
    )


def _remove_stale_uncertainty_reports(path):
    """Remove heteroscedastic reports when rebuilding a point-only run directory."""

    for filename in (
        "uncertainty_calibration.csv",
        "uncertainty_risk_coverage.csv",
    ):
        report_path = os.path.join(path, filename)
        if os.path.isfile(report_path):
            os.remove(report_path)


def _join_dataset_and_predictions(dataset_df, predictions_df, prediction_filename):
    if len(dataset_df) != len(predictions_df):
        raise ValueError(
            f"{prediction_filename} has {len(predictions_df)} rows, "
            f"but the matching dataset fold has {len(dataset_df)} rows. "
            "Use the same data/full_dataset_fold*.csv files that were used during training."
        )
    return pd.concat(
        [dataset_df.reset_index(drop=True), predictions_df.reset_index(drop=True)],
        axis=1,
    )


def _rename_prediction_columns(df):
    rename_map = {
        "Unnamed: 0": "index_pred",
        "0": "valence_pred",
        "1": "arousal_pred",
        "2": "valence_logvar_pred",
        "3": "arousal_logvar_pred",
    }
    return df.rename(columns=rename_map)


def _validate_prediction_frame(df, prediction_filename):
    """Validate mean-output columns and return whether raw uncertainty is present."""

    required_columns = {"valence_pred", "arousal_pred"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"{prediction_filename} is missing prediction columns: "
            + ", ".join(sorted(missing_columns))
        )
    return _has_heteroscedastic_predictions(df, prediction_filename)


def _prediction_output_columns(df):
    cols = [
        "index",
        "text",
        "dataset_of_origin",
        "valence_true",
        "arousal_true",
        "valence_pred",
        "arousal_pred",
    ]
    optional_cols = [
        "valence_logvar_pred",
        "arousal_logvar_pred",
        "valence_effective_logvar_pred",
        "arousal_effective_logvar_pred",
        "valence_variance_pred",
        "arousal_variance_pred",
    ]
    cols.extend([col for col in optional_cols if col in df.columns])
    return cols


# Code tested on script_sort_predictions_temp.ipynb
def create_prediction_tables(path, data_dir="data"):
    df_preds_fold1 = pd.read_csv(path + '/predictions_fold1.csv')
    df_preds_fold2 = pd.read_csv(path + '/predictions_fold2.csv')
    df_preds_fold1 = _rename_prediction_columns(df_preds_fold1)
    df_preds_fold2 = _rename_prediction_columns(df_preds_fold2)
    fold1_has_uncertainty = _validate_prediction_frame(
        df_preds_fold1,
        "predictions_fold1.csv",
    )
    fold2_has_uncertainty = _validate_prediction_frame(
        df_preds_fold2,
        "predictions_fold2.csv",
    )
    if fold1_has_uncertainty != fold2_has_uncertainty:
        raise ValueError(
            "predictions_fold1.csv and predictions_fold2.csv must use the same "
            "point-only or heteroscedastic output schema."
        )
    if fold1_has_uncertainty:
        logvar_min, logvar_max = _load_logvar_bounds(path)
    else:
        logvar_min, logvar_max = DEFAULT_LOGVAR_MIN, DEFAULT_LOGVAR_MAX
    
    # Import original dataset files to df
    df_dataset_fold1 = pd.read_csv(os.path.join(data_dir, 'full_dataset_fold1.csv'),sep='\t',
                    quotechar='"',
                    engine='python', 
                    quoting=csv.QUOTE_NONE,
                    escapechar='\\',
                    keep_default_na=False,
                    dtype={'index':np.int32,'text':str,'valence':np.float64, 'arousal':np.float64})
    df_dataset_fold2 = pd.read_csv(os.path.join(data_dir, 'full_dataset_fold2.csv'),sep='\t',
                    quotechar='"',
                    engine='python', 
                    quoting=csv.QUOTE_NONE,
                    escapechar='\\',
                    keep_default_na=False,
                    dtype={'index':np.int32,'text':str,'valence':np.float64, 'arousal':np.float64})
    df_dataset_fold1 = df_dataset_fold1.rename(columns={'valence' : 'valence_true', 'arousal' : 'arousal_true'})
    df_dataset_fold2 = df_dataset_fold2.rename(columns={'valence' : 'valence_true', 'arousal' : 'arousal_true'})

    # Merge the original dataset and the predicted values in the same dataframe

    # Fold 1
    df_fold1_join = _join_dataset_and_predictions(
        df_dataset_fold1, df_preds_fold1, "predictions_fold1.csv"
    )
    df_fold1_join = df_fold1_join.drop(columns=['index_pred'], errors='ignore')
    df_fold1_join = df_fold1_join[_prediction_output_columns(df_fold1_join)]

    # Fold 2
    df_fold2_join = _join_dataset_and_predictions(
        df_dataset_fold2, df_preds_fold2, "predictions_fold2.csv"
    )
    df_fold2_join = df_fold2_join.drop(columns=['index_pred'], errors='ignore')
    df_fold2_join = df_fold2_join[_prediction_output_columns(df_fold2_join)]
    
    df_join = pd.concat([df_fold1_join, df_fold2_join], axis=0, ignore_index=True)

    # Sort dataframe by index
    df_join = df_join.sort_values('index').reset_index(drop=True)
    if fold1_has_uncertainty:
        df_join = _add_effective_uncertainty_columns(
            df_join,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
        )
        df_join = df_join[_prediction_output_columns(df_join)]
    df_join.to_csv(path + "/all_predictions.csv", index=False)
    _write_out_of_fold_metrics(
        path,
        df_join,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    )
    if fold1_has_uncertainty:
        _write_uncertainty_reports(path, df_join)
    else:
        _remove_stale_uncertainty_reports(path)
    
    # A list with the name of all the datasets used 
    datasets_list = list(df_join.dataset_of_origin.unique())
    # len(datasets_list) # 33 datasets

    # words  dataset
    words_ds_list = ['ANEW to EP', 'ANGST', 'ANPW_R', 'BAWL_R', 'Cantonese Nouns','Chinese words', 'ChineseW11k', 'CroatianNorms', 'DutchAdj', 'FAN - french words', 'FEEL', 'FinnishNorms', 'FinnishNouns', 'German words', 'GlasgowNorms', 
    'Italian words', 'NAWL', 'nrc-vad', 'TurkishNorms', 'word ratings NL', 'word ratings ES', 'word ratings ENG']
    # sentences dataset
    sent_ds_list = ['ANET sentences', 'CVAI', 'CVAT', 'COMETA sentences', 'COMETA stories', 'Emobank', 'EmoTales sentences', 'fb', 'IEMOCAP sentences', 'MAS', 'PANIG sentences', 'Polish sentences']

    dataset_langs = {
        'ANGST': "German", 'BAWL_R': "German", 'German words': "German", 'COMETA sentences': "German", 'COMETA stories': "German", 'PANIG sentences': "German",
        'ANPW_R' : "Polish", 'NAWL' : "Polish", 'Polish sentences' : "Polish", 'Chinese words' : "Mandarin", 'ChineseW11k' : "Mandarin", 'CVAI' : "Mandarin",
        'CVAT' : "Mandarin", 'FAN - french words' : "French", 'FEEL' : "French", 'Italian words' : "Italian", 'CroatianNorms' : "Croatian", 'FinnishNorms' : 'Finnish',
        'FinnishNouns' : 'Finnish', 'TurkishNorms' : 'Turkish', 'word ratings NL' : "Dutch", 'DutchAdj' : "Dutch", 'GlasgowNorms' : 'English', 'nrc-vad' : 'English',
        'word ratings ENG' : 'English', 'ANET sentences' : 'English', 'Emobank' : 'English',  'EmoTales sentences' : 'English', 'fb' : 'English', 'IEMOCAP sentences' : 'English',
        'word ratings ES' : 'Spanish', 'Cantonese Nouns' : 'Cantonese', 'ANEW to EP' : 'Portuguese', 'MAS' : 'Portuguese'
    }

    # Keep only datasets available in the current run (e.g., English-only subsets).
    available_datasets = set(df_join.dataset_of_origin.unique())
    words_ds_list = [ds for ds in words_ds_list if ds in available_datasets]
    sent_ds_list = [ds for ds in sent_ds_list if ds in available_datasets]

    # add ds_type column
    temp_word = df_join[df_join['dataset_of_origin'].isin(words_ds_list)] #['ds_type'] = 'word'
    temp_word = temp_word.assign(ds_type = 'word')
    temp_sent = df_join[df_join['dataset_of_origin'].isin(sent_ds_list)] #['ds_type'] = 'word'
    temp_sent = temp_sent.assign(ds_type = 'sentence')
    full_df = pd.concat([temp_word, temp_sent], axis=0)
    
    # add language column
    german = ['ANGST', 'BAWL_R','German words', 'COMETA sentences', 'COMETA stories', 'PANIG sentences']
    polish = ['ANPW_R','NAWL', 'Polish sentences']
    mandarin = ['Chinese words','ChineseW11k','CVAI','CVAT']
    french = ['FAN - french words','FEEL']
    italian = ['Italian words']
    croatian = ['CroatianNorms']
    finnish = ['FinnishNorms','FinnishNouns']
    turkish = ['TurkishNorms']
    dutch = ['word ratings NL','DutchAdj']
    english = ['GlasgowNorms','nrc-vad','word ratings ENG','ANET sentences','Emobank', 'EmoTales sentences', 'fb', 'IEMOCAP sentences']
    spanish = ['word ratings ES']
    cantonese = ['Cantonese Nouns']
    portuguese = ['ANEW to EP','MAS']
    
    # Add columns language and type
    def add_column_lang(ds_origin):
        return dataset_langs.get(ds_origin, 'Unknown')

    # run add col lang function
    full_df['language'] = full_df.dataset_of_origin.apply(add_column_lang)
    
    # TABLES 1 - Word datasets

    # Array containing languages to fill df
    lang_array = []
    rmse_val_array = []
    rmse_aro_array = []
    mae_val_array = []
    mae_aro_array = []
    r_val_array = []
    r_aro_array = []

    if words_ds_list:
        for ds in words_ds_list:
            # language
            l = dataset_langs.get(ds, 'Unknown')
            lang_array.append(l)
            df_temp = full_df[full_df.dataset_of_origin == ds]
            rmse_valence = np.sqrt(mean_squared_error(df_temp.valence_true, df_temp.valence_pred))
            rmse_arousal = np.sqrt(mean_squared_error(df_temp.arousal_true, df_temp.arousal_pred))
            mae_valence = mean_absolute_error(df_temp.valence_true, df_temp.valence_pred)
            mae_arousal = mean_absolute_error(df_temp.arousal_true, df_temp.arousal_pred)
            r_valence = _safe_pearson_corr(
                df_temp.valence_true,
                df_temp.valence_pred,
            )
            r_arousal = _safe_pearson_corr(
                df_temp.arousal_true,
                df_temp.arousal_pred,
            )
           
            # Append values to its arrays
            rmse_val_array.append(round(rmse_valence,4))
            rmse_aro_array.append(round(rmse_arousal,4))
            mae_val_array.append(round(mae_valence,4))
            mae_aro_array.append(round(mae_arousal,4))
            r_val_array.append(round(r_valence,4))
            r_aro_array.append(round(r_arousal,4))

        # Arrays to put in the df
        ds_array = np.array(words_ds_list).reshape(len(words_ds_list), 1)
        lang_array = np.array(lang_array).reshape(len(lang_array), 1)
        rmse_val_array = np.array(rmse_val_array).reshape(len(rmse_val_array), 1)
        rmse_aro_array = np.array(rmse_aro_array).reshape(len(rmse_aro_array), 1)
        mae_val_array = np.array(mae_val_array).reshape(len(mae_val_array), 1)
        mae_aro_array = np.array(mae_aro_array).reshape(len(mae_aro_array), 1)
        r_val_array = np.array(r_val_array).reshape(len(r_val_array), 1)
        r_aro_array = np.array(r_aro_array).reshape(len(r_aro_array), 1)

        matrix = np.hstack((ds_array, lang_array, rmse_val_array, mae_val_array, r_val_array, rmse_aro_array, mae_aro_array, r_aro_array))
        # Putting the df together
        header = [np.array(['', '','Valence', 'Valence', 'Valence', 'Arousal', 'Arousal', 'Arousal']), 
        np.array(['Dataset','Language', 'RMSE', 'MAE', 'r', 'RMSE', 'MAE', 'r'])]

        df = pd.DataFrame(matrix, columns= header) #, index=ind

        def df_style(val):
            return "font-weight: bold"

        v_rmse_mean = np.mean(np.array(df.Valence.RMSE, dtype=float))
        v_mae_mean = np.mean(np.array(df.Valence.MAE, dtype=float))
        v_r_mean = np.mean(np.array(df.Valence.r, dtype=float))
        a_rmse_mean = np.mean(np.array(df.Arousal.RMSE, dtype=float))
        a_mae_mean = np.mean(np.array(df.Arousal.MAE, dtype=float))
        a_r_mean = np.mean(np.array(df.Arousal.r, dtype=float))
        df.loc[df.shape[0]] = ['Overall','', round(v_rmse_mean,4), round(v_mae_mean,4), round(v_r_mean,4), round(a_rmse_mean,4), round(a_mae_mean,4), round(a_r_mean,4)]
    else:
        df = pd.DataFrame()
    df.to_pickle(path + "/table1.pkl")

    # TABLE 2 - Sentence datasets
    # Array containing languages to fill df
    lang_array = []
    rmse_val_array = []
    rmse_aro_array = []
    mae_val_array = []
    mae_aro_array = []
    r_val_array = []
    r_aro_array = []

    if sent_ds_list:
        for ds in sent_ds_list:
            # language
            l = dataset_langs.get(ds, 'Unknown')
            lang_array.append(l)
            #get sub-df
            df_temp = full_df[full_df.dataset_of_origin == ds]
            rmse_valence = np.sqrt(mean_squared_error(df_temp.valence_true, df_temp.valence_pred))
            rmse_arousal = np.sqrt(mean_squared_error(df_temp.arousal_true, df_temp.arousal_pred))
            mae_valence = mean_absolute_error(df_temp.valence_true, df_temp.valence_pred)
            mae_arousal = mean_absolute_error(df_temp.arousal_true, df_temp.arousal_pred)
            r_valence = _safe_pearson_corr(
                df_temp.valence_true,
                df_temp.valence_pred,
            )
            r_arousal = _safe_pearson_corr(
                df_temp.arousal_true,
                df_temp.arousal_pred,
            )
            # Append values to its arrays
            rmse_val_array.append(round(rmse_valence,4))
            rmse_aro_array.append(round(rmse_arousal,4))
            mae_val_array.append(round(mae_valence,4))
            mae_aro_array.append(round(mae_arousal,4))
            r_val_array.append(round(r_valence,4))
            r_aro_array.append(round(r_arousal,4))

        # Arrays to put in the df
        ds_array = np.array(sent_ds_list).reshape(len(sent_ds_list), 1)
        lang_array = np.array(lang_array).reshape(len(lang_array), 1)
        rmse_val_array = np.array(rmse_val_array).reshape(len(rmse_val_array), 1)
        rmse_aro_array = np.array(rmse_aro_array).reshape(len(rmse_aro_array), 1)
        mae_val_array = np.array(mae_val_array).reshape(len(mae_val_array), 1)
        mae_aro_array = np.array(mae_aro_array).reshape(len(mae_aro_array), 1)
        r_val_array = np.array(r_val_array).reshape(len(r_val_array), 1)
        r_aro_array = np.array(r_aro_array).reshape(len(r_aro_array), 1)
        matrix = np.hstack((ds_array, lang_array, rmse_val_array, mae_val_array, r_val_array, rmse_aro_array, mae_aro_array, r_aro_array))
        # Putting the df together
        header = [np.array(['', '','Valence', 'Valence', 'Valence', 'Arousal', 'Arousal', 'Arousal']), 
        np.array(['Dataset','Language', 'RMSE', 'MAE', 'r', 'RMSE', 'MAE', 'r'])]

        df = pd.DataFrame(matrix, columns= header) #, index=ind

        v_rmse_mean = np.mean(np.array(df.Valence.RMSE, dtype=float))
        v_mae_mean = np.mean(np.array(df.Valence.MAE, dtype=float))
        v_r_mean = np.mean(np.array(df.Valence.r, dtype=float))
        a_rmse_mean = np.mean(np.array(df.Arousal.RMSE, dtype=float))
        a_mae_mean = np.mean(np.array(df.Arousal.MAE, dtype=float))
        a_r_mean = np.mean(np.array(df.Arousal.r, dtype=float))
        df.loc[df.shape[0]] = ['Overall','', round(v_rmse_mean,4), round(v_mae_mean,4), round(v_r_mean,4), round(a_rmse_mean,4), round(a_mae_mean,4), round(a_r_mean,4)]
    else:
        df = pd.DataFrame()
    df.to_pickle(path + "/table2.pkl")
   
   
   
   
   
def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val

def corrcoef(x):
    """
    Mimics `np.corrcoef`

    Arguments
    ---------
    x : 2D torch.Tensor
    
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)

    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref: 
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013

    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c
