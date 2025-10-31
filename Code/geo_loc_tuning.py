import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import uniform, randint
import xgboost as xgb
import lightgbm as lgb
from joblib import Parallel, delayed
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

def centered_log_ratio(data):
    """
    Performs the Centered Log-Ratio (CLR) transformation.
    A small pseudocount is added to handle zero values.
    """
    data_pseudo = data + 1e-6
    geometric_mean = np.exp(np.log(data_pseudo).mean(axis=1))
    clr_transformed = np.log(data_pseudo.divide(geometric_mean, axis=0))
    return clr_transformed

def isometric_log_ratio(data):
    """
    Performs a simplified Isometric Log-Ratio (ILR) transformation.
    This implementation uses a sequential binary partition approach.
    A small pseudocount is added to handle zero values.
    """
    data_pseudo = data + 1e-6
    n_features = data_pseudo.shape[1]
    
    ilr_features = pd.DataFrame(index=data_pseudo.index)
    for i in range(1, n_features):
        part1 = data_pseudo.iloc[:, :i]
        part2 = data_pseudo.iloc[:, i]
        
        gm1 = np.exp(np.log(part1).mean(axis=1))
        gm2 = part2
        
        balance = np.sqrt(i / (i + 1)) * np.log(gm1 / gm2)
        ilr_features[f'ilr_balance_{i}'] = balance
        
    return ilr_features


def apply_filter_strategy(otu_aligned, filter_name, filter_func):
    """Apply a single filtering strategy"""
    try:
        if filter_name in ['CLR_Transform', 'ILR_Transform']:
            otu_filtered = filter_func(otu_aligned)
        else:
            otu_filtered = filter_func(otu_aligned.copy())
        return otu_filtered
    except Exception as e:
        print(f"      ❌ Transformation or filtering failed for {filter_name}: {str(e)}")
        return None


def tune_and_evaluate_single_combination(filter_name, model_name, model, param_dist, X, y_final, combo_id, total_combos):
    """
    Tune and evaluate a single model-filter combination.
    This function is designed to be run in parallel.
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"[{timestamp}] 🔄 Starting [{combo_id}/{total_combos}]: {model_name} + {filter_name}")
    
    if X is None or X.shape[1] == 0:
        print(f"[{timestamp}] ⚠️  Skipping [{combo_id}/{total_combos}]: No features")
        return {
            'Target_Feature': 'Geographical Location',
            'Filtering_Strategy': filter_name,
            'Classifier': model_name,
            'CV_Accuracy_Mean': np.nan,
            'CV_Accuracy_Std': np.nan,
            'CV_F1_Mean': np.nan,
            'CV_AUC_Mean': np.nan,
            'Features_Count': 0,
            'Samples_Count': 0,
            'Best_Hyperparameters': 'No features',
            'Processing_Time_Seconds': 0
        }
    
    try:
        # Stratified K-Fold for robust tuning and CV
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Some models need scaled data
        X_model = X
        if model_name in ['SVM', 'LogisticRegression']:
            scaler = StandardScaler()
            X_model = scaler.fit_transform(X)
        
        print(f"[{timestamp}] 🔍 Hyperparameter tuning for [{combo_id}/{total_combos}]: {model_name}")
        
        # Randomized Search for Hyperparameter Tuning
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=20,
            cv=cv_inner,
            scoring='accuracy',
            n_jobs=1,  # Important: use 1 to avoid nested parallelism
            verbose=0,
            random_state=42,
            error_score='raise'  # Raise errors instead of silently failing
        )
        
        # Fit the search to find the best model
        random_search.fit(X_model, y_final)
        best_model = random_search.best_estimator_

        print(f"[{timestamp}] 📊 Cross-validating [{combo_id}/{total_combos}]: {model_name}")
        
        # Evaluate the best model with cross-validation
        cv_scores = cross_val_score(best_model, X_model, y_final, cv=cv_outer, scoring='accuracy', n_jobs=1)
        
        is_binary = y_final.nunique() == 2
        cv_roc_auc = [np.nan] * 5
        cv_f1 = [np.nan] * 5
        
        if is_binary:
            if hasattr(best_model, 'predict_proba'):
                cv_roc_auc = cross_val_score(best_model, X_model, y_final, cv=cv_outer, scoring='roc_auc', n_jobs=1)
            cv_f1 = cross_val_score(best_model, X_model, y_final, cv=cv_outer, scoring='f1', n_jobs=1)

        elapsed = time.time() - start_time
        timestamp_end = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp_end}] ✅ Completed [{combo_id}/{total_combos}]: {model_name} + {filter_name} | "
              f"Acc: {np.mean(cv_scores):.4f} | Time: {elapsed:.1f}s")

        return {
            'Target_Feature': 'Geographical Location',
            'Filtering_Strategy': filter_name,
            'Classifier': model_name,
            'CV_Accuracy_Mean': np.mean(cv_scores),
            'CV_Accuracy_Std': np.std(cv_scores),
            'CV_F1_Mean': np.mean(cv_f1),
            'CV_AUC_Mean': np.mean(cv_roc_auc),
            'Features_Count': X.shape[1],
            'Samples_Count': X.shape[0],
            'Best_Hyperparameters': str(random_search.best_params_),
            'Processing_Time_Seconds': elapsed
        }

    except Exception as e:
        elapsed = time.time() - start_time
        timestamp_err = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp_err}] ❌ ERROR [{combo_id}/{total_combos}]: {model_name} + {filter_name}")
        print(f"      Error details: {str(e)}")
        
        return {
            'Target_Feature': 'Geographical Location',
            'Filtering_Strategy': filter_name,
            'Classifier': model_name,
            'CV_Accuracy_Mean': np.nan,
            'CV_Accuracy_Std': np.nan,
            'CV_F1_Mean': np.nan,
            'CV_AUC_Mean': np.nan,
            'Features_Count': X.shape[1] if X is not None else 0,
            'Samples_Count': X.shape[0] if X is not None else 0,
            'Best_Hyperparameters': f'Error: {str(e)[:100]}',
            'Processing_Time_Seconds': elapsed
        }


def run_tuned_pipeline_for_geographical_location(n_jobs=40):
    """
    Main function to run the classification pipeline with hyperparameter tuning
    for the 'Geographical Location' metadata feature.
    
    Parameters:
    -----------
    n_jobs : int, default=40
        Number of parallel jobs for outer parallelization.
    """
    
    overall_start = time.time()
    
    # ------------------
    # 1. Load Data
    # ------------------
    print("="*80)
    print("📂 LOADING DATA")
    print("="*80)
    try:
        otu_raw = pd.read_excel("/home/user/Documents/Sandhya/AI_in_medicine_lab/AI project/abundance_only.xlsx", index_col=0)
        meta_raw = pd.read_excel("/home/user/Documents/Sandhya/AI_in_medicine_lab/AI project/metadata_only.xlsx")
        print(f"✅ Data loaded successfully")
        print(f"   OTU data shape: {otu_raw.shape}")
        print(f"   Metadata shape: {meta_raw.shape}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Preprocessing metadata
    meta = meta_raw.set_index(meta_raw.columns[0])
    
    # Define models and their hyperparameter search spaces
    models_and_params = {
        'HistGradientBoost_sklearn': (HistGradientBoostingClassifier(random_state=42), {
            'max_iter': randint(50, 300),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'min_samples_leaf': randint(1, 20),
            'max_leaf_nodes': randint(10, 50)
        }),
        'LightGBM': (lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs =1), {
            'n_estimators': randint(50, 300),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'num_leaves': randint(10, 50)
        }),
        'XGBoost': (xgb.XGBClassifier(eval_metric='logloss', random_state=42), {
            'n_estimators': randint(50, 300),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        }),
        'RandomForest': (RandomForestClassifier(random_state=42, n_jobs=1), {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None]
        }),
        'SVM': (SVC(random_state=42, probability=True), {
            'C': uniform(0.1, 10),
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'poly']
        }),
        'LogisticRegression': (LogisticRegression(random_state=42, max_iter=2000), {
            'C': uniform(0.1, 10),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        })
    }
    
    # Define filtering and transformation strategies
    filter_strategies = {
        'No_Filtering': lambda df: df,
        'Abundance_0.1%': lambda df: df.loc[:, (df > 0.001).any(axis=0)],
        'Prevalence_5%': lambda df: df.loc[:, (df > 0).mean(axis=0) >= 0.05],
        'Both_Filters': lambda df: df.loc[:, ((df > 0.001).any(axis=0) & ((df > 0).mean(axis=0) >= 0.05))],
        'CLR_Transform': centered_log_ratio,
        'ILR_Transform': isometric_log_ratio
    }
    
    # Target metadata feature
    target_col = 'Geographical Location'

    print("\n" + "="*80)
    print(f"🚀 RUNNING TUNED PIPELINE FOR: {target_col}")
    print("="*80)

    if target_col not in meta.columns:
        print(f"   ⚠️ Column '{target_col}' not found in metadata. Skipping.")
        return
    
    y = meta[target_col]
    y = y.dropna()

    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Align data after handling NaNs
    common_samples = list(set(otu_raw.index) & set(y.index))
    otu_aligned = otu_raw.loc[common_samples]
    y_series = pd.Series(y_encoded, index=y.index)
    y_final = y_series.loc[common_samples]

    if otu_aligned.shape[0] < 20 or y_final.nunique() <= 1:
        print(f"   ⚠️ Not enough samples or classes for target: {target_col}. Skipping.")
        return
        
    print(f"   📊 Dataset size: {otu_aligned.shape[0]} samples, {otu_aligned.shape[1]} features")
    print(f"   📊 Number of classes: {y_final.nunique()}")
    print(f"   📊 Class distribution: {dict(pd.Series(y_final).value_counts())}")

    # ------------------
    # 2. Prepare All Combinations
    # ------------------
    print("\n" + "="*80)
    print("🔧 PREPARING ALL MODEL-STRATEGY COMBINATIONS")
    print("="*80)
    
    combinations = []
    
    for filter_name, filter_func in filter_strategies.items():
        print(f"\n📋 Processing filter: {filter_name}")
        X = apply_filter_strategy(otu_aligned, filter_name, filter_func)
        
        if X is None or X.shape[1] == 0:
            print(f"   ⚠️ No features for {filter_name}, skipping this strategy")
            continue
        
        print(f"   ✅ Features after filtering: {X.shape[1]}")
        
        for model_name, (model, param_dist) in models_and_params.items():
            combinations.append((filter_name, model_name, model, param_dist, X, y_final))
    
    total_combos = len(combinations)
    
    print("\n" + "="*80)
    print("🚀 STARTING PARALLEL PROCESSING")
    print("="*80)
    print(f"📊 Total combinations: {total_combos}")
    print(f"💻 Parallel workers: {n_jobs}")
    print(f"⏱️  Estimated time: {total_combos * 5 / n_jobs:.1f} - {total_combos * 15 / n_jobs:.1f} minutes")
    print(f"🕐 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # ------------------
    # 3. Run in Parallel
    # ------------------
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(tune_and_evaluate_single_combination)(
            filter_name, model_name, model, param_dist, X, y_final, i+1, total_combos
        )
        for i, (filter_name, model_name, model, param_dist, X, y_final) in enumerate(combinations)
    )
    
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "="*80)
    print("✅ ALL PROCESSING COMPLETE!")
    print("="*80)
    print(f"🕐 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  Total time: {overall_elapsed/60:.1f} minutes ({overall_elapsed:.0f} seconds)")
    
    # ------------------
    # 4. Save Results to Excel
    # ------------------
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    output_file = "/home/user/Documents/Sandhya/AI_in_medicine_lab/AI project/geo_location_all_models_tuned.xlsx"
    
    try:
        results_df.to_excel(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
        
        print("\n" + "="*80)
        print("📊 RESULTS SUMMARY")
        print("="*80)
        
        print("\n📊 By Classifier:")
        summary = results_df.groupby('Classifier')['CV_Accuracy_Mean'].agg(['mean', 'max', 'min', 'std'])
        print(summary)
        
        print("\n📊 By Strategy:")
        summary2 = results_df.groupby('Filtering_Strategy')['CV_Accuracy_Mean'].agg(['mean', 'max', 'min'])
        print(summary2)
        
        print("\n🏆 Best Overall Result:")
        best_idx = results_df['CV_Accuracy_Mean'].idxmax()
        best_result = results_df.loc[best_idx]
        print(f"   Classifier: {best_result['Classifier']}")
        print(f"   Strategy: {best_result['Filtering_Strategy']}")
        print(f"   Accuracy: {best_result['CV_Accuracy_Mean']:.4f} ± {best_result['CV_Accuracy_Std']:.4f}")
        print(f"   Features: {best_result['Features_Count']}")
        print(f"   Time: {best_result['Processing_Time_Seconds']:.1f}s")
        
        print("\n⏱️  Processing Time Statistics:")
        print(f"   Average: {results_df['Processing_Time_Seconds'].mean():.1f}s")
        print(f"   Median: {results_df['Processing_Time_Seconds'].median():.1f}s")
        print(f"   Max: {results_df['Processing_Time_Seconds'].max():.1f}s")
        
    except Exception as e:
        print(f"❌ Could not save to Excel file. Error: {str(e)}")
        print("\nDisplaying results in console instead:")
        print(results_df)
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    # Using 40 cores for optimal balance
    run_tuned_pipeline_for_geographical_location(n_jobs=40)