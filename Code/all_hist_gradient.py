import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

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


def evaluate_single_combination(target_name, filter_name, X, y_final):
    """
    Evaluate a single target-filter combination.
    This function is designed to be run in parallel.
    """
    if X is None or X.shape[1] == 0:
        return {
            'Target_Feature': target_name,
            'Filtering_Strategy': filter_name,
            'Classifier': 'HistGradientBoost',
            'CV_Accuracy_Mean': np.nan,
            'CV_Accuracy_Std': np.nan,
            'CV_F1_Mean': np.nan,
            'CV_AUC_Mean': np.nan,
            'Features_Count': 0,
            'Samples_Count': 0
        }
    
    try:
        model = HistGradientBoostingClassifier(random_state=42)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Cross-validation for accuracy
        cv_scores = cross_val_score(model, X, y_final, cv=cv, scoring='accuracy', n_jobs=1)
        
        # F1 and AUC for binary classification
        is_binary = y_final.nunique() == 2
        if is_binary and hasattr(model, 'predict_proba'):
            cv_roc_auc = cross_val_score(model, X, y_final, cv=cv, scoring='roc_auc', n_jobs=1)
        else:
            cv_roc_auc = [np.nan] * 5
        
        if is_binary:
            cv_f1 = cross_val_score(model, X, y_final, cv=cv, scoring='f1', n_jobs=1)
        else:
            cv_f1 = [np.nan] * 5
        
        return {
            'Target_Feature': target_name,
            'Filtering_Strategy': filter_name,
            'Classifier': 'HistGradientBoost',
            'CV_Accuracy_Mean': np.mean(cv_scores),
            'CV_Accuracy_Std': np.std(cv_scores),
            'CV_F1_Mean': np.mean(cv_f1),
            'CV_AUC_Mean': np.mean(cv_roc_auc),
            'Features_Count': X.shape[1],
            'Samples_Count': X.shape[0]
        }
    
    except Exception as e:
        print(f"      ❌ Error with HistGradientBoost on {target_name} ({filter_name}): {str(e)}")
        return {
            'Target_Feature': target_name,
            'Filtering_Strategy': filter_name,
            'Classifier': 'HistGradientBoost',
            'CV_Accuracy_Mean': np.nan,
            'CV_Accuracy_Std': np.nan,
            'CV_F1_Mean': np.nan,
            'CV_AUC_Mean': np.nan,
            'Features_Count': X.shape[1] if X is not None else 0,
            'Samples_Count': X.shape[0] if X is not None else 0
        }


def run_classification_pipeline(n_jobs=-1):
    """
    Main function to run the classification pipeline for each metadata feature.
    
    Parameters:
    -----------
    n_jobs : int, default=-1
        Number of parallel jobs. -1 uses all available cores.
        Recommended: Use n_jobs=-2 to leave one core free for system.
    """
    
    # ------------------
    # 1. Load Data
    # ------------------
    print("📂 Loading data...")
    try:
        otu_raw = pd.read_excel("/home/user/Documents/Sandhya/AI_in_medicine_lab/AI project/abundance_only.xlsx", index_col=0)
        meta_raw = pd.read_excel("/home/user/Documents/Sandhya/AI_in_medicine_lab/AI project/metadata_only.xlsx")
    except FileNotFoundError:
        print("Error: Make sure your data files are in the specified path.")
        return
    
    # Preprocessing metadata
    meta = meta_raw.set_index(meta_raw.columns[0])
    
    # Define filtering and transformation strategies
    filter_strategies = {
        'No_Filtering': lambda df: df,
        'Abundance_0.1%': lambda df: df.loc[:, (df > 0.001).any(axis=0)],
        'Prevalence_5%': lambda df: df.loc[:, (df > 0).mean(axis=0) >= 0.05],
        'Both_Filters': lambda df: df.loc[:, ((df > 0.001).any(axis=0) & ((df > 0).mean(axis=0) >= 0.05))],
        'CLR_Transform': centered_log_ratio,
        'ILR_Transform': isometric_log_ratio
    }
    
    # Metadata targets
    metadata_targets = {
        'Geographical_Location': 'Geographical Location',
        'Geographical_Zone_in_India': 'Geographical zone in India',
        'Gender': 'Gender',
        'Age_in_years': 'Age in years',
        'Life_style_pattern': 'Life style pattern'
    }

    # ------------------
    # 2. Prepare All Combinations
    # ------------------
    print("🔧 Preparing all target-strategy combinations...")
    combinations = []
    
    for target_name, target_col in metadata_targets.items():
        if target_col not in meta.columns:
            print(f"   ⚠️ Column '{target_col}' not found in metadata. Skipping.")
            continue
        
        y = meta[target_col]
        y = y.dropna()
        
        if y.dtype == 'object' or y.nunique() > 2:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y

        common_samples = list(set(otu_raw.index) & set(y.index))
        otu_aligned = otu_raw.loc[common_samples]
        y_series = pd.Series(y_encoded, index=y.index)
        y_final = y_series.loc[common_samples]

        if otu_aligned.shape[0] < 20 or y_final.nunique() <= 1:
            print(f"   ⚠️ Not enough samples or classes for target: {target_name}. Skipping.")
            continue
        
        print(f"   📊 {target_name}: {otu_aligned.shape[0]} samples, {otu_aligned.shape[1]} features")
        
        # Apply all filtering strategies for this target
        for filter_name, filter_func in filter_strategies.items():
            X = apply_filter_strategy(otu_aligned, filter_name, filter_func)
            combinations.append((target_name, filter_name, X, y_final))
    
    print(f"\n🚀 Total combinations to process: {len(combinations)}")
    print(f"💻 Running in parallel with {n_jobs if n_jobs > 0 else 'all'} CPU cores...")
    
    # ------------------
    # 3. Run in Parallel
    # ------------------
    # Using backend='loky' for better memory management with many cores
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10, batch_size='auto')(
        delayed(evaluate_single_combination)(target_name, filter_name, X, y_final)
        for target_name, filter_name, X, y_final in combinations
    )
    
    # ------------------
    # 4. Save Results to Excel
    # ------------------
    results_df = pd.DataFrame(results)
    output_file = "/home/user/Documents/Sandhya/AI_in_medicine_lab/AI project/histgradient_only_results.xlsx"
    
    try:
        results_df.to_excel(output_file, index=False)
        print(f"\n✅ All results saved to: {output_file}")
        print(f"\n📊 Summary:")
        print(results_df.groupby('Target_Feature')['CV_Accuracy_Mean'].agg(['mean', 'max', 'min']))
    except Exception as e:
        print(f"\n❌ Could not save to Excel file. Error: {str(e)}")
        print("\nDisplaying results in console instead:")
        print(results_df)

if __name__ == '__main__':
    # Using 40 cores for optimal balance of speed and system resources
    run_classification_pipeline(n_jobs=40)