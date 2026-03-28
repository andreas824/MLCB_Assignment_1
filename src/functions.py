import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils import resample

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform

import optuna
from sklearn.model_selection import cross_val_score

from sklearn.base import clone

from mrmr import mrmr_regression
from mrmr import mrmr_classif

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,  # PR-AUC
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

def load_data(file_path):
    """Load data from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def get_feature_matrix_target_and_features(df, target_column):
    """Prepare the feature matrix X, target variable y, and identify numeric and categorical features."""
    # Separate features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify numeric and categorical columns i.e. features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    return X, y, numeric_features, categorical_features

def build_preprocessor(numeric_features, categorical_features):
    # Define transformers for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

def build_num_preprocessor():
    """Builds a preprocessing pipeline for numerical features only."""
    num_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    return num_preprocessor

def build_cat_preprocessor():
    """Builds a preprocessing pipeline for categorical features only."""
    cat_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    return cat_preprocessor


def split_stratified_data(X,y, n_bins, test_size=0.2, random_state=42):
    """Split the data into training and validation sets using stratified sampling.
    In the regression problem, where age is the target variable (continuous), 
    we can use KBinsDiscretizer to create bins for stratification."""
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    y_binned = discretizer.fit_transform(y.values.reshape(-1, 1)).ravel()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y_binned)
    return X_train, X_val, y_train, y_val


def build_pipeline(preprocessor, model):
    """Build a machine learning pipeline."""
    my_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return my_pipeline


def evaluate_model_bootstraping(y_val, y_pred, n_bootstraps=1000, random_state=42):
    """Evaluate the model using bootstrapping. Calculate RMSE, MAE,
    R-squared and Pearson's correlation coefficient with 95% confidence intervals.
    Also returns the metric scores RMSE,R-squared to create boxplots later."""
    np.random.seed(random_state)
    n_samples = len(y_val)
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    pearson_scores = []

    y_val_array = np.array(y_val)
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_val_sample = y_val_array[indices]
        y_pred_sample = y_pred[indices]

        rmse_scores.append(np.sqrt(mean_squared_error(y_val_sample, y_pred_sample)))
        mae_scores.append(mean_absolute_error(y_val_sample, y_pred_sample))
        r2_scores.append(r2_score(y_val_sample, y_pred_sample))
        pearson_scores.append(pearsonr(y_val_sample, y_pred_sample)[0])

    rmse_mean = np.mean(rmse_scores)
    rmse_ci = np.percentile(rmse_scores, [2.5, 97.5])
    
    mae_mean = np.mean(mae_scores)
    mae_ci = np.percentile(mae_scores, [2.5, 97.5])
    
    r2_mean = np.mean(r2_scores)
    r2_ci = np.percentile(r2_scores, [2.5, 97.5])
    
    pearson_mean = np.mean(pearson_scores)
    pearson_ci = np.percentile(pearson_scores, [2.5, 97.5])

    results = {
        'rmse': (rmse_mean, rmse_ci),
        'mae': (mae_mean, mae_ci),
        'r2': (r2_mean, r2_ci),
        'pearson': (pearson_mean, pearson_ci)
    }

    results_dict = {
        'rmse': rmse_scores,
        'r2': r2_scores,
    }

    return results, results_dict

def print_evaluation_results(model_name, results):
    """Print the evaluation results with confidence intervals."""
    print(f"\n{model_name} Results:")
    for metric, (mean, ci) in results.items():
        print(f"{metric.upper()}: {mean:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")

def get_boxplot(results_dict, metric):
    """Create a boxplot for a specific metric from the results dictionary."""
    data = [results_dict[model][metric] for model in results_dict]
    labels = list(results_dict.keys())
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.title(f'Boxplot of {metric.upper()} Scores')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.savefig(f'../figures/{metric}_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_comparison_boxplot(results_dict, save_path='../figures/comparison_boxplot.png'):
    """Create a side-by-side boxplot comparing RMSE and R² across all models."""
    labels = list(results_dict.keys())
    rmse_data = [results_dict[m]['rmse'] for m in labels]
    r2_data   = [results_dict[m]['r2']   for m in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.boxplot(rmse_data, labels=labels)
    ax1.set_title('RMSE Comparison')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y')

    ax2.boxplot(r2_data, labels=labels)
    ax2.set_title('R² Comparison')
    ax2.set_ylabel('R²')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def stability_selection(X, y, n_subsamples=50, subsample_fraction=0.8, top_k=200, random_state=42):
    """Perform Stability Selection to identify the most important features.
    Draws 50 subsamples of the 80% of the data, without replacement. For each
    subsample ranks the CpG features base on their absolute Spearman correlation
    with the target variable (age) and selects the top 200 features. Finally,
    it returns a list with the stable features (those that appear in more than 25 resamples)."""
    np.random.seed(random_state)
    n_samples = int(len(X) * subsample_fraction)
    feature_counts = {feat: 0 for feat in X.columns}

    for i in range(n_subsamples):
        X_sub, y_sub = resample(X, y, n_samples=n_samples, replace=False, random_state=random_state + i)
        
        spearman_corrs = []
        for feature in X_sub.columns:
            corr, _ = spearmanr(X_sub[feature], y_sub)
            spearman_corrs.append((feature, abs(corr)))
            
        spearman_corrs.sort(key=lambda x: x[1], reverse=True)
        top_features = [feat for feat, _ in spearman_corrs[:top_k]]
        
        for feat in top_features:
            feature_counts[feat] += 1
    
    threshold = n_subsamples / 2
    stable_features = [feat for feat, count in feature_counts.items() if count > threshold]

    return stable_features, feature_counts

def selection_frequency_distribution(feature_counts):
    """Plot and save the distribution of selection frequencies for all features."""
    frequencies = list(feature_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.hist(frequencies, bins=20, edgecolor='k')
    plt.axvline(x=25, color='red', linestyle='--', label='Stability Threshold (>25)')
    plt.title('Distribution of Feature Selection Frequencies')
    plt.xlabel('Selection Frequency')
    plt.ylabel('Number of Features')
    plt.legend()
    plt.grid(axis='y')
    
    plt.savefig('../figures/stability_selection_distribution.png', dpi=300, bbox_inches='tight')
    
    plt.show()


from sklearn.base import clone

def search_best_K(X_train_df, y_train, X_val_df, y_val, model, K_values, 
                  save_path=None, title_suffix=''):
    """Search for the best K in mRMR feature selection based on validation RMSE.
    
    Args:
        X_train_df:   preprocessed training features as DataFrame
        y_train:      training target
        X_val_df:     preprocessed validation features as DataFrame  
        y_val:        validation target
        model:        sklearn model instance (e.g. ElasticNet())
        K_values:     list of K values to search
        save_path:    path to save the plot (optional)
        title_suffix: suffix for the plot title e.g. 'broad' or 'fine-grained'
    
    Returns:
        val_rmse_per_K: dict {K: RMSE}
        best_K:         K with minimum validation RMSE
    """
    val_rmse_per_K = {}

    for K in K_values:
        selected = mrmr_regression(X=X_train_df, y=y_train, K=K)
        
        pipe = Pipeline([
            ('model', clone(model))  # fresh instance for each K
        ])
        pipe.fit(X_train_df[selected], y_train)
        y_pred_val = pipe.predict(X_val_df[selected])
        
        rmse = np.sqrt(mean_squared_error(np.array(y_val), y_pred_val))
        val_rmse_per_K[K] = rmse
        print(f"K={K:4d} → Val RMSE: {rmse:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(list(val_rmse_per_K.keys()), list(val_rmse_per_K.values()),
             marker='o', color='steelblue')
    plt.xlabel('K (number of features)')
    plt.ylabel('Validation RMSE')
    plt.title(f'mRMR: Validation RMSE vs K {title_suffix}')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    best_K = min(val_rmse_per_K, key=val_rmse_per_K.get)
    print(f"\nBest K (min RMSE): {best_K}")

    return val_rmse_per_K, best_K

def build_results_table(models_results: dict, stage: str) -> pd.DataFrame:
    """Build a results table in the required assignment format.
    
    Args:
        models_results: dict with {model_name: results} 
                        where results is the output of evaluate_model_bootstrapping
        stage: e.g. 'Baseline', 'FS-only', 'FS+Tuned'
    
    Returns:
        pd.DataFrame in the format required by the assignment
    """
    rows = []
    for model_name, results in models_results.items():
        rmse_mean, rmse_ci = results['rmse']
        mae_mean,  _       = results['mae']
        r2_mean,   _       = results['r2']
        pearson_mean, _    = results['pearson']
        
        rows.append({
            'Model':       model_name,
            'Stage':       stage,
            'RMSE (mean)': round(rmse_mean, 4),
            '95% CI':      f"[{rmse_ci[0]:.4f}, {rmse_ci[1]:.4f}]",
            'MAE':         round(mae_mean, 4),
            'R²':          round(r2_mean, 4),
            'Pearson r':   round(pearson_mean, 4),
        })
    
    return pd.DataFrame(rows).set_index('Model')




def randomized_search_tune(model_name, model, X_dev, y_dev, n_iter=40, cv=5, random_state=42):
    """Tune hyperparameters using RandomizedSearchCV with 5-fold CV on the full development set.
    
    Args:
        model_name:    'ElasticNet', 'SVR', or 'BayesianRidge'
        model:         sklearn model instance
        X_dev:         full development set features (train + val, already preprocessed)
        y_dev:         development set target
        n_iter:        number of random iterations (default 40)
        cv:            number of cross-validation folds (default 5)
        random_state:  seed for reproducibility (default 42)
    
    Returns:
        best_pipeline: fitted pipeline with best hyperparameters on full development set
        best_params:   best hyperparameter combination found
        cv_results:    full CV results dataframe
    """
    
    # Search spaces per model
    param_grids = {
        'ElasticNet': {
            'model__alpha':    loguniform(0.001, 10),
            'model__l1_ratio': uniform(0.1, 0.9),   # uniform(loc, scale) → [0.1, 1.0]
        },
        'SVR': {
            'model__C':       loguniform(0.1, 500),
            'model__epsilon': [0.01, 0.1, 0.5, 1.0],
            'model__kernel':  ['rbf', 'linear'],
        },
        'BayesianRidge': {
            'model__alpha_1':  loguniform(1e-7, 1e-3),
            'model__alpha_2':  loguniform(1e-7, 1e-3),
            'model__lambda_1': loguniform(1e-7, 1e-3),
            'model__lambda_2': loguniform(1e-7, 1e-3),
        }
    }
    
    if model_name not in param_grids:
        raise ValueError(f"model_name must be one of {list(param_grids.keys())}")
    
    # Build pipeline with preprocessor + model
    pipeline = Pipeline([
        ('preprocessor', build_num_preprocessor()),
        ('model', model)
    ])
    
    # RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grids[model_name],
        n_iter=n_iter,
        cv=cv,
        scoring='neg_root_mean_squared_error',
        random_state=random_state,
        refit=True,    # refit best model on full development set automatically
        n_jobs=-1,     # use all available CPU cores
        verbose=1
    )
    
    search.fit(X_dev, y_dev)
    
    best_params = search.best_params_
    cv_results = pd.DataFrame(search.cv_results_)
    
    print(f"\n{model_name} — Best parameters: {best_params}")
    print(f"{model_name} — Best CV RMSE: {-search.best_score_:.4f}")
    
    return search.best_estimator_, best_params, cv_results



def evaluate_model_bootstrapping_eval_set(y_true, y_pred, n_bootstraps=1000, random_state=42):
    """Evaluate the model on the evaluation set using bootstrapping.
    Same as evaluate_model_bootstrapping but also returns std for each metric,
    as required by Task 4.2.
    
    Args:
        y_true:       true labels (evaluation set)
        y_pred:       model predictions
        n_bootstraps: number of bootstrap resamples (default 1000)
        random_state: seed (default 42)
    
    Returns:
        results:      dict with {metric: (mean, std, ci)} for each metric
        results_dict: dict with {metric: [scores]} for boxplots
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    rmse_scores, mae_scores, r2_scores, pearson_scores = [], [], [], []

    y_true_arr = np.array(y_true)

    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_sample = y_true_arr[indices]
        y_pred_sample = y_pred[indices]

        rmse_scores.append(np.sqrt(mean_squared_error(y_true_sample, y_pred_sample)))
        mae_scores.append(mean_absolute_error(y_true_sample, y_pred_sample))
        r2_scores.append(r2_score(y_true_sample, y_pred_sample))
        pearson_scores.append(pearsonr(y_true_sample, y_pred_sample)[0])

    results = {
        'rmse':    (np.mean(rmse_scores),    np.std(rmse_scores),    np.percentile(rmse_scores,    [2.5, 97.5])),
        'mae':     (np.mean(mae_scores),     np.std(mae_scores),     np.percentile(mae_scores,     [2.5, 97.5])),
        'r2':      (np.mean(r2_scores),      np.std(r2_scores),      np.percentile(r2_scores,      [2.5, 97.5])),
        'pearson': (np.mean(pearson_scores), np.std(pearson_scores), np.percentile(pearson_scores, [2.5, 97.5])),
    }

    results_dict = {
        'rmse':    rmse_scores,
        'r2':      r2_scores,
        'mae':     mae_scores,
        'pearson': pearson_scores,
    }

    return results, results_dict



def build_results_table_eval(models_results: dict, stage: str) -> pd.DataFrame:
    """Build a results table in the required assignment format for Task 4.2.
    Uses the output of evaluate_model_bootstrapping_eval which includes std.
    
    Args:
        models_results: dict with {model_name: results} 
                        where results is the output of evaluate_model_bootstrapping_eval
        stage: e.g. 'Baseline', 'FS-only', 'FS+Tuned'
    
    Returns:
        pd.DataFrame in the format required by Task 4.2
    """
    rows = []
    for model_name, results in models_results.items():
        rmse_mean, rmse_std, rmse_ci = results['rmse']
        mae_mean,  mae_std,  mae_ci  = results['mae']
        r2_mean,   r2_std,   r2_ci   = results['r2']
        pearson_mean, pearson_std, _ = results['pearson']
        
        rows.append({
            'Model':       model_name,
            'Stage':       stage,
            'RMSE (mean)': round(rmse_mean, 4),
            'RMSE (std)':  round(rmse_std, 4),
            '95% CI':      f"[{rmse_ci[0]:.4f}, {rmse_ci[1]:.4f}]",
            'MAE':         round(mae_mean, 4),
            'R²':          round(r2_mean, 4),
            'Pearson r':   round(pearson_mean, 4),
        })
    
    return pd.DataFrame(rows).set_index('Model')



def optuna_tune_model(model_name, pipeline, X_train, y_train, n_trials=40, cv=5, random_state=42):
    """Tune hyperparameters using Optuna (TPE Bayesian optimization).
    
    Args:
        model_name:  'ElasticNet', 'SVR', or 'BayesianRidge'
        pipeline:    sklearn Pipeline with preprocessor + model 
        X_train:     training features (full development set, already preprocessed)
        y_train:     training target
        n_trials:    number of Optuna trials (default 40)
        cv:          number of CV folds (default 5)
        random_state: seed (default 42)
    
    Returns:
        best_pipeline: fitted pipeline with best hyperparameters
        study:         Optuna study object (for optimization history plot)
    """
    
    def objective(trial):
        # Search spaces 
        if model_name == 'ElasticNet':
            params = {
                'model__alpha':    trial.suggest_float('alpha', 0.001, 10, log=True),
                'model__l1_ratio': trial.suggest_float('l1_ratio', 0.1, 1.0),
            }
        elif model_name == 'SVR':
            params = {
                'model__C':       trial.suggest_float('C', 0.1, 500, log=True),
                'model__epsilon': trial.suggest_categorical('epsilon', [0.01, 0.1, 0.5, 1.0]),
                'model__kernel':  trial.suggest_categorical('kernel', ['rbf', 'linear']),
            }
        elif model_name == 'BayesianRidge':
            params = {
                'model__alpha_1':  trial.suggest_float('alpha_1',  1e-7, 1e-3, log=True),
                'model__alpha_2':  trial.suggest_float('alpha_2',  1e-7, 1e-3, log=True),
                'model__lambda_1': trial.suggest_float('lambda_1', 1e-7, 1e-3, log=True),
                'model__lambda_2': trial.suggest_float('lambda_2', 1e-7, 1e-3, log=True),
            }
        else:
            raise ValueError(f"model_name must be one of ['ElasticNet', 'SVR', 'BayesianRidge']")
        
        trial_pipeline = clone(pipeline)  # create a copy of the original pipeline
        trial_pipeline.set_params(**params)        # set the hyperparameters for this trial
        
        # 5-fold CV, objective: minimize RMSE
        scores = cross_val_score(
            trial_pipeline, X_train, y_train,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        return -scores.mean()  # Optuna minimizes, so we return positive RMSE
    
    # create study — TPE sampler by default
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    # Suppress Optuna logs for cleaner output
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study.optimize(objective, n_trials=n_trials)
    
    # Refit with best params on the full development set
    best_params = {k: v for k, v in study.best_params.items()}
    best_params_pipeline = {f'model__{k}': v for k, v in best_params.items()}
    pipeline.set_params(**best_params_pipeline)
    pipeline.fit(X_train, y_train)
    
    print(f"\n{model_name} — Best Optuna params: {study.best_params}")
    print(f"{model_name} — Best Optuna CV RMSE: {study.best_value:.4f}")
    
    return pipeline, study

def plot_optuna_history(study, model_name, save_path=None):
    """Plot Optuna optimization history (trial RMSE vs trial number).
    
    Args:
        study:      Optuna study object (returned by optuna_tune_model)
        model_name: plot title
        save_path:  path for saving (e.g., '../figures/optuna_history.png')
    """
    trials = study.trials
    trial_numbers = [t.number for t in trials]
    trial_values  = [t.value  for t in trials]
    
    # Running best to plot the best RMSE so far at each trial
    running_best = []
    current_best = float('inf')
    for v in trial_values:
        current_best = min(current_best, v)
        running_best.append(current_best)
    
    plt.figure(figsize=(10, 5))
    plt.scatter(trial_numbers, trial_values, alpha=0.5, label='Trial RMSE', color='steelblue')
    plt.plot(trial_numbers, running_best, color='red', linewidth=2, label='Best so far')
    plt.xlabel('Trial number')
    plt.ylabel('CV RMSE')
    plt.title(f'Optuna Optimization History — {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()



def classification_model_evaluate_bootstrapping(y_true, y_pred, y_prob, n_bootstraps=1000, random_state=42):
    """Evaluate classification model using bootstrapping. Calculate Accuracy, F1-score, MCC, 
    ROC-AUC, and PR-AUC with mean, std and 95% confidence intervals.
    Also returns the metric scores for boxplots later.
    
    Args:
        y_true:       true binary labels (0/1)
        y_pred:       predicted binary labels (0/1)
        y_prob:       predicted probabilities for class 1
        n_bootstraps: number of bootstrap resamples (default 1000)
        random_state: seed (default 42)
    
    Returns:
        results:      dict with {metric: (mean, std, ci)}
        results_dict: dict with {metric: [scores]} for boxplots
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    accuracy_scores = []
    f1_scores = []
    mcc_scores = []
    roc_auc_scores = []
    pr_auc_scores = []

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_prob_arr = np.array(y_prob)
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_sample = y_true_arr[indices]
        y_pred_sample = y_pred_arr[indices]
        y_prob_sample = y_prob_arr[indices]

        # skip if only one class in resample — metrics undefined
        if len(np.unique(y_true_sample)) < 2:
            continue

        accuracy_scores.append(accuracy_score(y_true_sample, y_pred_sample))
        f1_scores.append(f1_score(y_true_sample, y_pred_sample, zero_division=0))
        mcc_scores.append(matthews_corrcoef(y_true_sample, y_pred_sample))
        roc_auc_scores.append(roc_auc_score(y_true_sample, y_prob_sample))
        pr_auc_scores.append(average_precision_score(y_true_sample, y_prob_sample))

    results = {
        'accuracy': (np.mean(accuracy_scores), np.std(accuracy_scores), np.percentile(accuracy_scores, [2.5, 97.5])),
        'f1_score': (np.mean(f1_scores),       np.std(f1_scores),       np.percentile(f1_scores,       [2.5, 97.5])),
        'mcc':      (np.mean(mcc_scores),       np.std(mcc_scores),      np.percentile(mcc_scores,      [2.5, 97.5])),
        'roc_auc':  (np.mean(roc_auc_scores),  np.std(roc_auc_scores),  np.percentile(roc_auc_scores,  [2.5, 97.5])),
        'pr_auc':   (np.mean(pr_auc_scores),   np.std(pr_auc_scores),   np.percentile(pr_auc_scores,   [2.5, 97.5])),
    }

    results_dict = {
        'accuracy': accuracy_scores,
        'f1_score': f1_scores,
        'mcc':      mcc_scores,
        'roc_auc':  roc_auc_scores,
        'pr_auc':   pr_auc_scores,
    }

    return results, results_dict


def build_classification_results_table(models_results: dict) -> pd.DataFrame:
    """Build a results table in the required assignment format for Bonus B.
    
    Args:
        models_results: dict with {model_name: results}
                        where results is the output of classification_model_evaluate_bootstrapping
    
    Returns:
        pd.DataFrame in the format required by Bonus B
    """
    rows = []
    for model_name, results in models_results.items():
        acc_mean,     acc_std,     acc_ci     = results['accuracy']
        f1_mean,      f1_std,      f1_ci      = results['f1_score']
        mcc_mean,     mcc_std,     mcc_ci     = results['mcc']
        roc_auc_mean, roc_auc_std, roc_auc_ci = results['roc_auc']
        pr_auc_mean,  pr_auc_std,  pr_auc_ci  = results['pr_auc']

        rows.append({
            'Model':              model_name,
            'Accuracy (mean)':    round(acc_mean, 4),
            'Accuracy (95% CI)':  f"[{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]",
            'F1 (mean)':          round(f1_mean, 4),
            'F1 (95% CI)':        f"[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]",
            'MCC (mean)':         round(mcc_mean, 4),
            'MCC (95% CI)':       f"[{mcc_ci[0]:.4f}, {mcc_ci[1]:.4f}]",
            'ROC-AUC (mean)':     round(roc_auc_mean, 4),
            'ROC-AUC (95% CI)':   f"[{roc_auc_ci[0]:.4f}, {roc_auc_ci[1]:.4f}]",
            'PR-AUC (mean)':      round(pr_auc_mean, 4),
            'PR-AUC (95% CI)':    f"[{pr_auc_ci[0]:.4f}, {pr_auc_ci[1]:.4f}]",
        })

    return pd.DataFrame(rows).set_index('Model')



def search_best_K_classif(X_train_df, y_train, X_val_df, y_val, model, K_values,
                           save_path=None, title_suffix=''):
    """Search for best K in mRMR for classification based on validation F1 score."""

    val_f1_per_K = {}

    for K in K_values:
        selected = mrmr_classif(X=X_train_df, y=y_train, K=K)

        pipe = Pipeline([('model', clone(model))])
        pipe.fit(X_train_df[selected], y_train)
        y_pred_val = pipe.predict(X_val_df[selected])

        f1 = f1_score(y_val, y_pred_val)   # F1 robust choice
        val_f1_per_K[K] = f1
        print(f"K={K:4d} → Val F1: {f1:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(list(val_f1_per_K.keys()), list(val_f1_per_K.values()),
             marker='o', color='steelblue')
    plt.xlabel('K (number of features)')
    plt.ylabel('Validation F1')
    plt.title(f'mRMR: Validation F1 vs K {title_suffix}')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    best_K = max(val_f1_per_K, key=val_f1_per_K.get)  # maximize F1
    print(f"\nBest K (max F1): {best_K}")

    return val_f1_per_K, best_K