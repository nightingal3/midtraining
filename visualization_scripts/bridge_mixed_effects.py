#!/usr/bin/env python3
"""
Mixed Effects Analysis for Midtraining Bridge Effects

This script uses mixed effects models to properly control for task difficulty (C4→SFT gap)
while testing bridge hypotheses and domain matching effects.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, linregress
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import argparse
import re
import json
from collections import Counter

# Try to import statsmodels for mixed effects
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("WARNING: statsmodels not available. Install with: pip install statsmodels")
    print("Falling back to manual controls using sklearn...")
    STATSMODELS_AVAILABLE = False
    from sklearn.linear_model import LinearRegression


def load_similarity_matrix(filepath):
    """Load similarity matrix from CSV file."""
    print(f"Loading similarity matrix from {filepath}")
    
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, index_col=0, encoding=encoding)
                print(f"  Successfully loaded with {encoding} encoding")
                print(f"  Matrix shape: {df.shape}")
                print(f"  Datasets: {list(df.index)}")
                return df
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode {filepath} with any encoding")
        
    except Exception as e:
        print(f"Error loading similarity matrix: {e}")
        return None


def map_dataset_names(dataset_name):
    """Map dataset names from CSV to similarity matrix names."""
    if not isinstance(dataset_name, str):
        dataset_name = str(dataset_name)
    
    # Clean the name and remove percentages/parentheses
    clean_name = dataset_name.strip()
    clean_name = re.sub(r'\s*\(\d+%?\)', '', clean_name)
    clean_name = clean_name.strip()
    
    # EXCLUDE MOVIE REVIEWS - faulty entries
    if 'Movie Reviews' in clean_name or 'Movie Review' in clean_name:
        print(f"  EXCLUDING Movie Reviews from analysis: '{dataset_name}'")
        return None
    
    # EXCLUDE FLAN 15% - problematic entries
    if 'FLAN 15%' in dataset_name or 'FLAN (15%)' in dataset_name:
        print(f"  EXCLUDING FLAN 15% from analysis: '{dataset_name}'")
        return None
    
    # SPECIAL CASE: StarCoder (100%) should use StarCoder-cts (continuous pretraining version)
    if 'Starcoder (100%)' in dataset_name or 'StarCoder (100%)' in dataset_name:
        print(f"  Using StarCoder-cts for continuous pretraining: '{dataset_name}' → 'StarCoder-cts'")
        return 'StarCoder-cts'
    
    # Direct mapping for exact matches
    name_mapping = {
        'Pycode': 'PyCode',
        'PyCode': 'PyCode', 
        'Starcoder': 'StarCoder',
        'StarCoder': 'StarCoder',
        'Math': 'Math Combined',
        'Math Combined': 'Math Combined',
        'FLAN': 'FLAN Combined',
        'FLAN Combined': 'FLAN Combined',
        'C4': 'C4',
        'GSM8K': 'GSM8K',
        'GSM8k': 'GSM8K',
        'Social IQA': 'Social IQA',
        'LIMA': 'LIMA',
        'SciQ': 'SciQ',
        'TREC': 'TREC',
        'DCLM': 'DCLM',
        'KnowledgeQA': 'KnowledgeQA'
    }
    
    mapped_name = name_mapping.get(clean_name, clean_name)
    
    if clean_name != dataset_name:
        print(f"  Name mapping: '{dataset_name}' → '{clean_name}' → '{mapped_name}'")
    
    return mapped_name


def get_similarity(dataset1, dataset2, similarity_df):
    """Get similarity between two datasets."""
    mapped_ds1 = map_dataset_names(dataset1)
    mapped_ds2 = map_dataset_names(dataset2)
    
    # Handle excluded datasets
    if mapped_ds1 is None or mapped_ds2 is None:
        print(f"  EXCLUDED: Cannot compute similarity for {dataset1} → {dataset2} (one or both datasets excluded)")
        return None
    
    try:
        # DEBUG: Check if both directions exist and what values they have
        print(f"  DEBUG: Looking up similarity between '{mapped_ds1}' and '{mapped_ds2}'")
        print(f"  DEBUG: Matrix shape: {similarity_df.shape}")
        print(f"  DEBUG: '{mapped_ds1}' in index: {mapped_ds1 in similarity_df.index}")
        print(f"  DEBUG: '{mapped_ds2}' in columns: {mapped_ds2 in similarity_df.columns}")
        
        if mapped_ds1 in similarity_df.index and mapped_ds2 in similarity_df.columns:
            similarity = similarity_df.loc[mapped_ds1, mapped_ds2]
            print(f"  DEBUG: Found similarity value: {similarity}")
            
            if pd.isna(similarity):
                print(f"  DEBUG: Similarity is NaN, trying reverse lookup")
                similarity = similarity_df.loc[mapped_ds2, mapped_ds1]
                print(f"  DEBUG: Reverse similarity value: {similarity}")
            
            return similarity
        else:
            print(f"  DEBUG: Trying reverse lookup ({mapped_ds2} → {mapped_ds1})")
            similarity = similarity_df.loc[mapped_ds2, mapped_ds1]
            return similarity
            
    except KeyError as e:
        print(f"  ERROR: Could not find similarity for {dataset1} ({mapped_ds1}) → {dataset2} ({mapped_ds2})")
        print(f"  Available datasets: {list(similarity_df.index)}")
        print(f"  KeyError details: {e}")
        return None
    except Exception as e:
        print(f"  ERROR: Unexpected error in similarity lookup: {e}")
        return None


def classify_bridge_type(midtrain_dataset, sft_dataset):
    """Classify bridge into domain matching categories."""
    midtrain_clean = map_dataset_names(midtrain_dataset)
    sft_clean = map_dataset_names(sft_dataset)
    
    perfect_matches = [
        ('StarCoder', 'PyCode'),
        ('Math Combined', 'GSM8K'),
    ]
    
    partial_matches = [
        ('Math Combined', 'SciQ'),
        ('FLAN Combined', 'LIMA'),
        ('FLAN Combined', 'SciQ'),
        ('FLAN Combined', 'TREC'),
    ]
    
    if (midtrain_clean, sft_clean) in perfect_matches:
        return 'perfect_match'
    elif (midtrain_clean, sft_clean) in partial_matches:
        return 'partial_match'
    else:
        return 'cross_domain'


def load_and_process_experimental_data(csv_file_path, similarity_df, exclude_sciq=False, remove_outliers=False):
    """Load experimental results and compute bridge metrics, optionally excluding SciQ."""
    print(f"Loading experimental results from {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()
    
    # Exclude SciQ if requested
    if exclude_sciq:
        mask_mid = df['Pre/midtrain mix'].str.contains('SciQ', na=False)
        mask_sft = df['SFT dataset'].str.contains('SciQ', na=False)
        df = df[~(mask_mid | mask_sft)]
        print("  EXCLUDING all SciQ experiments from dataset")

    # Forward fill model size and SFT dataset
    df['Model size'] = df['Model size'].fillna(method='ffill')
    df['SFT dataset'] = df['SFT dataset'].fillna(method='ffill')
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Find C4 baseline
    c4_baseline = df[df['Pre/midtrain mix'] == 'C4'].copy()
    if exclude_sciq:
        c4_baseline = c4_baseline[~c4_baseline['SFT dataset'].str.contains('SciQ', na=False)]
    
    if len(c4_baseline) == 0:
        print("ERROR: No C4 baseline found!")
        return None
    
    print(f"Found {len(c4_baseline)} C4 baseline entries")
    
    # Create baseline lookup
    baseline_lookup = {}
    for _, row in c4_baseline.iterrows():
        key = (row['Model size'], row['SFT dataset'])
        baseline_lookup[key] = row['SFT val loss after FT']
    
    results = []
    skipped_experiments = []
    
    for _, row in df.iterrows():
        if row['Pre/midtrain mix'] == 'C4':
            continue
        if exclude_sciq and ('SciQ' in str(row['Pre/midtrain mix']) or 'SciQ' in str(row['SFT dataset'])):
            skipped_experiments.append(f"{row['Pre/midtrain mix']} → {row['SFT dataset']} (SciQ excluded)")
            print(f"  SKIPPING SciQ experiment: {row['Pre/midtrain mix']} → {row['SFT dataset']}")
            continue
        if pd.isna(row['SFT val loss after FT']) or pd.isna(row['Pre/midtrain mix']):
            continue
        if 'Movie Review' in str(row['SFT dataset']):
            skipped_experiments.append(f"{row['Pre/midtrain mix']} → {row['SFT dataset']} (Movie Reviews excluded)")
            print(f"  SKIPPING Movie Reviews experiment: {row['Pre/midtrain mix']} → {row['SFT dataset']}")
            continue
        if ('FLAN 15%' in str(row['Pre/midtrain mix']) or 'FLAN (15%)' in str(row['Pre/midtrain mix']) or 'FLAN 15%' in str(row['SFT dataset']) or 'FLAN (15%)' in str(row['SFT dataset'])):
            skipped_experiments.append(f"{row['Pre/midtrain mix']} → {row['SFT dataset']} (FLAN 15% excluded)")
            print(f"  SKIPPING FLAN 15% experiment: {row['Pre/midtrain mix']} → {row['SFT dataset']}")
            continue
        key = (row['Model size'], row['SFT dataset'])
        if key not in baseline_lookup:
            print(f"WARNING: No baseline found for {key}")
            continue
        baseline_loss = baseline_lookup[key]
        midtrain_loss = row['SFT val loss after FT']
        absolute_improvement = baseline_loss - midtrain_loss
        relative_improvement = absolute_improvement / baseline_loss
        
        c4_to_midtrain_sim = get_similarity('C4', row['Pre/midtrain mix'], similarity_df)
        midtrain_to_sft_sim = get_similarity(row['Pre/midtrain mix'], row['SFT dataset'], similarity_df)
        c4_to_sft_sim = get_similarity('C4', row['SFT dataset'], similarity_df)
        
        if c4_to_midtrain_sim is None or midtrain_to_sft_sim is None or c4_to_sft_sim is None:
            skipped_experiments.append(f"{row['Pre/midtrain mix']} → {row['SFT dataset']} (similarity calculation failed)")
            continue
        
        c4_to_midtrain_dist = 1 - c4_to_midtrain_sim
        midtrain_to_sft_dist = 1 - midtrain_to_sft_sim
        c4_to_sft_dist = 1 - c4_to_sft_sim
        bridge_length_total = c4_to_midtrain_dist + midtrain_to_sft_dist
        bridge_length_max = max(c4_to_midtrain_dist, midtrain_to_sft_dist)
        bridge_type = classify_bridge_type(row['Pre/midtrain mix'], row['SFT dataset'])
        
        results.append({
            'model_size': row['Model size'],
            'sft_dataset': row['SFT dataset'],
            'midtrain_dataset': row['Pre/midtrain mix'],
            'baseline_loss': baseline_loss,
            'midtrain_loss': midtrain_loss,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement,
            'c4_to_midtrain_dist': c4_to_midtrain_dist,
            'midtrain_to_sft_dist': midtrain_to_sft_dist,
            'c4_to_sft_dist': c4_to_sft_dist,
            'bridge_length_total': bridge_length_total,
            'bridge_length_max': bridge_length_max,
            'bridge_type': bridge_type,
            'c4_loss_after_ft': row.get('C4 val loss after FT', np.nan)
        })
    
    analysis_df = pd.DataFrame(results)
    
    print(f"Processed {len(analysis_df)} valid experiments")
    if skipped_experiments:
        print(f"Skipped {len(skipped_experiments)} experiments:")
        for exp in skipped_experiments[:10]:
            print(f"  {exp}")
        if len(skipped_experiments) > 10:
            print(f"  ... and {len(skipped_experiments) - 10} more")
    
    # OUTLIER DETECTION AND HANDLING
    print(f"\n" + "="*40)
    print("OUTLIER DETECTION")
    print("="*40)
    
    # Check for extreme outliers in relative improvement
    improvement_values = analysis_df['relative_improvement']
    q1 = improvement_values.quantile(0.25)
    q3 = improvement_values.quantile(0.75)
    iqr = q3 - q1
    
    # Define outliers as beyond 3*IQR (very conservative)
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr
    
    outliers = analysis_df[
        (analysis_df['relative_improvement'] < lower_bound) | 
        (analysis_df['relative_improvement'] > upper_bound)
    ]
    
    print(f"Improvement statistics before outlier removal:")
    print(f"  Mean: {improvement_values.mean():+.1%}")
    print(f"  Median: {improvement_values.median():+.1%}")
    print(f"  Std: {improvement_values.std():.1%}")
    print(f"  Range: {improvement_values.min():+.1%} to {improvement_values.max():+.1%}")
    print(f"  Q1: {q1:+.1%}, Q3: {q3:+.1%}, IQR: {iqr:.1%}")
    print(f"  Outlier bounds: {lower_bound:+.1%} to {upper_bound:+.1%}")
    
    if len(outliers) > 0 and remove_outliers:
        print(f"\nFound {len(outliers)} extreme outliers:")
        for _, outlier in outliers.iterrows():
            print(f"  {outlier['midtrain_dataset']:15} → {outlier['sft_dataset']:10} ({outlier['model_size']}): {outlier['relative_improvement']:+.1%}")
            print(f"    Baseline: {outlier['baseline_loss']:.4f}, Midtrain: {outlier['midtrain_loss']:.4f}")
        
        # Ask user what to do with outliers
        print(f"\nOptions for handling outliers:")
        print(f"  1. Remove outliers (recommended for clean analysis)")
        print(f"  2. Cap outliers at bounds (winsorize)")
        print(f"  3. Keep outliers (may skew results)")
        
        # For automated analysis, remove extreme outliers
        print(f"\nAUTOMATIC DECISION: Removing extreme outliers for clean analysis")
        analysis_df_clean = analysis_df[
            (analysis_df['relative_improvement'] >= lower_bound) & 
            (analysis_df['relative_improvement'] <= upper_bound)
        ]
        
        print(f"Removed {len(outliers)} outliers, {len(analysis_df_clean)} experiments remaining")
        
        # Show cleaned statistics
        clean_improvements = analysis_df_clean['relative_improvement']
        print(f"\nImprovement statistics after outlier removal:")
        print(f"  Mean: {clean_improvements.mean():+.1%}")
        print(f"  Median: {clean_improvements.median():+.1%}")
        print(f"  Std: {clean_improvements.std():.1%}")
        print(f"  Range: {clean_improvements.min():+.1%} to {clean_improvements.max():+.1%}")
        
        # Use cleaned data
        analysis_df = analysis_df_clean
        
    else:
        print("No extreme outliers detected")
    
    return analysis_df


def run_mixed_effects_models(analysis_df):
    """Run mixed effects models to test bridge hypotheses."""
    print(f"\n" + "="*60)
    print("MIXED EFFECTS MODEL ANALYSIS")
    print("="*60)
    print("Testing bridge hypotheses while controlling for task difficulty...")
    
    if not STATSMODELS_AVAILABLE:
        return run_manual_controls(analysis_df)
    
    # Prepare data
    df = analysis_df.copy()
    print(f"Analyzing {len(df)} experiments across {df['sft_dataset'].nunique()} SFT datasets")
    
    # Create dummy variables for modeling
    df['perfect_match'] = (df['bridge_type'] == 'perfect_match').astype(int)
    df['partial_match'] = (df['bridge_type'] == 'partial_match').astype(int)
    
    results = {}
    
    # Model 1: Simple bridge length effect
    print(f"\n" + "="*40)
    print("MODEL 1: Bridge Length Effect")
    print("="*40)
    print("Formula: improvement ~ bridge_length_max + (1|sft_dataset)")
    print("Tests: Does bridge length matter after controlling for task difficulty?")
    
    try:
        model1 = mixedlm("relative_improvement ~ bridge_length_max", 
                        df, groups=df["sft_dataset"]).fit()
        
        print(f"\nFixed Effects:")
        print(f"  Intercept: {model1.params['Intercept']:.4f} (p = {model1.pvalues['Intercept']:.4f})")
        print(f"  Bridge Length: {model1.params['bridge_length_max']:.4f} (p = {model1.pvalues['bridge_length_max']:.4f})")
        
        bridge_coef = model1.params['bridge_length_max']
        bridge_p = model1.pvalues['bridge_length_max']
        
        if bridge_p < 0.05:
            direction = "POSITIVE" if bridge_coef > 0 else "NEGATIVE"
            print(f"\n✓ SIGNIFICANT {direction} bridge effect!")
            print(f"  → 0.1 increase in bridge length → {bridge_coef*0.1:+.1%} improvement change")
        else:
            print(f"\n✗ No significant bridge effect (p = {bridge_p:.4f})")
        
        results['model1'] = {
            'bridge_coef': bridge_coef,
            'bridge_p': bridge_p,
            'aic': model1.aic,
            'significant': bridge_p < 0.05
        }
        
    except Exception as e:
        print(f"Error fitting Model 1: {e}")
        results['model1'] = None
    
    # Model 2: Bridge components
    print(f"\n" + "="*40)
    print("MODEL 2: Bridge Components")
    print("="*40)
    print("Formula: improvement ~ c4_to_midtrain_dist + midtrain_to_sft_dist + (1|sft_dataset)")
    print("Tests: Which part of the bridge matters?")
    
    try:
        model2 = mixedlm("relative_improvement ~ c4_to_midtrain_dist + midtrain_to_sft_dist", 
                        df, groups=df["sft_dataset"]).fit()
        
        print(f"\nFixed Effects:")
        print(f"  Intercept: {model2.params['Intercept']:.4f} (p = {model2.pvalues['Intercept']:.4f})")
        print(f"  C4→Midtrain: {model2.params['c4_to_midtrain_dist']:.4f} (p = {model2.pvalues['c4_to_midtrain_dist']:.4f})")
        print(f"  Midtrain→SFT: {model2.params['midtrain_to_sft_dist']:.4f} (p = {model2.pvalues['midtrain_to_sft_dist']:.4f})")
        
        c4_mid_coef = model2.params['c4_to_midtrain_dist']
        c4_mid_p = model2.pvalues['c4_to_midtrain_dist']
        mid_sft_coef = model2.params['midtrain_to_sft_dist']
        mid_sft_p = model2.pvalues['midtrain_to_sft_dist']
        
        print(f"\nComponent Significance:")
        if c4_mid_p < 0.05:
            direction = "POSITIVE" if c4_mid_coef > 0 else "NEGATIVE"
            print(f"  ✓ C4→Midtrain: SIGNIFICANT {direction} effect")
        else:
            print(f"  ✗ C4→Midtrain: Not significant")
            
        if mid_sft_p < 0.05:
            direction = "POSITIVE" if mid_sft_coef > 0 else "NEGATIVE"
            print(f"  ✓ Midtrain→SFT: SIGNIFICANT {direction} effect")
        else:
            print(f"  ✗ Midtrain→SFT: Not significant")
        
        results['model2'] = {
            'c4_mid_coef': c4_mid_coef,
            'c4_mid_p': c4_mid_p,
            'mid_sft_coef': mid_sft_coef,
            'mid_sft_p': mid_sft_p,
            'aic': model2.aic
        }
        
    except Exception as e:
        print(f"Error fitting Model 2: {e}")
        results['model2'] = None
    
    # Model 3: Domain matching vs Bridge length
    print(f"\n" + "="*40)
    print("MODEL 3: Domain Matching vs Bridge Length")
    print("="*40)
    print("Formula: improvement ~ bridge_length_max + perfect_match + (1|sft_dataset)")
    print("Tests: Is it bridge length or domain matching that matters?")
    
    try:
        model3 = mixedlm("relative_improvement ~ bridge_length_max + perfect_match", 
                        df, groups=df["sft_dataset"]).fit()
        
        print(f"\nFixed Effects:")
        print(f"  Intercept: {model3.params['Intercept']:.4f} (p = {model3.pvalues['Intercept']:.4f})")
        print(f"  Bridge Length: {model3.params['bridge_length_max']:.4f} (p = {model3.pvalues['bridge_length_max']:.4f})")
        print(f"  Perfect Match: {model3.params['perfect_match']:.4f} (p = {model3.pvalues['perfect_match']:.4f})")
        
        bridge_controlled_coef = model3.params['bridge_length_max']
        bridge_controlled_p = model3.pvalues['bridge_length_max']
        match_coef = model3.params['perfect_match']
        match_p = model3.pvalues['perfect_match']
        
        print(f"\nControlled Effects:")
        if bridge_controlled_p < 0.05:
            print(f"  ✓ Bridge length remains significant after controlling for domain matching")
        else:
            print(f"  ✗ Bridge length not significant when controlling for domain matching")
            
        if match_p < 0.05:
            print(f"  ✓ Perfect domain matching: {match_coef:+.1%} improvement boost")
        else:
            print(f"  ✗ Domain matching effect not significant")
        
        results['model3'] = {
            'bridge_controlled_coef': bridge_controlled_coef,
            'bridge_controlled_p': bridge_controlled_p,
            'match_coef': match_coef,
            'match_p': match_p,
            'aic': model3.aic
        }
        
    except Exception as e:
        print(f"Error fitting Model 3: {e}")
        results['model3'] = None
    
    # Model 4: Full model with all domain types
    print(f"\n" + "="*40)
    print("MODEL 4: Full Domain Model")
    print("="*40)
    print("Formula: improvement ~ c4_to_midtrain_dist + perfect_match + partial_match + (1|sft_dataset)")
    print("Tests: C4→Midtrain distance vs domain matching effects")
    
    try:
        model4 = mixedlm("relative_improvement ~ c4_to_midtrain_dist + perfect_match + partial_match", 
                        df, groups=df["sft_dataset"]).fit()
        
        print(f"\nFixed Effects:")
        print(f"  Intercept: {model4.params['Intercept']:.4f} (p = {model4.pvalues['Intercept']:.4f})")
        print(f"  C4→Midtrain: {model4.params['c4_to_midtrain_dist']:.4f} (p = {model4.pvalues['c4_to_midtrain_dist']:.4f})")
        print(f"  Perfect Match: {model4.params['perfect_match']:.4f} (p = {model4.pvalues['perfect_match']:.4f})")
        print(f"  Partial Match: {model4.params['partial_match']:.4f} (p = {model4.pvalues['partial_match']:.4f})")
        
        results['model4'] = {
            'c4_mid_final_coef': model4.params['c4_to_midtrain_dist'],
            'c4_mid_final_p': model4.pvalues['c4_to_midtrain_dist'],
            'perfect_final_coef': model4.params['perfect_match'],
            'perfect_final_p': model4.pvalues['perfect_match'],
            'partial_final_coef': model4.params['partial_match'],
            'partial_final_p': model4.pvalues['partial_match'],
            'aic': model4.aic
        }
        
    except Exception as e:
        print(f"Error fitting Model 4: {e}")
        results['model4'] = None
    
    return results


def run_manual_controls(analysis_df):
    """Manual control analysis using residualization when statsmodels unavailable."""
    print(f"\n" + "="*40)
    print("MANUAL CONTROL ANALYSIS")
    print("="*40)
    print("Using residualization to control for task difficulty...")
    
    # Step 1: Remove effect of task difficulty
    X_difficulty = analysis_df[['c4_to_sft_dist']].values
    y_improvement = analysis_df['relative_improvement'].values
    
    difficulty_model = LinearRegression().fit(X_difficulty, y_improvement)
    improvement_residuals = y_improvement - difficulty_model.predict(X_difficulty)
    
    print(f"Task difficulty explains {difficulty_model.score(X_difficulty, y_improvement):.1%} of improvement variance")
    
    # Step 2: Test effects on residualized improvement
    bridge_corr, bridge_p = pearsonr(analysis_df['bridge_length_max'], improvement_residuals)
    c4_mid_corr, c4_mid_p = pearsonr(analysis_df['c4_to_midtrain_dist'], improvement_residuals)
    mid_sft_corr, mid_sft_p = pearsonr(analysis_df['midtrain_to_sft_dist'], improvement_residuals)
    
    # Domain matching effect
    perfect_mask = analysis_df['bridge_type'] == 'perfect_match'
    if perfect_mask.sum() > 0:
        perfect_mean = improvement_residuals[perfect_mask].mean()
        other_mean = improvement_residuals[~perfect_mask].mean()
        domain_effect = perfect_mean - other_mean
    else:
        domain_effect = 0
    
    print(f"\nEffects on Residualized Improvement (controlling for task difficulty):")
    print(f"  Bridge Length: r = {bridge_corr:+.3f} (p = {bridge_p:.4f})")
    print(f"  C4→Midtrain:   r = {c4_mid_corr:+.3f} (p = {c4_mid_p:.4f})")
    print(f"  Midtrain→SFT:  r = {mid_sft_corr:+.3f} (p = {mid_sft_p:.4f})")
    print(f"  Domain Match:  Δ = {domain_effect:+.1%}")
    
    return {
        'manual_bridge_corr': bridge_corr,
        'manual_bridge_p': bridge_p,
        'manual_c4_mid_corr': c4_mid_corr,
        'manual_c4_mid_p': c4_mid_p,
        'manual_domain_effect': domain_effect
    }


def interpret_results(results, analysis_df):
    """Interpret the mixed effects model results and provide conclusions."""
    print(f"\n" + "="*60)
    print("RESULTS INTERPRETATION")
    print("="*60)
    
    if not STATSMODELS_AVAILABLE:
        print("Manual control results:")
        if results.get('manual_bridge_p', 1) < 0.05:
            print("✓ Bridge length effect survives task difficulty control")
        else:
            print("✗ Bridge length effect disappears after controlling for task difficulty")
        return
    
    # Check what we found
    model1 = results.get('model1')
    model2 = results.get('model2') 
    model3 = results.get('model3')
    model4 = results.get('model4')
    
    print("Summary of Findings:")
    
    # Bridge length findings
    if model1 and model1['significant']:
        if model3 and model3['bridge_controlled_p'] >= 0.05:
            print("\n🔍 FINDING 1: Bridge effect is confounded with domain matching")
            print("   → Bridge length appears significant in isolation")
            print("   → But disappears when controlling for domain matching")
            print("   → CONCLUSION: Domain matching explains the apparent bridge effect")
            conclusion = "domain_matching_explains_bridge"
        else:
            print("\n✓ FINDING 1: Bridge length has genuine effect")
            print("   → Survives controls for both task difficulty and domain matching")
            print("   → CONCLUSION: Bridge optimization may be worthwhile")
            conclusion = "bridge_effect_real"
    else:
        print("\n✗ FINDING 1: No significant bridge length effect")
        print("   → Bridge optimization not supported")
        conclusion = "no_bridge_effect"
    
    # Component findings
    if model2:
        if model2['c4_mid_p'] < 0.05 and model2['mid_sft_p'] >= 0.05:
            print("\n🎯 FINDING 2: C4→Midtrain distance drives results")
            print("   → First bridge segment matters, second doesn't")
            print("   → CONCLUSION: Focus on midtrain dataset selection")
        elif model2['mid_sft_p'] < 0.05 and model2['c4_mid_p'] >= 0.05:
            print("\n🎯 FINDING 2: Midtrain→SFT distance drives results")
            print("   → Second bridge segment matters, first doesn't")
            print("   → CONCLUSION: Focus on midtrain-to-task alignment")
        elif model2['c4_mid_p'] < 0.05 and model2['mid_sft_p'] < 0.05:
            print("\n🎯 FINDING 2: Both bridge components matter")
            print("   → Full bridge architecture is important")
        else:
            print("\n❓ FINDING 2: No clear component effects")
    
    # Domain matching findings
    if model3 and model3['match_p'] < 0.05:
        effect_size = model3['match_coef']
        print(f"\n⭐ FINDING 3: Domain matching has large effect")
        print(f"   → Perfect matches provide {effect_size:+.1%} improvement")
        print(f"   → This is the strongest predictor we found")
    
    # Overall recommendations
    print(f"\n" + "="*40)
    print("PRACTICAL RECOMMENDATIONS")
    print("="*40)
    
    if conclusion == "domain_matching_explains_bridge":
        print("1. PRIORITIZE DOMAIN MATCHING over bridge optimization")
        print("2. Use your similarity matrices to identify domain-matched datasets")
        print("3. Don't worry about bridge length or smoothness")
        
    elif conclusion == "bridge_effect_real":
        print("1. Bridge optimization IS worthwhile")
        print("2. Use similarity matrices to optimize bridge architecture")
        print("3. Consider multi-stage midtraining approaches")
        
    elif conclusion == "no_bridge_effect":
        print("1. Bridge optimization appears ineffective")
        print("2. Focus on other aspects (domain matching, scale, etc.)")
        
    # Domain matching always matters if we found it
    if model3 and model3['match_p'] < 0.05:
        print("4. Domain matching provides substantial benefits")
        print("5. Perfect matches should be prioritized when available")
    
    # Effect sizes
    perfect_matches = analysis_df[analysis_df['bridge_type'] == 'perfect_match']
    others = analysis_df[analysis_df['bridge_type'] != 'perfect_match']
    
    if len(perfect_matches) > 0:
        perfect_mean = perfect_matches['relative_improvement'].mean()
        other_mean = others['relative_improvement'].mean()
        print(f"\n📊 EFFECT SIZES:")
        print(f"   Perfect matches: {perfect_mean:+.1%} average improvement")
        print(f"   Other combinations: {other_mean:+.1%} average improvement")
        print(f"   Domain matching advantage: {perfect_mean - other_mean:+.1%}")

def create_enhanced_visualizations(analysis_df, results):
    """Create enhanced plots for bridge hypothesis analysis."""
    print(f"\n" + "="*40)
    print("CREATING ENHANCED VISUALIZATIONS")
    print("="*40)
    
    # Set global font size for all text
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'legend.fontsize': 14, 'legend.title_fontsize': 15})

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(22, 17))
    
    # 1. Original scatter plots (top row)
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 2)
    
    # Recreate your original plots but larger markers
    create_original_scatter_plots(analysis_df, ax1, ax2, marker_size=120)
    
    # 2. 2D Bridge Space Visualization
    ax3 = plt.subplot(3, 3, 3)
    create_2d_bridge_space(analysis_df, ax3, marker_size=120)
    
    # 3. Bridge Score Analysis
    ax4 = plt.subplot(3, 3, 4)
    create_bridge_score_analysis(analysis_df, ax4)
    
    # 4. Domain-Separated Analysis
    ax5 = plt.subplot(3, 3, 5)
    ax6 = plt.subplot(3, 3, 6)
    create_domain_separated_analysis(analysis_df, ax5, ax6, marker_size=120)
    
    # 5. Heatmap of improvements
    ax7 = plt.subplot(3, 3, 7)
    create_improvement_heatmap(analysis_df, ax7)
    
    # 6. Path shortening analysis
    ax8 = plt.subplot(3, 3, 8)
    create_path_shortening_analysis(analysis_df, ax8)
    
    # 7. Box plot by domain matching
    ax9 = plt.subplot(3, 3, 9)
    create_domain_matching_boxplot(analysis_df, ax9)
    
    plt.tight_layout()
    plt.savefig('enhanced_bridge_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved enhanced visualizations to 'enhanced_bridge_analysis.png'")
    
    # Also create individual high-quality versions of key plots
    create_key_individual_plots(analysis_df)


def create_original_scatter_plots(analysis_df, ax1, ax2, marker_size=120):
    """Recreate original scatter plots in subplot format, separated by model size."""
    # Define marker shapes for MIDTRAIN datasets (swapped from SFT)
    midtrain_datasets = sorted(analysis_df['midtrain_dataset'].unique())
    
    # Manually assign distinctive shapes to midtrain datasets
    shape_assignment = {
        'Starcoder (20%)': 'h',
        'Starcoder (100%)': '8',
        'Math (12%)': 's',    # square  
        'FLAN (5%)': 'X',    # triangle up
        'DCLM (20%)': 'd',            # hexagon (very distinct)
        'KnowledgeQA (20%)': 'v',     # diamond 
        'C4': 'P',              # plus (filled) - if C4 appears as midtrain
    }
    
    # Fallback shapes if we have more datasets
    fallback_shapes = ['v', 'X', '<', '>', '8', 'p', 'd', '1', '2', '3', '4']
    
    midtrain_marker_map = {}
    for i, midtrain in enumerate(midtrain_datasets):
        if midtrain in shape_assignment:
            midtrain_marker_map[midtrain] = shape_assignment[midtrain]
        else:
            midtrain_marker_map[midtrain] = fallback_shapes[i % len(fallback_shapes)]
    
    # Define colors for SFT datasets (swapped from midtrain)
    sft_datasets = sorted(analysis_df['sft_dataset'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(sft_datasets)))
    sft_color_map = {sft: colors[i] for i, sft in enumerate(sft_datasets)}
    
    # Get model sizes and sort them for consistent ordering
    model_sizes = sorted(analysis_df['model_size'].unique())
    print(f"Model sizes found: {model_sizes}")
    
    # Helper function to plot data for a specific model size
    def plot_model_size_data(ax, df_subset, x_col, title_prefix, model_size):
        if len(df_subset) == 0:
            ax.text(0.5, 0.5, f'No data for {model_size}', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=16)
            ax.set_title(f'{title_prefix} ({model_size}) - No Data')
            return
        
        # Plot points - NOW: colors for SFT datasets, shapes for midtrain datasets
        for _, row in df_subset.iterrows():
            ax.scatter(row[x_col], row['relative_improvement'], 
                      color=sft_color_map[row['sft_dataset']],  # Color by SFT dataset
                      marker=midtrain_marker_map[row['midtrain_dataset']],  # Shape by midtrain dataset
                      alpha=0.7, s=marker_size, edgecolors='black', linewidth=1)
        
        # Add trend line if we have enough data
        if len(df_subset) >= 3:
            if 'proximity_advantage' in x_col:
                # Linear fit for proximity advantage
                slope, intercept, r_value, p_value, _ = linregress(
                    df_subset[x_col], df_subset['relative_improvement']
                )
                x_trend = np.linspace(df_subset[x_col].min(), df_subset[x_col].max(), 100)
                y_trend = slope * x_trend + intercept
                ax.plot(x_trend, y_trend, 'r--', alpha=0.8, linewidth=2)
                stats_text = f'(r={r_value:.3f}, p={p_value:.3f})'
            else:
                # Quadratic fit for the first two columns
                x_vals = df_subset[x_col].values
                y_vals = df_subset['relative_improvement'].values
                coeffs = np.polyfit(x_vals, y_vals, 2)
                poly = np.poly1d(coeffs)
                x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
                y_trend = poly(x_trend)
                ax.plot(x_trend, y_trend, 'r--', alpha=0.8, linewidth=2)
                y_pred = poly(x_vals)
                ss_res = np.sum((y_vals - y_pred) ** 2)
                ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
                stats_text = f'(deg2 $R^2$={r2:.3f})'
        else:
            stats_text = f'(n={len(df_subset)})'
        
        # Formatting
        if 'c4_to_midtrain' in x_col:
            x_label = 'C4 → Midtrain Distance'
        elif 'midtrain_to_sft' in x_col:
            x_label = 'Midtrain → SFT Distance'
        elif 'proximity_advantage' in x_col:
            x_label = 'Proximity Advantage\n(dist(C4→SFT) - dist(Midtrain→SFT))'
        else:
            x_label = x_col.replace('_', ' ').title()
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Relative Improvement', fontsize=12)
        ax.set_title(f'{title_prefix} ({model_size})\n{stats_text}', fontsize=13)
        ax.grid(True, alpha=0.3)
    


    # --- NEW LOGIC: Save first two columns as a separate figure with one row per model size ---
    n_models = len(model_sizes)
    fig12, axes12 = plt.subplots(n_models, 2, figsize=(14, 6 * n_models), squeeze=False)
    legend_marker_size_first_two = 8
    legend_marker_size = 18  # for proximity advantage row
    sft_handles_first_two = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=colors[i], markersize=legend_marker_size_first_two,
                              label=sft, markeredgecolor='black')
                   for i, sft in enumerate(sft_datasets)]
    midtrain_handles_first_two = [plt.Line2D([0], [0], marker=midtrain_marker_map[midtrain], color='w',
                                 markerfacecolor='gray', markersize=legend_marker_size_first_two, label=midtrain,
                                 markeredgecolor='black')
                      for midtrain in midtrain_datasets]
    # For proximity advantage row, use larger handles (already handled below)
    for i, model_size in enumerate(model_sizes):
        model_data = analysis_df[analysis_df['model_size'] == model_size]
        # Plot 1: C4→Midtrain
        ax1_new = axes12[i, 0]
        for _, row in model_data.iterrows():
            ax1_new.scatter(row['c4_to_midtrain_dist'], row['relative_improvement'],
                           color=sft_color_map[row['sft_dataset']],
                           marker=midtrain_marker_map[row['midtrain_dataset']],
                           alpha=0.7, s=marker_size, edgecolors='black', linewidth=1)
        if len(model_data) > 1:
            # Quadratic fit for C4→Midtrain
            x = model_data['c4_to_midtrain_dist']
            y = model_data['relative_improvement']
            coeffs = np.polyfit(x, y, 2)
            x_trend = np.linspace(x.min(), x.max(), 100)
            y_trend = coeffs[0] * x_trend**2 + coeffs[1] * x_trend + coeffs[2]
            ax1_new.plot(x_trend, y_trend, 'r--', alpha=0.8, linewidth=2)
            # Calculate R^2 for quadratic fit
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            title_stats = f'(quad $R^2$={r2:.3f})'
        else:
            title_stats = f'(n={len(model_data)})'
        ax1_new.set_xlabel('C4 → Midtrain Distance', fontsize=20)
        ax1_new.set_ylabel('Relative Improvement', fontsize=20)
        ax1_new.set_title(f'{model_size}: C4→Midtrain vs. Benefit\n{title_stats}', fontsize=18)
        ax1_new.grid(True, alpha=0.3)
        ax1_new.legend(handles=sft_handles_first_two, bbox_to_anchor=(1.05, 1), loc='upper left',
                      title='SFT Dataset', fontsize=10, title_fontsize=11)
        # Plot 2: Midtrain→SFT
        ax2_new = axes12[i, 1]
        for _, row in model_data.iterrows():
            ax2_new.scatter(row['midtrain_to_sft_dist'], row['relative_improvement'],
                           color=sft_color_map[row['sft_dataset']],
                           marker=midtrain_marker_map[row['midtrain_dataset']],
                           alpha=0.7, s=marker_size, edgecolors='black', linewidth=1)
        if len(model_data) > 1:
            # Quadratic fit for Midtrain→SFT
            x2 = model_data['midtrain_to_sft_dist']
            y2 = model_data['relative_improvement']
            coeffs2 = np.polyfit(x2, y2, 2)
            x_trend2 = np.linspace(x2.min(), x2.max(), 100)
            y_trend2 = coeffs2[0] * x_trend2**2 + coeffs2[1] * x_trend2 + coeffs2[2]
            ax2_new.plot(x_trend2, y_trend2, 'r--', alpha=0.8, linewidth=2)
            # Calculate R^2 for quadratic fit
            y_pred2 = np.polyval(coeffs2, x2)
            ss_res2 = np.sum((y2 - y_pred2) ** 2)
            ss_tot2 = np.sum((y2 - np.mean(y2)) ** 2)
            r2_2 = 1 - ss_res2 / ss_tot2 if ss_tot2 != 0 else 0
            title_stats2 = f'(quad $R^2$={r2_2:.3f})'
        else:
            title_stats2 = f'(n={len(model_data)})'
        ax2_new.set_xlabel('Midtrain → SFT Distance', fontsize=20)
        ax2_new.set_ylabel('Relative Improvement', fontsize=20)
        ax2_new.set_title(f'{model_size}: Midtrain→SFT vs. Benefit\n{title_stats2}', fontsize=18)
        ax2_new.grid(True, alpha=0.3)
        ax2_new.legend(handles=midtrain_handles_first_two, bbox_to_anchor=(1.05, 1), loc='upper left',
                      title='Midtrain Dataset', fontsize=10, title_fontsize=11)
    plt.tight_layout()
    fig12.savefig('bridge_scatter_first_two_cols.png', dpi=300, bbox_inches='tight')
    fig12.savefig('bridge_scatter_first_two_cols.pdf', dpi=300, bbox_inches='tight')
    print('✓ Saved first two scatter columns to bridge_scatter_first_two_cols.png/pdf')

    # --- NEW LOGIC: Plot the third column (Midtrain Proximity Advantage) as a single row ---
    # Sort model sizes smallest to largest (assuming they are strings like '70m', '160m', etc.)
    def model_size_sort_key(ms):
        import re
        match = re.match(r'(\d+)', ms)
        return int(match.group(1)) if match else 0
    model_sizes_sorted = sorted(model_sizes, key=model_size_sort_key)
    n_models = len(model_sizes_sorted)
    fig3, axes3 = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)
    axes3 = axes3[0]  # 1D array
    for i, model_size in enumerate(model_sizes_sorted):
        ax = axes3[i]
        model_data = analysis_df[analysis_df['model_size'] == model_size].copy()
        model_data['midtrain_proximity_advantage'] = model_data['c4_to_sft_dist'] - model_data['midtrain_to_sft_dist']
        # Custom plotting for proximity advantage with compact title
        # Plot points
        for _, row in model_data.iterrows():
            ax.scatter(row['midtrain_proximity_advantage'], row['relative_improvement'],
                       color=sft_color_map[row['sft_dataset']],
                       marker=midtrain_marker_map[row['midtrain_dataset']],
                       alpha=0.7, s=marker_size, edgecolors='black', linewidth=1)
        # Add x=0 and y=0 reference lines (light grey, dashed)
        ax.axhline(0, color='#bbbbbb', linestyle='--', linewidth=1.5, zorder=0)
        ax.axvline(0, color='#bbbbbb', linestyle='--', linewidth=1.5, zorder=0)
        # Trend line and stats
        if len(model_data) >= 3:
            x_vals = model_data['midtrain_proximity_advantage']
            y_vals = model_data['relative_improvement']
            slope, intercept, r_value, p_value, _ = linregress(x_vals, y_vals)
            x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_trend = slope * x_trend + intercept
            ax.plot(x_trend, y_trend, 'r--', alpha=0.8, linewidth=2)
            stats_text = f'(r={r_value:.3f}, p={p_value:.3f})'
        else:
            stats_text = f'(n={len(model_data)})'
        ax.set_xlabel('Proximity Advantage', fontsize=22)
        ax.set_ylabel('Relative Improvement (%)', fontsize=22)
        ax.set_title(f'Proximity Advantage vs. Benefit ({model_size}\n{stats_text})', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # Format y-axis as percent
        import matplotlib.ticker as mticker
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    # Create horizontal legends at the bottom with larger font
    # Add space at the bottom for two separate horizontal legends, left and right
    fig3.subplots_adjust(bottom=0.33)
    # Define legend handles for proximity advantage row (large markers)
    sft_handles = [plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=colors[i], markersize=legend_marker_size,
                              label=sft, markeredgecolor='black')
                   for i, sft in enumerate(sft_datasets)]
    midtrain_handles = [plt.Line2D([0], [0], marker=midtrain_marker_map[midtrain], color='w',
                                   markerfacecolor='gray', markersize=legend_marker_size, label=midtrain,
                                   markeredgecolor='black')
                      for midtrain in midtrain_datasets]
    sft_ncol = min(len(sft_handles), 4)
    midtrain_ncol = min(len(midtrain_handles), 4)
    # SFT legend: bottom center, with frame, further down, max 4 per row
    sft_legend = fig3.legend(
        handles=sft_handles,
        labels=[h.get_label() for h in sft_handles],
        loc='lower center',
        bbox_to_anchor=(0.25, -0.23),
        ncol=sft_ncol,
        fontsize=18,
        frameon=True,
        fancybox=True,
        shadow=False,
        borderpad=0.5,
        title='SFT Dataset',
        title_fontsize=20,
        alignment='center'
    )
    fig3.add_artist(sft_legend)
    # Midtrain legend: bottom center, with frame, further down, max 4 per row, align top with SFT
    # Calculate number of rows for midtrain legend
    import math
    n_midtrain = len(midtrain_handles)
    ncol = midtrain_ncol
    nrows = math.ceil(n_midtrain / ncol)
    # Center the last row by adding empty proxy handles if needed
    proxy_handles = list(midtrain_handles)
    proxy_labels = [h.get_label() for h in midtrain_handles]
    remainder = n_midtrain % ncol
    if remainder != 0:
        n_to_add = ncol - remainder
        from matplotlib.lines import Line2D
        for _ in range(n_to_add):
            proxy_handles.append(Line2D([], [], linestyle='None', marker=None, alpha=0))
            proxy_labels.append("")
    # Move the midtrain legend box down and align its top with SFT legend
    midtrain_legend = fig3.legend(
        handles=proxy_handles,
        labels=proxy_labels,
        loc='lower center',
        bbox_to_anchor=(0.75, -0.23 - (nrows-1)*0.06),
        ncol=ncol,
        fontsize=18,
        frameon=True,
        fancybox=True,
        shadow=False,
        borderpad=0.5,
        title='Midtrain Dataset',
        title_fontsize=20,
        alignment='center'
    )
    fig3.add_artist(midtrain_legend)
    plt.tight_layout(rect=[0,0.22,1,1])
    fig3.savefig('bridge_scatter_proximity_advantage_row.png', dpi=300, bbox_inches='tight')
    fig3.savefig('bridge_scatter_proximity_advantage_row.pdf', dpi=300, bbox_inches='tight')
    print('✓ Saved proximity advantage row to bridge_scatter_proximity_advantage_row.png/pdf')


def create_2d_bridge_space(analysis_df, ax, marker_size=120):
    """Create 2D visualization of bridge space with improvement as color."""
    scatter = ax.scatter(analysis_df['c4_to_midtrain_dist'], 
                        analysis_df['midtrain_to_sft_dist'],
                        c=analysis_df['relative_improvement'],
                        cmap='RdYlGn', s=marker_size, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Relative Improvement', rotation=270, labelpad=20)
    
    # Annotate some interesting points
    for idx, row in analysis_df.iterrows():
        if row['bridge_type'] == 'perfect_match' or abs(row['relative_improvement']) > 0.05:
            ax.annotate(f"{row['midtrain_dataset'][:4]}→{row['sft_dataset'][:4]}", 
                       (row['c4_to_midtrain_dist'], row['midtrain_to_sft_dist']),
                       fontsize=6, alpha=0.7)
    
    ax.set_xlabel('C4 → Midtrain Distance', fontsize=20)
    ax.set_ylabel('Midtrain → SFT Distance', fontsize=20)
    ax.set_title('2D Bridge Space\n(color = improvement)', fontsize=22)
    ax.grid(True, alpha=0.3)


def create_bridge_score_analysis(analysis_df, ax):
    """Analyze different bridge score formulations."""
    # Calculate different bridge scores
    df = analysis_df.copy()
    
    # Score 1: Minimum distance (bottleneck)
    df['bridge_score_min'] = df[['c4_to_midtrain_dist', 'midtrain_to_sft_dist']].min(axis=1)
    
    # Score 2: Path shortening ratio
    df['bridge_score_ratio'] = df['c4_to_midtrain_dist'] / df['bridge_length_total']
    
    # Score 3: Balance score (how balanced the two segments are)
    df['bridge_score_balance'] = 1 - abs(df['c4_to_midtrain_dist'] - df['midtrain_to_sft_dist']) / df['bridge_length_total']
    
    # Plot correlations
    scores = ['bridge_score_min', 'bridge_score_ratio', 'bridge_score_balance']
    score_names = ['Min Distance\n(Bottleneck)', 'Path Shortening\nRatio', 'Balance Score']
    correlations = []
    
    for score in scores:
        corr, p = pearsonr(df[score], df['relative_improvement'])
        correlations.append((corr, p))
    
    # Bar plot of correlations
    x = np.arange(len(scores))
    corr_values = [c[0] for c in correlations]
    p_values = [c[1] for c in correlations]
    
    bars = ax.bar(x, corr_values, alpha=0.7)
    
    # Color bars by significance
    for i, (bar, p) in enumerate(zip(bars, p_values)):
        if p < 0.01:
            bar.set_color('darkgreen')
        elif p < 0.05:
            bar.set_color('lightgreen')
        else:
            bar.set_color('gray')
    
    # Add p-values as text
    for i, (corr, p) in enumerate(correlations):
        ax.text(i, corr + 0.01 if corr > 0 else corr - 0.01, 
                f'p={p:.3f}', ha='center', va='bottom' if corr > 0 else 'top', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(score_names, fontsize=12)
    ax.set_ylabel('Correlation with Improvement', fontsize=16)
    ax.set_title('Bridge Score Formulations', fontsize=18)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylim(-0.6, 0.6)
    ax.grid(True, alpha=0.3, axis='y')


def create_domain_separated_analysis(analysis_df, ax1, ax2, marker_size=120):
    """Separate analysis for math/code vs general domains."""
    # Classify datasets
    math_code_midtrain = ['StarCoder', 'Math Combined', 'PyCode']
    math_code_sft = ['GSM8K', 'PyCode']
    
    df = analysis_df.copy()
    df['is_math_code'] = (
        (df['midtrain_dataset'].isin(math_code_midtrain)) | 
        (df['sft_dataset'].isin(math_code_sft))
    )
    
    # Plot 1: Math/Code experiments
    math_code_df = df[df['is_math_code']]
    general_df = df[~df['is_math_code']]
    
    # Math/Code subplot
    ax1.scatter(math_code_df['c4_to_midtrain_dist'], 
               math_code_df['relative_improvement'],
               alpha=0.7, s=marker_size, label='Math/Code')
    
    if len(math_code_df) > 2:
        slope1, intercept1, r1, p1, _ = linregress(
            math_code_df['c4_to_midtrain_dist'], 
            math_code_df['relative_improvement']
        )
        x_trend = np.linspace(math_code_df['c4_to_midtrain_dist'].min(), 
                             math_code_df['c4_to_midtrain_dist'].max(), 100)
        ax1.plot(x_trend, slope1 * x_trend + intercept1, 'r--', alpha=0.8)
        ax1.text(0.05, 0.95, f'r={r1:.3f}\np={p1:.3f}', 
                transform=ax1.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
    
    ax1.set_xlabel('C4 → Midtrain Distance', fontsize=20)
    ax1.set_ylabel('Relative Improvement', fontsize=20)
    ax1.set_title('Math/Code Experiments', fontsize=22)
    ax1.grid(True, alpha=0.3)
    
    # General domain subplot
    ax2.scatter(general_df['c4_to_midtrain_dist'], 
               general_df['relative_improvement'],
               alpha=0.7, s=marker_size, label='General', color='green')
    
    if len(general_df) > 2:
        slope2, intercept2, r2, p2, _ = linregress(
            general_df['c4_to_midtrain_dist'], 
            general_df['relative_improvement']
        )
        x_trend2 = np.linspace(general_df['c4_to_midtrain_dist'].min(), 
                              general_df['c4_to_midtrain_dist'].max(), 100)
        ax2.plot(x_trend2, slope2 * x_trend2 + intercept2, 'g--', alpha=0.8)
        ax2.text(0.05, 0.95, f'r={r2:.3f}\np={p2:.3f}', 
                transform=ax2.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top')
    
    ax2.set_xlabel('C4 → Midtrain Distance', fontsize=20)
    ax2.set_ylabel('Relative Improvement', fontsize=20)
    ax2.set_title('General Domain Experiments', fontsize=22)
    ax2.grid(True, alpha=0.3)


def create_improvement_heatmap(analysis_df, ax):
    """Create heatmap of improvements by midtrain-SFT combination."""
    # Pivot table
    pivot = analysis_df.pivot_table(
        values='relative_improvement',
        index='midtrain_dataset',
        columns='sft_dataset',
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', 
                center=0, ax=ax, cbar_kws={'label': 'Relative Improvement'})
    
    ax.set_title('Average Improvement by Combination', fontsize=22)
    ax.set_xlabel('SFT Dataset', fontsize=20)
    ax.set_ylabel('Midtrain Dataset', fontsize=20)
    
    # Rotate labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)


def create_path_shortening_analysis(analysis_df, ax):
    """Analyze how much midtraining shortens the path from C4 to SFT."""
    df = analysis_df.copy()
    
    # Calculate path shortening
    df['path_shortening'] = (df['c4_to_sft_dist'] - df['bridge_length_total']) / df['c4_to_sft_dist']
    
    # Plot path shortening vs improvement
    scatter = ax.scatter(df['path_shortening'], df['relative_improvement'],
                        c=df['bridge_type'].map({'perfect_match': 2, 'partial_match': 1, 'cross_domain': 0}),
                        cmap='viridis', s=60, alpha=0.7)
    
    # Add trend line
    slope, intercept, r, p, _ = linregress(df['path_shortening'], df['relative_improvement'])
    x_trend = np.linspace(df['path_shortening'].min(), df['path_shortening'].max(), 100)
    ax.plot(x_trend, slope * x_trend + intercept, 'r--', alpha=0.8)
    
    ax.set_xlabel('Path Shortening Ratio', fontsize=20)
    ax.set_ylabel('Relative Improvement', fontsize=20)
    ax.set_title(f'Path Shortening vs. Improvement\n(r={r:.3f}, p={p:.3f})', fontsize=22)
    ax.grid(True, alpha=0.3)
    
    # Add text box with insights
    ax.text(0.05, 0.95, f'Avg shortening: {df["path_shortening"].mean():.1%}', 
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top')


def create_domain_matching_boxplot(analysis_df, ax):
    """Create box plot of improvements by domain matching type."""
    # Order for display
    order = ['perfect_match', 'partial_match', 'cross_domain']
    
    # Create box plot
    df = analysis_df.copy()
    df['improvement_pct'] = df['relative_improvement'] * 100  # Convert to percentage
    
    box_data = [df[df['bridge_type'] == bt]['improvement_pct'].values for bt in order]
    
    bp = ax.boxplot(box_data, labels=['Perfect\nMatch', 'Partial\nMatch', 'Cross\nDomain'],
                    patch_artist=True, showmeans=True)
    
    # Color the boxes
    colors = ['lightgreen', 'yellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add sample sizes
    for i, bt in enumerate(order):
        n = len(df[df['bridge_type'] == bt])
        ax.text(i+1, ax.get_ylim()[0] + 1, f'n={n}', ha='center', fontsize=8)
    
    ax.set_ylabel('Relative Improvement (%)', fontsize=20)
    ax.set_title('Improvement by Domain Matching Type', fontsize=22)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)


def create_key_individual_plots(analysis_df):
    """Create individual high-quality versions of the most important plots."""
    
    # 1. 2D Bridge Space (Main figure for paper)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter with consistent styling
    scatter = ax.scatter(analysis_df['c4_to_midtrain_dist'], 
                        analysis_df['midtrain_to_sft_dist'],
                        c=analysis_df['relative_improvement'] * 100,  # Convert to percentage
                        cmap='RdYlGn', s=150, alpha=0.8, 
                        edgecolors='black', linewidth=1,
                        vmin=-10, vmax=10)  # Set consistent color scale
    
    # Improve colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Relative Improvement (%)', rotation=270, labelpad=25, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add diagonal line (where distances are equal)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)
    
    # Annotate perfect matches
    for idx, row in analysis_df.iterrows():
        if row['bridge_type'] == 'perfect_match':
            ax.scatter(row['c4_to_midtrain_dist'], row['midtrain_to_sft_dist'],
                      s=300, facecolors='none', edgecolors='blue', linewidth=2)
            ax.annotate(f"{row['midtrain_dataset'][:6]}→{row['sft_dataset'][:6]}", 
                       (row['c4_to_midtrain_dist'], row['midtrain_to_sft_dist']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    ax.set_xlabel('C4 → Midtrain Distance', fontsize=12)
    ax.set_ylabel('Midtrain → SFT Distance', fontsize=12)
    ax.set_title('Bridge Architecture in 2D Distance Space', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax.text(0.95, 0.05, 'Far from C4\nClose to SFT', transform=ax.transAxes,
            fontsize=9, ha='right', va='bottom', alpha=0.6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.text(0.05, 0.95, 'Close to C4\nFar from SFT', transform=ax.transAxes,
            fontsize=9, ha='left', va='top', alpha=0.6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('bridge_2d_space_high_quality.png', dpi=300, bbox_inches='tight')
    print("✓ Saved high-quality 2D bridge space to 'bridge_2d_space_high_quality.png'")
    
    # 2. Domain matching comparison (for paper)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data
    improvements_by_type = {
        'Perfect Match': analysis_df[analysis_df['bridge_type'] == 'perfect_match']['relative_improvement'] * 100,
        'Partial Match': analysis_df[analysis_df['bridge_type'] == 'partial_match']['relative_improvement'] * 100,
        'Cross Domain': analysis_df[analysis_df['bridge_type'] == 'cross_domain']['relative_improvement'] * 100
    }
    
    # Create violin plot instead of box plot for better visualization
    positions = [1, 2, 3]
    parts = ax.violinplot([improvements_by_type[k].values for k in improvements_by_type.keys()],
                         positions=positions, showmeans=True, showmedians=True)
    
    # Customize violin colors
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Add individual points
    for i, (label, data) in enumerate(improvements_by_type.items()):
        x = np.random.normal(positions[i], 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.4, s=30, color='black')
    
    # Statistics
    for i, (label, data) in enumerate(improvements_by_type.items()):
        mean_val = data.mean()
        ax.text(positions[i], ax.get_ylim()[1] * 0.95, f'μ={mean_val:.1f}%\nn={len(data)}',
                ha='center', va='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    ax.set_xticks(positions)
    ax.set_xticklabels(improvements_by_type.keys(), fontsize=12)
    ax.set_ylabel('Relative Improvement (%)', fontsize=12)
    ax.set_title('Midtraining Effectiveness by Domain Matching', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('domain_matching_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved domain matching comparison to 'domain_matching_comparison.png'")


def create_visualizations(analysis_df, results):
    """Create all visualizations including enhanced ones."""
    # Check how many model sizes we have
    model_sizes = sorted(analysis_df['model_size'].unique())
    print(f"Creating visualizations for model sizes: {model_sizes}")
    
    # Determine figure layout based on number of model sizes
    if len(model_sizes) >= 2:
        # Create figure with rows for each model size, columns for C4→Midtrain, Midtrain→SFT, and Proximity
        fig_height = 5 * len(model_sizes)  # 5 inches per model size row
        fig = plt.figure(figsize=(18, fig_height))  # Wider for 3 columns
        
        # Create dummy axes for the function call (will be replaced by create_original_scatter_plots)
        ax1 = fig.add_subplot(len(model_sizes), 3, 1)
        ax2 = fig.add_subplot(len(model_sizes), 3, 2)
        
        # Create the model-size separated plots
        create_original_scatter_plots(analysis_df, ax1, ax2, marker_size=120)
        
        plt.tight_layout()
        plt.savefig('bridge_distance_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('bridge_distance_analysis.pdf', dpi=300, bbox_inches='tight')
        print(f"✓ Saved model-size separated plots ({len(model_sizes)} model sizes) to 'bridge_distance_analysis.png'")
        plt.close()
    else:
        # Single model size - use original layout
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        create_original_scatter_plots(analysis_df, axes[0], axes[1], marker_size=120)
        plt.tight_layout()
        plt.savefig('bridge_distance_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved single model size plots to 'bridge_distance_analysis.png'")
        plt.close()
    
    # Create enhanced visualizations
    create_enhanced_visualizations(analysis_df, results)

def main():
    p=argparse.ArgumentParser(description='Mixed effects bridge analysis')
    p.add_argument('--results_csv',required=True)
    p.add_argument('--similarity_csv',required=True)
    p.add_argument('--output_csv',default='mixed_effects_results.csv')
    p.add_argument('--create_plots',action='store_true')
    p.add_argument('--exclude_sciq',action='store_true',help='Exclude SciQ')
    p.add_argument('--remove_outliers',action='store_true',
                   help='Filter out extreme improvement outliers')
    args=p.parse_args()

    sim_df=load_similarity_matrix(args.similarity_csv)
    if sim_df is None: return
    analysis_df=load_and_process_experimental_data(
        args.results_csv, sim_df,
        exclude_sciq=args.exclude_sciq,
        remove_outliers=args.remove_outliers
    )
    if analysis_df is None:
        print("Failed to process experimental data. Exiting.")
        return
    
    # Show data summary
    print(f"\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Total experiments: {len(analysis_df)}")
    print(f"SFT datasets: {analysis_df['sft_dataset'].nunique()}")
    print(f"Midtrain datasets: {analysis_df['midtrain_dataset'].nunique()}")
    print(f"Model sizes: {list(analysis_df['model_size'].unique())}")
    
    print(f"\nDomain matching distribution:")
    domain_counts = analysis_df['bridge_type'].value_counts()
    for domain_type, count in domain_counts.items():
        pct = count / len(analysis_df) * 100
        print(f"  {domain_type}: {count} ({pct:.1f}%)")
    
    print(f"\nImprovement statistics:")
    print(f"  Mean: {analysis_df['relative_improvement'].mean():+.1%}")
    print(f"  Std:  {analysis_df['relative_improvement'].std():.1%}")
    print(f"  Range: {analysis_df['relative_improvement'].min():+.1%} to {analysis_df['relative_improvement'].max():+.1%}")
    
    # Run mixed effects analysis
    results = run_mixed_effects_models(analysis_df)
    
    # Interpret results
    interpret_results(results, analysis_df)
    
    # Create visualizations if requested
    if args.create_plots:
        create_visualizations(analysis_df, results)
    
    # Save processed data
    analysis_df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Saved processed data to '{args.output_csv}'")
    
    # Save model results
    if STATSMODELS_AVAILABLE and any(results.values()):
        results_summary = {}
        for model_name, model_results in results.items():
            if model_results:
                for key, value in model_results.items():
                    results_summary[f"{model_name}_{key}"] = value
        
        results_df = pd.DataFrame([results_summary])
        results_file = args.output_csv.replace('.csv', '_model_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"✓ Saved model results to '{results_file}'")
    
    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    # Final summary
    if STATSMODELS_AVAILABLE and results.get('model3'):
        model3 = results['model3']
        if model3['match_p'] < 0.05:
            print(f"🎯 KEY FINDING: Domain matching provides {model3['match_coef']:+.1%} improvement boost")
        
        if model3['bridge_controlled_p'] < 0.05:
            print(f"🔧 Bridge optimization may be worthwhile (p = {model3['bridge_controlled_p']:.4f})")
        else:
            print(f"❌ Bridge optimization not supported (p = {model3['bridge_controlled_p']:.4f})")
    
    print("\nRecommendation: Focus on domain matching and midtraining dataset selection")
    print("over abstract bridge architecture optimization.")


if __name__ == "__main__":
    main()