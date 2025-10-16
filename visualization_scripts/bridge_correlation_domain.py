import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

def parse_similarity_matrix(matrix_text):
    """Parse similarity matrix from text format"""
    lines = matrix_text.strip().split('\n')
    
    # Find the header line (starts with "Dataset,")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('Dataset,'):
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError("Could not find header line starting with 'Dataset,'")
    
    # Extract matrix data starting from header
    matrix_lines = lines[header_idx:]
    matrix_text_clean = '\n'.join(matrix_lines)
    
    # Parse as CSV
    df = pd.read_csv(StringIO(matrix_text_clean), index_col=0)
    
    return df

def load_similarity_matrices(embedding_csv=None, ngram_csv=None):
    """Load similarity matrices from CSV files or use defaults"""
    
    if embedding_csv and ngram_csv:
        print(f"Loading embedding similarity matrix from {embedding_csv}")
        print(f"Loading n-gram similarity matrix from {ngram_csv}")
        
        try:
            embedding_df = pd.read_csv(embedding_csv, index_col=0)
            ngram_df = pd.read_csv(ngram_csv, index_col=0)
            
            print(f"Loaded embedding matrix shape: {embedding_df.shape}")
            print(f"Loaded n-gram matrix shape: {ngram_df.shape}")
            print(f"Embedding datasets: {list(embedding_df.index)}")
            print(f"N-gram datasets: {list(ngram_df.index)}")
            
            return embedding_df, ngram_df
            
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            print("Falling back to matrices from images...")
    
    print("Using similarity matrices extracted from your images")
    
    # From Image 2: Embedding-based similarity matrix (centroid, instructional)
    embedding_data = {
        'C4': [1.0, 0.66, 0.48, 0.72, 0.45, 0.37, 0.70, 0.84, 0.62, 0.73, 0.76],
        'StarCoder': [0.66, 1.0, 0.42, 0.53, 0.34, 0.36, 0.50, 0.63, 0.50, 0.54, 0.61],
        'Math Combined': [0.48, 0.42, 1.0, 0.43, 0.84, 0.49, 0.46, 0.55, 0.44, 0.45, 0.48],
        'FLAN Combined': [0.72, 0.53, 0.43, 1.0, 0.38, 0.37, 0.63, 0.66, 0.69, 0.65, 0.83],
        'GSM8K': [0.45, 0.34, 0.84, 0.38, 1.0, 0.42, 0.43, 0.45, 0.36, 0.36, 0.38],
        'PyCode': [0.37, 0.36, 0.49, 0.37, 0.42, 1.0, 0.33, 0.41, 0.33, 0.36, 0.44],
        'Social IQA': [0.70, 0.50, 0.46, 0.63, 0.43, 0.33, 1.0, 0.76, 0.57, 0.71, 0.68],
        'LIMA': [0.84, 0.63, 0.55, 0.66, 0.45, 0.41, 0.76, 1.0, 0.67, 0.72, 0.75],
        'SciQ': [0.62, 0.50, 0.44, 0.69, 0.36, 0.33, 0.57, 0.67, 1.0, 0.55, 0.72],
        'Movie Reviews': [0.73, 0.54, 0.45, 0.65, 0.36, 0.36, 0.71, 0.72, 0.55, 1.0, 0.72],
        'TREC': [0.76, 0.61, 0.48, 0.83, 0.38, 0.44, 0.68, 0.75, 0.72, 0.72, 1.0]
    }
    
    # From Image 1: Syntactic Multi-level similarity matrix (char 1-10 grams)
    ngram_data = {
        'C4': [1.0, 0.12, 0.60, 0.48, 0.58, 0.13, 0.49, 0.85, 0.74, 0.51, 0.44],
        'StarCoder': [0.12, 1.0, 0.20, 0.11, 0.11, 0.99, 0.11, 0.30, 0.11, 0.11, 0.11],
        'Math Combined': [0.60, 0.20, 1.0, 0.56, 0.63, 0.21, 0.38, 0.60, 0.59, 0.42, 0.40],
        'FLAN Combined': [0.48, 0.11, 0.56, 1.0, 0.39, 0.12, 0.35, 0.45, 0.49, 0.35, 0.41],
        'GSM8K': [0.58, 0.11, 0.63, 0.39, 1.0, 0.13, 0.40, 0.55, 0.54, 0.39, 0.37],
        'PyCode': [0.13, 0.99, 0.21, 0.12, 0.13, 1.0, 0.12, 0.32, 0.13, 0.12, 0.12],
        'Social IQA': [0.49, 0.11, 0.38, 0.35, 0.40, 0.12, 1.0, 0.47, 0.42, 0.35, 0.34],
        'LIMA': [0.85, 0.30, 0.60, 0.45, 0.55, 0.32, 0.47, 1.0, 0.70, 0.48, 0.41],
        'SciQ': [0.74, 0.11, 0.59, 0.49, 0.54, 0.13, 0.42, 0.70, 1.0, 0.46, 0.44],
        'Movie Reviews': [0.51, 0.11, 0.42, 0.35, 0.39, 0.12, 0.35, 0.48, 0.46, 1.0, 0.34],
        'TREC': [0.44, 0.11, 0.40, 0.41, 0.37, 0.12, 0.34, 0.41, 0.44, 0.34, 1.0]
    }
    
    # Create DataFrames
    datasets = ['C4', 'StarCoder', 'Math Combined', 'FLAN Combined', 'GSM8K', 'PyCode', 'Social IQA', 'LIMA', 'SciQ', 'Movie Reviews', 'TREC']
    
    embedding_df = pd.DataFrame(embedding_data, index=datasets)
    ngram_df = pd.DataFrame(ngram_data, index=datasets)
    
    print(f"Embedding matrix shape: {embedding_df.shape}")
    print(f"N-gram matrix shape: {ngram_df.shape}")
    print(f"Available datasets: {datasets}")
    
    return embedding_df, ngram_df

def map_dataset_names(dataset_name):
    """Map dataset names from CSV to similarity matrix names"""
    if not isinstance(dataset_name, str):
        dataset_name = str(dataset_name)
    
    # Clean the name and remove percentages/parentheses
    clean_name = dataset_name.strip()
    
    # Remove percentage indicators like (20%), (12%), etc.
    import re
    clean_name = re.sub(r'\s*\(\d+%?\)', '', clean_name)
    clean_name = clean_name.strip()
    
    # Direct mapping for exact matches - updated for new matrix
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
        'GSM8k': 'GSM8K',  # Handle case difference
        'Social IQA': 'Social IQA',
        'LIMA': 'LIMA',
        'SciQ': 'SciQ',
        'Movie Reviews': 'Movie Reviews',
        'TREC': 'TREC'
        # Note: DCLM and KnowledgeQA removed since not in new matrices
    }
    
    mapped_name = name_mapping.get(clean_name, clean_name)
    
    # Debug output
    if clean_name != dataset_name:
        print(f"  Name mapping: '{dataset_name}' → '{clean_name}' → '{mapped_name}'")
    
    return mapped_name

def get_combined_similarity(dataset1, dataset2, embedding_df, ngram_df, weight=0.5):
    """Get combined similarity between two datasets"""
    mapped_ds1 = map_dataset_names(dataset1)
    mapped_ds2 = map_dataset_names(dataset2)
    
    try:
        emb_sim = embedding_df.loc[mapped_ds1, mapped_ds2]
        ngram_sim = ngram_df.loc[mapped_ds1, mapped_ds2]
        combined_sim = weight * emb_sim + (1 - weight) * ngram_sim
        print(f"  Similarity {dataset1} → {dataset2}: emb={emb_sim:.3f}, ngram={ngram_sim:.3f}, combined={combined_sim:.3f}")
        return combined_sim
    except KeyError as e:
        print(f"  ERROR: Could not find similarity for {dataset1} ({mapped_ds1}) → {dataset2} ({mapped_ds2})")
        print(f"  Available datasets in matrix: {list(embedding_df.index)}")
        return 0.0

def calculate_bridge_length(midtrain_dataset, sft_dataset, embedding_df, ngram_df, method='total'):
    """Calculate bridge length using different methods"""
    c4_to_midtrain_sim = get_combined_similarity('C4', midtrain_dataset, embedding_df, ngram_df)
    midtrain_to_sft_sim = get_combined_similarity(midtrain_dataset, sft_dataset, embedding_df, ngram_df)
    
    # Convert similarities to distances
    c4_to_midtrain_dist = 1 - c4_to_midtrain_sim
    midtrain_to_sft_dist = 1 - midtrain_to_sft_sim
    
    if method == 'total':
        # Original: sum of distances
        bridge_length = c4_to_midtrain_dist + midtrain_to_sft_dist
    elif method == 'max':
        # Alternative: max of distances (weakest link)
        bridge_length = max(c4_to_midtrain_dist, midtrain_to_sft_dist)
    elif method == 'min':
        # Alternative: min of distances (strongest link)
        bridge_length = min(c4_to_midtrain_dist, midtrain_to_sft_dist)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return bridge_length, c4_to_midtrain_dist, midtrain_to_sft_dist

def analyze_midtraining_bridge_effect(csv_file_path, embedding_csv=None, ngram_csv=None):
    """Main analysis function"""
    
    # Load similarity matrices
    print("Loading similarity matrices...")
    embedding_df, ngram_df = load_similarity_matrices(embedding_csv, ngram_csv)
    
    print(f"Embedding matrix shape: {embedding_df.shape}")
    print(f"N-gram matrix shape: {ngram_df.shape}")
    print(f"Available datasets: {list(embedding_df.index)}")
    
    # Load experimental results
    print(f"\nLoading experimental results from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    print("Dataset columns:", df.columns.tolist())
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    # Fix the CSV formatting issue - forward fill model size and SFT dataset
    print("\nFixing CSV formatting (forward filling model size and SFT dataset)...")
    df['Model size'] = df['Model size'].fillna(method='ffill')
    df['SFT dataset'] = df['SFT dataset'].fillna(method='ffill')
    
    print("After fixing:")
    print(df.head(10))
    
    # Find C4 baseline
    c4_baseline = df[df['Pre/midtrain mix'] == 'C4'].copy()
    
    if len(c4_baseline) == 0:
        print("Error: No C4 baseline found!")
        print("Available Pre/midtrain mix values:", df['Pre/midtrain mix'].unique())
        return None
    
    print(f"\nFound {len(c4_baseline)} C4 baseline entries")
    print("C4 baseline rows:")
    print(c4_baseline[['Model size', 'SFT dataset', 'Pre/midtrain mix', 'SFT val loss after FT']])
    
    # Create baseline lookup
    baseline_lookup = {}
    for _, row in c4_baseline.iterrows():
        key = (row['Model size'], row['SFT dataset'])
        baseline_lookup[key] = row['SFT val loss after FT']
        print(f"Baseline for {key}: {row['SFT val loss after FT']}")
    
    # Get default baseline assuming all experiments are 70m + PyCode
    if ('70m', 'PyCode') in baseline_lookup:
        default_baseline = baseline_lookup[('70m', 'PyCode')]
        print(f"Using default baseline for all experiments: {default_baseline}")
    else:
        default_baseline = None
    
    # Analyze midtraining experiments
    results = []
    skipped_experiments = []
    
    for _, row in df.iterrows():
        if row['Pre/midtrain mix'] == 'C4':
            continue  # Skip baseline
            
        # Skip rows with missing values
        if pd.isna(row['SFT val loss after FT']) or pd.isna(row['Pre/midtrain mix']):
            continue
            
        print(f"Processing: {row['Model size']} | {row['SFT dataset']} | {row['Pre/midtrain mix']}")
        
        # Check if datasets are available in our similarity matrices
        midtrain_mapped = map_dataset_names(row['Pre/midtrain mix'])
        sft_mapped = map_dataset_names(row['SFT dataset'])
        
        if midtrain_mapped not in embedding_df.index:
            print(f"  SKIPPING: Midtrain dataset '{midtrain_mapped}' not in similarity matrix")
            skipped_experiments.append(f"{row['Pre/midtrain mix']} → {row['SFT dataset']}")
            continue
            
        if sft_mapped not in embedding_df.columns:
            print(f"  SKIPPING: SFT dataset '{sft_mapped}' not in similarity matrix")
            skipped_experiments.append(f"{row['Pre/midtrain mix']} → {row['SFT dataset']}")
            continue
        
        # Get baseline
        key = (row['Model size'], row['SFT dataset'])
        if key in baseline_lookup:
            baseline_loss = baseline_lookup[key]
        elif default_baseline is not None:
            baseline_loss = default_baseline
            print(f"  Using default baseline: {default_baseline}")
        else:
            print(f"  Warning: No baseline found for {key}")
            continue
            
        midtrain_loss = row['SFT val loss after FT']
        improvement = baseline_loss - midtrain_loss
        
        print(f"  Baseline: {baseline_loss:.4f}, Midtrain: {midtrain_loss:.4f}, Improvement: {improvement:.4f}")
        
        # Calculate bridge length using different methods
        bridge_total, c4_dist, mid_dist = calculate_bridge_length(
            row['Pre/midtrain mix'], 
            row['SFT dataset'], 
            embedding_df, 
            ngram_df,
            method='total'
        )
        
        bridge_max, _, _ = calculate_bridge_length(
            row['Pre/midtrain mix'], 
            row['SFT dataset'], 
            embedding_df, 
            ngram_df,
            method='max'
        )
        
        print(f"  C4→{row['Pre/midtrain mix']} dist: {c4_dist:.3f}, {row['Pre/midtrain mix']}→{row['SFT dataset']} dist: {mid_dist:.3f}")
        print(f"  Bridge total: {bridge_total:.4f}, Bridge max: {bridge_max:.4f}")
        print(f"  Expected: shorter bridge = better improvement (negative correlation)")
        
        results.append({
            'model_size': row['Model size'],
            'sft_dataset': row['SFT dataset'],
            'midtrain_dataset': row['Pre/midtrain mix'],
            'baseline_loss': baseline_loss,
            'midtrain_loss': midtrain_loss,
            'improvement': improvement,
            'bridge_length_total': bridge_total,
            'bridge_length_max': bridge_max,
            'c4_to_midtrain_dist': c4_dist,
            'midtrain_to_sft_dist': mid_dist,
            'c4_to_midtrain_sim': get_combined_similarity('C4', row['Pre/midtrain mix'], embedding_df, ngram_df),
            'midtrain_to_sft_sim': get_combined_similarity(row['Pre/midtrain mix'], row['SFT dataset'], embedding_df, ngram_df)
        })
    
    print(f"\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Total experiments processed: {len(results)}")
    print(f"Experiments skipped: {len(skipped_experiments)}")
    if skipped_experiments:
        print("Skipped experiments:")
        for exp in skipped_experiments[:10]:  # Show first 10
            print(f"  {exp}")
        if len(skipped_experiments) > 10:
            print(f"  ... and {len(skipped_experiments) - 10} more")
    
    if len(results) == 0:
        print("Error: No valid experiments found!")
        return None
    
    # Convert to DataFrame
    analysis_df = pd.DataFrame(results)
    
    print(f"\nAnalyzing {len(analysis_df)} midtraining experiments")
    
    # Calculate correlations for different bridge methods
    corr_total, p_total = pearsonr(analysis_df['bridge_length_total'], analysis_df['improvement'])
    corr_max, p_max = pearsonr(analysis_df['bridge_length_max'], analysis_df['improvement'])
    
    print(f"\n" + "="*50)
    print("BRIDGE LENGTH CORRELATION ANALYSIS - ALL DATA")
    print("="*50)
    print(f"TOTAL bridge length: r = {corr_total:.4f}, p = {p_total:.4f}")
    print(f"MAX bridge length:   r = {corr_max:.4f}, p = {p_max:.4f}")
    print(f"Sample size: {len(analysis_df)}")
    
    # Analyze by model size separately
    model_sizes = analysis_df['model_size'].unique()
    model_results = {}
    
    for model_size in model_sizes:
        if pd.isna(model_size):
            continue
            
        model_data = analysis_df[analysis_df['model_size'] == model_size]
        if len(model_data) < 3:  # Need minimum samples
            continue
            
        print(f"\n" + "="*30)
        print(f"MODEL SIZE: {model_size}")
        print("="*30)
        
        # Calculate correlations for this model size
        model_corr_total, model_p_total = pearsonr(model_data['bridge_length_total'], model_data['improvement'])
        model_corr_max, model_p_max = pearsonr(model_data['bridge_length_max'], model_data['improvement'])
        
        print(f"TOTAL bridge: r = {model_corr_total:.4f}, p = {model_p_total:.4f} (n={len(model_data)})")
        print(f"MAX bridge:   r = {model_corr_max:.4f}, p = {model_p_max:.4f} (n={len(model_data)})")
        
        model_results[model_size] = {
            'data': model_data,
            'corr_total': model_corr_total,
            'p_total': model_p_total,
            'corr_max': model_corr_max,
            'p_max': model_p_max
        }
        
        # Show best/worst for this model size
        print(f"\nBest bridges for {model_size}:")
        best_for_model = model_data.nsmallest(5, 'bridge_length_max')
        for _, row in best_for_model.iterrows():
            print(f"  {row['midtrain_dataset']:15} → {row['sft_dataset']:10} | Max: {row['bridge_length_max']:.3f} | Improvement: {row['improvement']:.4f}")
    
    # Choose the best overall method
    if abs(corr_max) > abs(corr_total):
        correlation, p_value = corr_max, p_max
        best_method = "MAX"
        bridge_col = 'bridge_length_max'
        print(f"\nBest method overall: MAX bridge length")
    else:
        correlation, p_value = corr_total, p_total
        best_method = "TOTAL"
        bridge_col = 'bridge_length_total'
        print(f"\nBest method overall: TOTAL bridge length")
    
    if p_value < 0.001:
        sig_level = "***"
    elif p_value < 0.01:
        sig_level = "**"
    elif p_value < 0.05:
        sig_level = "*"
    else:
        sig_level = "ns"
    
    print(f"Best correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significance: {sig_level}")
    
    # Interpretation - expecting NEGATIVE correlation (shorter bridge = better)
    print(f"\nExpected: NEGATIVE correlation (shorter bridge length = better improvement)")
    if correlation < -0.4 and p_value < 0.05:
        print("✓ STRONG EVIDENCE supports the bridge hypothesis (negative correlation)")
    elif correlation < -0.2 and p_value < 0.05:
        print("✓ MODERATE EVIDENCE supports the bridge hypothesis (negative correlation)")
    elif correlation < -0.15 and p_value < 0.1:
        print("? WEAK EVIDENCE for the bridge hypothesis (negative correlation)")
    elif correlation > 0.4 and p_value < 0.05:
        print("⚠️  STRONG POSITIVE correlation - unexpected! (longer bridge = better?)")
    elif correlation > 0.2 and p_value < 0.05:
        print("⚠️  MODERATE POSITIVE correlation - unexpected! (longer bridge = better?)")
    else:
        print("✗ NO CLEAR EVIDENCE for the bridge hypothesis")
        
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Total bridge length correlation
    axes[0,0].scatter(analysis_df['bridge_length_total'], analysis_df['improvement'], alpha=0.6)
    axes[0,0].set_xlabel('Total Bridge Length')
    axes[0,0].set_ylabel('Improvement')
    axes[0,0].set_title(f'Total Bridge: r = {corr_total:.3f}, p = {p_total:.3f}')
    z = np.polyfit(analysis_df['bridge_length_total'], analysis_df['improvement'], 1)
    p_fit = np.poly1d(z)
    axes[0,0].plot(analysis_df['bridge_length_total'], p_fit(analysis_df['bridge_length_total']), "r--", alpha=0.8)
    
    # Max bridge length correlation
    axes[0,1].scatter(analysis_df['bridge_length_max'], analysis_df['improvement'], alpha=0.6, color='orange')
    axes[0,1].set_xlabel('Max Bridge Length')
    axes[0,1].set_ylabel('Improvement')
    axes[0,1].set_title(f'Max Bridge: r = {corr_max:.3f}, p = {p_max:.3f}')
    z = np.polyfit(analysis_df['bridge_length_max'], analysis_df['improvement'], 1)
    p_fit = np.poly1d(z)
    axes[0,1].plot(analysis_df['bridge_length_max'], p_fit(analysis_df['bridge_length_max']), "r--", alpha=0.8)
    
    # Component correlations
    corr_c4_mid, p_c4_mid = pearsonr(analysis_df['c4_to_midtrain_sim'], analysis_df['improvement'])
    corr_mid_sft, p_mid_sft = pearsonr(analysis_df['midtrain_to_sft_sim'], analysis_df['improvement'])
    
    axes[0,2].scatter(analysis_df['c4_to_midtrain_sim'], analysis_df['improvement'], alpha=0.6, color='green')
    axes[0,2].set_xlabel('C4→Midtrain Similarity')
    axes[0,2].set_ylabel('Improvement')
    axes[0,2].set_title(f'C4→Midtrain: r = {corr_c4_mid:.3f}')
    
    axes[1,0].scatter(analysis_df['midtrain_to_sft_sim'], analysis_df['improvement'], alpha=0.6, color='purple')
    axes[1,0].set_xlabel('Midtrain→SFT Similarity')
    axes[1,0].set_ylabel('Improvement')
    axes[1,0].set_title(f'Midtrain→SFT: r = {corr_mid_sft:.3f}')
    
    # Distribution of improvements
    axes[1,1].hist(analysis_df['improvement'], bins=20, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Improvement')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Improvements')
    axes[1,1].axvline(analysis_df['improvement'].mean(), color='red', linestyle='--', label='Mean')
    axes[1,1].legend()
    
    # Bridge length comparison
    axes[1,2].scatter(analysis_df['bridge_length_total'], analysis_df['bridge_length_max'], alpha=0.6)
    axes[1,2].set_xlabel('Total Bridge Length')
    axes[1,2].set_ylabel('Max Bridge Length')
    axes[1,2].set_title('Total vs Max Bridge Length')
    
    plt.tight_layout()
    plt.show()
    
    # Save results
    analysis_df.to_csv('midtraining_bridge_analysis_results.csv', index=False)
    print(f"\nDetailed results saved to 'midtraining_bridge_analysis_results.csv'")
    
            # Save results
    analysis_df.to_csv('midtraining_bridge_analysis_results.csv', index=False)
    print(f"\nDetailed results saved to 'midtraining_bridge_analysis_results.csv'")
    
    # Add domain matching analysis
    analysis_df = analyze_domain_matching_effect(analysis_df)
    
    return analysis_df, correlation, p_value, best_method, model_results, model_results
    
    # Component analysis
    print(f"\n" + "="*50)
    print("COMPONENT ANALYSIS")
    print("="*50)
    
    corr_c4_mid, p_c4_mid = pearsonr(analysis_df['c4_to_midtrain_sim'], analysis_df['improvement'])
    corr_mid_sft, p_mid_sft = pearsonr(analysis_df['midtrain_to_sft_sim'], analysis_df['improvement'])
    
    print(f"C4→Midtrain similarity: r = {corr_c4_mid:.4f}, p = {p_c4_mid:.4f}")
    print(f"Midtrain→SFT similarity: r = {corr_mid_sft:.4f}, p = {p_mid_sft:.4f}")
    
    # Show detailed results
    print(f"\n" + "="*50)
    print(f"DETAILED RESULTS (sorted by {best_method} bridge length)")
    print("="*50)
    
    # Sort by the better method
    analysis_df_sorted = analysis_df.sort_values(bridge_col, ascending=True)
    print("All experiments sorted by bridge length (shortest first):")
    for _, row in analysis_df_sorted.iterrows():
        print(f"{row['midtrain_dataset']:15} → {row['sft_dataset']:10} | Total: {row['bridge_length_total']:.4f} | Max: {row['bridge_length_max']:.4f} | Improvement: {row['improvement']:.4f}")
    
def classify_bridge_type(midtrain_dataset, sft_dataset):
    """Classify bridge into domain matching categories"""
    
    # Clean dataset names
    midtrain_clean = map_dataset_names(midtrain_dataset)
    sft_clean = map_dataset_names(sft_dataset)
    
    # Define domain matching rules
    perfect_matches = [
        ('StarCoder', 'PyCode'),  # Code domain
        ('Math Combined', 'GSM8K'),  # Math domain
    ]
    
    partial_matches = [
        ('Math Combined', 'SciQ'),  # Both knowledge/reasoning heavy
        ('FLAN Combined', 'LIMA'),  # Both instruction following
        ('FLAN Combined', 'SciQ'),  # Both QA format
        ('FLAN Combined', 'TREC'),  # Both QA format
    ]
    
    # Check for perfect domain match
    if (midtrain_clean, sft_clean) in perfect_matches:
        return 'perfect_match'
    
    # Check for partial domain match
    if (midtrain_clean, sft_clean) in partial_matches:
        return 'partial_match'
    
    # Check if both are general/instruction datasets
    general_datasets = ['C4', 'FLAN Combined', 'LIMA']
    if midtrain_clean in general_datasets and sft_clean in general_datasets:
        return 'general_to_general'
    
    # Everything else is cross-domain
    return 'cross_domain'

def analyze_domain_matching_effect(analysis_df):
    """Analyze bridge effectiveness by domain matching"""
    
    # Add bridge type classification
    analysis_df['bridge_type'] = analysis_df.apply(
        lambda row: classify_bridge_type(row['midtrain_dataset'], row['sft_dataset']), 
        axis=1
    )
    
    print(f"\n" + "="*50)
    print("DOMAIN MATCHING ANALYSIS")
    print("="*50)
    
    # Group by bridge type
    bridge_groups = analysis_df.groupby('bridge_type')
    
    for bridge_type, group in bridge_groups:
        if len(group) == 0:
            continue
            
        print(f"\n{bridge_type.upper().replace('_', ' ')} (n={len(group)}):")
        print(f"  Mean improvement: {group['improvement'].mean():.4f}")
        print(f"  Std improvement: {group['improvement'].std():.4f}")
        print(f"  Best improvement: {group['improvement'].max():.4f}")
        print(f"  Worst improvement: {group['improvement'].min():.4f}")
        
        # Show all experiments in this category
        for _, row in group.sort_values('improvement', ascending=False).iterrows():
            print(f"    {row['midtrain_dataset']:15} → {row['sft_dataset']:10} | {row['model_size']} | Improvement: {row['improvement']:.4f}")
    
    # Statistical comparison
    print(f"\n" + "="*30)
    print("STATISTICAL COMPARISONS")
    print("="*30)
    
    # Perfect matches vs others
    perfect_matches = analysis_df[analysis_df['bridge_type'] == 'perfect_match']
    others = analysis_df[analysis_df['bridge_type'] != 'perfect_match']
    
    if len(perfect_matches) > 0 and len(others) > 0:
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(perfect_matches['improvement'], others['improvement'])
        
        print(f"Perfect domain matches vs Others:")
        print(f"  Perfect matches mean: {perfect_matches['improvement'].mean():.4f} (n={len(perfect_matches)})")
        print(f"  Others mean: {others['improvement'].mean():.4f} (n={len(others)})")
        print(f"  T-test: t={t_stat:.3f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"  ✗ Not significant (p ≥ 0.05)")
    
    # Effect size analysis
    print(f"\n" + "="*30)
    print("EFFECT SIZE ANALYSIS")
    print("="*30)
    
    overall_mean = analysis_df['improvement'].mean()
    
    for bridge_type, group in bridge_groups:
        if len(group) == 0:
            continue
        group_mean = group['improvement'].mean()
        effect_size = group_mean - overall_mean
        
        print(f"{bridge_type:20}: Δ = {effect_size:+.4f} vs overall mean")
    
    return analysis_df
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Total bridge length correlation
    axes[0,0].scatter(analysis_df['bridge_length_total'], analysis_df['improvement'], alpha=0.6)
    axes[0,0].set_xlabel('Total Bridge Length')
    axes[0,0].set_ylabel('Improvement')
    axes[0,0].set_title(f'Total Bridge: r = {corr_total:.3f}, p = {p_total:.3f}')
    z = np.polyfit(analysis_df['bridge_length_total'], analysis_df['improvement'], 1)
    p_fit = np.poly1d(z)
    axes[0,0].plot(analysis_df['bridge_length_total'], p_fit(analysis_df['bridge_length_total']), "r--", alpha=0.8)
    
    # Max bridge length correlation
    axes[0,1].scatter(analysis_df['bridge_length_max'], analysis_df['improvement'], alpha=0.6, color='orange')
    axes[0,1].set_xlabel('Max Bridge Length')
    axes[0,1].set_ylabel('Improvement')
    axes[0,1].set_title(f'Max Bridge: r = {corr_max:.3f}, p = {p_max:.3f}')
    z = np.polyfit(analysis_df['bridge_length_max'], analysis_df['improvement'], 1)
    p_fit = np.poly1d(z)
    axes[0,1].plot(analysis_df['bridge_length_max'], p_fit(analysis_df['bridge_length_max']), "r--", alpha=0.8)
    
    # Component correlations
    corr_c4_mid, p_c4_mid = pearsonr(analysis_df['c4_to_midtrain_sim'], analysis_df['improvement'])
    corr_mid_sft, p_mid_sft = pearsonr(analysis_df['midtrain_to_sft_sim'], analysis_df['improvement'])
    
    axes[0,2].scatter(analysis_df['c4_to_midtrain_sim'], analysis_df['improvement'], alpha=0.6, color='green')
    axes[0,2].set_xlabel('C4→Midtrain Similarity')
    axes[0,2].set_ylabel('Improvement')
    axes[0,2].set_title(f'C4→Midtrain: r = {corr_c4_mid:.3f}')
    
    axes[1,0].scatter(analysis_df['midtrain_to_sft_sim'], analysis_df['improvement'], alpha=0.6, color='purple')
    axes[1,0].set_xlabel('Midtrain→SFT Similarity')
    axes[1,0].set_ylabel('Improvement')
    axes[1,0].set_title(f'Midtrain→SFT: r = {corr_mid_sft:.3f}')
    
    # Distribution of improvements
    axes[1,1].hist(analysis_df['improvement'], bins=20, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Improvement')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Improvements')
    axes[1,1].axvline(analysis_df['improvement'].mean(), color='red', linestyle='--', label='Mean')
    axes[1,1].legend()
    
    # Bridge length comparison
    axes[1,2].scatter(analysis_df['bridge_length_total'], analysis_df['bridge_length_max'], alpha=0.6)
    axes[1,2].set_xlabel('Total Bridge Length')
    axes[1,2].set_ylabel('Max Bridge Length')
    axes[1,2].set_title('Total vs Max Bridge Length')
    
    plt.tight_layout()
    plt.show()
    
    return analysis_df, correlation, p_value, best_method

if __name__ == "__main__":
    # Run the analysis
    csv_file_path = "/projects/bfcu/mliu7/all_in_one_pretrainingvisualization_scripts/final_step_ft_results.csv"
    
    # Optional: provide paths to your new similarity matrices
    # embedding_csv = "path/to/your/embedding_similarity.csv"
    # ngram_csv = "path/to/your/syntactic_similarity.csv"
    embedding_csv = None  # Set to None to use defaults
    ngram_csv = None      # Set to None to use defaults
    
    try:
        results_df, correlation, p_value, best_method, model_results = analyze_midtraining_bridge_effect(
            csv_file_path, embedding_csv, ngram_csv
        )
        
        if results_df is not None:
            print(f"\n" + "="*50)
            print("FINAL SUMMARY")
            print("="*50)
            print(f"Best method: {best_method} bridge length")
            print(f"Overall bridge hypothesis correlation: {correlation:.4f} (p = {p_value:.4f})")
            
            # Summary by model size
            print(f"\nBy model size:")
            for model_size, results in model_results.items():
                best_corr = results['corr_max'] if abs(results['corr_max']) > abs(results['corr_total']) else results['corr_total']
                best_p = results['p_max'] if abs(results['corr_max']) > abs(results['corr_total']) else results['p_total']
                method = "MAX" if abs(results['corr_max']) > abs(results['corr_total']) else "TOTAL"
                
                if abs(best_corr) > 0.3 and best_p < 0.05:
                    strength = "STRONG"
                elif abs(best_corr) > 0.2 and best_p < 0.05:
                    strength = "MODERATE"
                elif abs(best_corr) > 0.15 and best_p < 0.1:
                    strength = "WEAK"
                else:
                    strength = "NONE"
                    
                print(f"  {model_size}: {method} r = {best_corr:.3f} (p = {best_p:.3f}) - {strength}")
            
            if abs(correlation) > 0.4 and p_value < 0.05:
                evidence = "Strong"
            elif abs(correlation) > 0.2 and p_value < 0.05:
                evidence = "Moderate"
            elif abs(correlation) > 0.15 and p_value < 0.1:
                evidence = "Weak"
            else:
                evidence = "None"
            print(f"\nOverall evidence strength: {evidence}")
            
            # Save results
            results_df.to_csv('midtraining_bridge_analysis_results.csv', index=False)
            print(f"\nDetailed results saved to 'midtraining_bridge_analysis_results.csv'")
            
    except FileNotFoundError:
        print(f"Error: Could not find the file '{csv_file_path}'")
        print("Please make sure the file exists and update the path in the script")
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()