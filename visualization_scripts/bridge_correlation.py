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
    """Load similarity matrices from CSV or Excel files"""
    
    if embedding_csv and ngram_csv:
        print(f"Loading embedding similarity matrix from {embedding_csv}")
        print(f"Loading n-gram similarity matrix from {ngram_csv}")
        
        try:
            # Try to determine file type and load accordingly
            def load_file(filepath):
                if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                    print(f"  Loading as Excel file: {filepath}")
                    return pd.read_excel(filepath, index_col=0)
                elif filepath.endswith('.csv'):
                    print(f"  Loading as CSV file: {filepath}")
                    # Try different encodings
                    for encoding in ['utf-8', 'latin1', 'cp1252']:
                        try:
                            return pd.read_csv(filepath, index_col=0, encoding=encoding)
                        except UnicodeDecodeError:
                            continue
                    raise ValueError(f"Could not decode {filepath} with any encoding")
                else:
                    # Try CSV first, then Excel
                    try:
                        return pd.read_csv(filepath, index_col=0, encoding='utf-8')
                    except:
                        try:
                            return pd.read_excel(filepath, index_col=0)
                        except:
                            raise ValueError(f"Could not load {filepath} as CSV or Excel")
            
            embedding_df = load_file(embedding_csv)
            ngram_df = load_file(ngram_csv)
            
            print(f"Loaded embedding matrix shape: {embedding_df.shape}")
            print(f"Loaded n-gram matrix shape: {ngram_df.shape}")
            print(f"Embedding datasets: {list(embedding_df.index)}")
            print(f"N-gram datasets: {list(ngram_df.index)}")
            
            # Show sample values for verification
            print(f"\nSample embedding values:")
            print(f"  C4 → StarCoder: {embedding_df.loc['C4', 'StarCoder'] if 'C4' in embedding_df.index and 'StarCoder' in embedding_df.columns else 'N/A'}")
            print(f"  StarCoder → PyCode: {embedding_df.loc['StarCoder', 'PyCode'] if 'StarCoder' in embedding_df.index and 'PyCode' in embedding_df.columns else 'N/A'}")
            
            print(f"\nSample n-gram values:")
            print(f"  C4 → StarCoder: {ngram_df.loc['C4', 'StarCoder'] if 'C4' in ngram_df.index and 'StarCoder' in ngram_df.columns else 'N/A'}")
            print(f"  StarCoder → PyCode: {ngram_df.loc['StarCoder', 'PyCode'] if 'StarCoder' in ngram_df.index and 'PyCode' in ngram_df.columns else 'N/A'}")
            
            # Verify matrices are symmetric and have same datasets
            if not embedding_df.index.equals(embedding_df.columns):
                print("WARNING: Embedding matrix is not symmetric (index ≠ columns)")
            if not ngram_df.index.equals(ngram_df.columns):
                print("WARNING: N-gram matrix is not symmetric (index ≠ columns)")
            if not embedding_df.index.equals(ngram_df.index):
                print("WARNING: Embedding and n-gram matrices have different datasets")
                
            return embedding_df, ngram_df
            
        except Exception as e:
            print(f"Error loading files: {e}")
            print("Please check file paths and format")
            print("Supported formats: .csv, .xlsx, .xls")
            return None, None
    
    else:
        print("ERROR: Both embedding_csv and ngram_csv paths must be provided")
        print("Usage: analyze_midtraining_bridge_effect(csv_file, embedding_csv, ngram_csv)")
        return None, None

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
    
    # TEMPORARILY EXCLUDE MOVIE REVIEWS FROM ANALYSIS
    if 'Movie Reviews' in clean_name or 'Movie Review' in clean_name:
        print(f"  EXCLUDING Movie Reviews from analysis: '{dataset_name}'")
        return None
    
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
        'Movie Reviews': None,  # Explicitly exclude
        'TREC': 'TREC'
        # Note: DCLM and KnowledgeQA removed since not in new matrices
    }
    
    mapped_name = name_mapping.get(clean_name, clean_name)
    
    # Debug output
    if clean_name != dataset_name:
        print(f"  Name mapping: '{dataset_name}' → '{clean_name}' → '{mapped_name}'")
    
    return mapped_name

def get_combined_similarity(dataset1, dataset2, embedding_df, ngram_df, weight=0.5, similarity_mode='combined'):
    """Get similarity between two datasets with different modes"""
    mapped_ds1 = map_dataset_names(dataset1)
    mapped_ds2 = map_dataset_names(dataset2)
    
    # Handle excluded datasets (mapped to None)
    if mapped_ds1 is None or mapped_ds2 is None:
        print(f"  EXCLUDED: Cannot compute similarity for {dataset1} → {dataset2} (one or both datasets excluded)")
        return 0.0  # Return neutral similarity for excluded datasets
    
    try:
        emb_sim = embedding_df.loc[mapped_ds1, mapped_ds2]
        ngram_sim = ngram_df.loc[mapped_ds1, mapped_ds2]
        
        if similarity_mode == 'semantic_only':
            combined_sim = emb_sim
            mode_desc = "semantic"
        elif similarity_mode == 'syntactic_only':
            combined_sim = ngram_sim
            mode_desc = "syntactic"
        else:  # combined
            combined_sim = weight * emb_sim + (1 - weight) * ngram_sim
            mode_desc = "combined"
            
        print(f"  Similarity {dataset1} → {dataset2} ({mode_desc}): emb={emb_sim:.3f}, ngram={ngram_sim:.3f}, final={combined_sim:.3f}")
        return combined_sim
    except KeyError as e:
        print(f"  ERROR: Could not find similarity for {dataset1} ({mapped_ds1}) → {dataset2} ({mapped_ds2})")
        print(f"  Available datasets in matrix: {list(embedding_df.index)}")
        return 0.0

def calculate_bridge_length(midtrain_dataset, sft_dataset, embedding_df, ngram_df, method='total', similarity_mode='combined'):
    """Calculate bridge length using DISTANCES: (1-sim(C4,Midtrain)) + (1-sim(Midtrain,SFT))"""
    c4_to_midtrain_sim = get_combined_similarity('C4', midtrain_dataset, embedding_df, ngram_df, similarity_mode=similarity_mode)
    midtrain_to_sft_sim = get_combined_similarity(midtrain_dataset, sft_dataset, embedding_df, ngram_df, similarity_mode=similarity_mode)
    
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

def analyze_catastrophic_forgetting(analysis_df, original_df):
    """
    Test catastrophic forgetting hypothesis using C4 validation loss.
    We need the original_df to access the C4 baseline rows that were filtered out.
    """
    print(f"\n" + "="*50)
    print("CATASTROPHIC FORGETTING ANALYSIS")
    print("="*50)
    print("Testing: Do larger C4→Midtrain gaps cause forgetting of general capabilities?")
    
    # Filter to only experiments with C4 loss data from the processed analysis_df
    c4_data = analysis_df[analysis_df['c4_loss_after_ft'].notna()].copy()
    
    if len(c4_data) == 0:
        print("ERROR: No C4 loss data available in processed experiments")
        return analysis_df, None
    
    print(f"Analyzing {len(c4_data)} midtraining experiments with C4 loss data")
    
    # Get C4 baselines from the original dataframe
    print(f"\nExtracting C4 baselines from original data...")
    
    # Clean column names in original data
    original_df_clean = original_df.copy()
    original_df_clean.columns = original_df_clean.columns.str.strip()
    
    # Forward fill model size and SFT dataset in original data
    original_df_clean['Model size'] = original_df_clean['Model size'].fillna(method='ffill')
    original_df_clean['SFT dataset'] = original_df_clean['SFT dataset'].fillna(method='ffill')
    
    # Find C4 baseline rows
    c4_baseline_rows = original_df_clean[original_df_clean['Pre/midtrain mix'] == 'C4'].copy()
    
    print(f"Found {len(c4_baseline_rows)} C4 baseline rows in original data")
    
    if len(c4_baseline_rows) == 0:
        print("ERROR: No C4 baseline rows found in original data!")
        return analysis_df, None
    
    # Extract baselines
    c4_baselines = {}
    for _, row in c4_baseline_rows.iterrows():
        if pd.notna(row['C4 val loss after FT']):
            key = (row['Model size'], row['SFT dataset'])
            c4_baselines[key] = row['C4 val loss after FT']
            print(f"  Baseline {key}: {row['C4 val loss after FT']:.3f}")
    
    if len(c4_baselines) == 0:
        print("ERROR: No C4 baselines have valid C4 loss data!")
        return analysis_df, None
    
    print(f"\nC4 baselines extracted: {len(c4_baselines)}")
    
    # Calculate degradation for each midtraining experiment
    degradations = []
    
    print(f"\nCalculating C4 degradation for each experiment:")
    for idx, row in c4_data.iterrows():
        key = (row['model_size'], row['sft_dataset'])
        if key in c4_baselines:
            baseline = c4_baselines[key]
            degradation = row['c4_loss_after_ft'] - baseline
            degradations.append(degradation)
            print(f"  {row['midtrain_dataset']:15} → {row['sft_dataset']:10} ({row['model_size']}): "
                  f"{row['c4_loss_after_ft']:.3f} - {baseline:.3f} = {degradation:+.4f}")
        else:
            print(f"  WARNING: No baseline found for {key}")
            degradations.append(np.nan)
    
    c4_data['c4_degradation'] = degradations
    
    # Remove experiments without valid degradation calculations
    c4_midtrain = c4_data[c4_data['c4_degradation'].notna()].copy()
    
    if len(c4_midtrain) == 0:
        print("ERROR: No valid degradation calculations")
        return analysis_df, None
    
    print(f"\nAnalyzing {len(c4_midtrain)} experiments with valid C4 degradation:")
    
    # Show degradation by experiment
    print(f"\nC4 Degradation Summary:")
    print(f"{'SFT Dataset':12} {'Midtrain':15} {'C4→Mid Dist':>11} {'C4 Degradation':>14} {'SFT Improvement':>15}")
    print("-" * 75)
    
    for _, row in c4_midtrain.sort_values('c4_degradation').iterrows():
        print(f"{row['sft_dataset']:12} {row['midtrain_dataset'][:15]:15} "
              f"{row['c4_to_midtrain_dist']:11.3f} {row['c4_degradation']:14.4f} "
              f"{row['relative_improvement']:15.3f}")
    
    # Test catastrophic forgetting hypothesis
    from scipy.stats import pearsonr
    
    # Key test: Do larger C4→Midtrain gaps cause more C4 degradation?
    corr_gap_degrade, p_gap_degrade = pearsonr(c4_midtrain['c4_to_midtrain_dist'], c4_midtrain['c4_degradation'])
    
    print(f"\n" + "="*30)
    print("CATASTROPHIC FORGETTING HYPOTHESIS TEST")
    print("="*30)
    
    print(f"H1: Larger C4→Midtrain gaps → More C4 degradation")
    print(f"    Correlation: r = {corr_gap_degrade:+.3f}, p = {p_gap_degrade:.4f}")
    
    if corr_gap_degrade > 0.2 and p_gap_degrade < 0.05:
        print(f"    ✓ CATASTROPHIC FORGETTING CONFIRMED")
        print(f"      → Larger gaps cause more forgetting of general capabilities")
        forgetting_evidence = "CONFIRMED"
    elif corr_gap_degrade > 0.1 and p_gap_degrade < 0.1:
        print(f"    ? WEAK EVIDENCE for catastrophic forgetting")
        forgetting_evidence = "WEAK"
    else:
        print(f"    ✗ NO EVIDENCE of catastrophic forgetting")
        print(f"      → Large gaps don't hurt general capabilities")
        forgetting_evidence = "NONE"
    
    # Test trade-off hypothesis: SFT improvement vs C4 degradation
    corr_improvement_degrade, p_improvement_degrade = pearsonr(c4_midtrain['relative_improvement'], c4_midtrain['c4_degradation'])
    
    print(f"\nH2: SFT improvement comes at cost of C4 degradation")
    print(f"    Correlation: r = {corr_improvement_degrade:+.3f}, p = {p_improvement_degrade:.4f}")
    
    if corr_improvement_degrade > 0.2 and p_improvement_degrade < 0.05:
        print(f"    ✓ TRADE-OFF CONFIRMED")
        print(f"      → Better SFT performance causes more general forgetting")
        tradeoff_evidence = "CONFIRMED"
    elif corr_improvement_degrade < -0.2 and p_improvement_degrade < 0.05:
        print(f"    ✓ SYNERGY FOUND")
        print(f"      → Better SFT performance with less general forgetting")
        tradeoff_evidence = "SYNERGY"
    else:
        print(f"    ✗ NO CLEAR TRADE-OFF")
        print(f"      → SFT improvement independent of general forgetting")
        tradeoff_evidence = "INDEPENDENT"
    
    # Efficiency analysis: SFT improvement per unit of C4 degradation
    # Handle negative degradations (improvements) by using absolute value + small constant
    c4_midtrain['efficiency'] = c4_midtrain['relative_improvement'] / (abs(c4_midtrain['c4_degradation']) + 0.001)
    
    print(f"\n" + "="*30)
    print("MIDTRAINING EFFICIENCY ANALYSIS")
    print("="*30)
    print("Efficiency = SFT Improvement / |C4 Degradation|")
    print("(Higher is better: more task improvement per unit of general capability change)")
    
    efficiency_sorted = c4_midtrain.sort_values('efficiency', ascending=False)
    print(f"\nMost Efficient Midtraining Choices:")
    for _, row in efficiency_sorted.head(5).iterrows():
        print(f"  {row['midtrain_dataset']:15} → {row['sft_dataset']:10} | "
              f"Efficiency: {row['efficiency']:6.2f} | "
              f"SFT: {row['relative_improvement']:+.1%} | "
              f"C4 change: {row['c4_degradation']:+.4f}")
    
    print(f"\nLeast Efficient Midtraining Choices:")
    for _, row in efficiency_sorted.tail(3).iterrows():
        print(f"  {row['midtrain_dataset']:15} → {row['sft_dataset']:10} | "
              f"Efficiency: {row['efficiency']:6.2f} | "
              f"SFT: {row['relative_improvement']:+.1%} | "
              f"C4 change: {row['c4_degradation']:+.4f}")
    
    # Check if efficiency correlates with gap size
    corr_gap_efficiency, p_gap_efficiency = pearsonr(c4_midtrain['c4_to_midtrain_dist'], c4_midtrain['efficiency'])
    
    print(f"\nDoes gap size predict efficiency?")
    print(f"  C4→Midtrain gap vs Efficiency: r = {corr_gap_efficiency:+.3f}, p = {p_gap_efficiency:.4f}")
    
    if corr_gap_efficiency > 0.2 and p_gap_efficiency < 0.05:
        print(f"  ✓ Larger gaps are MORE efficient")
    elif corr_gap_efficiency < -0.2 and p_gap_efficiency < 0.05:
        print(f"  ✓ Smaller gaps are MORE efficient")
    else:
        print(f"  ✗ Gap size doesn't predict efficiency")
    
    # Summary insights
    print(f"\n" + "="*30)
    print("CATASTROPHIC FORGETTING INSIGHTS")
    print("="*30)
    
    # Reconcile with earlier findings
    print(f"Reconciling with earlier findings:")
    print(f"  Earlier: Larger C4→Midtrain gaps → Better SFT performance")
    print(f"  Now: Larger C4→Midtrain gaps → {forgetting_evidence.lower()} general forgetting")
    
    if forgetting_evidence == "CONFIRMED" and tradeoff_evidence == "CONFIRMED":
        print(f"\n✓ CLASSIC TRADE-OFF:")
        print(f"  → Larger gaps help task performance but hurt general capabilities")
        print(f"  → Catastrophic forgetting theory validated")
        print(f"  → Recommendation: Optimize for efficiency, not just task performance")
        
    elif forgetting_evidence == "NONE" and tradeoff_evidence in ["INDEPENDENT", "SYNERGY"]:
        print(f"\n✓ EFFICIENT SPECIALIZATION:")
        print(f"  → Midtraining can improve task performance without hurting general capabilities")
        print(f"  → Catastrophic forgetting not a major concern at this scale")
        print(f"  → Recommendation: Focus on maximizing task improvement")
        
    else:
        print(f"\n? MIXED RESULTS:")
        print(f"  → Some evidence for both benefits and costs")
        print(f"  → May depend on specific dataset combinations")
        print(f"  → Recommendation: Case-by-case optimization")
    
    # Add to main dataframe for return
    analysis_df = analysis_df.merge(
        c4_midtrain[['model_size', 'sft_dataset', 'midtrain_dataset', 'c4_degradation', 'efficiency']],
        on=['model_size', 'sft_dataset', 'midtrain_dataset'],
        how='left'
    )
    
    return analysis_df, {
        'forgetting_evidence': forgetting_evidence,
        'tradeoff_evidence': tradeoff_evidence,
        'corr_gap_degrade': corr_gap_degrade,
        'p_gap_degrade': p_gap_degrade,
        'corr_improvement_degrade': corr_improvement_degrade,
        'p_improvement_degrade': p_improvement_degrade,
        'corr_gap_efficiency': corr_gap_efficiency,
        'p_gap_efficiency': p_gap_efficiency
    }
    
    # Remove baseline experiments and NaN degradations
    c4_midtrain = c4_data[(c4_data['midtrain_dataset'] != 'C4') & (c4_data['c4_degradation'].notna())].copy()
    
    if len(c4_midtrain) == 0:
        print("ERROR: No valid degradation calculations")
        return analysis_df
    
    print(f"\nAnalyzing {len(c4_midtrain)} midtraining experiments:")
    
    # Show degradation by experiment
    print(f"\nC4 Degradation by Experiment:")
    print(f"{'SFT Dataset':12} {'Midtrain':15} {'C4→Mid Dist':>11} {'C4 Degradation':>14} {'SFT Improvement':>15}")
    print("-" * 75)
    
    for _, row in c4_midtrain.sort_values('c4_degradation').iterrows():
        print(f"{row['sft_dataset']:12} {row['midtrain_dataset'][:15]:15} "
              f"{row['c4_to_midtrain_dist']:11.3f} {row['c4_degradation']:14.3f} "
              f"{row['relative_improvement']:15.3f}")
    
    # Test catastrophic forgetting hypothesis
    from scipy.stats import pearsonr
    
    # Key test: Do larger C4→Midtrain gaps cause more C4 degradation?
    corr_gap_degrade, p_gap_degrade = pearsonr(c4_midtrain['c4_to_midtrain_dist'], c4_midtrain['c4_degradation'])
    
    print(f"\n" + "="*30)
    print("CATASTROPHIC FORGETTING HYPOTHESIS TEST")
    print("="*30)
    
    print(f"H1: Larger C4→Midtrain gaps → More C4 degradation")
    print(f"    Correlation: r = {corr_gap_degrade:+.3f}, p = {p_gap_degrade:.4f}")
    
    if corr_gap_degrade > 0.2 and p_gap_degrade < 0.05:
        print(f"    ✓ CATASTROPHIC FORGETTING CONFIRMED")
        print(f"      → Larger gaps cause more forgetting of general capabilities")
        forgetting_evidence = "CONFIRMED"
    elif corr_gap_degrade > 0.1 and p_gap_degrade < 0.1:
        print(f"    ? WEAK EVIDENCE for catastrophic forgetting")
        forgetting_evidence = "WEAK"
    else:
        print(f"    ✗ NO EVIDENCE of catastrophic forgetting")
        print(f"      → Large gaps don't hurt general capabilities")
        forgetting_evidence = "NONE"
    
    # Test trade-off hypothesis: SFT improvement vs C4 degradation
    corr_improvement_degrade, p_improvement_degrade = pearsonr(c4_midtrain['relative_improvement'], c4_midtrain['c4_degradation'])
    
    print(f"\nH2: SFT improvement comes at cost of C4 degradation")
    print(f"    Correlation: r = {corr_improvement_degrade:+.3f}, p = {p_improvement_degrade:.4f}")
    
    if corr_improvement_degrade > 0.2 and p_improvement_degrade < 0.05:
        print(f"    ✓ TRADE-OFF CONFIRMED")
        print(f"      → Better SFT performance causes more general forgetting")
        tradeoff_evidence = "CONFIRMED"
    elif corr_improvement_degrade < -0.2 and p_improvement_degrade < 0.05:
        print(f"    ✓ SYNERGY FOUND")
        print(f"      → Better SFT performance with less general forgetting")
        tradeoff_evidence = "SYNERGY"
    else:
        print(f"    ✗ NO CLEAR TRADE-OFF")
        print(f"      → SFT improvement independent of general forgetting")
        tradeoff_evidence = "INDEPENDENT"
    
    # Efficiency analysis: SFT improvement per unit of C4 degradation
    c4_midtrain['efficiency'] = c4_midtrain['relative_improvement'] / (c4_midtrain['c4_degradation'] + 0.001)  # Avoid division by zero
    
    print(f"\n" + "="*30)
    print("MIDTRAINING EFFICIENCY ANALYSIS")
    print("="*30)
    print("Efficiency = SFT Improvement / C4 Degradation")
    print("(Higher is better: more task improvement per unit of general forgetting)")
    
    efficiency_sorted = c4_midtrain.sort_values('efficiency', ascending=False)
    print(f"\nMost Efficient Midtraining Choices:")
    for _, row in efficiency_sorted.head(5).iterrows():
        print(f"  {row['midtrain_dataset']:15} → {row['sft_dataset']:10} | "
              f"Efficiency: {row['efficiency']:6.2f} | "
              f"SFT: {row['relative_improvement']:+.1%} | "
              f"C4 cost: {row['c4_degradation']:+.3f}")
    
    print(f"\nLeast Efficient Midtraining Choices:")
    for _, row in efficiency_sorted.tail(3).iterrows():
        print(f"  {row['midtrain_dataset']:15} → {row['sft_dataset']:10} | "
              f"Efficiency: {row['efficiency']:6.2f} | "
              f"SFT: {row['relative_improvement']:+.1%} | "
              f"C4 cost: {row['c4_degradation']:+.3f}")
    
    # Check if efficiency correlates with gap size
    corr_gap_efficiency, p_gap_efficiency = pearsonr(c4_midtrain['c4_to_midtrain_dist'], c4_midtrain['efficiency'])
    
    print(f"\nDoes gap size predict efficiency?")
    print(f"  C4→Midtrain gap vs Efficiency: r = {corr_gap_efficiency:+.3f}, p = {p_gap_efficiency:.4f}")
    
    if corr_gap_efficiency > 0.2 and p_gap_efficiency < 0.05:
        print(f"  ✓ Larger gaps are MORE efficient")
    elif corr_gap_efficiency < -0.2 and p_gap_efficiency < 0.05:
        print(f"  ✓ Smaller gaps are MORE efficient")
    else:
        print(f"  ✗ Gap size doesn't predict efficiency")
    
    # Summary insights
    print(f"\n" + "="*30)
    print("CATASTROPHIC FORGETTING INSIGHTS")
    print("="*30)
    
    # Reconcile with earlier findings
    print(f"Reconciling with earlier findings:")
    print(f"  Earlier: Larger C4→Midtrain gaps → Better SFT performance")
    print(f"  Now: Larger C4→Midtrain gaps → {forgetting_evidence.lower()} general forgetting")
    
    if forgetting_evidence == "CONFIRMED" and tradeoff_evidence == "CONFIRMED":
        print(f"\n✓ CLASSIC TRADE-OFF:")
        print(f"  → Larger gaps help task performance but hurt general capabilities")
        print(f"  → Catastrophic forgetting theory validated")
        print(f"  → Recommendation: Optimize for efficiency, not just task performance")
        
    elif forgetting_evidence == "NONE" and tradeoff_evidence in ["INDEPENDENT", "SYNERGY"]:
        print(f"\n✓ EFFICIENT SPECIALIZATION:")
        print(f"  → Midtraining can improve task performance without hurting general capabilities")
        print(f"  → Catastrophic forgetting not a major concern at this scale")
        print(f"  → Recommendation: Focus on maximizing task improvement")
        
    else:
        print(f"\n? MIXED RESULTS:")
        print(f"  → Some evidence for both benefits and costs")
        print(f"  → May depend on specific dataset combinations")
        print(f"  → Recommendation: Case-by-case optimization")
    
    # Add to dataframe for return
    analysis_df = analysis_df.merge(
        c4_midtrain[['model_size', 'sft_dataset', 'midtrain_dataset', 'c4_degradation', 'efficiency']],
        on=['model_size', 'sft_dataset', 'midtrain_dataset'],
        how='left'
    )
    
    return analysis_df, {
        'forgetting_evidence': forgetting_evidence,
        'tradeoff_evidence': tradeoff_evidence,
        'corr_gap_degrade': corr_gap_degrade,
        'p_gap_degrade': p_gap_degrade,
        'corr_improvement_degrade': corr_improvement_degrade,
        'p_improvement_degrade': p_improvement_degrade,
        'corr_gap_efficiency': corr_gap_efficiency,
        'p_gap_efficiency': p_gap_efficiency
    }

def analyze_within_sft_dataset_groups(analysis_df):
    """
    Control for task difficulty by comparing bridge effects WITHIN each SFT dataset.
    This removes the C4→SFT distance confound.
    """
    print(f"\n" + "="*50)
    print("WITHIN-SFT DATASET ANALYSIS")
    print("="*50)
    print("Controlling for task difficulty by comparing within each SFT dataset...")
    print("This removes the C4→SFT distance confound")
    
    # Group by SFT dataset
    sft_groups = analysis_df.groupby('sft_dataset')
    
    within_group_results = []
    
    print(f"\nAnalyzing {len(sft_groups)} SFT dataset groups:")
    
    for sft_dataset, group in sft_groups:
        if len(group) < 2:  # Need at least 2 experiments to compare
            print(f"\n{sft_dataset}: SKIPPED (only {len(group)} experiment)")
            continue
            
        print(f"\n{sft_dataset.upper()} (n={len(group)}):")
        print(f"  Task difficulty (C4→SFT): {group['c4_to_sft_direct_dist'].iloc[0]:.3f} (constant)")
        
        # Show all experiments for this SFT dataset
        group_sorted = group.sort_values('relative_improvement', ascending=False)
        print(f"  Experiments ranked by improvement:")
        for _, row in group_sorted.iterrows():
            print(f"    {row['midtrain_dataset']:15} | Bridge: {row['bridge_length_max']:.3f} | "
                  f"C4→Mid: {row['c4_to_midtrain_dist']:.3f} | Improvement: {row['relative_improvement']:+.3f} ({row['relative_improvement']*100:+.1f}%)")
        
        # Within-group correlations (if enough samples)
        if len(group) >= 3:
            from scipy.stats import pearsonr
            
            corr_bridge, p_bridge = pearsonr(group['bridge_length_max'], group['relative_improvement'])
            corr_c4_mid, p_c4_mid = pearsonr(group['c4_to_midtrain_dist'], group['relative_improvement'])
            corr_mid_sft, p_mid_sft = pearsonr(group['midtrain_to_sft_dist'], group['relative_improvement'])
            
            print(f"  Within-group correlations:")
            print(f"    Bridge length:     r = {corr_bridge:+.3f} (p = {p_bridge:.3f})")
            print(f"    C4→Midtrain dist:  r = {corr_c4_mid:+.3f} (p = {p_c4_mid:.3f})")
            print(f"    Midtrain→SFT dist: r = {corr_mid_sft:+.3f} (p = {p_mid_sft:.3f})")
            
            # Store results for meta-analysis
            within_group_results.append({
                'sft_dataset': sft_dataset,
                'n_experiments': len(group),
                'task_difficulty': group['c4_to_sft_direct_dist'].iloc[0],
                'corr_bridge': corr_bridge,
                'p_bridge': p_bridge,
                'corr_c4_mid': corr_c4_mid,
                'p_c4_mid': p_c4_mid,
                'corr_mid_sft': corr_mid_sft,
                'p_mid_sft': p_mid_sft,
                'improvement_range': group['relative_improvement'].max() - group['relative_improvement'].min(),
                'best_midtrain': group_sorted.iloc[0]['midtrain_dataset'],
                'best_improvement': group_sorted.iloc[0]['relative_improvement']
            })
            
        else:
            print(f"  Within-group correlation: N/A (need ≥3 experiments)")
            
        # Identify best midtraining choice for this SFT dataset
        best_row = group_sorted.iloc[0]
        worst_row = group_sorted.iloc[-1]
        improvement_gap = best_row['relative_improvement'] - worst_row['relative_improvement']
        
        print(f"  Best choice: {best_row['midtrain_dataset']} ({best_row['relative_improvement']:+.1%})")
        print(f"  Worst choice: {worst_row['midtrain_dataset']} ({worst_row['relative_improvement']:+.1%})")
        print(f"  Choice matters: {improvement_gap:.1%} gap")
        
        if improvement_gap > 0.02:  # 2 percentage points
            print(f"  → SUBSTANTIAL midtraining choice effect for {sft_dataset}")
        elif improvement_gap > 0.01:  # 1 percentage point
            print(f"  → MODERATE midtraining choice effect for {sft_dataset}")
        else:
            print(f"  → MINIMAL midtraining choice effect for {sft_dataset}")
    
    # Meta-analysis across groups
    if len(within_group_results) >= 2:
        print(f"\n" + "="*30)
        print("META-ANALYSIS ACROSS SFT DATASETS")
        print("="*30)
        
        results_df = pd.DataFrame(within_group_results)
        
        # Average correlations across groups (simple meta-analysis)
        avg_corr_bridge = results_df['corr_bridge'].mean()
        avg_corr_c4_mid = results_df['corr_c4_mid'].mean()
        avg_corr_mid_sft = results_df['corr_mid_sft'].mean()
        
        print(f"Average within-group correlations:")
        print(f"  Bridge length:     r = {avg_corr_bridge:+.3f}")
        print(f"  C4→Midtrain dist:  r = {avg_corr_c4_mid:+.3f}")
        print(f"  Midtrain→SFT dist: r = {avg_corr_mid_sft:+.3f}")
        
        # Count significant correlations
        sig_bridge = (results_df['p_bridge'] < 0.05).sum()
        sig_c4_mid = (results_df['p_c4_mid'] < 0.05).sum()
        sig_mid_sft = (results_df['p_mid_sft'] < 0.05).sum()
        
        print(f"\nSignificant correlations (p < 0.05):")
        print(f"  Bridge length: {sig_bridge}/{len(results_df)} groups")
        print(f"  C4→Midtrain: {sig_c4_mid}/{len(results_df)} groups")
        print(f"  Midtrain→SFT: {sig_mid_sft}/{len(results_df)} groups")
        
        # Effect size analysis
        print(f"\nEffect sizes by task difficulty:")
        results_df_sorted = results_df.sort_values('task_difficulty')
        for _, row in results_df_sorted.iterrows():
            print(f"  {row['sft_dataset']:15} (difficulty: {row['task_difficulty']:.3f}) | "
                  f"Choice gap: {row['improvement_range']:+.1%} | "
                  f"Best: {row['best_midtrain']}")
        
        # Key insights
        print(f"\n" + "="*30)
        print("KEY INSIGHTS FROM WITHIN-GROUP ANALYSIS")
        print("="*30)
        
        if abs(avg_corr_bridge) > 0.2:
            print(f"✓ Bridge length still matters WITHIN task difficulty groups")
            print(f"  → Bridge optimization is genuinely useful")
            if avg_corr_bridge > 0:
                print(f"  → Longer bridges appear better (controlling for difficulty)")
            else:
                print(f"  → Shorter bridges appear better (controlling for difficulty)")
        else:
            print(f"✗ Bridge length has minimal effect within task difficulty groups")
            print(f"  → Bridge optimization may not be worthwhile")
        
        if abs(avg_corr_c4_mid) > abs(avg_corr_mid_sft) + 0.1:
            print(f"✓ C4→Midtrain distance still drives results within groups")
            print(f"  → Even controlling for task difficulty, C4 distance matters")
        elif abs(avg_corr_mid_sft) > abs(avg_corr_c4_mid) + 0.1:
            print(f"✓ Midtrain→SFT similarity is the key factor within groups")
            print(f"  → Domain matching quality drives results")
        else:
            print(f"? Mixed evidence within groups")
        
        # Practical recommendations
        large_choice_effects = results_df[results_df['improvement_range'] > 0.02]
        if len(large_choice_effects) > 0:
            print(f"\nTasks where midtraining choice matters most:")
            for _, row in large_choice_effects.sort_values('improvement_range', ascending=False).iterrows():
                print(f"  {row['sft_dataset']}: {row['improvement_range']:+.1%} gap (use {row['best_midtrain']})")
        
        return results_df
    else:
        print(f"\nInsufficient data for meta-analysis ({len(within_group_results)} groups with ≥3 experiments)")
        return pd.DataFrame(within_group_results)

def analyze_midtraining_bridge_effect(csv_file_path, embedding_csv, ngram_csv, similarity_mode='combined'):
    """Main analysis function"""
    
    # Load similarity matrices
    print("Loading similarity matrices...")
    embedding_df, ngram_df = load_similarity_matrices(embedding_csv, ngram_csv)
    
    if embedding_df is None or ngram_df is None:
        print("Failed to load similarity matrices. Exiting.")
        return None
        
    print(f"\nUsing similarity mode: {similarity_mode}")
    if similarity_mode == 'semantic_only':
        print("  Using ONLY semantic/embedding similarities")
    elif similarity_mode == 'syntactic_only':
        print("  Using ONLY syntactic/n-gram similarities")
    else:
        print("  Using COMBINED semantic + syntactic similarities (50/50)")
    
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
        
        # Skip experiments involving Movie Reviews (mapped to None)
        if midtrain_mapped is None:
            print(f"  SKIPPING: Midtrain dataset '{row['Pre/midtrain mix']}' excluded from analysis")
            skipped_experiments.append(f"{row['Pre/midtrain mix']} → {row['SFT dataset']} (midtrain excluded)")
            continue
            
        if sft_mapped is None:
            print(f"  SKIPPING: SFT dataset '{row['SFT dataset']}' excluded from analysis")
            skipped_experiments.append(f"{row['Pre/midtrain mix']} → {row['SFT dataset']} (SFT excluded)")
            continue
        
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
        absolute_improvement = baseline_loss - midtrain_loss
        relative_improvement = absolute_improvement / baseline_loss  # Relative improvement as fraction
        
        print(f"  Baseline: {baseline_loss:.4f}, Midtrain: {midtrain_loss:.4f}")
        print(f"  Absolute improvement: {absolute_improvement:.4f}")
        print(f"  Relative improvement: {relative_improvement:.4f} ({relative_improvement*100:.1f}%)")
        
        # Calculate bridge length using different methods
        bridge_total, c4_dist, mid_dist = calculate_bridge_length(
            row['Pre/midtrain mix'], 
            row['SFT dataset'], 
            embedding_df, 
            ngram_df,
            method='total',
            similarity_mode=similarity_mode
        )
        
        bridge_max, _, _ = calculate_bridge_length(
            row['Pre/midtrain mix'], 
            row['SFT dataset'], 
            embedding_df, 
            ngram_df,
            method='max',
            similarity_mode=similarity_mode
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
            'c4_loss_after_ft': row['C4 val loss after FT'] if pd.notna(row['C4 val loss after FT']) else np.nan,
            'absolute_improvement': absolute_improvement,
            'relative_improvement': relative_improvement,  # This is now our main metric
            'bridge_length_total': bridge_total,
            'bridge_length_max': bridge_max,
            'c4_to_midtrain_dist': c4_dist,
            'midtrain_to_sft_dist': mid_dist,
            'c4_to_midtrain_sim': get_combined_similarity('C4', row['Pre/midtrain mix'], embedding_df, ngram_df, similarity_mode=similarity_mode),
            'midtrain_to_sft_sim': get_combined_similarity(row['Pre/midtrain mix'], row['SFT dataset'], embedding_df, ngram_df, similarity_mode=similarity_mode)
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
    
    print(f"\nInitial dataset: {len(analysis_df)} experiments")
    
    print(f"\nAnalyzing {len(analysis_df)} experiments")
    
    # Calculate correlations for different bridge methods using relative improvement
    corr_total, p_total = pearsonr(analysis_df['bridge_length_total'], analysis_df['relative_improvement'])
    corr_max, p_max = pearsonr(analysis_df['bridge_length_max'], analysis_df['relative_improvement'])
    
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
        
        # Calculate correlations for this model size using relative improvement
        model_corr_total, model_p_total = pearsonr(model_data['bridge_length_total'], model_data['relative_improvement'])
        model_corr_max, model_p_max = pearsonr(model_data['bridge_length_max'], model_data['relative_improvement'])
        
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
            print(f"  {row['midtrain_dataset']:15} → {row['sft_dataset']:10} | Max: {row['bridge_length_max']:.3f} | Rel. Improvement: {row['relative_improvement']:.3f} ({row['relative_improvement']*100:.1f}%)")
    
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


        # BRIDGE COMPONENT ANALYSIS - Test for selection bias
    print(f"\n" + "="*50)
    print("BRIDGE COMPONENT ANALYSIS")
    print("="*50)
    print("Testing if results are driven by selection bias...")
    print("(Midtraining is only applied to datasets far from C4)")
    
    # Correlations with relative improvement
    
    corr_c4_mid_dist, p_c4_mid_dist = pearsonr(analysis_df['c4_to_midtrain_dist'], analysis_df['relative_improvement'])
    corr_mid_sft_dist, p_mid_sft_dist = pearsonr(analysis_df['midtrain_to_sft_dist'], analysis_df['relative_improvement'])
    
    # Also check similarities (inverse relationship expected)
    corr_c4_mid_sim, p_c4_mid_sim = pearsonr(analysis_df['c4_to_midtrain_sim'], analysis_df['relative_improvement'])
    corr_mid_sft_sim, p_mid_sft_sim = pearsonr(analysis_df['midtrain_to_sft_sim'], analysis_df['relative_improvement'])
    
    print(f"\nBridge Component Correlations with Relative Improvement:")
    print(f"  C4 → Midtrain DISTANCE:    r = {corr_c4_mid_dist:.4f}, p = {p_c4_mid_dist:.4f}")
    print(f"  Midtrain → SFT DISTANCE:   r = {corr_mid_sft_dist:.4f}, p = {p_mid_sft_dist:.4f}")
    print(f"  C4 → Midtrain SIMILARITY:  r = {corr_c4_mid_sim:.4f}, p = {p_c4_mid_sim:.4f}")
    print(f"  Midtrain → SFT SIMILARITY: r = {corr_mid_sft_sim:.4f}, p = {p_mid_sft_sim:.4f}")
    
    # Interpretation guide
    print(f"\nInterpretation:")
    print(f"  If SELECTION BIAS is the main driver:")
    print(f"    • C4→Midtrain distance should correlate POSITIVELY with improvement")
    print(f"    • (Datasets far from C4 need midtraining AND benefit more)")
    print(f"    • Midtrain→SFT distance correlation should be weaker/different")
    print(f"")
    print(f"  If BRIDGE QUALITY matters independently:")
    print(f"    • Both components should show similar correlation patterns")
    print(f"    • Midtrain→SFT similarity should matter even controlling for C4 distance")
    
    # Identify the stronger driver
    abs_c4_mid = abs(corr_c4_mid_dist)
    abs_mid_sft = abs(corr_mid_sft_dist)
    
    if abs_c4_mid > abs_mid_sft + 0.1:  # Meaningful difference
        print(f"\n➤ C4→Midtrain distance is the STRONGER predictor")
        print(f"  This suggests SELECTION BIAS: datasets far from C4 benefit more from midtraining")
        bias_evidence = "C4_DISTANCE"
    elif abs_mid_sft > abs_c4_mid + 0.1:
        print(f"\n➤ Midtrain→SFT distance is the STRONGER predictor")
        print(f"  This suggests BRIDGE QUALITY matters beyond just C4 distance")
        bias_evidence = "BRIDGE_QUALITY"
    else:
        print(f"\n➤ Both components have similar predictive power")
        print(f"  Mixed evidence - both selection bias and bridge quality may matter")
        bias_evidence = "MIXED"
    
    # Check significance levels
    significant_c4_mid = p_c4_mid_dist < 0.05
    significant_mid_sft = p_mid_sft_dist < 0.05
    
    if significant_c4_mid and not significant_mid_sft:
        print(f"  STATISTICAL SUPPORT for selection bias (only C4→Mid significant)")
    elif significant_mid_sft and not significant_c4_mid:
        print(f"  STATISTICAL SUPPORT for bridge quality (only Mid→SFT significant)")
    elif significant_c4_mid and significant_mid_sft:
        print(f"  BOTH components statistically significant")
    else:
        print(f"  NEITHER component reaches significance")
    
    # Additional insight: check if C4→SFT direct distance predicts improvement
    print(f"\n" + "="*30)
    print("DIRECT C4→SFT BASELINE CHECK")
    print("="*30)
    
    # Calculate direct C4 to SFT distances for comparison
    c4_to_sft_distances = []
    for _, row in analysis_df.iterrows():
        c4_sft_sim = get_combined_similarity('C4', row['sft_dataset'], embedding_df, ngram_df, similarity_mode=similarity_mode)
        c4_sft_dist = 1 - c4_sft_sim
        c4_to_sft_distances.append(c4_sft_dist)
    
    analysis_df['c4_to_sft_direct_dist'] = c4_to_sft_distances
    
    corr_c4_sft_direct, p_c4_sft_direct = pearsonr(analysis_df['c4_to_sft_direct_dist'], analysis_df['relative_improvement'])
    
    print(f"Direct C4 → SFT distance correlation: r = {corr_c4_sft_direct:.4f}, p = {p_c4_sft_direct:.4f}")
    print(f"This shows how much the 'task difficulty' alone predicts improvement")
    
    if abs(corr_c4_sft_direct) > max(abs_c4_mid, abs_mid_sft):
        print(f"➤ DIRECT C4→SFT distance is the strongest predictor!")
        print(f"  This strongly suggests SELECTION BIAS: harder tasks benefit more from midtraining")
        bias_evidence = "DIRECT_DISTANCE"
    
    # Show the raw data for manual inspection
    print(f"\n" + "="*30)
    print("RAW COMPONENT DATA")
    print("="*30)
    print("Manual inspection of which component drives the effect...")
    
    analysis_df_sorted = analysis_df.sort_values('relative_improvement', ascending=False)

    within_group_results = analyze_within_sft_dataset_groups(analysis_df)
    if 'C4 val loss after FT' in df.columns:
        analysis_df, forgetting_results = analyze_catastrophic_forgetting(analysis_df, df)  # Pass original df
    else:
        forgetting_results = None
        print("WARNING: No C4 loss data found, skipping catastrophic forgetting analysis")
    
    #breakpoint()
    
    print(f"{'Midtrain→SFT':15} {'C4→Mid Dist':>10} {'Mid→SFT Dist':>12} {'C4→SFT Direct':>13} {'Total Bridge':>12} {'Improvement':>11}")
    print("-" * 85)
    for _, row in analysis_df_sorted.iterrows():
        print(f"{row['midtrain_dataset'][:15]:15} "
              f"{row['c4_to_midtrain_dist']:10.3f} "
              f"{row['midtrain_to_sft_dist']:12.3f} "
              f"{row['c4_to_sft_direct_dist']:13.3f} "
              f"{row['bridge_length_total']:12.3f} "
              f"{row['relative_improvement']:11.3f}")
    
    # Store component analysis results for later use
    component_analysis = {
        'corr_c4_mid_dist': corr_c4_mid_dist,
        'p_c4_mid_dist': p_c4_mid_dist,
        'corr_mid_sft_dist': corr_mid_sft_dist,
        'p_mid_sft_dist': p_mid_sft_dist,
        'corr_c4_sft_direct': corr_c4_sft_direct,
        'p_c4_sft_direct': p_c4_sft_direct,
        'bias_evidence': bias_evidence
    }
    
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
    axes[0,0].scatter(analysis_df['bridge_length_total'], analysis_df['relative_improvement'], alpha=0.6)
    axes[0,0].set_xlabel('Total Bridge Length')
    axes[0,0].set_ylabel('Relative Improvement')
    axes[0,0].set_title(f'Total Bridge: r = {corr_total:.3f}, p = {p_total:.3f}')
    z = np.polyfit(analysis_df['bridge_length_total'], analysis_df['relative_improvement'], 1)
    p_fit = np.poly1d(z)
    axes[0,0].plot(analysis_df['bridge_length_total'], p_fit(analysis_df['bridge_length_total']), "r--", alpha=0.8)
    
    # Max bridge length correlation
    axes[0,1].scatter(analysis_df['bridge_length_max'], analysis_df['relative_improvement'], alpha=0.6, color='orange')
    axes[0,1].set_xlabel('Max Bridge Length')
    axes[0,1].set_ylabel('Relative Improvement')
    axes[0,1].set_title(f'Max Bridge: r = {corr_max:.3f}, p = {p_max:.3f}')
    z = np.polyfit(analysis_df['bridge_length_max'], analysis_df['relative_improvement'], 1)
    p_fit = np.poly1d(z)
    axes[0,1].plot(analysis_df['bridge_length_max'], p_fit(analysis_df['bridge_length_max']), "r--", alpha=0.8)
    
    # Component correlations
    corr_c4_mid, p_c4_mid = pearsonr(analysis_df['c4_to_midtrain_sim'], analysis_df['relative_improvement'])
    corr_mid_sft, p_mid_sft = pearsonr(analysis_df['midtrain_to_sft_sim'], analysis_df['relative_improvement'])
    
    axes[0,2].scatter(analysis_df['c4_to_midtrain_sim'], analysis_df['relative_improvement'], alpha=0.6, color='green')
    axes[0,2].set_xlabel('C4→Midtrain Similarity')
    axes[0,2].set_ylabel('Relative Improvement')
    axes[0,2].set_title(f'C4→Midtrain: r = {corr_c4_mid:.3f}')
    
    axes[1,0].scatter(analysis_df['midtrain_to_sft_sim'], analysis_df['relative_improvement'], alpha=0.6, color='purple')
    axes[1,0].set_xlabel('Midtrain→SFT Similarity')
    axes[1,0].set_ylabel('Relative Improvement')
    axes[1,0].set_title(f'Midtrain→SFT: r = {corr_mid_sft:.3f}')
    
    # Distribution of improvements
    axes[1,1].hist(analysis_df['relative_improvement'], bins=20, alpha=0.7, edgecolor='black')
    axes[1,1].set_xlabel('Relative Improvement')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Relative Improvements')
    axes[1,1].axvline(analysis_df['relative_improvement'].mean(), color='red', linestyle='--', label='Mean')
    axes[1,1].legend()
    
    # Bridge length comparison
    axes[1,2].scatter(analysis_df['bridge_length_total'], analysis_df['bridge_length_max'], alpha=0.6)
    axes[1,2].set_xlabel('Total Bridge Length')
    axes[1,2].set_ylabel('Max Bridge Length')
    axes[1,2].set_title('Total vs Max Bridge Length')
    
    plt.tight_layout()
    plt.savefig('midtraining_bridge_analysis_visualization.png')

    # Save results
    analysis_df.to_csv('midtraining_bridge_analysis_results.csv', index=False)
    print(f"\nDetailed results saved to 'midtraining_bridge_analysis_results.csv'")
    
            # Save results
    analysis_df.to_csv('midtraining_bridge_analysis_results.csv', index=False)
    print(f"\nDetailed results saved to 'midtraining_bridge_analysis_results.csv'")
    
    # Add domain matching analysis
    analysis_df = analyze_domain_matching_effect(analysis_df)
    
    return analysis_df, correlation, p_value, best_method, model_results, component_analysis
    
def analyze_threshold_effects(analysis_df):
    """Test for threshold effects in similarity vs improvement"""
    
    print(f"\n" + "="*50)
    print("THRESHOLD EFFECT ANALYSIS")
    print("="*50)
    
    # Test different similarity thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        # High similarity group
        high_sim = analysis_df[
            (analysis_df['c4_to_midtrain_sim'] > threshold) | 
            (analysis_df['midtrain_to_sft_sim'] > threshold)
        ]
        
        # Low similarity group  
        low_sim = analysis_df[
            (analysis_df['c4_to_midtrain_sim'] <= threshold) & 
            (analysis_df['midtrain_to_sft_sim'] <= threshold)
        ]
        
        if len(high_sim) > 0 and len(low_sim) > 0:
            from scipy.stats import ttest_ind
            t_stat, p_value = ttest_ind(high_sim['improvement'], low_sim['improvement'])
            
            print(f"\nThreshold = {threshold}:")
            print(f"  High similarity (>{threshold}): mean = {high_sim['improvement'].mean():.4f} (n={len(high_sim)})")
            print(f"  Low similarity (≤{threshold}): mean = {low_sim['improvement'].mean():.4f} (n={len(low_sim)})")
            print(f"  Difference: {high_sim['improvement'].mean() - low_sim['improvement'].mean():.4f}")
            print(f"  T-test: p = {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  ✓ SIGNIFICANT threshold effect!")
    
    # Test max similarity threshold (best single connection)
    print(f"\n" + "="*30)
    print("MAX SIMILARITY THRESHOLD")
    print("="*30)
    
    # Calculate max similarity for each experiment
    analysis_df['max_similarity'] = analysis_df[['c4_to_midtrain_sim', 'midtrain_to_sft_sim']].max(axis=1)
    
    for threshold in [0.7, 0.8, 0.9]:
        high_max = analysis_df[analysis_df['max_similarity'] > threshold]
        low_max = analysis_df[analysis_df['max_similarity'] <= threshold]
        
        if len(high_max) > 0 and len(low_max) > 0:
            t_stat, p_value = ttest_ind(high_max['improvement'], low_max['improvement'])
            
            print(f"\nMax similarity > {threshold}:")
            print(f"  High: {high_max['improvement'].mean():.4f} (n={len(high_max)})")
            print(f"  Low: {low_max['improvement'].mean():.4f} (n={len(low_max)})")
            print(f"  T-test: p = {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  ✓ SIGNIFICANT threshold at {threshold}!")
                
                # Show what's above threshold
                print(f"  Above threshold experiments:")
                for _, row in high_max.sort_values('improvement', ascending=False).iterrows():
                    print(f"    {row['midtrain_dataset']:15} → {row['sft_dataset']:10} | Max sim: {row['max_similarity']:.3f} | Improvement: {row['improvement']:.4f}")
    
    return analysis_df
    
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
        print(f"  Mean relative improvement: {group['relative_improvement'].mean():.4f} ({group['relative_improvement'].mean()*100:.1f}%)")
        print(f"  Std relative improvement: {group['relative_improvement'].std():.4f}")
        print(f"  Best relative improvement: {group['relative_improvement'].max():.4f} ({group['relative_improvement'].max()*100:.1f}%)")
        print(f"  Worst relative improvement: {group['relative_improvement'].min():.4f} ({group['relative_improvement'].min()*100:.1f}%)")
        
        # Show all experiments in this category
        for _, row in group.sort_values('relative_improvement', ascending=False).iterrows():
            print(f"    {row['midtrain_dataset']:15} → {row['sft_dataset']:10} | {row['model_size']} | Rel. Improvement: {row['relative_improvement']:.4f} ({row['relative_improvement']*100:.1f}%)")
    
    # Statistical comparison
    print(f"\n" + "="*30)
    print("STATISTICAL COMPARISONS")
    print("="*30)
    
    # Perfect matches vs others
    perfect_matches = analysis_df[analysis_df['bridge_type'] == 'perfect_match']
    others = analysis_df[analysis_df['bridge_type'] != 'perfect_match']
    
    if len(perfect_matches) > 0 and len(others) > 0:
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(perfect_matches['relative_improvement'], others['relative_improvement'])
        
        print(f"Perfect domain matches vs Others:")
        print(f"  Perfect matches mean: {perfect_matches['relative_improvement'].mean():.4f} ({perfect_matches['relative_improvement'].mean()*100:.1f}%) (n={len(perfect_matches)})")
        print(f"  Perfect matches std: {perfect_matches['relative_improvement'].std():.4f}")
        print(f"  Others mean: {others['relative_improvement'].mean():.4f} ({others['relative_improvement'].mean()*100:.1f}%) (n={len(others)})")
        print(f"  Others std: {others['relative_improvement'].std():.4f}")
        print(f"  T-test: t={t_stat:.3f}, p={p_value:.4f}")
        print(f"  Effect size (Cohen's d): {(perfect_matches['relative_improvement'].mean() - others['relative_improvement'].mean()) / others['relative_improvement'].std():.3f}")
        
        if p_value < 0.05:
            print(f"  ✓ SIGNIFICANT difference (p < 0.05)")
        elif p_value < 0.1:
            print(f"  ? MARGINALLY SIGNIFICANT (p < 0.1)")
        else:
            print(f"  ✗ Not significant (p ≥ 0.05)")
            
        # Show individual perfect match values
        print(f"\n  Perfect match details:")
        for _, row in perfect_matches.iterrows():
            print(f"    {row['midtrain_dataset']:15} → {row['sft_dataset']:10} | {row['model_size']} | {row['relative_improvement']:.3f} ({row['relative_improvement']*100:.1f}%)")
        
        # Power analysis insight
        print(f"\n  Statistical power note:")
        print(f"    With n={len(perfect_matches)} vs n={len(others)}, large effect sizes needed for significance")
        print(f"    Effect size is {(perfect_matches['relative_improvement'].mean() - others['relative_improvement'].mean())*100:.1f} percentage points")
        print(f"    This is a practically meaningful difference despite p={p_value:.3f}")
    
    # Effect size analysis
    print(f"\n" + "="*30)
    print("EFFECT SIZE ANALYSIS")
    print("="*30)
    
    overall_mean = analysis_df['relative_improvement'].mean()
    
    for bridge_type, group in bridge_groups:
        if len(group) == 0:
            continue
        group_mean = group['relative_improvement'].mean()
        effect_size = group_mean - overall_mean
        
        print(f"{bridge_type:20}: Δ = {effect_size:+.4f} ({effect_size*100:+.1f}%) vs overall mean")
    
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
    plt.savefig('midtraining_bridge_analysis_visualization.png')
    
    return analysis_df, correlation, p_value, best_method

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze midtraining bridge effects')
    parser.add_argument('--similarity_mode', 
                       choices=['combined', 'semantic_only', 'syntactic_only'],
                       default='combined',
                       help='Similarity mode to use (default: combined)')
    
    args = parser.parse_args()
    
    # Run the analysis
    csv_file_path = "/projects/bfcu/mliu7/all_in_one_pretrainingvisualization_scripts/final_step_ft_results.csv"
    
    # Your similarity matrix file paths
    embedding_csv = "/projects/bfcu/mliu7/all_in_one_pretrainingutil_scripts/similarity_results_07_14_extended.csv" 
    #ngram_csv = "/projects/bfcu/mliu7/all_in_one_pretraining07_15_sim_ngram.csv"
    ngram_csv = "/projects/bfcu/mliu7/all_in_one_pretrainingdataset_similarity_matrix_syntactic_multilevel_char1to5_plain.csv"
    
    print(f"Running analysis with similarity mode: {args.similarity_mode}")
    
    try:
        results_df, correlation, p_value, best_method, model_results, component_analysis = analyze_midtraining_bridge_effect(
            csv_file_path, embedding_csv, ngram_csv, args.similarity_mode
        )
        
        if results_df is not None:
            print(f"\n" + "="*50)
            print("FINAL SUMMARY")
            print("="*50)
            print(f"Best method: {best_method} bridge length")
            print(f"Overall bridge hypothesis correlation: {correlation:.4f} (p = {p_value:.4f})")
            
            # Bridge component breakdown
            print(f"\nBridge Component Analysis:")
            print(f"  C4 → Midtrain distance:    r = {component_analysis['corr_c4_mid_dist']:+.4f} (p = {component_analysis['p_c4_mid_dist']:.4f})")
            print(f"  Midtrain → SFT distance:   r = {component_analysis['corr_mid_sft_dist']:+.4f} (p = {component_analysis['p_mid_sft_dist']:.4f})")
            print(f"  Direct C4 → SFT distance:  r = {component_analysis['corr_c4_sft_direct']:+.4f} (p = {component_analysis['p_c4_sft_direct']:.4f})")
            
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
                    
                print(f"  {model_size}: {method} r = {best_corr:+.3f} (p = {best_p:.3f}) - {strength}")
            
            # Overall evidence strength
            if abs(correlation) > 0.4 and p_value < 0.05:
                evidence = "Strong"
            elif abs(correlation) > 0.2 and p_value < 0.05:
                evidence = "Moderate"
            elif abs(correlation) > 0.15 and p_value < 0.1:
                evidence = "Weak"
            else:
                evidence = "None"
            print(f"\nOverall evidence strength: {evidence}")
            
            # Selection bias assessment - key insight
            bias_evidence = component_analysis['bias_evidence']
            c4_mid_stronger = abs(component_analysis['corr_c4_mid_dist']) > abs(component_analysis['corr_mid_sft_dist']) + 0.1
            direct_strongest = abs(component_analysis['corr_c4_sft_direct']) > max(
                abs(component_analysis['corr_c4_mid_dist']), 
                abs(component_analysis['corr_mid_sft_dist'])
            )
            
            print(f"\nKey Finding:")
            if bias_evidence in ['C4_DISTANCE', 'DIRECT_DISTANCE'] or direct_strongest:
                print(f"  ⚠️  SELECTION BIAS detected - bridge effect may be spurious")
                print(f"  → C4 distance (task difficulty) drives improvement, not bridge quality")
                if direct_strongest:
                    print(f"  → Direct C4→SFT distance is strongest predictor (r = {component_analysis['corr_c4_sft_direct']:+.3f})")
                print(f"  → Recommendation: Focus on task difficulty rather than bridge optimization")
                
            elif bias_evidence == 'BRIDGE_QUALITY':
                print(f"  ✓ BRIDGE QUALITY matters independently")
                print(f"  → Midtrain→SFT similarity drives improvement beyond C4 distance")
                print(f"  → Recommendation: Optimize bridge architecture for target domain")
                
            else:  # MIXED
                print(f"  ? MIXED EVIDENCE - both task difficulty and bridge quality matter")
                print(f"  → Recommendation: Consider both factors in midtraining design")
            
            # Show best examples
            perfect_matches = results_df[results_df['bridge_type'] == 'perfect_match'] if 'bridge_type' in results_df.columns else pd.DataFrame()
            if len(perfect_matches) > 0:
                print(f"\nDomain Matching Results:")
                perfect_mean = perfect_matches['relative_improvement'].mean()
                others = results_df[results_df['bridge_type'] != 'perfect_match']
                others_mean = others['relative_improvement'].mean()
                print(f"  Perfect domain matches: {perfect_mean:.1%} improvement (n={len(perfect_matches)})")
                print(f"  Other combinations: {others_mean:.1%} improvement (n={len(others)})")
                
                if perfect_mean > others_mean + 0.005:  # 0.5 percentage point difference
                    print(f"  → Domain matching provides meaningful benefit")
                else:
                    print(f"  → Domain matching shows limited benefit")
            
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