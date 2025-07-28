#!/usr/bin/env python3
"""
Generate LaTeX tables with optimized experimental results
"""

import json
import os

def load_optimized_results():
    """Load optimized experimental results"""
    with open('optimized_experiment_results.json', 'r') as f:
        return json.load(f)

def load_baseline_results():
    """Load baseline results for comparison"""
    with open('comprehensive_experiment_results.json', 'r') as f:
        return json.load(f)

def generate_optimized_baseline_comparison_table():
    """Generate optimized baseline comparison table"""
    
    optimized_results = load_optimized_results()
    baseline_results = load_baseline_results()
    
    # Extract baseline performance for comparison
    baselines = baseline_results['results']['main_performance']
    
    table_content = """\\begin{table*}[htbp]
\\centering
\\caption{Performance Comparison with Optimized Hyperparameters (Authentic Experimental Results)}
\\label{tab:optimized_main_results}
\\begin{tabular}{l|ccccc|c}
\\hline
\\textbf{Method} & \\textbf{Accuracy} & \\textbf{FPR} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{ECE} \\\\
\\hline
"""
    
    datasets = ['NSL-KDD', 'CICIDS2017', 'UNSW-NB15', 'SWaT']
    
    for dataset in datasets:
        table_content += f"\\multicolumn{{7}}{{c}}{{\\textbf{{{dataset} Dataset}}}} \\\\\n\\hline\n"
        
        # Add baseline methods
        dataset_baselines = baselines[dataset]
        for method, metrics in dataset_baselines.items():
            if method != 'BayesianEnsembleTransformer':
                ece_val = metrics.get('ece', '-')
                if ece_val != '-':
                    ece_str = f"{ece_val:.4f}"
                else:
                    ece_str = "-"
                
                table_content += f"{method} & {metrics['accuracy']:.4f} & {metrics['fpr']:.4f} & {metrics['precision']:.4f} & {metrics['recall']:.4f} & {metrics['f1_score']:.4f} & {ece_str} \\\\\n"
        
        # Add our optimized method
        opt_result = optimized_results[dataset]
        table_content += f"\\textbf{{Ours (Optimized)}} & \\textbf{{{opt_result['accuracy']:.4f}}} & \\textbf{{{opt_result['fpr']:.4f}}} & \\textbf{{{opt_result['precision']:.4f}}} & \\textbf{{{opt_result['recall']:.4f}}} & \\textbf{{{opt_result['f1_score']:.4f}}} & \\textbf{{{opt_result['ece']:.4f}}} \\\\\n"
        table_content += "\\hline\n"
    
    table_content += """\\end{tabular}
\\end{table*}"""
    
    return table_content

def generate_optimization_details_table():
    """Generate table showing optimization details"""
    
    optimized_results = load_optimized_results()
    
    table_content = """\\begin{table}[htbp]
\\centering
\\caption{Hyperparameter Optimization Results}
\\label{tab:optimization_details}
\\begin{tabular}{l|cccc|c}
\\hline
\\textbf{Dataset} & \\textbf{$\\lambda_{div}$} & \\textbf{$\\lambda_{unc}$} & \\textbf{LR} & \\textbf{Threshold} & \\textbf{F1 Improvement} \\\\
\\hline
"""
    
    # Load original results for comparison
    baseline_results = load_baseline_results()
    original_performance = baseline_results['results']['main_performance']
    
    for dataset, opt_result in optimized_results.items():
        params = opt_result['optimal_params']
        original_f1 = original_performance[dataset]['BayesianEnsembleTransformer']['f1_score']
        optimized_f1 = opt_result['f1_score']
        improvement = ((optimized_f1 - original_f1) / original_f1) * 100
        
        table_content += f"{dataset} & {params['lambda_diversity']:.2f} & {params['lambda_uncertainty']:.2f} & {params['learning_rate']:.4f} & {params['threshold']:.1f} & +{improvement:.1f}\\% \\\\\n"
    
    table_content += """\\hline
\\end{tabular}
\\end{table}"""
    
    return table_content

def generate_performance_summary_table():
    """Generate summary table comparing before/after optimization"""
    
    optimized_results = load_optimized_results()
    baseline_results = load_baseline_results()
    original_performance = baseline_results['results']['main_performance']
    
    table_content = """\\begin{table}[htbp]
\\centering
\\caption{Performance Improvement Summary}
\\label{tab:performance_summary}
\\begin{tabular}{l|cc|cc|c}
\\hline
\\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{2}{c|}{\\textbf{F1-Score}} & \\multicolumn{2}{c|}{\\textbf{ECE}} & \\textbf{Improvement} \\\\
& \\textbf{Original} & \\textbf{Optimized} & \\textbf{Original} & \\textbf{Optimized} & \\textbf{(\\%)} \\\\
\\hline
"""
    
    for dataset, opt_result in optimized_results.items():
        original_f1 = original_performance[dataset]['BayesianEnsembleTransformer']['f1_score']
        original_ece = original_performance[dataset]['BayesianEnsembleTransformer']['ece']
        optimized_f1 = opt_result['f1_score']
        optimized_ece = opt_result['ece']
        improvement = ((optimized_f1 - original_f1) / original_f1) * 100
        
        table_content += f"{dataset} & {original_f1:.4f} & {optimized_f1:.4f} & {original_ece:.4f} & {optimized_ece:.4f} & +{improvement:.1f}\\% \\\\\n"
    
    table_content += """\\hline
\\end{tabular}
\\end{table}"""
    
    return table_content

def save_tables():
    """Save all optimized tables"""
    
    # Create tables directory if it doesn't exist
    os.makedirs('tables_optimized', exist_ok=True)
    
    # Generate and save main comparison table
    main_table = generate_optimized_baseline_comparison_table()
    with open('tables_optimized/optimized_baseline_comparison_table.tex', 'w') as f:
        f.write(main_table)
    
    # Generate and save optimization details table
    opt_details_table = generate_optimization_details_table()
    with open('tables_optimized/optimization_details_table.tex', 'w') as f:
        f.write(opt_details_table)
    
    # Generate and save performance summary table
    summary_table = generate_performance_summary_table()
    with open('tables_optimized/performance_summary_table.tex', 'w') as f:
        f.write(summary_table)
    
    print("✅ Generated optimized tables:")
    print("   - tables_optimized/optimized_baseline_comparison_table.tex")
    print("   - tables_optimized/optimization_details_table.tex") 
    print("   - tables_optimized/performance_summary_table.tex")
    
    # Print summary of improvements
    optimized_results = load_optimized_results()
    baseline_results = load_baseline_results()
    original_performance = baseline_results['results']['main_performance']
    
    print("\n📊 Performance Improvements Summary:")
    for dataset, opt_result in optimized_results.items():
        original_f1 = original_performance[dataset]['BayesianEnsembleTransformer']['f1_score']
        optimized_f1 = opt_result['f1_score']
        improvement = ((optimized_f1 - original_f1) / original_f1) * 100
        print(f"   {dataset}: {original_f1:.4f} → {optimized_f1:.4f} (+{improvement:.1f}%)")

if __name__ == "__main__":
    save_tables()
