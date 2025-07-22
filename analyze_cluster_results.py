#!/usr/bin/env python3
"""
Results Analysis Script for Cluster Experiments
Generates LaTeX tables and summary reports for the requested metrics
"""

import json
import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_results(results_file='experiment_results/all_results.json'):
    """Load experimental results"""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def generate_latex_table(results, dataset_name):
    """Generate LaTeX table for a specific dataset"""
    if dataset_name not in results:
        print(f"Dataset {dataset_name} not found in results")
        return ""
    
    dataset_results = results[dataset_name]
    
    # LaTeX table header
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Performance Results on {dataset_name.upper().replace('_', '-')} Dataset}}\n"
    latex += f"\\label{{tab:{dataset_name}_results}}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\hline\n"
    latex += "Method & FPR & Precision & Recall & F1 \\\\\n"
    latex += "\\hline\n"
    
    # Add results for each method
    for method, metrics in dataset_results.items():
        fpr = metrics.get('fpr', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        
        # Format method name for LaTeX
        method_name = method.replace('_', ' ')
        if method == 'Bayesian Ensemble Transformer':
            method_name = "\\textbf{" + method_name + "}"
        
        latex += f"{method_name} & {fpr:.4f} & {precision:.4f} & {recall:.4f} & {f1:.4f} \\\\\n"
    
    latex += "\\hline\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

def generate_combined_table(results):
    """Generate combined LaTeX table for all datasets"""
    latex = "\\begin{table*}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Performance Comparison Across All Datasets}\n"
    latex += "\\label{tab:all_results}\n"
    latex += "\\begin{tabular}{llcccc}\n"
    latex += "\\hline\n"
    latex += "Dataset & Method & FPR & Precision & Recall & F1 \\\\\n"
    latex += "\\hline\n"
    
    for dataset_name, dataset_results in results.items():
        dataset_display = dataset_name.upper().replace('_', '-')
        
        for i, (method, metrics) in enumerate(dataset_results.items()):
            fpr = metrics.get('fpr', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
            
            # Format method name for LaTeX
            method_name = method.replace('_', ' ')
            if method == 'Bayesian Ensemble Transformer':
                method_name = "\\textbf{" + method_name + "}"
            
            # Only show dataset name for first method
            dataset_cell = dataset_display if i == 0 else ""
            
            latex += f"{dataset_cell} & {method_name} & {fpr:.4f} & {precision:.4f} & {recall:.4f} & {f1:.4f} \\\\\n"
        
        latex += "\\hline\n"
    
    latex += "\\end{tabular}\n"
    latex += "\\end{table*}\n"
    
    return latex

def generate_summary_report(results):
    """Generate a comprehensive summary report"""
    report = []
    report.append("="*80)
    report.append("UNCERTAINTY-AWARE INTRUSION DETECTION EXPERIMENTAL RESULTS")
    report.append("="*80)
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Overall summary
    total_datasets = len(results)
    total_methods = len(next(iter(results.values()))) if results else 0
    
    report.append(f"Summary:")
    report.append(f"- Datasets evaluated: {total_datasets}")
    report.append(f"- Methods compared: {total_methods}")
    report.append("")
    
    # Best performing method per dataset
    report.append("Best Performing Method per Dataset (by F1-score):")
    report.append("-" * 50)
    
    for dataset_name, dataset_results in results.items():
        best_method = max(dataset_results.items(), key=lambda x: x[1].get('f1', 0))
        method_name, metrics = best_method
        
        report.append(f"{dataset_name.upper():<15}: {method_name:<30} (F1: {metrics['f1']:.4f})")
    
    report.append("")
    
    # Detailed results per dataset
    for dataset_name, dataset_results in results.items():
        report.append(f"\n{dataset_name.upper()} Dataset Results:")
        report.append("-" * 60)
        report.append(f"{'Method':<30} {'FPR':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
        report.append("-" * 60)
        
        # Sort by F1 score (descending)
        sorted_methods = sorted(dataset_results.items(), key=lambda x: x[1].get('f1', 0), reverse=True)
        
        for method, metrics in sorted_methods:
            fpr = metrics.get('fpr', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
            
            report.append(f"{method:<30} {fpr:<8.4f} {precision:<10.4f} {recall:<8.4f} {f1:<8.4f}")
    
    # Method comparison across datasets
    report.append("\n\nMethod Performance Across Datasets:")
    report.append("="*80)
    
    # Get all unique methods
    all_methods = set()
    for dataset_results in results.values():
        all_methods.update(dataset_results.keys())
    
    for method in sorted(all_methods):
        report.append(f"\n{method}:")
        report.append("-" * 40)
        report.append(f"{'Dataset':<15} {'FPR':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
        report.append("-" * 40)
        
        method_f1_scores = []
        for dataset_name, dataset_results in results.items():
            if method in dataset_results:
                metrics = dataset_results[method]
                fpr = metrics.get('fpr', 0)
                precision = metrics.get('precision', 0)
                recall = metrics.get('recall', 0)
                f1 = metrics.get('f1', 0)
                method_f1_scores.append(f1)
                
                report.append(f"{dataset_name:<15} {fpr:<8.4f} {precision:<10.4f} {recall:<8.4f} {f1:<8.4f}")
        
        # Calculate average performance
        if method_f1_scores:
            avg_f1 = np.mean(method_f1_scores)
            std_f1 = np.std(method_f1_scores)
            report.append("-" * 40)
            report.append(f"{'Average':<15} {'':<8} {'':<10} {'':<8} {avg_f1:<8.4f}")
            report.append(f"{'Std Dev':<15} {'':<8} {'':<10} {'':<8} {std_f1:<8.4f}")
    
    return "\n".join(report)

def save_results_to_files(results, output_dir='experiment_results'):
    """Save all analysis results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save individual dataset tables
    for dataset_name in results.keys():
        latex_table = generate_latex_table(results, dataset_name)
        table_file = os.path.join(output_dir, f'{dataset_name}_table.tex')
        with open(table_file, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved: {table_file}")
    
    # Generate and save combined table
    combined_table = generate_combined_table(results)
    combined_file = os.path.join(output_dir, 'combined_results_table.tex')
    with open(combined_file, 'w') as f:
        f.write(combined_table)
    print(f"Combined LaTeX table saved: {combined_file}")
    
    # Generate and save summary report
    summary_report = generate_summary_report(results)
    report_file = os.path.join(output_dir, 'experimental_summary.txt')
    with open(report_file, 'w') as f:
        f.write(summary_report)
    print(f"Summary report saved: {report_file}")
    
    # Save results as CSV for easy analysis
    csv_data = []
    for dataset_name, dataset_results in results.items():
        for method, metrics in dataset_results.items():
            row = {
                'Dataset': dataset_name,
                'Method': method,
                'FPR': metrics.get('fpr', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1': metrics.get('f1', 0),
                'Accuracy': metrics.get('accuracy', 0),
                'AUC': metrics.get('auc', 0)
            }
            csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_file = os.path.join(output_dir, 'all_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"CSV results saved: {csv_file}")

def main():
    """Main analysis function"""
    print("Analyzing cluster experiment results...")
    
    # Load results
    results = load_results()
    if results is None:
        print("No results to analyze. Run experiments first.")
        return
    
    print(f"Loaded results for {len(results)} datasets")
    
    # Generate analysis
    save_results_to_files(results)
    
    # Print summary to console
    summary = generate_summary_report(results)
    print("\n" + summary)
    
    print("\nAnalysis complete! Check experiment_results/ directory for output files.")

if __name__ == "__main__":
    main()
