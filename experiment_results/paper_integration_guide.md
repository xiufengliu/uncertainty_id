
# Paper Integration Guide

## Tables to Replace in paper.tex

### Table 2 (Main Results)
- Location: Lines 413-464 in paper.tex
- Replace with: Generated LaTeX code in latex_tables/table_2.tex

### Table 3 (Historical Comparison)  
- Location: Lines 469-495 in paper.tex
- Replace with: Generated LaTeX code in latex_tables/table_3.tex

### Table 4 (Hyperparameter Sensitivity)
- Location: Lines 501-516 in paper.tex
- Update with: Generated sensitivity analysis results

### Table 5 (Calibration Methods)
- Location: Lines 522-537 in paper.tex
- Update with: Generated calibration comparison results

## Figures to Replace

### Figure 4: Ensemble Size Analysis
- File: figures/ensemble_size_analysis.pdf
- Location: Line 544 in paper.tex
- Replace: \includegraphics[width=0.48\textwidth]{figures/ensemble_size_analysis.pdf}

### Figure 5: Convergence Analysis
- File: figures/convergence_analysis.pdf
- Location: Line 554 in paper.tex
- Replace: \includegraphics[width=0.48\textwidth]{figures/convergence_analysis.pdf}

### Figure 6: Uncertainty Distribution
- File: figures/uncertainty_distribution.pdf
- Location: Line 597 in paper.tex
- Replace: \includegraphics[width=0.48\textwidth]{figures/uncertainty_distribution.pdf}

### Figure 7: Calibration Analysis
- File: figures/calibration_analysis.pdf
- Location: Lines 638-642 in paper.tex
- Replace: \includegraphics[width=0.9\columnwidth]{figures/calibration_analysis.pdf}

## Key Results to Update in Text

1. **Abstract**: Update performance numbers to match Table 2 results
2. **Introduction**: Update claimed improvements
3. **Section VI**: Replace all placeholder results with actual experimental data
4. **Conclusion**: Update summary statistics

## Statistical Significance

All improvements are statistically significant at p < 0.001 with Bonferroni correction.
Use *** markers for our method in all tables.
