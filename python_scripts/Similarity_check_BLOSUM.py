#!/usr/bin/env python3
import os
from Bio import AlignIO
from Bio.Align import AlignInfo
from Bio.Align import substitution_matrices
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Define input and output paths
alignment_file = "/mnt/c/Users/javed/Downloads/quality/mutation_analysis/analysis_results_7/b.aln"
output_dir = "/mnt/c/Users/javed/Downloads/quality/mutation_analysis/analysis_results_7/analysis_results"
os.makedirs(output_dir, exist_ok=True)

print(f"Starting analysis of {alignment_file}")

# --- 1. Read Alignment ---
try:
    alignment = AlignIO.read(alignment_file, "clustal")
    print(f"Successfully read alignment with {len(alignment)} sequences, length {alignment.get_alignment_length()}.")
except Exception as e:
    print(f"Error reading alignment file: {e}")
    exit()

seq_names = [record.id for record in alignment]
num_sequences = len(alignment)
alignment_length = alignment.get_alignment_length()

# --- 2. Calculate Conservation Scores ---
print("Calculating conservation scores...")
conservation_scores = []
most_common_residues = []
for i in range(alignment_length):
    column = alignment[:, i]
    counts = {}
    num_non_gap_in_column = 0
    for residue in column:
        if residue != '-':
            counts[residue] = counts.get(residue, 0) + 1
            num_non_gap_in_column += 1
    
    if not counts: # All gaps in this column
        most_common_residue = '-'
        max_freq_score = 0.0 
    else:
        most_common_residue = max(counts, key=counts.get)
        if num_non_gap_in_column > 0:
            max_freq_score = counts[most_common_residue] / num_non_gap_in_column
        else:
            max_freq_score = 0.0
        
    conservation_scores.append(max_freq_score)
    most_common_residues.append(most_common_residue)

conservation_df = pd.DataFrame({
    'Position': range(1, alignment_length + 1),
    'MostCommonResidue': most_common_residues,
    'ConservationScore': conservation_scores
})
conservation_csv_path = os.path.join(output_dir, "conservation_scores.csv")
conservation_df.to_csv(conservation_csv_path, index=False)
print(f"Conservation scores saved to {conservation_csv_path}")

# --- 3. Calculate Pairwise Identity Matrix ---
print("Calculating pairwise identity matrix...")
identity_matrix = np.zeros((num_sequences, num_sequences))
for i in range(num_sequences):
    for j in range(i, num_sequences):
        seq1_str = str(alignment[i].seq)
        seq2_str = str(alignment[j].seq)
        identicals = 0
        valid_comparison_positions = 0
        
        for k in range(alignment_length):
            res1 = seq1_str[k]
            res2 = seq2_str[k]
            
            if res1 == '-' and res2 == '-':
                continue
            
            valid_comparison_positions += 1
            if res1 == res2 and res1 != '-':
                identicals += 1
        
        if valid_comparison_positions > 0:
            identity = (identicals / valid_comparison_positions) * 100.0
        else:
            identity = 100.0 if i == j else 0.0
            
        identity_matrix[i, j] = identity
        identity_matrix[j, i] = identity

identity_df = pd.DataFrame(identity_matrix, index=seq_names, columns=seq_names)
identity_csv_path = os.path.join(output_dir, "pairwise_identity_matrix.csv")
identity_df.to_csv(identity_csv_path)
print(f"Pairwise identity matrix saved to {identity_csv_path}")

# --- 4. Calculate Pairwise Similarity Matrix (using BLOSUM62) ---
print("Calculating pairwise similarity matrix (BLOSUM62 based)...")
blosum62 = substitution_matrices.load("BLOSUM62")

similarity_matrix_blosum_score = np.zeros((num_sequences, num_sequences))
similarity_matrix_percent = np.zeros((num_sequences, num_sequences))

for i in range(num_sequences):
    for j in range(i, num_sequences):
        seq1_str = str(alignment[i].seq)
        seq2_str = str(alignment[j].seq)
        
        current_blosum_sum = 0
        positive_scoring_pairs = 0
        num_aa_pairs = 0

        for k in range(alignment_length):
            res1 = seq1_str[k]
            res2 = seq2_str[k]

            if res1 != '-' and res2 != '-':
                num_aa_pairs += 1
                pair_score = 0
                try:
                    pair_score = blosum62[(res1, res2)]
                except KeyError:
                    try:
                        pair_score = blosum62[(res2, res1)]
                    except KeyError:
                        pass 
                
                current_blosum_sum += pair_score
                if pair_score > 0:
                    positive_scoring_pairs += 1
        
        similarity_matrix_blosum_score[i, j] = current_blosum_sum
        similarity_matrix_blosum_score[j, i] = current_blosum_sum
        
        if num_aa_pairs > 0:
            similarity_percent = (positive_scoring_pairs / num_aa_pairs) * 100.0
        else:
            similarity_percent = 100.0 if i == j and num_sequences > 0 and alignment[i].seq.strip('-') else 0.0

        similarity_matrix_percent[i, j] = similarity_percent
        similarity_matrix_percent[j, i] = similarity_percent

similarity_blosum_score_df = pd.DataFrame(similarity_matrix_blosum_score, index=seq_names, columns=seq_names)
similarity_blosum_csv_path = os.path.join(output_dir, "pairwise_similarity_blosum_scores.csv")
similarity_blosum_score_df.to_csv(similarity_blosum_csv_path)
print(f"Pairwise BLOSUM62 scores matrix saved to {similarity_blosum_csv_path}")

similarity_percent_df = pd.DataFrame(similarity_matrix_percent, index=seq_names, columns=seq_names)
similarity_percent_csv_path = os.path.join(output_dir, "pairwise_similarity_percent_matrix.csv")
similarity_percent_df.to_csv(similarity_percent_csv_path)
print(f"Pairwise similarity percent matrix saved to {similarity_percent_csv_path}")

# --- 5. Generate Plots ---
print("Generating plots...")
# Plot 1: Conservation Scores
plt.figure(figsize=(max(15, alignment_length / 8), 6)) 
plt.plot(conservation_df['Position'], conservation_df['ConservationScore'], marker='.', linestyle='-', markersize=4)
plt.title('Sequence Conservation Scores')
plt.xlabel('Position in Alignment')
plt.ylabel('Conservation Score (Freq. of Most Common AA in Non-Gap Column)')
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
conservation_plot_path = os.path.join(output_dir, "conservation_scores_plot.png")
plt.savefig(conservation_plot_path)
plt.close()
print(f"Conservation scores plot saved to {conservation_plot_path}")

# Plot 2: Pairwise Identity Heatmap (Updated with fixed scale)
plt.figure(figsize=(max(8, num_sequences / 1.8), max(6, num_sequences / 2.2)))
sns.heatmap(identity_df, annot=True, cmap="viridis", fmt=".1f", linewidths=.5, cbar_kws={'label': 'Identity (%)'}, vmin=97, vmax=100)
plt.title('Pairwise Sequence Identity Matrix (Scale: 97-100%)', fontsize=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
identity_heatmap_path = os.path.join(output_dir, "identity_matrix_heatmap_updated.png") # Changed filename
plt.savefig(identity_heatmap_path)
plt.close()
print(f"Updated identity matrix heatmap saved to {identity_heatmap_path}")

# Plot 3: Pairwise Similarity (Percent Positive BLOSUM62) Heatmap (Updated with fixed scale)
plt.figure(figsize=(max(12, num_sequences / 1.8), max(10, num_sequences / 2.2)))
sns.heatmap(similarity_percent_df, annot=False, cmap="magma", fmt=".1f", linewidths=.5, cbar_kws={'label': 'Similarity (%, Positive BLOSUM62 Pairs)'}, vmin=96, vmax=100)
plt.title('Pairwise Sequence Similarity Matrix', fontsize=42)
plt.xticks(rotation=45, ha='right', fontsize=24)
plt.yticks(rotation=0, fontsize=24)
plt.tight_layout()
similarity_heatmap_path = os.path.join(output_dir, "similarity_matrix_heatmap_updated.png") # Changed filename
plt.savefig(similarity_heatmap_path)
plt.close()
print(f"Updated similarity matrix heatmap saved to {similarity_heatmap_path}")

print(f"Analysis complete. All results saved in {output_dir}")

