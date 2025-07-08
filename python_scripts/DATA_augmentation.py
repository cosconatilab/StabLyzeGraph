# import pandas as pd

# # Mapping of amino acids to corresponding mutated residues (possible substitutions)
# decoy_mutations = {
#     'A': ['W', 'S'],
#     'C': ['A'],
#     'D': ['K'],
#     'E': ['R'],
#     'F': ['S', 'A'],
#     'G': ['P'],
#     'H': ['A'],
#     'I': ['S'],
#     'K': ['D'],
#     'L': ['S'],
#     'M': ['S'],
#     'N': ['A'],
#     'P': ['G'],
#     'Q': ['A'],
#     'R': ['E'],
#     'S': ['I', 'F'],
#     'T': ['I', 'F'],
#     'V': ['W'],
#     'W': ['S', 'A'],
#     'Y': ['A'],
# }

# # Function to generate substituted mutants based on defined mapping
# def generate_substitution_mutants(sequence, mutation):
#     original_aa = mutation[0]              
#     position = int(mutation[1:-1]) - 1     
#     mutant_aa = mutation[-1]               

#     # Verify original AA matches the sequence
#     if sequence[position] != original_aa:
#         raise ValueError(f"Mismatch at position {position+1}: sequence has '{sequence[position]}', mutation specifies '{original_aa}'.")

#     if mutant_aa in decoy_mutations:
#         new_mutants = decoy_mutations[mutant_aa]
#         mutant_sequences = []
#         for aa in new_mutants:
#             mutated_seq = sequence[:position] + aa + sequence[position + 1:]
#             mutant_sequences.append(mutated_seq)
#         return mutant_sequences
#     else:
#         raise ValueError(f"Mutated amino acid '{mutant_aa}' not recognized in decoy mapping.")


# # Handle CSV file containing mutations
# def augment_mutations_from_csv(input_file):
#     df = pd.read_csv(input_file)

#     if 'Mutations' not in df.columns or 'sequence' not in df.columns:
#         raise ValueError("CSV must have columns 'Mutations' and 'sequence'.")

#     augmented_rows = []

#     for idx, row in df.iterrows():
#         original_sequence = row['sequence']
#         mutations = row['Mutations'].split(';')

#         # Start with the original sequence
#         sequences_to_process = [original_sequence]

#         for mutation in mutations:
#             mutation = mutation.strip()
#             new_sequences = []

#             for seq in sequences_to_process:
#                 mutants = generate_substitution_mutants(seq, mutation)
#                 new_sequences.extend(mutants)

#             # Update list with newly created sequences
#             sequences_to_process = new_sequences

#         # Each final sequence after applying all mutations
#         for seq in sequences_to_process:
#             augmented_rows.append({
#                 'Original Mutations': row['Mutations'],
#                 'Original Sequence': original_sequence,
#                 'Augmented Sequence': seq
#             })

#     augmented_df = pd.DataFrame(augmented_rows)
#     return augmented_df

# # Saving results to CSV
# def save_augmented_data(df, output_file):
#     df.to_csv(output_file, index=False)

# # Main execution
# if __name__ == "__main__":
#     input_file = "protein_mutations.csv"  
#     output_file = "augmented_protein_mutations.csv"

#     try:
#         augmented_data = augment_mutations_from_csv(input_file)
#         save_augmented_data(augmented_data, output_file)
#         print(f"Augmented data saved to {output_file}")
#     except Exception as e:
#         print(f"Error: {e}")


import pandas as pd

# Mapping of mutated amino acids to corresponding substitution residues
decoy_mutations = {
    'A': ['W', 'S'],
    'C': ['A'],
    'D': ['K'],
    'E': ['R'],
    'F': ['S', 'A'],
    'G': ['P'],
    'H': ['A'],
    'I': ['S'],
    'K': ['D'],
    'L': ['S'],
    'M': ['S'],
    'N': ['A'],
    'P': ['G'],
    'Q': ['A'],
    'R': ['E'],
    'S': ['I', 'F'],
    'T': ['I', 'F'],
    'V': ['W'],
    'W': ['S', 'A'],
    'Y': ['A'],
}

# Function to generate augmented mutations
def generate_augmented_mutations(mutation):
    original_aa = mutation[0]                # Original amino acid
    position = mutation[1:-1]                # Position
    mutant_aa = mutation[-1]                 # Mutated amino acid

    if mutant_aa in decoy_mutations:
        substitutions = decoy_mutations[mutant_aa]
        augmented = [f"{original_aa}{position}{sub_aa}" for sub_aa in substitutions]
        return augmented
    else:
        # If mutant amino acid isn't in mapping, return empty list or original mutation
        return []

# Main function to read CSV and generate augmented mutations
def augment_mutations_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    if 'Mutations' not in df.columns:
        raise ValueError("CSV must have a 'Mutations' column.")

    augmented_data = []

    for mut in df['Mutations']:
        augmented_mutations = generate_augmented_mutations(mut)
        for augmented_mut in augmented_mutations:
            augmented_data.append({
                'Original Mutation': mut,
                'Augmented Mutation': augmented_mut
            })

    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(output_csv, index=False)
    print(f"Augmented mutations saved to '{output_csv}'")

# Execute script
if __name__ == "__main__":
    input_file = "Actives.csv"
    output_file = "augmented_mutations.csv"
    augment_mutations_csv(input_file, output_file)
