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
