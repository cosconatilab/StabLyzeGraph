import csv
import re

def apply_mutations(wild_type, mutation_line):
    """
    Apply multiple mutations from a line to the wild-type sequence.
    """
    mutated_sequence = list(wild_type)  # Convert sequence to list for easy mutation
    mutations = mutation_line.split(';')  # Split multiple mutations by semicolons
    valid_mutations = []

    for mutation in mutations:
        mutation = mutation.strip()  # Remove extra spaces
        # Match mutation format (e.g., A2C)
        match = re.match(r"^([A-Z])(\d+)([A-Z])$", mutation)
        if not match:
            print(f"Skipping invalid mutation format: {mutation}")
            continue

        original, position, new = match.groups()
        position = int(position)  # Convert position to integer

        # Check if the position is valid
        if position < 1 or position > len(wild_type):
            print(f"Skipping out-of-bounds mutation: {mutation}")
            continue

        # Check if the original amino acid matches
        if mutated_sequence[position - 1] != original:
            print(f"Warning: Mutation {mutation} does not match wild type at position {position}.")
            continue

        # Apply the mutation
        mutated_sequence[position - 1] = new
        valid_mutations.append(mutation)

    return "".join(mutated_sequence), "; ".join(valid_mutations)

def read_mutations_from_csv(file_path):
    """
    Read mutations from a CSV file. Each line can have one or more mutations.
    """
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        mutations = [row[0].strip() for row in reader if row]  # Read each line
    return mutations

def main():
    # Input wild-type sequence
    wild_type = input("Enter the wild-type protein sequence: ").strip()
    csv_path = input("Enter the path to the CSV file with mutations: ").strip()

    # Read mutation lines from CSV
    mutation_lines = read_mutations_from_csv(csv_path)

    # Apply mutations and generate results
    mutants = []
    for mutation_line in mutation_lines:
        mutated_sequence, valid_mutations = apply_mutations(wild_type, mutation_line)
        if valid_mutations:
            mutants.append((valid_mutations, mutated_sequence))

    # Print results
    print("\nGenerated Mutants:")
    for mutations, sequence in mutants:
        print(f"Mutations: {mutations} -> Mutant Sequence: {sequence}")

    # Save results to a CSV file
    save_path = input("Enter the path to save the mutants (or press Enter to skip): ").strip()
    if save_path:
        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Mutations", "Mutant Sequence"])
            writer.writerows(mutants)
        print(f"Mutants saved to {save_path}")

if __name__ == "__main__":
    main()
