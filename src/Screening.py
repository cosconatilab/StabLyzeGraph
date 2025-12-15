import gc
import os
import numpy as np
import pandas as pd
import torch
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    SAGPooling,
    TopKPooling,
)
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq # Added import
from Bio.Align import AlignInfo
from Bio.PDB import PDBParser
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import random
import subprocess
import logging
from joblib import Parallel, delayed
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, product
import argparse
from tqdm import tqdm
import time # Added for GUI integration
import json # Added for GUI integration
import sys # Added for GUI integration

matplotlib.use("Agg")  # Use non-GUI backend for rendering to files.

# Set up logging for better tracking
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

# Default random seed (can be overridden by argument)
DEFAULT_SEED = 42
optimal_threshold = 0.85 # Default optimal threshold

# Set seed for CPU operations initially
torch.manual_seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU operations have a separate seed we also want to set
if device.type == "cuda":
    torch.cuda.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

########################
### MODEL DEFINITION ###
########################

class GNNFeatureExtractor(torch.nn.Module):
    """GNN model for graph-level classification."""

    def __init__(self, in_features, hidden_features, dropout_rate, ratio):
        super(GNNFeatureExtractor, self).__init__()
        print(f"DEBUG: Initializing model with in_features={in_features}, hidden_features={hidden_features}") # DEBUG
        self.gat1 = GATConv(
            in_features, int(hidden_features / 4), heads=4
        )
        self.sage1 = SAGEConv(
            hidden_features,
            hidden_features,
            aggr="max",
            normalize=True,
        )
        self.sagpool = SAGPooling(hidden_features, ratio=ratio)
        self.lin = torch.nn.Linear(hidden_features, int(hidden_features / 2))
        self.fc = torch.nn.Linear(int(hidden_features / 2), 1)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.gcn_batch_norm = BatchNorm1d(hidden_features)
        self.gat_batch_norm = BatchNorm1d(hidden_features)
        self.sage_batch_norm = BatchNorm1d(hidden_features)
        self.graph_batch_norm = BatchNorm1d(hidden_features)
        self.lin_batch_norm = BatchNorm1d(int(hidden_features / 2))
        print("DEBUG: Model layers initialized") # DEBUG

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gat1(x, edge_index, edge_attr)
        x = self.gat_batch_norm(x)
        x = F.leaky_relu(x)

        x, edge_index, edge_attr, batch, _, _ = self.sagpool(
            x, edge_index, edge_attr, batch=batch
        )

        x = self.sage1(x, edge_index)
        x = self.sage_batch_norm(x)
        x = F.leaky_relu(x)

        x = global_add_pool(x, batch)
        x = self.graph_batch_norm(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        fingerprint_layer = self.lin(x)
        fingerprint_layer = self.lin_batch_norm(fingerprint_layer)
        fingerprint_layer = F.tanh(fingerprint_layer)
        fingerprint_layer = self.dropout(fingerprint_layer)

        single_output = self.fc(fingerprint_layer)

        return single_output, fingerprint_layer

print("DEBUG: Defining helper functions") # DEBUG
########################
### HELPER FUNCTIONS ###
########################

# Added for GUI integration
def update_progress_file(progress, message, file_path):
    """Writes progress updates to a JSON file."""
    if not file_path:
        return
    try:
        with open(file_path, "w") as f:
            json.dump({"progress": int(progress), "message": str(message)}, f)
    except Exception as e:
        logging.warning(f"Could not write progress to {file_path}: {e}")

def relabel_node_indices(data):
    """
    Relabels node indices in a batched PyTorch Geometric Data object to be contiguous.
    """
    if isinstance(data, Batch):
        batch = data.batch
        edge_index = data.edge_index.clone()
        num_nodes = data.num_nodes
        node_idx = torch.arange(num_nodes, device=edge_index.device)
        cum_nodes = torch.cat([
            torch.tensor([0], device=edge_index.device),
            torch.cumsum(torch.bincount(batch), dim=0),
        ])
        for i in range(len(cum_nodes) - 1):
            start = cum_nodes[i]
            end = cum_nodes[i + 1]
            edge_index[:, (edge_index[0] >= start) & (edge_index[0] < end)] -= start
            edge_index[:, (edge_index[1] >= start) & (edge_index[1] < end)] -= start
        data.edge_index = edge_index
        return data
    elif isinstance(data, Data):
        return data # No relabeling needed for single graphs
    else:
        raise TypeError(
            "Input data must be either torch_geometric.data.Data or torch_geometric.data.Batch"
        )

def read_sequence_file(file_path, is_active=None):
    """Reads sequences from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    df = pd.read_csv(file_path)
    sequences = df.iloc[:, 0].tolist()
    if is_active is not None:
        activity_values = [1 if is_active else 0] * len(sequences)
        return sequences, activity_values
    return sequences

def read_fasta(file_path):
    """Reads sequences from a fasta file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    wild_type_sequence = [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]
    if not wild_type_sequence:
        raise ValueError(f"No sequences found in FASTA file: {file_path}")
    return wild_type_sequence[0]

def load_dictionaries(csv_file):
    """Dynamically load feature dictionaries from a CSV file."""
    df = pd.read_csv(csv_file, index_col=0)
    scaler = StandardScaler()
    standardized_values = scaler.fit_transform(df.values)
    feature_dicts = {
        row: {col: val for col, val in zip(df.columns, standardized_values[idx])}
        for idx, row in enumerate(df.index)
    }
    return feature_dicts

def extract_coordinates_from_pdb(pdb_file, sequences, wild_type_sequence, seed=DEFAULT_SEED):
    """Extracts 3D coordinates from the PDB file for wild type and generates coordinates for other sequences."""
    np.random.seed(seed)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("wild_type", pdb_file)
    wild_type_coords_list = [
        residue["CA"].get_vector().get_array()
        for model in structure
        for chain in model
        for residue in chain
        if residue.has_id("CA")
    ]
    if not wild_type_coords_list:
        raise ValueError(f"No CA atoms found in PDB file: {pdb_file}")
    
    wild_type_coords_array = np.array(wild_type_coords_list)

    if len(wild_type_coords_array) != len(wild_type_sequence):
        logging.warning(f"Mismatch between wild type PDB CA coordinates ({len(wild_type_coords_array)}) and FASTA sequence length ({len(wild_type_sequence)}). Using shorter length for alignment.")
        min_len = min(len(wild_type_coords_array), len(wild_type_sequence))
        wild_type_coords_array = wild_type_coords_array[:min_len]
        wild_type_sequence_adjusted = wild_type_sequence[:min_len]
    else:
        wild_type_sequence_adjusted = wild_type_sequence

    coordinates = []
    for seq in sequences:
        seq_adjusted = seq[:len(wild_type_sequence_adjusted)] # Ensure seq is not longer than adjusted WT
        seq_coords = np.zeros((len(wild_type_sequence_adjusted), 3))
        for i, (wild_aa, seq_aa) in enumerate(zip(wild_type_sequence_adjusted, seq_adjusted)):
            if i < len(wild_type_coords_array):
                if wild_aa == seq_aa:
                    seq_coords[i] = wild_type_coords_array[i]
                else:
                    seq_coords[i] = wild_type_coords_array[i] + np.random.normal(0, 1, size=3)
            else:
                 # Should not happen if lengths are managed correctly, but as a fallback
                seq_coords[i] = np.random.normal(0, 1, size=3) 
        coordinates.append(seq_coords)
    return coordinates

def calculate_conservation_scores(sequences_file, output_dir):
    """Calculate conservation scores using multiple sequence alignment via Clustal Omega."""
    aligned_file = os.path.join(output_dir, "aligned_mutated_sequences.aln")
    logging.info(f"Running Clustal Omega for alignment. Input: {sequences_file}, Output: {aligned_file}")
    try:
        subprocess.run(
            ["clustalo", "-i", sequences_file, "-o", aligned_file, "--force", "--outfmt=clu"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running Clustal Omega: {e.stderr.decode()}")
        raise e
    try:
        alignment = AlignIO.read(aligned_file, "clustal")
    except Exception as e:
        raise RuntimeError(f"Error reading alignment file: {aligned_file}. Details: {str(e)}")
    summary_align = AlignInfo.SummaryInfo(alignment)
    consensus = summary_align.dumb_consensus()
    conservation_scores = [sum(1 for aa in alignment[:, i] if aa == consensus[i]) / len(alignment[:, i]) for i in range(len(consensus))]
    return conservation_scores

def sequence_to_graph(seq, properties, conservation_scores, pdb_coords, distance_threshold=10.0):
    """Converts protein sequence into a graph representation."""
    num_nodes = len(seq)
    if len(pdb_coords) < num_nodes:
        # This can happen if PDB is shorter than sequence after alignment for conservation
        # logging.warning(f"PDB coordinates length ({len(pdb_coords)}) is less than sequence length ({num_nodes}). Truncating sequence for graph.")
        num_nodes = len(pdb_coords)
        seq = seq[:num_nodes]
    if len(conservation_scores) < num_nodes:
        # logging.warning(f"Conservation scores length ({len(conservation_scores)}) is less than sequence length ({num_nodes}). Truncating sequence for graph.")
        num_nodes = len(conservation_scores)
        seq = seq[:num_nodes]
        pdb_coords = pdb_coords[:num_nodes]

    node_features_list = []
    list_aa = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    for idx, aa in enumerate(seq):
        if aa not in list_aa:
            # logging.warning(f"Unknown amino acid 	'{aa}	' at position {idx} replaced with 'A' for feature generation.")
            aa = "A" # Replace unknown/gap with a common AA for feature purposes
        features = [properties[dict_name].get(aa, 0) for dict_name in properties]
        features.append(conservation_scores[idx])
        node_features_list.append(features)
    
    node_features = torch.tensor(node_features_list, dtype=torch.float).to(device)
    if node_features.numel() > 0:
        node_features = (node_features - node_features.mean(dim=0)) / (node_features.std(dim=0) + 1e-6)
    else:
        # Handle case with no node features (e.g. empty sequence after truncation)
        # This should ideally not happen with proper input validation
        return Data(x=torch.empty((0, len(properties) + 1), dtype=torch.float).to(device), edge_index=torch.empty((2,0), dtype=torch.long).to(device), edge_attr=torch.empty((0), dtype=torch.float).to(device))

    edge_index_list = []
    edge_attr_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = np.linalg.norm(pdb_coords[i] - pdb_coords[j])
            if distance <= distance_threshold:
                edge_index_list.extend([[i, j], [j, i]])
                edge_attr_list.extend([np.exp(-distance)] * 2)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).to(device)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

def identify_mutations(wild_type_sequence, sequences_to_compare):
    """Identify mutations between wild type and a list of sequences."""
    all_mutations = set()
    for seq in sequences_to_compare:
        for i, (wt_aa, seq_aa) in enumerate(zip(wild_type_sequence, seq)):
            if i < len(wild_type_sequence) and i < len(seq): # Ensure within bounds
                 if wt_aa != seq_aa:
                    all_mutations.add((i, wt_aa, seq_aa))
    return list(all_mutations)

def generate_mutants_combinatorial(wild_type_sequence, active_mutations, inactive_mutations=None, num_mutations=2, max_mutants=5000, mutation_source="active"):
    mutant_sequences = []
    mutation_descriptions = []
    if mutation_source == "active":
        selected_mutations = active_mutations
    elif mutation_source == "inactive":
        if inactive_mutations is None: raise ValueError("Inactive mutations needed for 'inactive' source.")
        selected_mutations = inactive_mutations
    elif mutation_source == "paired":
        if inactive_mutations is None: raise ValueError("Inactive mutations needed for 'paired' source.")
        combined_mutations = active_mutations + inactive_mutations
        all_combos = list(combinations(combined_mutations, num_mutations))
        active_set, inactive_set = set(active_mutations), set(inactive_mutations)
        mutation_combinations = [combo for combo in all_combos if any(m in active_set for m in combo) and any(m in inactive_set for m in combo)]
    else:
        raise ValueError(f"Unknown mutation source: {mutation_source}")
    
    if mutation_source != "paired": # For active/inactive, just combine the selected mutations
        if not selected_mutations: return [], []
        mutation_combinations = list(combinations(selected_mutations, num_mutations))

    if len(mutation_combinations) > max_mutants:
        logging.info(f"Sampling {max_mutants} from {len(mutation_combinations)} combinations.")
        mutation_combinations = random.sample(mutation_combinations, max_mutants)
    
    for combo in mutation_combinations:
        mutant = list(wild_type_sequence)
        desc = []
        valid_combo = True
        for pos, wt_aa, new_aa in combo:
            if pos < len(mutant):
                mutant[pos] = new_aa
                desc.append(f"{wt_aa}{pos+1}{new_aa}")
            else:
                valid_combo = False; break # Position out of bounds
        if valid_combo:
            mutant_seq = "".join(mutant)
            if mutant_seq not in mutant_sequences:
                mutant_sequences.append(mutant_seq)
                mutation_descriptions.append(";".join(desc))
    return mutant_sequences, mutation_descriptions

def generate_mutants_weighted(wild_type_sequence, mutations, num_mutations, weights, max_mutants=5000):
    mutant_sequences, mutation_descriptions = [], []
    positions = {}
    for pos, wt_aa, mut_aa in mutations: positions.setdefault(pos, []).append((wt_aa, mut_aa))
    position_weights = {pos: weights.get(pos, 1.0) for pos in positions.keys()}
    total_weight = sum(position_weights.values())
    if total_weight == 0: return [], [] # Avoid division by zero if no weights
    normalized_weights = {pos: w/total_weight for pos, w in position_weights.items()}
    weighted_positions, weighted_probs = list(normalized_weights.keys()), list(normalized_weights.values())
    
    for _ in range(min(max_mutants, 5000)):
        sampled_positions, remaining_positions, remaining_probs = [], weighted_positions[:], weighted_probs[:]
        for _ in range(min(num_mutations, len(weighted_positions))):
            if not remaining_positions: break
            total_prob = sum(remaining_probs)
            if total_prob == 0: break
            norm_probs = [p/total_prob for p in remaining_probs]
            idx = np.random.choice(len(remaining_positions), p=norm_probs)
            pos = remaining_positions.pop(idx); remaining_probs.pop(idx)
            sampled_positions.append(pos)
        
        mutant, desc = list(wild_type_sequence), []
        valid_mutant = True
        for pos in sampled_positions:
            if pos < len(wild_type_sequence) and pos in positions and positions[pos]:
                wt_aa_orig, mut_aa = random.choice(positions[pos]) # wt_aa_orig is the original WT at this pos from mutation list
                mutant[pos] = mut_aa
                desc.append(f"{wild_type_sequence[pos]}{pos+1}{mut_aa}") # Use actual WT for description
            else:
                valid_mutant = False; break
        if valid_mutant:
            mutant_seq = "".join(mutant)
            if mutant_seq not in mutant_sequences:
                mutant_sequences.append(mutant_seq); mutation_descriptions.append(";".join(desc))
    return mutant_sequences, mutation_descriptions

def generate_mutants_evolutionary(wild_type_sequence, active_sequences, num_generations=5, population_size=100, mutation_rate=0.05, max_mutants=5000):
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    population = active_sequences[:]
    if len(population) < population_size:
        population.append(wild_type_sequence)
        while len(population) < population_size:
            mutant = list(wild_type_sequence)
            for _ in range(random.randint(1,3)):
                pos = random.randint(0, len(wild_type_sequence)-1)
                new_aa = random.choice(amino_acids)
                while new_aa == mutant[pos]: new_aa = random.choice(amino_acids)
                mutant[pos] = new_aa
            mutant_seq = "".join(mutant)
            if mutant_seq not in population: population.append(mutant_seq)
    
    all_mutants, mutation_descriptions = set(population), []
    for _ in range(num_generations):
        offspring = []
        while len(offspring) < population_size:
            p1, p2 = random.sample(population, 2)
            cp = random.randint(1, len(wild_type_sequence)-2)
            child_list = list(p1[:cp] + p2[cp:])
            for i in range(len(child_list)):
                if random.random() < mutation_rate:
                    new_aa = random.choice(amino_acids)
                    while new_aa == child_list[i]: new_aa = random.choice(amino_acids)
                    child_list[i] = new_aa
            child = "".join(child_list)
            if child not in all_mutants: offspring.append(child); all_mutants.add(child)
        population = offspring if offspring else population # Avoid empty population
        if not population: break # Stop if population dies out
        
    mutant_sequences_final = list(all_mutants)[:max_mutants]
    for mutant in mutant_sequences_final:
        desc = [f"{wt_aa}{i+1}{mut_aa}" for i, (wt_aa, mut_aa) in enumerate(zip(wild_type_sequence, mutant)) if wt_aa != mut_aa]
        mutation_descriptions.append(";".join(desc))
    return mutant_sequences_final, mutation_descriptions

def screen_mutants(model, mutant_features_graphs, batch_size=64):
    if not mutant_features_graphs: return [], [] # Handle empty list
    mutant_loader = DataLoader(mutant_features_graphs, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probabilities, all_fingerprints = [], []
    with torch.no_grad():
        for batch_data in mutant_loader:
            batch_data = batch_data.to(device)
            # batch_data = relabel_node_indices(batch_data) # Relabeling might not be needed if graphs are processed carefully
            outputs, fingerprints = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch)
            all_probabilities.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            all_fingerprints.extend(fingerprints.cpu().numpy())
    return all_probabilities, all_fingerprints

def calculate_position_weights(wild_type_sequence, active_sequences, inactive_sequences=None):
    active_mut_counts, inactive_mut_counts = {}, {}
    for seq_list, counts_dict in [(active_sequences, active_mut_counts), (inactive_sequences or [], inactive_mut_counts)]:
        for seq in seq_list:
            for i, (wt_aa, seq_aa) in enumerate(zip(wild_type_sequence, seq)):
                if i < len(wild_type_sequence) and i < len(seq) and wt_aa != seq_aa:
                    counts_dict[i] = counts_dict.get(i, 0) + 1
    weights = {}
    all_positions = set(active_mut_counts.keys()) | set(inactive_mut_counts.keys())
    for pos in all_positions:
        act_c, inact_c = active_mut_counts.get(pos,0), inactive_mut_counts.get(pos,0)
        act_freq = act_c / len(active_sequences) if active_sequences else 0
        inact_freq = inact_c / len(inactive_sequences) if inactive_sequences else 0
        weights[pos] = (act_freq + 0.01) / (inact_freq + 0.01)
    return weights

def visualize_mutant_probabilities(probabilities, mutation_descriptions, output_dir, optimal_threshold=optimal_threshold):
    """Visualize the distribution of mutant probabilities and top mutations."""
    if not probabilities: 
        logging.info("No probabilities to visualize.")
        return None, None

    prob_hist_path = os.path.join(output_dir, "mutant_probabilities_histogram.png")
    top_mut_path = os.path.join(output_dir, "mutant_probabilities_top20.png")

    plt.figure(figsize=(10, 6))
    plt.hist(probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel("Predicted Probability", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of Mutant Probabilities", fontsize=14)
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(prob_hist_path, dpi=300)
    plt.close()
    logging.info(f"Saved probability distribution histogram to {prob_hist_path}")

    if len(probabilities) > 0:
        # Sort by probability to get top N
        sorted_indices = np.argsort(probabilities)[::-1] # Descending
        top_n = min(20, len(probabilities))
        top_indices_n = sorted_indices[:top_n]
        
        top_probs = [probabilities[i] for i in top_indices_n]
        top_mutations_desc = [mutation_descriptions[i] if i < len(mutation_descriptions) else "N/A" for i in top_indices_n]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_mutations_desc)), top_probs, color='lightcoral', edgecolor='black')
        plt.yticks(range(len(top_mutations_desc)), top_mutations_desc, fontsize=10)
        plt.xlabel("Predicted Probability", fontsize=12)
        plt.ylabel("Mutation", fontsize=12)
        plt.title(f"Top {top_n} Predicted Mutations", fontsize=14)
        plt.gca().invert_yaxis() # Display highest probability at the top
        plt.grid(axis='x', alpha=0.75)
        plt.tight_layout()
        plt.savefig(top_mut_path, dpi=300)
        plt.close()
        logging.info(f"Saved top mutations plot to {top_mut_path}")
        return prob_hist_path, top_mut_path
    return prob_hist_path, None

##########################
### MAIN PIPELINE CODE ###
##########################

def run_screening_pipeline(
    active_file, inactive_file, wild_type_file, pdb_file, properties_file, model_path,
    output_dir, progress_file, hidden_dim=128, dropout_rate=0.25, ratio=0.70,
    num_mutations=2, generation_method="combinatorial", max_mutants=5000, 
    optimal_threshold=optimal_threshold, # Added optimal_threshold
    seed=DEFAULT_SEED, num_cores=-1
):
    logging.info("Starting screening pipeline.")
    update_progress_file(0, "Starting...", progress_file)
    
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if device.type == "cuda": torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    logging.info(f"Using SEED: {seed}, DEVICE: {device}, Optimal Threshold: {optimal_threshold}")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    update_progress_file(5, "Reading input files...", progress_file)
    active_sequences, _ = read_sequence_file(active_file, is_active=True)
    inactive_sequences, _ = read_sequence_file(inactive_file, is_active=False)
    wild_type_sequence = read_fasta(wild_type_file)

    update_progress_file(10, "Identifying mutations...", progress_file)
    active_mutations = identify_mutations(wild_type_sequence, active_sequences)
    inactive_mutations = identify_mutations(wild_type_sequence, inactive_sequences)
    logging.info(f"Identified {len(active_mutations)} active & {len(inactive_mutations)} inactive mutations.")

    update_progress_file(15, f"Generating mutants ({generation_method})...", progress_file)
    if generation_method == "combinatorial":
        mutant_sequences, mutation_descriptions = generate_mutants_combinatorial(
            wild_type_sequence, active_mutations, inactive_mutations, num_mutations, max_mutants, mutation_source="active"
        )
    elif generation_method == "weighted":
        # Calculate weights if using weighted generation
        weights = calculate_position_weights(wild_type_sequence, active_sequences, inactive_sequences)
        mutant_sequences, mutation_descriptions = generate_mutants_weighted(
            wild_type_sequence, active_mutations + inactive_mutations, num_mutations, weights, max_mutants
        )
    elif generation_method == "evolutionary": # Corrected case
        mutant_sequences, mutation_descriptions = generate_mutants_evolutionary(
            wild_type_sequence, active_sequences, max_mutants=max_mutants
        )
    else:
        raise ValueError(f"Unknown generation method: {generation_method}")
    logging.info(f"Generated {len(mutant_sequences)} unique mutants.")

    if not mutant_sequences:
        logging.warning("No mutants were generated. Skipping prediction.")
        ranked_mutants_path = os.path.join(output_dir, "ranked_mutants.csv")
        pd.DataFrame(columns=["rank", "mutation", "probability"]).to_csv(ranked_mutants_path, index=False)
        # Create empty plot files for consistency in results.json
        prob_hist_path = os.path.join(output_dir, "mutant_probabilities_histogram.png")
        top_mut_path = os.path.join(output_dir, "mutant_probabilities_top20.png")
        open(prob_hist_path, 'a').close()
        open(top_mut_path, 'a').close()
        update_progress_file(100, "Screening complete (no mutants generated).", progress_file)
        return ranked_mutants_path, None, prob_hist_path, top_mut_path, None # Added None for mutant_fasta_path

    # Save generated mutant sequences to FASTA
    mutant_fasta_path = os.path.join(output_dir, "generated_mutant_sequences.fasta")
    with open(mutant_fasta_path, "w") as f_mut_fasta:
        for i, seq_mut in enumerate(mutant_sequences):
            f_mut_fasta.write(f">mutant_{i}|{mutation_descriptions[i]}\n{seq_mut}\n")
    logging.info(f"Saved generated mutant sequences to {mutant_fasta_path}")

    update_progress_file(30, "Extracting coordinates...", progress_file)
    mutant_coords_list = Parallel(n_jobs=num_cores, backend="multiprocessing")(
        delayed(extract_coordinates_from_pdb)(pdb_file, [seq], wild_type_sequence, seed) for seq in mutant_sequences
    )
    mutant_coords = [coords[0] for coords in mutant_coords_list if coords] # Ensure coords is not empty
    if len(mutant_coords) != len(mutant_sequences):
        logging.warning(f"Coordinate extraction mismatch: {len(mutant_coords)} coords for {len(mutant_sequences)} sequences. This might lead to issues.")
        # Filter sequences/descriptions to match available coordinates
        valid_indices = [i for i, mc in enumerate(mutant_coords_list) if mc]
        mutant_sequences = [mutant_sequences[i] for i in valid_indices]
        mutation_descriptions = [mutation_descriptions[i] for i in valid_indices]
        if not mutant_sequences: # If no valid coords, stop
            logging.error("No valid coordinates could be extracted for any mutant. Aborting.")
            # Similar to no mutants generated case
            ranked_mutants_path = os.path.join(output_dir, "ranked_mutants.csv")
            pd.DataFrame(columns=["rank", "mutation", "probability"]).to_csv(ranked_mutants_path, index=False)
            prob_hist_path = os.path.join(output_dir, "mutant_probabilities_histogram.png")
            top_mut_path = os.path.join(output_dir, "mutant_probabilities_top20.png")
            open(prob_hist_path, 'a').close(); open(top_mut_path, 'a').close()
            update_progress_file(100, "Screening failed (coordinate extraction error).", progress_file)
            return ranked_mutants_path, None, prob_hist_path, top_mut_path, mutant_fasta_path

    update_progress_file(40, "Loading properties & calculating conservation...", progress_file)
    properties = load_dictionaries(properties_file)
    
    # Use the generated mutant FASTA for conservation
    conservation_scores = calculate_conservation_scores(mutant_fasta_path, output_dir)

    update_progress_file(60, "Generating graphs...", progress_file)
    # Adjust conservation scores length if needed (due to potential PDB coord mismatch)
    # This logic needs to be robust, considering sequence lengths can vary
    # The sequence_to_graph function now handles internal truncation based on shortest feature
    mutant_graphs = Parallel(n_jobs=num_cores, backend="multiprocessing")(
        delayed(sequence_to_graph)(seq, properties, conservation_scores, coords)
        for seq, coords in zip(tqdm(mutant_sequences, desc="Graph Generation", leave=False), mutant_coords)
    )
    mutant_graphs = [g for g in mutant_graphs if g.num_nodes > 0] # Filter out empty graphs
    if not mutant_graphs:
        logging.error("No valid graphs generated for mutants. Aborting.")
        # Similar to no mutants generated case
        ranked_mutants_path = os.path.join(output_dir, "ranked_mutants.csv")
        pd.DataFrame(columns=["rank", "mutation", "probability"]).to_csv(ranked_mutants_path, index=False)
        prob_hist_path = os.path.join(output_dir, "mutant_probabilities_histogram.png")
        top_mut_path = os.path.join(output_dir, "mutant_probabilities_top20.png")
        open(prob_hist_path, 'a').close(); open(top_mut_path, 'a').close()
        update_progress_file(100, "Screening failed (graph generation error).", progress_file)
        return ranked_mutants_path, None, prob_hist_path, top_mut_path, mutant_fasta_path

    update_progress_file(75, "Loading model...", progress_file)
    in_features = mutant_graphs[0].num_node_features
    model = GNNFeatureExtractor(in_features, hidden_dim, dropout_rate, ratio).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    update_progress_file(80, "Predicting probabilities...", progress_file)
    all_preds, _ = screen_mutants(model, mutant_graphs) # Fingerprints not used here
    if len(all_preds) != len(mutant_sequences):
        logging.warning(f"Prediction/mutant count mismatch: {len(all_preds)} preds for {len(mutant_sequences)} mutants. Truncating to shortest.")
        min_len = min(len(all_preds), len(mutant_sequences))
        all_preds = all_preds[:min_len]
        # Ensure mutation_descriptions also matches this length for results_df
        mutation_descriptions_for_df = mutation_descriptions[:min_len]
    else:
        mutation_descriptions_for_df = mutation_descriptions

    update_progress_file(95, "Ranking mutants & visualizing...", progress_file)
    results_df = pd.DataFrame({"mutation": mutation_descriptions_for_df, "probability": all_preds})
    results_df = results_df.sort_values(by="probability", ascending=False)
    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    ranked_mutants_path = os.path.join(output_dir, "ranked_mutants.csv")
    results_df.to_csv(ranked_mutants_path, index=False)
    logging.info(f"Ranked mutants saved to {ranked_mutants_path}")

    # Visualize probabilities
    prob_hist_path, top_mut_path = visualize_mutant_probabilities(all_preds, mutation_descriptions_for_df, output_dir, optimal_threshold)

    wild_type_pred = None
    try:
        # Predict wild type probability for reference
        wt_coords_single = extract_coordinates_from_pdb(pdb_file, [wild_type_sequence], wild_type_sequence, seed)[0]
        # Use the same conservation scores context as mutants for WT graph
        wt_graph = sequence_to_graph(wild_type_sequence, properties, conservation_scores, wt_coords_single)
        if wt_graph.num_nodes > 0:
            wt_graph_batch = Batch.from_data_list([wt_graph]).to(device)
            with torch.no_grad():
                wt_output, _ = model(wt_graph_batch.x, wt_graph_batch.edge_index, wt_graph_batch.edge_attr, wt_graph_batch.batch)
                wild_type_pred = torch.sigmoid(wt_output).cpu().item()
            logging.info(f"Wild Type Predicted Probability: {wild_type_pred:.4f}")
    except Exception as wt_e:
        logging.warning(f"Could not predict wild type probability: {wt_e}")

    update_progress_file(100, "Screening complete.", progress_file)
    logging.info("Screening pipeline finished successfully.")
    return ranked_mutants_path, wild_type_pred, prob_hist_path, top_mut_path, mutant_fasta_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StablyzeGraph Screening Pipeline")
    parser.add_argument("--active", required=True, help="Path to active sequences CSV file")
    parser.add_argument("--inactive", required=True, help="Path to inactive sequences CSV file")
    parser.add_argument("--wild_type", required=True, help="Path to wild type sequence FASTA file")
    parser.add_argument("--pdb", required=True, help="Path to PDB structure file")
    parser.add_argument("--properties", required=True, help="Path to amino acid properties CSV file")
    parser.add_argument("--model_path", required=True, help="Path to the trained GNN model (.pth file)")
    parser.add_argument("--output", required=True, help="Directory to save output files")
    parser.add_argument("--progress_file", required=True, help="Path to JSON file for progress updates")
    
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--dropout_rate", type=float, default=0.25, help="Dropout rate")
    parser.add_argument("--ratio", type=float, default=0.70, help="Pooling ratio")
    
    parser.add_argument("--num_mutations", type=int, default=2, help="Number of mutations")
    parser.add_argument("--generation_method", choices=["combinatorial", "weighted", "evolutionary"], default="combinatorial", help="Mutant generation method") # Corrected 'Evolutionary' to 'evolutionary'
    parser.add_argument("--max_mutants", type=int, default=5000, help="Max mutants to generate")
    parser.add_argument("--optimal_threshold", type=float, default=optimal_threshold, help="Optimal threshold for visualization") # Added
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--num_cores", type=int, default=-1, help="Number of CPU cores")

    args = parser.parse_args()
    result_file = os.path.join(args.output, "results.json")
    
    try:
        ranked_csv, wt_pred, hist_png, top_png, mut_fasta = run_screening_pipeline(
            args.active, args.inactive, args.wild_type, args.pdb, args.properties, args.model_path,
            args.output, args.progress_file, args.hidden_dim, args.dropout_rate, args.ratio,
            args.num_mutations, args.generation_method, args.max_mutants, args.optimal_threshold, # Passed optimal_threshold
            args.seed, args.num_cores
        )
        result_data = {
            "status": "success", "message": "Screening completed successfully.",
            "ranked_mutants_file": ranked_csv,
            "wild_type_prediction": f"{wt_pred:.4f}" if wt_pred is not None else "N/A",
            "probability_histogram_file": hist_png,
            "top_mutations_plot_file": top_png,
            "generated_mutants_fasta_file": mut_fasta
        }
        with open(result_file, "w") as f: json.dump(result_data, f, indent=4)
            
    except Exception as e:
        logging.exception("Error in screening pipeline:")
        result_data = {"status": "error", "error": str(e), "timestamp": time.time()}
        try:
            with open(result_file, "w") as f: json.dump(result_data, f, indent=4)
        except Exception as write_e:
            logging.error(f"Critical: Could not write error status to results file {result_file}: {write_e}")
        sys.exit(1)
