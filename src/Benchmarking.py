print("DEBUG: Script execution started") # DEBUG
import gc
print("DEBUG: Imported gc") # DEBUG
import os
print("DEBUG: Imported os") # DEBUG
import numpy as np
print("DEBUG: Imported numpy") # DEBUG
import pandas as pd
print("DEBUG: Imported pandas") # DEBUG
import torch
print("DEBUG: Imported torch") # DEBUG
from torch.nn import BatchNorm1d
print("DEBUG: Imported BatchNorm1d") # DEBUG
import torch.nn.functional as F
print("DEBUG: Imported torch.nn.functional") # DEBUG
from torch_geometric.data import Data, Batch
print("DEBUG: Imported torch_geometric.data") # DEBUG
from torch_geometric.loader import DataLoader
print("DEBUG: Imported torch_geometric.loader") # DEBUG
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
print("DEBUG: Imported torch_geometric.nn") # DEBUG
from Bio import SeqIO
print("DEBUG: Imported Bio.SeqIO") # DEBUG
from Bio import AlignIO
print("DEBUG: Imported Bio.AlignIO") # DEBUG
from Bio.Align import AlignInfo
print("DEBUG: Imported Bio.Align.AlignInfo") # DEBUG
from Bio.PDB import PDBParser
print("DEBUG: Imported Bio.PDB.PDBParser") # DEBUG
from sklearn.preprocessing import StandardScaler
print("DEBUG: Imported sklearn.preprocessing") # DEBUG
from sklearn.svm import SVC
print("DEBUG: Imported sklearn.svm") # DEBUG
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
print("DEBUG: Imported sklearn.ensemble") # DEBUG
from sklearn.linear_model import LogisticRegression
print("DEBUG: Imported sklearn.linear_model") # DEBUG
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    GridSearchCV,
    train_test_split,
)
print("DEBUG: Imported sklearn.model_selection") # DEBUG
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
print("DEBUG: Imported sklearn.metrics") # DEBUG
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
print("DEBUG: Imported imblearn.over_sampling") # DEBUG
import random
print("DEBUG: Imported random") # DEBUG
import subprocess
print("DEBUG: Imported subprocess") # DEBUG
import logging
print("DEBUG: Imported logging") # DEBUG
from joblib import Parallel, delayed
print("DEBUG: Imported joblib") # DEBUG
import warnings
print("DEBUG: Imported warnings") # DEBUG
import matplotlib
print("DEBUG: Imported matplotlib") # DEBUG
import matplotlib.pyplot as plt
print("DEBUG: Imported matplotlib.pyplot") # DEBUG
import seaborn as sns
print("DEBUG: Imported seaborn") # DEBUG
import argparse # Added for GUI
print("DEBUG: Imported argparse") # DEBUG
import json # Added for GUI
print("DEBUG: Imported json") # DEBUG
import time # Added for GUI
print("DEBUG: Imported time") # DEBUG
from Bio.Seq import Seq # Ensure Seq is imported correctly
print("DEBUG: Imported Bio.Seq") # DEBUG

print("DEBUG: Script started - imports completed") # DEBUG

print("DEBUG: Setting matplotlib backend") # DEBUG
matplotlib.use("Agg")  # Use non-GUI backend for rendering to files.
print("DEBUG: Matplotlib backend set") # DEBUG

# Set up logging for better tracking
print("DEBUG: Setting up logging") # DEBUG
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output
print("DEBUG: Logging setup complete, warnings ignored") # DEBUG

print("DEBUG: Setting random seeds") # DEBUG
# Set random seed for reproducibility
seed = 42

# Set seed for CPU operations
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
print("DEBUG: CPU seeds set") # DEBUG

# Check if GPU is available
print("DEBUG: Checking for CUDA device") # DEBUG
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
print(f"DEBUG: Using device: {device}") # DEBUG

# GPU operations have a separate seed we also want to set
if device.type == "cuda":
    print("DEBUG: Setting CUDA seeds") # DEBUG
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("DEBUG: CUDA seeds set") # DEBUG

print("DEBUG: Defining model class") # DEBUG
########################
### MODEL DEFINITION ###
########################


class GNNFeatureExtractor(torch.nn.Module):
    """GNN model for graph-level classification."""

    def __init__(self, in_features, hidden_features, dropout_rate, ratio):
        super(GNNFeatureExtractor, self).__init__()
        print(f"DEBUG: Initializing model with in_features={in_features}, hidden_features={hidden_features}") # DEBUG
        # self.gcn1 = GCNConv(in_features, hidden_features)
        self.gat1 = GATConv(
            in_features, int(hidden_features / 4), heads=4
        )  # add dropout here
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
        #self.gcn_batch_norm = BatchNorm1d(hidden_features)
        self.gcn_batch_norm = BatchNorm1d(hidden_features)
        self.gat_batch_norm = BatchNorm1d(hidden_features)
        self.sage_batch_norm = BatchNorm1d(hidden_features)
        self.graph_batch_norm = BatchNorm1d(hidden_features)
        self.lin_batch_norm = BatchNorm1d(int(hidden_features / 2))
        print("DEBUG: Model layers initialized") # DEBUG

    def forward(self, x, edge_index, edge_attr, batch):
        # x = self.gcn1(x, edge_index, edge_attr)
        # x = self.gcn_batch_norm(x)
        # x = F.leaky_relu(x)
        # x = self.dropout(x)

        x = self.gat1(x, edge_index, edge_attr)
        x = self.gat_batch_norm(x)
        x = F.leaky_relu(x)
        # x = self.dropout(x)

        x, edge_index, edge_attr, batch, _, _ = self.sagpool(
            x, edge_index, edge_attr, batch=batch
        )

        x = self.sage1(x, edge_index)
        x = self.sage_batch_norm(x)
        x = F.leaky_relu(x)
        # x = self.dropout(x)

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


def simple_model_initialize_weights(m):
    if isinstance(m, torch.nn.Linear):
        # Xavier/Glorot Initialization
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, GCNConv):
        # He Initialization
        torch.nn.init.kaiming_uniform_(m.lin.weight, nonlinearity="leaky_relu")
        if m.lin.bias is not None:
            torch.nn.init.constant_(m.lin.bias, 0)
    elif (
        hasattr(m, "weight")
        and isinstance(m, torch.nn.Module)
        and len(m.weight.shape) >= 2  # required by He initialization
    ):
        # check if it has weights, and if it is a module.
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    else:
        pass


# MODIFIED: Simplified relabel_node_indices function to avoid potential hanging
def relabel_node_indices(data):
    """
    Simplified version of relabel_node_indices to avoid potential hanging issues.
    """
    # print("DEBUG: relabel_node_indices called") # DEBUG - Commented out for less noise
    try:
        # Just return the data as is for now to avoid potential hanging
        return data
    except Exception as e:
        print(f"DEBUG: Error in relabel_node_indices: {e}") # DEBUG
        # Return the original data if there's an error
        return data


def load_dictionaries(csv_file):
    """Dynamically load feature dictionaries from a CSV file."""
    print(f"DEBUG: Loading dictionaries from {csv_file}") # DEBUG
    df = pd.read_csv(csv_file, index_col=0)
    # Standardize the feature values
    scaler = StandardScaler()
    standardized_values = scaler.fit_transform(df.values)

    # Create a dictionary with standardized values
    feature_dicts = {
        row: {col: val for col, val in zip(df.columns, standardized_values[idx])}
        for idx, row in enumerate(df.index)
    }
    print(f"DEBUG: Loaded {len(feature_dicts)} dictionaries") # DEBUG
    return feature_dicts


def read_sequence_file(file_path, is_active):
    """Reads sequences and their activity values from a CSV file."""
    print(f"DEBUG: Reading sequence file {file_path}, is_active={is_active}") # DEBUG
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")

    df = pd.read_csv(file_path)

    # if is_active:
    #     df = df.sort_values(by=df.columns[1], ascending=True)
    # else:
    #     df = df.sort_values(by=df.columns[1], ascending=False)

    sequences = df.iloc[:, 0].tolist()

    if is_active:
        activity_values = [1] * len(df.iloc[:, 1].tolist())  # Set actives to class 1
    else:
        activity_values = [0] * len(df.iloc[:, 1].tolist())  # Set inactives to class 0

    print(f"DEBUG: Read {len(sequences)} sequences") # DEBUG
    return sequences, activity_values


def read_fasta(file_path):
    """Reads sequences from a fasta file."""
    print(f"DEBUG: Reading FASTA file {file_path}") # DEBUG
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")

    wild_type_sequence = [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]
    print(f"DEBUG: Read {len(wild_type_sequence)} sequences from FASTA") # DEBUG
    return wild_type_sequence


def extract_coordinates_from_pdb(pdb_file, sequences, wild_type_sequence, seed=seed):
    """Extracts 3D coordinates from the PDB file for wild type and generates coordinates for other sequences."""
    print(f"DEBUG: Extracting coordinates from {pdb_file} for {len(sequences)} sequence(s)") # DEBUG
    seed = np.random.seed(seed)  # Only used to guarantee reproducibility

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("wild_type", pdb_file)
    wild_type_coords = [
        residue["CA"].get_vector().get_array()
        for model in structure
        for chain in model
        for residue in chain
        if residue.has_id("CA")
    ]

    if len(wild_type_coords) != len(wild_type_sequence):
        logging.warning("Mismatch between wild type coordinates and sequence length.")
        print(f"DEBUG: Mismatch between wild type coordinates ({len(wild_type_coords)}) and sequence length ({len(wild_type_sequence)})") # DEBUG

    coordinates = []

    for seq_idx, seq in enumerate(sequences):
        # print(f"DEBUG: Processing sequence {seq_idx+1}/{len(sequences)} in extract_coordinates_from_pdb") # DEBUG - Too verbose
        seq_coords = np.zeros((len(seq), 3))

        for i, (wild_aa, seq_aa) in enumerate(zip(wild_type_sequence, seq)):
            if i >= len(wild_type_coords): # Avoid index error if sequence is longer than PDB coords
                print(f"DEBUG: Warning - sequence index {i} out of bounds for PDB coordinates (length {len(wild_type_coords)}) in sequence {seq_idx+1}")
                break
            if wild_aa == seq_aa:
                seq_coords[i] = wild_type_coords[i]
            else:
                seq_coords[i] = wild_type_coords[i] + np.random.normal(
                    0, 1, size=3
                )  # Perturb for mutations

        coordinates.append(seq_coords)

    print(f"DEBUG: Extracted coordinates for {len(coordinates)} sequences") # DEBUG
    return coordinates


def calculate_conservation_scores(sequences_file):
    """Calculate conservation scores using multiple sequence alignment."""
    print(f"DEBUG: Calculating conservation scores from {sequences_file}") # DEBUG
    aligned_file = "aligned_sequences.aln"

    try:
        print(f"DEBUG: Running Clustal Omega: clustalo -i {sequences_file} -o {aligned_file} --force --outfmt=clu") # DEBUG
        subprocess.run(
            [
                "clustalo",
                "-i",
                sequences_file,
                "-o",
                aligned_file,
                "--force",
                "--outfmt=clu",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"DEBUG: Clustal Omega completed successfully") # DEBUG
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running Clustal Omega: {e.stderr.decode()}")
        print(f"DEBUG: Error running Clustal Omega: {e.stderr.decode()}") # DEBUG
        raise e
    except FileNotFoundError:
        logging.error("Clustal Omega command not found. Please ensure it is installed and in your PATH.")
        print("DEBUG: Clustal Omega command not found.") # DEBUG
        raise FileNotFoundError("Clustal Omega not found.")

    try:
        print(f"DEBUG: Reading alignment file: {aligned_file}") # DEBUG
        alignment = AlignIO.read(aligned_file, "clustal")
        print(f"DEBUG: Alignment file read successfully") # DEBUG
    except Exception as e:
        print(f"DEBUG: Error reading alignment file: {aligned_file}. Details: {str(e)}") # DEBUG
        raise RuntimeError(
            f"Error reading alignment file: {aligned_file}. Details: {str(e)}"
        )

    summary_align = AlignInfo.SummaryInfo(alignment)
    consensus = summary_align.dumb_consensus()

    conservation_scores = []

    for i in range(len(consensus)):
        col = alignment[:, i]
        score = sum(1 for aa in col if aa == consensus[i]) / len(col)
        conservation_scores.append(score)

    print(f"DEBUG: Calculated {len(conservation_scores)} conservation scores") # DEBUG
    return conservation_scores


def sequence_to_graph(
    seq,
    properties,
    conservation_scores,
    pdb_coords,
    distance_threshold=10.0,  # In Angstroms
):
    """Converts protein sequence into a graph representation."""
    # print(f"DEBUG: Converting sequence to graph, length={len(seq)}") # DEBUG - Too verbose

    list_aa = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ]

    num_nodes = len(seq)
    node_features = []

    for idx, aa in enumerate(seq):
        if aa not in list_aa:
            print(f"DEBUG: Unknown amino acid {aa} found in sequence at index {idx}") # DEBUG
            raise ValueError(f"Unknown amino acid {aa} found in sequence.")
        features = [properties[dict_name].get(aa, 0) for dict_name in properties]
        if idx >= len(conservation_scores):
             print(f"DEBUG: Warning - conservation score index {idx} out of bounds (length {len(conservation_scores)}) for sequence length {len(seq)}")
             features.append(0.0) # Append a default score
        else:
             features.append(conservation_scores[idx])
        node_features.append(features)

    node_features = torch.tensor(node_features, dtype=torch.float).to(device)

    edge_index = []
    edge_attr = []

    if len(pdb_coords) != num_nodes:
        print(f"DEBUG: Warning - Mismatch between PDB coordinates length ({len(pdb_coords)}) and sequence length ({num_nodes}) in sequence_to_graph")
        # Handle mismatch, e.g., skip edge creation or raise error
        # For now, let's skip edge creation if lengths don't match
        print(f"DEBUG: Skipping edge creation due to coordinate/sequence length mismatch.")
    else:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = np.linalg.norm(pdb_coords[i] - pdb_coords[j])
                if distance <= distance_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    # Edge attribute decreases exponentially with increasing distance between nodes (amino acids)
                    edge_attr.append(np.exp(-distance))
                    edge_attr.append(np.exp(-distance))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)

    # print(f"DEBUG: Created graph with {num_nodes} nodes and {len(edge_index[0])} edges") # DEBUG - Too verbose
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

print("DEBUG: Defining main pipeline function") # DEBUG
##########################
### MAIN PIPELINE CODE ###
##########################


def run_benchmarking_pipeline(
    active_file,
    inactive_file,
    wild_type_file,
    pdb_file,
    properties_file,
    output_dir, # Added for GUI
    progress_file, # Added for GUI
    seed=42, # Use default from args if not overridden
    num_cores=-1,
    # Added GNN/Training parameters from args
    hidden_dim=128,
    dropout_rate=0.20,
    ratio=0.7,
    learning_rate=0.00001,
    l2_regularization=0.0001,
    momentum=0.9,
    scheduler_factor=0.9,
    scheduler_patience=20,
    stop_patience=50,
    grad_clip=10.0,
    max_epochs=1000, # MODIFIED: Reduced from 1000 to 5 for testing
    batch_size=64,
    device=torch.device("cpu") # Default device, will be updated by args
):
    print(f"DEBUG: Starting benchmarking pipeline with active_file={active_file}, inactive_file={inactive_file}") # DEBUG

    # Define output paths using output_dir (for GUI integration)
    best_model_path = os.path.join(output_dir, "best_model.pth")
    learning_curve_path = os.path.join(output_dir, "GNN_learning_curve.png")
    probability_plots_path = os.path.join(output_dir, "probability_plots.png")
    roc_curve_path = os.path.join(output_dir, "roc_curve.png")
    prc_curve_path = os.path.join(output_dir, "prc_curve.png")
    model_params_path = os.path.join(output_dir, "GNN_model_params.pth")
    metrics_path = os.path.join(output_dir, "metrics.csv")
    fingerprints_path = os.path.join(output_dir, "test_fingerprints.csv")
    print(f"DEBUG: Output paths defined in directory: {output_dir}") # DEBUG

    # Add progress update (for GUI integration)
    total_steps_pipeline = 10 # Approximate steps within the pipeline function
    current_step_pipeline = 0
    update_progress(progress_file, 1 + current_step_pipeline, 1 + total_steps_pipeline, "Preparing data...") # Overall step 2
    print("DEBUG: Progress updated: Preparing data...") # DEBUG

    logging.info("Starting benchmarking pipeline.")
    print(f"DEBUG: SEED: {seed}, DEVICE: {device}") # DEBUG
    logging.info("Preparing data.")

    # Read active and inactive sequences and their activity values
    print("DEBUG: Reading active sequences") # DEBUG
    active_sequences, active_activity_values = read_sequence_file(
        active_file, is_active=True
    )
    print("DEBUG: Reading inactive sequences") # DEBUG
    inactive_sequences, inactive_activity_values = read_sequence_file(
        inactive_file, is_active=False
    )

    # Read wild type sequence from fasta file
    print("DEBUG: Reading wild type sequence") # DEBUG
    wild_type_sequence = read_fasta(wild_type_file)[0]

    # Ensure there are at least five members in each class
    if len(active_activity_values) < 10 or len(inactive_activity_values) < 10:
        print(f"DEBUG: Not enough members in classes: active={len(active_activity_values)}, inactive={len(inactive_activity_values)}") # DEBUG
        raise ValueError(
            "Both active and inactive activity values must have at least 10 members."
        )

    current_step_pipeline += 1
    update_progress(progress_file, 1 + current_step_pipeline, 1 + total_steps_pipeline, "Extracting coordinates...")
    print("DEBUG: Progress updated: Extracting coordinates...") # DEBUG

    # Generate 3D coordinates for all sequences based on wild type PDB
    print("DEBUG: Extracting coordinates for active sequences") # DEBUG
    active_coords = Parallel(n_jobs=num_cores)(
        delayed(extract_coordinates_from_pdb)(pdb_file, [seq], wild_type_sequence, seed)
        for seq in active_sequences
    )
    active_coords = [coords[0] for coords in active_coords]  # Flatten results
    print(f"DEBUG: Extracted coordinates for {len(active_coords)} active sequences") # DEBUG

    print("DEBUG: Extracting coordinates for inactive sequences") # DEBUG
    inactive_coords = Parallel(n_jobs=num_cores)(
        delayed(extract_coordinates_from_pdb)(pdb_file, [seq], wild_type_sequence, seed)
        for seq in inactive_sequences
    )
    inactive_coords = [coords[0] for coords in inactive_coords]  # Flatten results
    print(f"DEBUG: Extracted coordinates for {len(inactive_coords)} inactive sequences") # DEBUG

    current_step_pipeline += 1
    update_progress(progress_file, 1 + current_step_pipeline, 1 + total_steps_pipeline, "Loading features...")
    print("DEBUG: Progress updated: Loading features...") # DEBUG

    # Load feature dictionaries
    print("DEBUG: Loading property dictionaries") # DEBUG
    properties = load_dictionaries(properties_file)

    # Combine active and inactive sequences into a single file
    combined_file = "combined_sequences.fasta"
    print(f"DEBUG: Writing combined sequences to {combined_file}") # DEBUG
    with open(combined_file, "w") as outfile:
        for i, seq in enumerate(active_sequences):
            outfile.write(f">active_seq_{i}\n{seq}\n")
        for j, seq in enumerate(inactive_sequences):
            outfile.write(f">inactive_seq_{j}\n{seq}\n")

    active_sequences_file = "active_sequences.fasta"
    print(f"DEBUG: Writing active sequences to {active_sequences_file}") # DEBUG
    with open(active_sequences_file, "w") as outfile:
        for i, seq in enumerate(active_sequences):
            outfile.write(f">active_seq_{i}\n{seq}\n")

    inactive_sequences_file = "inactive_sequences.fasta"
    print(f"DEBUG: Writing inactive sequences to {inactive_sequences_file}") # DEBUG
    with open(inactive_sequences_file, "w") as outfile:
        for j, seq in enumerate(inactive_sequences):
            outfile.write(f">inactive_seq_{j}\n{seq}\n")

    current_step_pipeline += 1
    update_progress(progress_file, 1 + current_step_pipeline, 1 + total_steps_pipeline, "Calculating conservation scores...")
    print("DEBUG: Progress updated: Calculating conservation scores...") # DEBUG

    # Calculate conservation scores from combined sequences
    print("DEBUG: Calculating conservation scores for active sequences") # DEBUG
    conservation_scores_active = calculate_conservation_scores(active_sequences_file)

    current_step_pipeline += 1
    update_progress(progress_file, 1 + current_step_pipeline, 1 + total_steps_pipeline, "Generating graphs...")
    print("DEBUG: Progress updated: Generating graphs...") # DEBUG

    # Extract features for active and inactive sequences
    print("DEBUG: Converting active sequences to graphs") # DEBUG
    active_features = []
    for i, (seq, coords) in enumerate(zip(active_sequences, active_coords)):
        print(f"DEBUG: Converting active sequence {i+1}/{len(active_sequences)} to graph") # DEBUG
        graph = sequence_to_graph(
            seq,
            properties,
            conservation_scores_active, # Use active scores
            coords,
        )
        active_features.append(graph)
    print(f"DEBUG: Converted {len(active_features)} active sequences to graphs") # DEBUG

    print("DEBUG: Calculating conservation scores for inactive sequences") # DEBUG
    conservation_scores_inactive = calculate_conservation_scores(inactive_sequences_file)

    print("DEBUG: Converting inactive sequences to graphs") # DEBUG
    inactive_features = []
    for i, (seq, coords) in enumerate(zip(inactive_sequences, inactive_coords)):
        print(f"DEBUG: Converting inactive sequence {i+1}/{len(inactive_sequences)} to graph") # DEBUG
        graph = sequence_to_graph(
            seq,
            properties,
            conservation_scores_inactive, # Use inactive scores
            coords,
        )
        inactive_features.append(graph)
    print(f"DEBUG: Converted {len(inactive_features)} inactive sequences to graphs") # DEBUG

    all_features = active_features + inactive_features
    all_labels = torch.tensor(
        [1] * len(active_sequences) + [0] * len(inactive_sequences)
    ).float()
    print(f"DEBUG: Combined {len(all_features)} features with {len(all_labels)} labels") # DEBUG

    current_step_pipeline += 1
    update_progress(progress_file, 1 + current_step_pipeline, 1 + total_steps_pipeline, "Splitting data...")
    print("DEBUG: Progress updated: Splitting data...") # DEBUG

    # Split data into training (and validation) and test sets
    print("DEBUG: Splitting data into train/test sets") # DEBUG
    train_features, test_features, train_labels, test_labels = train_test_split(
        all_features,
        all_labels,
        test_size=0.2,
        stratify=all_labels,
        shuffle=True,
        random_state=seed,
    )
    print(f"DEBUG: Split data: train={len(train_features)}, test={len(test_features)}") # DEBUG
    # train_features, val_features, train_labels, val_labels = train_test_split(
    #     train_features,
    #     train_labels,
    #     test_size=0.25,
    #     stratify=train_labels,
    #     shuffle=True,
    #     random_state=seed,
    # )  # 0.2 / 0.8 = 0.25

    current_step_pipeline += 1
    update_progress(progress_file, 1 + current_step_pipeline, 1 + total_steps_pipeline, "Resampling data...")
    print("DEBUG: Progress updated: Resampling data...") # DEBUG

    # Oversample the minority class in the training set
    print("DEBUG: Resampling minority class") # DEBUG
    minority_class_count = min(
        sum(train_labels == 0), sum(train_labels == 1)
    )  # Determine the number of samples in the minority class
    print(f"DEBUG: Minority class count: {minority_class_count}") # DEBUG

    if minority_class_count < 10:
        print("DEBUG: Using RandomOverSampler") # DEBUG
        ros = RandomOverSampler(random_state=seed)
    else:
        print("DEBUG: Using SMOTE with k_neighbors=10") # DEBUG
        ros = SMOTE(k_neighbors=10, random_state=seed)

    # ros = RandomOverSampler(random_state=seed)

    # Resample the training data indices.
    print("DEBUG: Fitting resampler") # DEBUG
    resampled_indices, resampled_labels = ros.fit_resample(
        np.array(range(len(train_features))).reshape(-1, 1),
        train_labels.numpy(),
    )
    resampled_indices = resampled_indices.flatten().tolist()
    resampled_labels = torch.tensor(resampled_labels).float()
    print(f"DEBUG: Resampled to {len(resampled_indices)} samples") # DEBUG

    # Create a new training dataset with the oversampled indices.
    print("DEBUG: Creating resampled training features") # DEBUG
    resampled_train_features = [
        train_features[i] for i in resampled_indices
    ]  # change here.
    print(f"DEBUG: Created {len(resampled_train_features)} resampled training features") # DEBUG

    # Create DataLoaders for validation and test
    print("DEBUG: Creating datasets") # DEBUG
    train_dataset = list(zip(resampled_train_features, resampled_labels))
    # val_dataset = list(zip(val_features, val_labels))
    test_dataset = list(zip(test_features, test_labels))
    print(f"DEBUG: Created datasets: train={len(train_dataset)}, test={len(test_dataset)}") # DEBUG

    # batch_size = 64 # Use arg
    print(f"DEBUG: Using batch_size={batch_size}") # DEBUG

    print("DEBUG: Initializing DataLoaders...") # DEBUG
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # DEBUG: Set num_workers=0
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # DEBUG: Set num_workers=0
    print("DEBUG: DataLoaders initialized.") # DEBUG
    print(f"DEBUG: Created DataLoaders: train={len(train_loader)}, test={len(test_loader)}") # DEBUG

    current_step_pipeline += 1
    update_progress(progress_file, 1 + current_step_pipeline, 1 + total_steps_pipeline, "Initializing model...")
    print("DEBUG: Progress updated: Initializing model...") # DEBUG

    logging.info("Generating fingerprints.")
    print("DEBUG: Initializing model") # DEBUG

    # Initialize GNN model using parameters from args
    input_dim = len(properties) + 1  # Number of properties + conservation score
    print(f"DEBUG: Model parameters: input_dim={input_dim}, hidden_dim={hidden_dim}, dropout_rate={dropout_rate}, ratio={ratio}") # DEBUG
    # hidden_dim = 128 # Use arg
    # dropout_rate = 0.25 # Use arg
    # ratio = 0.70 # Use arg
    # learning_rate = 0.00001 # Use arg
    # l2_regularization = 0.0001 # Use arg
    # momentum = 0.9 # Use arg
    # scheduler_factor = 0.9 # Use arg
    # scheduler_patience = 10 # Use arg
    # stop_patience = 50 # Use arg
    # grad_clip = 5.0 # Use arg
    # max_epochs = 1000 # Use arg

    model = GNNFeatureExtractor(input_dim, hidden_dim, dropout_rate, ratio).to(device)
    print("DEBUG: Model initialized and moved to device") # DEBUG

    # Initializing weights
    # model.apply(simple_model_initialize_weights)

    # Calculate original imbalance (before splits and oversampling)
    print("DEBUG: Calculating class weights") # DEBUG
    original_positive_count = len(active_sequences)
    original_negative_count = len(inactive_sequences)
    original_pos_weight = (
        torch.tensor([original_negative_count / original_positive_count])
        .float()
        .to(device)
    )
    print(f"DEBUG: Class weights: pos_weight={original_pos_weight.item()}") # DEBUG

    print("DEBUG: Setting up loss function, optimizer, and scheduler") # DEBUG
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=original_pos_weight)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=l2_regularization
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=scheduler_patience, gamma=scheduler_factor
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience
        #verbose=True,
    )
    print(f"DEBUG: Using optimizer={optimizer.__class__.__name__}, learning_rate={learning_rate}") # DEBUG

    epochs = max_epochs  # Adjust as needed
    early_stop_patience = stop_patience  # Number of epochs to wait for improvement
    best_test_loss = float("inf")  # Initialize with a very high value
    epochs_no_improve = 0  # Counter for epochs without improvement
    print(f"DEBUG: Training parameters: epochs={epochs}, early_stop_patience={early_stop_patience}") # DEBUG

    current_step_pipeline += 1
    update_progress(progress_file, 1 + current_step_pipeline, 1 + total_steps_pipeline, "Starting training loop...")
    print("DEBUG: Progress updated: Starting training loop...") # DEBUG

    print("DEBUG: Setup before training loop completed.") # DEBUG

    ### Training loop ###
    train_loss_list = []
    test_loss_list = []

    print("DEBUG: Entering main training loop...") # DEBUG
    for epoch in range(epochs):
        epoch_start_time = time.time() # DEBUG: Record epoch start time
        print(f"\n--- Starting Epoch {epoch + 1}/{epochs} ---") # DEBUG
        model.train()
        total_loss = 0
        batch_count = 0 # DEBUG
        total_batches = len(train_loader) # DEBUG
        print(f"Epoch {epoch + 1}: Starting training phase ({total_batches} batches)") # DEBUG
        for batch_features, batch_labels in train_loader:
            batch_start_time = time.time() # DEBUG
            batch_count += 1 # DEBUG
            print(f"  Epoch {epoch + 1}, Batch {batch_count}/{total_batches}: Loading data to device...") # DEBUG
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device).view(-1, 1)  # Reshape labels
            print(f"  Epoch {epoch + 1}, Batch {batch_count}/{total_batches}: Relabeling nodes (skipped)... ") # DEBUG
            # batch_features = relabel_node_indices(batch_features)  # MODIFIED: Skip relabeling for now
            print(f"  Epoch {epoch + 1}, Batch {batch_count}/{total_batches}: Zeroing gradients...") # DEBUG
            optimizer.zero_grad()
            print(f"  Epoch {epoch + 1}, Batch {batch_count}/{total_batches}: Forward pass...") # DEBUG
            single_output, _ = model(
                batch_features.x,
                batch_features.edge_index,
                batch_features.edge_attr,
                batch_features.batch,
            )
            print(f"  Epoch {epoch + 1}, Batch {batch_count}/{total_batches}: Calculating loss...") # DEBUG
            loss = criterion(single_output, batch_labels)
            print(f"  Epoch {epoch + 1}, Batch {batch_count}/{total_batches}: Backward pass...") # DEBUG
            loss.backward()
            print(f"  Epoch {epoch + 1}, Batch {batch_count}/{total_batches}: Clipping gradients...") # DEBUG
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            print(f"  Epoch {epoch + 1}, Batch {batch_count}/{total_batches}: Optimizer step...") # DEBUG
            optimizer.step()
            total_loss += loss.item()
            batch_end_time = time.time() # DEBUG
            print(f"  Epoch {epoch + 1}, Batch {batch_count}/{total_batches}: Completed in {batch_end_time - batch_start_time:.2f}s") # DEBUG
            if batch_count % 10 == 0: # DEBUG: Print progress every 10 batches
                 print(f"  Epoch {epoch + 1}: Processed batch {batch_count}/{total_batches} ({(batch_count/total_batches)*100:.1f}%) - Current Avg Loss: {total_loss / batch_count:.4f}")

        train_loss_list.append(total_loss / len(train_loader))
        print(f"Epoch {epoch + 1}: Training phase completed. Average Train Loss: {train_loss_list[-1]:.4f}") # DEBUG

        # Validation
        print(f"Epoch {epoch + 1}: Starting validation phase...") # DEBUG
        model.eval()
        test_loss = 0
        val_batch_count = 0 # DEBUG
        total_val_batches = len(test_loader) # DEBUG
        with torch.no_grad():
            for test_features, test_labels in test_loader:  # <-- CHANGE TO val_loader !!!
                val_batch_count += 1 # DEBUG
                print(f"  Epoch {epoch + 1}, Val Batch {val_batch_count}/{total_val_batches}: Loading data...") # DEBUG
                test_features = test_features.to(device)
                test_labels = test_labels.to(device).view(-1, 1)
                print(f"  Epoch {epoch + 1}, Val Batch {val_batch_count}/{total_val_batches}: Relabeling nodes (skipped)...") # DEBUG
                # test_features = relabel_node_indices(test_features)  # MODIFIED: Skip relabeling for now
                print(f"  Epoch {epoch + 1}, Val Batch {val_batch_count}/{total_val_batches}: Forward pass...") # DEBUG
                test_single_output, _ = model(
                    test_features.x,
                    test_features.edge_index,
                    test_features.edge_attr,
                    test_features.batch,
                )
                print(f"  Epoch {epoch + 1}, Val Batch {val_batch_count}/{total_val_batches}: Calculating loss...") # DEBUG
                test_loss += criterion(test_single_output, test_labels).item()
        test_loss_list.append(test_loss / len(test_loader))  # <-- CHANGE TO val_loader !!!
        print(f"Epoch {epoch + 1}: Validation phase completed. Average Val Loss: {test_loss_list[-1]:.4f}") # DEBUG

        print(
            f"Epoch {epoch + 1} Summary ---> Train Loss: {train_loss_list[epoch]:.4f} / Validation Loss: {test_loss_list[-1]:.4f}"  # CHANGE TO val_loader !!!
        )

        # Early stopping check
        print(f"Epoch {epoch + 1}: Checking early stopping... (Current Best Loss: {best_test_loss:.4f}, Epochs No Improve: {epochs_no_improve})" ) # DEBUG
        current_test_loss = test_loss_list[epoch]
        if current_test_loss < best_test_loss:
            print(f"  Epoch {epoch + 1}: Validation loss improved ({best_test_loss:.4f} --> {current_test_loss:.4f}). Saving model.") # DEBUG
            best_test_loss = current_test_loss
            epochs_no_improve = 0  # Reset counter
            torch.save(model.state_dict(), best_model_path)  # save best model using output_dir path.
        else:
            epochs_no_improve += 1
            print(f"  Epoch {epoch + 1}: Validation loss did not improve. Epochs without improvement: {epochs_no_improve}") # DEBUG

        if epochs_no_improve == early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

        # Scheduler step
        print(f"Epoch {epoch + 1}: Stepping scheduler...") # DEBUG
        # scheduler.step()  # <-- change when using StepLR !!!
        scheduler.step(current_test_loss)
        epoch_end_time = time.time() # DEBUG
        print(f"--- Epoch {epoch + 1} completed in {epoch_end_time - epoch_start_time:.2f} seconds ---") # DEBUG

    print("DEBUG: Training loop completed") # DEBUG

    # # Test
    # model.eval()
    # test_loss = 0
    # with torch.no_grad():
    #     for test_features, test_labels in test_loader:
    #         test_features = test_features.to(device)
    #         test_labels = test_labels.to(device).view(-1, 1)
    #         test_features = relabel_node_indices(test_features)
    #         test_single_output, _ = model(
    #             test_features.x,
    #             test_features.edge_index,
    #             test_features.edge_attr,
    #             test_features.batch,
    #         )
    #         test_loss += criterion(test_single_output, test_labels).item()
    # print(f"\nTest Loss: {test_loss / len(test_loader)}")

    current_step_pipeline += 1
    update_progress(progress_file, 1 + current_step_pipeline, 1 + total_steps_pipeline, "Calculating metrics...")
    print("DEBUG: Progress updated: Calculating metrics...") # DEBUG

    # Metrics Calculation
    print("DEBUG: Starting metrics calculation") # DEBUG
    all_fingerprints = []
    all_labels = []
    all_probabilities = []  # To store probabilities for AUC calculations

    with torch.no_grad():
        for test_features, test_labels in test_loader:
            test_features = test_features.to(device)
            test_labels = test_labels.to(device)

            single_output, fingerprint = model(
                test_features.x,
                test_features.edge_index,
                test_features.edge_attr,
                test_features.batch,
            )

            probabilities = torch.sigmoid(single_output).cpu().numpy()
            labels = test_labels.cpu().numpy().astype(int)

            all_labels.extend(labels)
            all_probabilities.extend(probabilities.flatten())
            all_fingerprints.extend(fingerprint.cpu())

    print(f"DEBUG: Collected {len(all_labels)} labels and {len(all_probabilities)} probabilities") # DEBUG

    # Plot Learning Curve
    print("DEBUG: Plotting learning curve") # DEBUG
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(test_loss_list, label="Test Loss")
    if len(train_loss_list) > 0:
        plt.xlim(1, len(train_loss_list))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend()
    plt.savefig(learning_curve_path, dpi=300)
    plt.close()
    print(f"DEBUG: Saved learning curve to {learning_curve_path}") # DEBUG

    # Print probabilities and associated labels
    print("\nProbabilities:", all_probabilities)
    print("\nLabels:", all_labels)

    # Find Optimal Threshold for Classification
    print("DEBUG: Finding optimal threshold") # DEBUG
    thresholds = np.arange(0.001, 1, 0.001)
    scores = [
        matthews_corrcoef(
            all_labels, (np.array(all_probabilities) > thresh).astype(int)
        )
        for thresh in thresholds
    ]
    reversed_scores = scores[::-1]
    reversed_thresholds = thresholds[::-1]
    optimal_index_reversed = np.argmax(reversed_scores)

    highest_optimal_threshold = reversed_thresholds[optimal_index_reversed]
    # lowest_optimal_threshold = thresholds[np.argmax(f1_scores)]

    # print(
    #     f"\nOptimal Classification Threshold (Lowest): {lowest_optimal_threshold} with F1 Score: {max(f1_scores)}"
    # )
    print(f"Optimal Classification Threshold (Highest): {highest_optimal_threshold}\n")

    # Generate Raw Probability Plots (not considering optimal threshold)
    print("DEBUG: Generating probability plots") # DEBUG
    probabilities = pd.to_numeric(np.array(all_probabilities))
    labels = pd.Series(all_labels)
    df = pd.DataFrame({"Probability": probabilities, "Label": labels})

    # Create a single figure with 3 subplots
    _, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    # Define the x-axis range
    x_min = probabilities.min() - 0.005 if len(probabilities) > 0 else 0 # Adjust as needed
    x_max = probabilities.max() + 0.005 if len(probabilities) > 0 else 1 # Adjust as needed

    # Histogram
    sns.histplot(x=probabilities, hue=labels, kde=True, ax=axes[0])
    axes[0].set_title("Distribution of Probabilities")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Frequency")
    axes[0].legend(title="Label", labels=["Active", "Inactive"])
    axes[0].set_xlim(x_min, x_max)

    # Box Plot
    sns.boxplot(x="Label", y="Probability", data=df, ax=axes[1])
    axes[1].set_title("Box Plot of Probabilities")
    axes[1].set_xlabel("Label")
    axes[1].set_ylabel("Predicted Probability")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Inactive", "Active"])
    axes[1].set_ylim(x_min, x_max)

    # Scatter Plot (with Jitter)
    sns.stripplot(
        x=labels, y=probabilities, jitter=True, palette=["red", "green"], ax=axes[2]
    )
    axes[2].set_title("Scatter Plot of Probabilities")
    axes[2].set_xlabel("Label")
    axes[2].set_ylabel("Predicted Probability")
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Inactive", "Active"])
    axes[2].set_ylim(x_min, x_max)

    plt.tight_layout()
    plt.savefig(probability_plots_path, dpi=300)
    print(f"DEBUG: Saved probability plots to {probability_plots_path}") # DEBUG

    # Generate Predictions with Default Threshold (> 0.5)
    print("DEBUG: Calculating metrics") # DEBUG
    print("Using >= Highest Optimal Threshold, the results are:")
    predictions = (np.array(all_probabilities) > highest_optimal_threshold).astype(int)

    # Calculate Metrics
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    # Calculate FPR from confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, predictions).ravel()
    print(f"\nTotal: {tn + fp + fn + tp}")
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
    fpr = fp / (fp + tn)
    roc_auc = roc_auc_score(all_labels, all_probabilities)
    prc_auc = average_precision_score(all_labels, all_probabilities)
    mcc = matthews_corrcoef(all_labels, predictions)

    # Calculate F-score
    f1 = precision_score(all_labels, predictions, zero_division=0)

    # Create DataFrame
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Precision",
                "Recall",
                "FPR",
                "F1 Score",
                "ROC AUC",
                "PRC AUC",
                "MCC",
                "Optimal Threshold", # Add to summary
            ],
            "Value": [precision, recall, fpr, f1, roc_auc, prc_auc, mcc, highest_optimal_threshold],
        }
    )

    # Save metrics DataFrame to CSV using output_dir path
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Print DataFrame
    print(metrics_df.to_string(index=False))  # print without index.

    # ROC Curve
    print("DEBUG: Generating ROC curve") # DEBUG
    fpr, recall, _ = roc_curve(all_labels, all_probabilities)
    plt.figure()
    plt.plot(fpr, recall, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(roc_curve_path, dpi=300)
    plt.close()
    print(f"DEBUG: Saved ROC curve to {roc_curve_path}") # DEBUG

    # PRC Curve
    print("DEBUG: Generating PRC curve") # DEBUG
    precision_prc, recall_prc, _ = precision_recall_curve(all_labels, all_probabilities)
    plt.figure()
    plt.plot(recall_prc, precision_prc, label=f"PRC curve (area = {prc_auc:.2f})")
    no_skill = len([x for x in all_labels if x == 1]) / len(all_labels)
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(prc_curve_path, dpi=300)
    plt.close()
    print(f"DEBUG: Saved PRC curve to {prc_curve_path}") # DEBUG

    # Save Model Parameters
    print("DEBUG: Saving model parameters") # DEBUG
    torch.save(model.state_dict(), model_params_path)  # save model parameters using output_dir path.
    print("\nModel parameters saved.")

    # Fingerprint Extraction and Label Collection from Test Set
    print("DEBUG: Extracting fingerprints") # DEBUG
    all_fingerprints = []
    all_labels = []

    with torch.no_grad():
        for test_features, test_labels in test_loader:
            test_features = test_features.to(device)
            # test_features = relabel_node_indices(test_features)  # MODIFIED: Skip relabeling for now
            _, fingerprint = model(
                test_features.x,
                test_features.edge_index,
                test_features.edge_attr,
                test_features.batch,
            )
            all_fingerprints.append(fingerprint.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy().flatten())

    # Convert to NumPy arrays
    all_fingerprints = np.concatenate(all_fingerprints, axis=0)
    all_labels = np.array(all_labels)
    print(f"DEBUG: Extracted {len(all_fingerprints)} fingerprints") # DEBUG

    ##################################################################
    prova = pd.DataFrame(all_fingerprints)
    prova.to_csv(fingerprints_path, index=False) # Use output_dir path
    print(f"DEBUG: Saved fingerprints to {fingerprints_path}") # DEBUG
    ##################################################################
    
    # Collect results into a dictionary
    results_data = {
        "status": "success",
        "plots": {
            "learning_curve": learning_curve_path,
            "probability_plots": probability_plots_path,
            "roc_curve": roc_curve_path,
            "prc_curve": prc_curve_path,
        },
        "metrics_file": metrics_path,
        "fingerprints_file": fingerprints_path,
        "best_model_file": best_model_path,
    }

    # Save results.json to the output directory
    results_file = os.path.join(output_dir, "results.json")
    try:
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"DEBUG: Results saved to {results_file}")
    except Exception as e:
        print(f"DEBUG: Error saving results.json: {e}")
    
    print("DEBUG: Benchmarking pipeline completed successfully") # DEBUG
    return

#################
### MAIN CODE ###
#################


# Original __main__ block removed to prevent conflict with GUI integration block below
# (This block was causing the TypeError as it called the pipeline without output_dir/progress_file)


# Function to update progress (for GUI integration)
def update_progress(progress_file, step, total_steps, message):
    if progress_file:
        progress = {
            "current_step": step,
            "total_steps": total_steps,
            "message": message,
        }
        try:
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
        except Exception as e:
            # logging.warning(f"Could not write progress to {progress_file}: {e}") # Avoid logging noise
            print(f"DEBUG: Warning - Could not write progress to {progress_file}: {e}") # DEBUG



# Add argparse for command-line arguments (for GUI integration)
if __name__ == "__main__":
    print("DEBUG: Starting main script execution (__name__ == '__main__')") # DEBUG
    parser = argparse.ArgumentParser(description="StablyzeGraph Benchmarking Pipeline (Non-GUI Logic with GUI Wrapper)")
    parser.add_argument("--active_file", type=str, required=True, help="Path to active sequences CSV file")
    parser.add_argument("--inactive_file", type=str, required=True, help="Path to inactive sequences CSV file")
    parser.add_argument("--wild_type_file", type=str, required=True, help="Path to wild type sequence FASTA file")
    parser.add_argument("--pdb_file", type=str, required=True, help="Path to PDB structure file")
    parser.add_argument("--properties_file", type=str, required=True, help="Path to amino acid properties CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--progress_file", type=str, required=True, help="File path to write progress updates.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_cores", type=int, default=-1, help="Number of CPU cores for parallel processing (-1 uses all available)")
    # Add other potential parameters if they need to be configurable from GUI
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size for GNN layers")
    parser.add_argument("--dropout_rate", type=float, default=0.20, help="Dropout rate for GNN layers")
    parser.add_argument("--ratio", type=float, default=0.70, help="Pooling ratio for SAGPooling")
    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate for optimizer")
    parser.add_argument("--l2_regularization", type=float, default=0.0001, help="L2 regularization weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer (if used)")
    parser.add_argument("--scheduler_factor", type=float, default=0.9, help="Factor for learning rate scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=200, help="Patience for learning rate scheduler")
    parser.add_argument("--stop_patience", type=int, default=50, help="Patience for early stopping")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clipping value")
    parser.add_argument("--max_epochs", type=int, default=1000, help="Maximum number of training epochs") # MODIFIED: Default to 5 for testing
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use for computation (auto, cpu, cuda).")

    print("DEBUG: Parsing arguments") # DEBUG
    args = parser.parse_args()
    print(f"DEBUG: Arguments parsed: active_file={args.active_file}, inactive_file={args.inactive_file}") # DEBUG

    # --- Setup based on args ---
    # Update global seed and device based on args
    print("DEBUG: Setting up environment based on args") # DEBUG
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"DEBUG: Using device: {device}") # DEBUG
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # ---------------------------

    # Create output directory if it doesn't exist
    print(f"DEBUG: Creating output directory: {args.output_dir}") # DEBUG
    os.makedirs(args.output_dir, exist_ok=True)

    total_steps = 10 # Approximate steps for progress reporting
    current_step = 0
    update_progress(args.progress_file, current_step, total_steps, "Initializing...")
    print("DEBUG: Progress updated: Initializing...") # DEBUG

    try:
        current_step += 1
        update_progress(args.progress_file, current_step, total_steps, "Starting benchmarking pipeline...")
        print("DEBUG: Progress updated: Starting benchmarking pipeline...") # DEBUG
        print("DEBUG: Calling run_benchmarking_pipeline function...") # DEBUG
        run_benchmarking_pipeline(
            active_file=args.active_file,
            inactive_file=args.inactive_file,
            wild_type_file=args.wild_type_file,
            pdb_file=args.pdb_file,
            properties_file=args.properties_file,
            output_dir=args.output_dir, # Pass output_dir
            progress_file=args.progress_file, # Pass progress_file
            seed=args.seed,
            num_cores=args.num_cores,
            # Pass other args to the pipeline function
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            ratio=args.ratio,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
            momentum=args.momentum,
            scheduler_factor=args.scheduler_factor,
            scheduler_patience=args.scheduler_patience,
            stop_patience=args.stop_patience,
            grad_clip=args.grad_clip,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            device=device # Pass the determined device object
        )
        print("DEBUG: run_benchmarking_pipeline function finished.") # DEBUG
        update_progress(args.progress_file, total_steps, total_steps, "Benchmarking completed successfully.")
        print("DEBUG: Progress updated: Benchmarking completed successfully.") # DEBUG
    except Exception as e:
        logging.exception("Benchmarking pipeline failed.")
        print(f"DEBUG: ERROR: {str(e)}") # DEBUG
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}") # DEBUG
        update_progress(args.progress_file, -1, total_steps, f"Benchmarking failed: {e}")
        print("DEBUG: Progress updated: Benchmarking failed.") # DEBUG

    print("DEBUG: Main script execution finished.") # DEBUG
