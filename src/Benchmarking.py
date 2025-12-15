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
import argparse
print("DEBUG: Imported argparse") # DEBUG
import json
print("DEBUG: Imported json") # DEBUG
import time
print("DEBUG: Imported time") # DEBUG
from Bio.Seq import Seq
print("DEBUG: Imported Bio.Seq") # DEBUG
import optuna
print("DEBUG: Imported optuna") # DEBUG

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
    print(f"DEBUG: Read {len(wild_type_sequence)} wild type sequences") # DEBUG
    return wild_type_sequence


def extract_coordinates_from_pdb(pdb_file, sequences, wild_type_sequence, seed=seed):
    """Extract 3D coordinates from a PDB file for given sequences."""
    # print(f"DEBUG: Extracting coordinates from {pdb_file}") # DEBUG - Commented out for less noise
    seed = np.random.seed(seed)

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
        print(f"DEBUG: Warning - Wild type coords length: {len(wild_type_coords)}, sequence length: {len(wild_type_sequence)}") # DEBUG

    coordinates = []

    for seq_idx, seq in enumerate(sequences):
        seq_coords = np.zeros((len(seq), 3))

        for i, (wild_aa, seq_aa) in enumerate(zip(wild_type_sequence, seq)):
            if i >= len(wild_type_coords):
                break
            if wild_aa == seq_aa:
                seq_coords[i] = wild_type_coords[i]
            else:
                seq_coords[i] = wild_type_coords[i] + np.random.normal(
                    0, 1, size=3
                )

        coordinates.append(seq_coords)

    # print(f"DEBUG: Extracted coordinates for {len(coordinates)} sequences") # DEBUG - Commented out for less noise
    return coordinates


def calculate_conservation_scores(sequences_file):
    """Calculate conservation scores using Clustal Omega alignment."""
    print(f"DEBUG: Calculating conservation scores from {sequences_file}") # DEBUG
    aligned_file = "aligned_sequences.aln"

    try:
        print("DEBUG: Running Clustal Omega") # DEBUG
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
        print("DEBUG: Clustal Omega completed successfully") # DEBUG
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running Clustal Omega: {e.stderr.decode()}")
        print(f"DEBUG: ERROR - Clustal Omega failed: {e.stderr.decode()}") # DEBUG
        raise e
    except FileNotFoundError:
        logging.error("Clustal Omega command not found. Please ensure it is installed and in your PATH.")
        print("DEBUG: ERROR - Clustal Omega not found in PATH") # DEBUG
        raise FileNotFoundError("Clustal Omega not found.")

    try:
        print(f"DEBUG: Reading alignment file {aligned_file}") # DEBUG
        alignment = AlignIO.read(aligned_file, "clustal")
        print(f"DEBUG: Alignment read successfully, length: {alignment.get_alignment_length()}") # DEBUG
    except Exception as e:
        raise RuntimeError(
            f"Error reading alignment file: {aligned_file}. Details: {str(e)}"
        )

    summary_align = AlignInfo.SummaryInfo(alignment)

    # Safe consensus builder (works across Biopython versions)
    try:
        consensus = str(summary_align.dumb_consensus())
    except AttributeError:
        try:
            consensus = str(summary_align.gap_consensus())
        except Exception:
            from collections import Counter
            L = alignment.get_alignment_length()
            col_consensus = []
            for i in range(L):
                col = [aa for aa in alignment[:, i] if aa != "-"]
                if not col:
                    col_consensus.append("X")
                else:
                    col_consensus.append(Counter(col).most_common(1)[0][0])
            consensus = "".join(col_consensus)

    print(f"DEBUG: Consensus sequence calculated, length: {len(consensus)}") # DEBUG

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
    distance_threshold=10.0,
):
    """Convert a protein sequence to a graph representation."""
    # print(f"DEBUG: Converting sequence to graph") # DEBUG - Commented out for less noise
    list_aa = [
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    ]

    num_nodes = len(seq)
    node_features = []

    for idx, aa in enumerate(seq):
        if aa not in list_aa:
            raise ValueError(f"Unknown amino acid {aa} found in sequence.")
        features = [properties[dict_name].get(aa, 0) for dict_name in properties]
        if idx >= len(conservation_scores):
             features.append(0.0)
        else:
             features.append(conservation_scores[idx])
        node_features.append(features)

    node_features = torch.tensor(node_features, dtype=torch.float).to(device)

    edge_index = []
    edge_attr = []

    if len(pdb_coords) != num_nodes:
        print(f"DEBUG: Warning - Mismatch between PDB coordinates length ({len(pdb_coords)}) and sequence length ({num_nodes}) in sequence_to_graph") # DEBUG
    else:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = np.linalg.norm(pdb_coords[i] - pdb_coords[j])
                if dist < distance_threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_attr.append([dist])
                    edge_attr.append([dist])

    if not edge_index:
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)
        edge_attr = torch.empty((0, 1), dtype=torch.float).to(device)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)

    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    # print(f"DEBUG: Graph created with {graph.num_nodes} nodes and {graph.num_edges} edges") # DEBUG - Commented out for less noise
    return graph


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
            print(f"DEBUG: Warning - Could not write progress to {progress_file}: {e}") # DEBUG


print("DEBUG: Defining main pipeline function") # DEBUG
#######################
### MAIN PIPELINE ###
#######################


def run_benchmarking_pipeline(
    active_file,
    inactive_file,
    wild_type_file,
    pdb_file,
    properties_file,
    output_dir,
    progress_file,
    seed=42,
    num_cores=-1,
    hidden_dim=128,
    dropout_rate=0.30,
    ratio=0.60,
    learning_rate=0.00001,
    l2_regularization=0.005,
    momentum=0.9,
    scheduler_factor=0.9,
    scheduler_patience=200,
    stop_patience=200,
    grad_clip=10.0,
    max_epochs=1000,
    batch_size=64,
    device=device,
    n_splits=5,
    n_repeats=1,
    trial=None,
    all_trial_metrics=None,
):
    print(f"DEBUG: Starting benchmarking pipeline with active_file={active_file}, inactive_file={inactive_file}") # DEBUG
    print(f"DEBUG: Parameters - hidden_dim={hidden_dim}, dropout_rate={dropout_rate}, learning_rate={learning_rate}, l2_regularization={l2_regularization}") # DEBUG

    # Define output paths using output_dir
    print(f"DEBUG: Output directory: {output_dir}") # DEBUG

    # Add progress update
    total_steps_pipeline = 10
    current_step_pipeline = 0
    update_progress(progress_file, 1 + current_step_pipeline, 1 + total_steps_pipeline, "Preparing data...")
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

    # Ensure there are at least ten members in each class
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
        if (i + 1) % 10 == 0 or (i + 1) == len(active_sequences):
            print(f"DEBUG: Converting active sequence {i+1}/{len(active_sequences)} to graph") # DEBUG
        graph = sequence_to_graph(
            seq,
            properties,
            conservation_scores_active,
            coords,
        )
        active_features.append(graph)
    print(f"DEBUG: Converted {len(active_features)} active sequences to graphs") # DEBUG

    print("DEBUG: Calculating conservation scores for inactive sequences") # DEBUG
    conservation_scores_inactive = calculate_conservation_scores(inactive_sequences_file)

    print("DEBUG: Converting inactive sequences to graphs") # DEBUG
    inactive_features = []
    for i, (seq, coords) in enumerate(zip(inactive_sequences, inactive_coords)):
        if (i + 1) % 10 == 0 or (i + 1) == len(inactive_sequences):
            print(f"DEBUG: Converting inactive sequence {i+1}/{len(inactive_sequences)} to graph") # DEBUG
        graph = sequence_to_graph(
            seq,
            properties,
            conservation_scores_inactive,
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

    # Calculate input dimension
    input_dim = len(properties) + 1  # Number of properties + conservation score
    print(f"DEBUG: Input dimension: {input_dim}") # DEBUG

    # Calculate original imbalance (before splits and oversampling)
    print("DEBUG: Calculating class weights") # DEBUG
    original_positive_count = len(active_sequences)
    original_negative_count = len(inactive_sequences)
    original_pos_weight = (
        torch.tensor([original_negative_count / original_positive_count])
        .float()
        .to(device)
    )
    print(f"DEBUG: Class weights - positive: {original_positive_count}, negative: {original_negative_count}, pos_weight: {original_pos_weight.item()}") # DEBUG

    # K-fold Cross-Validation Setup
    print(f"DEBUG: Setting up K-fold cross-validation with n_splits={n_splits}, n_repeats={n_repeats}") # DEBUG
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    fold_metrics = []

    for fold, (train_index, test_index) in enumerate(rskf.split(all_features, all_labels)):
        logging.info(f"Starting Fold {fold + 1}/{n_splits * n_repeats}")
        print(f"DEBUG: ========== Starting Fold {fold + 1}/{n_splits * n_repeats} ==========") # DEBUG
        fold_output_dir = os.path.join(output_dir, f"fold_{fold + 1}")
        os.makedirs(fold_output_dir, exist_ok=True)
        print(f"DEBUG: Fold output directory: {fold_output_dir}") # DEBUG

        print(f"DEBUG: Splitting data for fold {fold + 1}") # DEBUG
        train_features = [all_features[i] for i in train_index]
        test_features = [all_features[i] for i in test_index]
        train_labels = all_labels[train_index]
        test_labels = all_labels[test_index]
        print(f"DEBUG: Fold {fold + 1} - train size: {len(train_features)}, test size: {len(test_features)}") # DEBUG

        # Reset model and optimizer for each fold
        print(f"DEBUG: Initializing model for fold {fold + 1}") # DEBUG
        model = GNNFeatureExtractor(input_dim, hidden_dim, dropout_rate, ratio).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=l2_regularization
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience
        )
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=original_pos_weight)
        print(f"DEBUG: Model, optimizer, scheduler, and criterion initialized for fold {fold + 1}") # DEBUG

        # Oversample the minority class in the training set
        print(f"DEBUG: Resampling minority class for fold {fold + 1}") # DEBUG
        minority_class_count = min(
            sum(train_labels == 0), sum(train_labels == 1)
        )
        print(f"DEBUG: Minority class count: {minority_class_count}") # DEBUG

        if minority_class_count < 10:
            print("DEBUG: Using RandomOverSampler") # DEBUG
            ros = RandomOverSampler(random_state=seed)
        else:
            print("DEBUG: Using SMOTE with k_neighbors=10") # DEBUG
            ros = SMOTE(k_neighbors=10, random_state=seed)

        resampled_indices, resampled_labels = ros.fit_resample(
            np.array(range(len(train_features))).reshape(-1, 1),
            train_labels.numpy(),
        )
        resampled_indices = resampled_indices.flatten().tolist()
        resampled_labels = torch.tensor(resampled_labels).float()
        print(f"DEBUG: Resampled to {len(resampled_indices)} samples") # DEBUG

        resampled_train_features = [
            train_features[i] for i in resampled_indices
        ]

        train_dataset = list(zip(resampled_train_features, resampled_labels))
        test_dataset = list(zip(test_features, test_labels))
        print(f"DEBUG: Created datasets for fold {fold + 1}: train={len(train_dataset)}, test={len(test_dataset)}") # DEBUG

        print(f"DEBUG: Creating DataLoaders for fold {fold + 1}") # DEBUG
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"DEBUG: DataLoaders created for fold {fold + 1}") # DEBUG

        best_fold_model_path = os.path.join(fold_output_dir, "best_model.pth")
        fold_learning_curve_path = os.path.join(fold_output_dir, "GNN_learning_curve.png")
        fold_probability_plots_path = os.path.join(fold_output_dir, "probability_plots.png")
        fold_roc_curve_path = os.path.join(fold_output_dir, "roc_curve.png")
        fold_prc_curve_path = os.path.join(fold_output_dir, "prc_curve.png")
        fold_metrics_path = os.path.join(fold_output_dir, "metrics.csv")

        # Training loop for the current fold
        print(f"DEBUG: Starting training loop for fold {fold + 1}") # DEBUG
        train_loss_list = []
        test_loss_list = []
        best_test_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            model.train()
            total_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device).view(-1, 1)
                optimizer.zero_grad()
                single_output, _ = model(
                    batch_features.x,
                    batch_features.edge_index,
                    batch_features.edge_attr,
                    batch_features.batch,
                )
                loss = criterion(single_output, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_loss_list.append(avg_train_loss)

            model.eval()
            total_test_loss = 0
            all_probabilities_fold = []
            all_labels_fold = []
            with torch.no_grad():
                for test_features_fold, test_labels_fold in test_loader:
                    test_features_fold = test_features_fold.to(device)
                    single_output_fold, _ = model(
                        test_features_fold.x,
                        test_features_fold.edge_index,
                        test_features_fold.edge_attr,
                        test_features_fold.batch,
                    )
                    loss_fold = criterion(single_output_fold, test_labels_fold.view(-1, 1))
                    total_test_loss += loss_fold.item()
                    probabilities_fold = torch.sigmoid(single_output_fold).cpu().numpy().flatten()
                    all_probabilities_fold.extend(probabilities_fold)
                    all_labels_fold.extend(test_labels_fold.cpu().numpy().flatten())

            avg_test_loss = total_test_loss / len(test_loader)
            test_loss_list.append(avg_test_loss)

            scheduler.step(avg_test_loss)

            roc_auc_fold = roc_auc_score(all_labels_fold, all_probabilities_fold)

            if (epoch + 1) % 50 == 0 or (epoch + 1) == max_epochs:
                logging.info(
                    f"Fold {fold + 1}, Epoch {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test ROC AUC: {roc_auc_fold:.4f}"
                )
                print(f"DEBUG: Fold {fold + 1}, Epoch {epoch+1}/{max_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Test ROC AUC: {roc_auc_fold:.4f}") # DEBUG

            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_fold_model_path)
                print(f"DEBUG: Saved best model for fold {fold + 1} at epoch {epoch + 1}") # DEBUG
            else:
                epochs_no_improve += 1
                if epochs_no_improve == stop_patience:
                    logging.info(f"Early stopping triggered for fold {fold + 1} at epoch {epoch + 1}.")
                    print(f"DEBUG: Early stopping triggered for fold {fold + 1} at epoch {epoch + 1}") # DEBUG
                    break

            if trial:
                trial.report(roc_auc_fold, epoch)
                if trial.should_prune():
                    print(f"DEBUG: Trial pruned at fold {fold + 1}, epoch {epoch + 1}") # DEBUG
                    raise optuna.exceptions.TrialPruned()

        # Evaluate best model for the fold
        print(f"DEBUG: Loading best model for fold {fold + 1}") # DEBUG
        model.load_state_dict(torch.load(best_fold_model_path))
        model.eval()

        all_probabilities_final = []
        all_labels_final = []
        with torch.no_grad():
            for test_features_final, test_labels_final in test_loader:
                test_features_final = test_features_final.to(device)
                single_output_final, _ = model(
                    test_features_final.x,
                    test_features_final.edge_index,
                    test_features_final.edge_attr,
                    test_features_final.batch,
                )
                probabilities_final = torch.sigmoid(single_output_final).cpu().numpy().flatten()
                all_probabilities_final.extend(probabilities_final)
                all_labels_final.extend(test_labels_final.cpu().numpy().flatten())

        print(f"DEBUG: Calculating metrics for fold {fold + 1}") # DEBUG
        fpr_values, tpr_values, thresholds = roc_curve(all_labels_final, all_probabilities_final)
        optimal_idx = np.argmax(tpr_values - fpr_values)
        highest_optimal_threshold = thresholds[optimal_idx]
        print(f"DEBUG: Optimal threshold for fold {fold + 1}: {highest_optimal_threshold}") # DEBUG

        predictions = (np.array(all_probabilities_final) > highest_optimal_threshold).astype(int)

        precision = precision_score(all_labels_final, predictions, zero_division=0)
        recall = recall_score(all_labels_final, predictions, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(all_labels_final, predictions).ravel()
        fpr = fp / (fp + tn)
        roc_auc = roc_auc_score(all_labels_final, all_probabilities_final)
        prc_auc = average_precision_score(all_labels_final, all_probabilities_final)
        mcc = matthews_corrcoef(all_labels_final, predictions)
        f1 = f1_score(all_labels_final, predictions, zero_division=0)

        print(f"DEBUG: Fold {fold + 1} metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, FPR: {fpr:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}, PRC AUC: {prc_auc:.4f}, MCC: {mcc:.4f}") # DEBUG

        fold_metrics.append({
            "fold": fold + 1,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "l2_regularization": l2_regularization,
            "Precision": precision,
            "Recall": recall,
            "FPR": fpr,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "PRC AUC": prc_auc,
            "MCC": mcc,
            "Optimal Threshold": highest_optimal_threshold,
        })

        # Save plots for the current fold
        print(f"DEBUG: Generating plots for fold {fold + 1}") # DEBUG
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label="Train Loss")
        plt.plot(range(1, len(test_loss_list) + 1), test_loss_list, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(fold_learning_curve_path, dpi=300)
        plt.close()
        print(f"DEBUG: Saved learning curve for fold {fold + 1}") # DEBUG

        # Probability plots
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        df = pd.DataFrame({"Probability": all_probabilities_final, "Label": all_labels_final})
        x_min, x_max = 0, 1

        sns.histplot(data=df, x="Probability", hue="Label", kde=True, ax=axes[0], stat="density", common_norm=False)
        axes[0].set_title("Distribution of Predicted Probabilities")
        axes[0].set_xlabel("Predicted Probability")
        axes[0].set_ylabel("Density")
        axes[0].set_xlim(x_min, x_max)

        sns.boxplot(x="Label", y="Probability", data=df, ax=axes[1])
        axes[1].set_title("Box Plot of Probabilities")
        axes[1].set_xlabel("Label")
        axes[1].set_ylabel("Predicted Probability")
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["Inactive", "Active"])
        axes[1].set_ylim(x_min, x_max)

        sns.stripplot(
            x=all_labels_final, y=all_probabilities_final, jitter=True, palette=["red", "green"], ax=axes[2]
        )
        axes[2].set_title("Scatter Plot of Probabilities")
        axes[2].set_xlabel("Label")
        axes[2].set_ylabel("Predicted Probability")
        axes[2].set_xticks([0, 1])
        axes[2].set_xticklabels(["Inactive", "Active"])
        axes[2].set_ylim(x_min, x_max)

        plt.tight_layout()
        plt.savefig(fold_probability_plots_path, dpi=300)
        plt.close()
        print(f"DEBUG: Saved probability plots for fold {fold + 1}") # DEBUG

        # ROC Curve
        fpr_curve, tpr_curve, _ = roc_curve(all_labels_final, all_probabilities_final)
        plt.figure()
        plt.plot(fpr_curve, tpr_curve, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(fold_roc_curve_path, dpi=300)
        plt.close()
        print(f"DEBUG: Saved ROC curve for fold {fold + 1}") # DEBUG

        # PRC Curve
        precision_prc, recall_prc, _ = precision_recall_curve(all_labels_final, all_probabilities_final)
        plt.figure()
        plt.plot(recall_prc, precision_prc, label=f"PRC curve (area = {prc_auc:.2f})")
        no_skill = len([x for x in all_labels_final if x == 1]) / len(all_labels_final)
        plt.plot([0, 1], [no_skill, no_skill], linestyle="--")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.savefig(fold_prc_curve_path, dpi=300)
        plt.close()
        print(f"DEBUG: Saved PRC curve for fold {fold + 1}") # DEBUG

        # Save fold metrics
        fold_metrics_df = pd.DataFrame([fold_metrics[-1]])
        fold_metrics_df.to_csv(fold_metrics_path, index=False)
        print(f"DEBUG: Saved metrics for fold {fold + 1}") # DEBUG

        print(f"DEBUG: ========== Completed Fold {fold + 1}/{n_splits * n_repeats} ==========") # DEBUG

    # Aggregate metrics across folds
    print("DEBUG: Aggregating metrics across all folds") # DEBUG
    metrics_df = pd.DataFrame(fold_metrics)
    
    # Calculate the average metrics
    avg_metrics = metrics_df.drop(columns=["fold"]).mean().to_dict()
    avg_metrics["fold"] = "Average"
    avg_metrics["dropout_rate"] = dropout_rate
    avg_metrics["learning_rate"] = learning_rate
    avg_metrics["l2_regularization"] = l2_regularization
    
    # Append the average to the DataFrame
    metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics])], ignore_index=True)
    
    # Save the metrics to a CSV file
    all_metrics_path = os.path.join(output_dir, "all_fold_metrics.csv")
    metrics_df.to_csv(all_metrics_path, index=False)
    logging.info(f"All fold metrics saved to {all_metrics_path}")
    print(f"DEBUG: All fold metrics saved to {all_metrics_path}") # DEBUG

    # If this is part of an Optuna trial, append to all_trial_metrics
    if all_trial_metrics is not None:
        trial_summary = avg_metrics.copy()
        if trial:
            trial_summary["trial_number"] = trial.number
        all_trial_metrics.append(trial_summary)
        print(f"DEBUG: Appended trial metrics to all_trial_metrics") # DEBUG

    # Return the average ROC AUC for Optuna optimization
    avg_roc_auc = avg_metrics["ROC AUC"]
    print(f"DEBUG: Average ROC AUC across all folds: {avg_roc_auc:.4f}") # DEBUG
    print("DEBUG: Benchmarking pipeline completed successfully") # DEBUG
    
    # --- NEW: Generate results.json for GUI display ---
    # The GUI expects paths to the fold_1 plots and the all_fold_metrics.csv
    
    # Paths for fold 1 (assuming fold_1 is the first fold, i.e., fold=0)
    fold_1_dir = os.path.join(output_dir, "fold_1")
    gnn_learning_curve_path = os.path.join(fold_1_dir, "GNN_learning_curve.png")
    probability_plot_path = os.path.join(fold_1_dir, "probability_plots.png")
    all_metrics_path = os.path.join(output_dir, "all_fold_metrics.csv")
    
    results_data = {
        "status": "success", # Explicitly set status for main.py WorkerThread
        "GNN_learning_curve": gnn_learning_curve_path.replace(os.sep, "/"),
        "probability_plots": probability_plot_path.replace(os.sep, "/"),
        "all_fold_metrics": all_metrics_path.replace(os.sep, "/"),
    }
    
    results_file = os.path.join(output_dir, "results.json")
    try:
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=4)
        logging.info(f"Successfully created results.json at {results_file}")
        print(f"DEBUG: Successfully created results.json at {results_file}") # DEBUG
    except Exception as e:
        logging.error(f"Failed to create results.json: {e}")
        print(f"DEBUG: ERROR: Failed to create results.json: {e}") # DEBUG
    # --- END NEW ---

    # --- F1 Score Discrepancy Debugging ---
    # 1. Extract Unrounded Confusion Matrix Values
    tn, fp, fn, tp = confusion_matrix(all_labels_final, predictions).ravel()
    print(f"DEBUG_METRIC: Confusion Matrix Values - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    # 2. Explicitly Verify Positive Label
    # Set pos_label explicitly to ensure consistency
    precision_1 = precision_score(all_labels_final, predictions, zero_division=0, pos_label=1)
    recall_1 = recall_score(all_labels_final, predictions, zero_division=0, pos_label=1)
    f1_1_check = f1_score(all_labels_final, predictions, zero_division=0, pos_label=1)

    # Check the other class just in case your labels are reversed
    precision_0 = precision_score(all_labels_final, predictions, zero_division=0, pos_label=0)
    recall_0 = recall_score(all_labels_final, predictions, zero_division=0, pos_label=0)
    f1_0_check = f1_score(all_labels_final, predictions, zero_division=0, pos_label=0)

    # Manual F1 Calculation using unrounded values for pos_label=1
    # NOTE: Add a try/except for ZeroDivisionError if precision_1 + recall_1 is zero
    try:
        manual_f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)
    except ZeroDivisionError:
        manual_f1_1 = 0.0

    # Manual F1 Calculation using unrounded values for pos_label=0
    try:
        manual_f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
    except ZeroDivisionError:
        manual_f1_0 = 0.0

    # 3. Report Findings and Correct Metric
    # Determine the correct F1 score
    correct_f1 = f1_1_check if np.isclose(f1_1_check, manual_f1_1) else f1_0_check

    # Find the original F1 score calculation in the code (assuming it's one of the existing metrics)
    # Since we don't know the original F1, we'll use the one that matches the manual calculation as the 'Correct F1'
    # For the report, we'll print all values and let the user compare with their 'Original F1'
    
    report_message = (
        f"DEBUG_METRIC: F1 Discrepancy Analysis:\n"
        f"  - Confusion Matrix (TN, FP, FN, TP): {tn}, {fp}, {fn}, {tp}\n"
        f"  - pos_label=1: P={precision_1:.4f}, R={recall_1:.4f}, F1_check={f1_1_check:.4f}, Manual_F1={manual_f1_1:.4f}\n"
        f"  - pos_label=0: P={precision_0:.4f}, R={recall_0:.4f}, F1_check={f1_0_check:.4f}, Manual_F1={manual_f1_0:.4f}\n"
        f"  - Correct F1 (Matching P/R manual calc): {correct_f1:.4f}"
    )
    print(report_message)
    logging.info(report_message)
    
    # Update results_data with the detailed debug information
    results_data["f1_debug_report"] = {
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
        "pos_label_1": {
            "precision": float(precision_1),
            "recall": float(recall_1),
            "f1_check": float(f1_1_check),
            "manual_f1": float(manual_f1_1),
            "match": bool(np.isclose(f1_1_check, manual_f1_1))
        },
        "pos_label_0": {
            "precision": float(precision_0),
            "recall": float(recall_0),
            "f1_check": float(f1_0_check),
            "manual_f1": float(manual_f1_0),
            "match": bool(np.isclose(f1_0_check, manual_f1_0))
        },
        "correct_f1_value": float(correct_f1)
    }
    # --- END F1 Score Discrepancy Debugging ---
    
    return avg_roc_auc


print("DEBUG: Defining Optuna objective function") # DEBUG
def objective(trial, args):
    """Optuna objective function for hyperparameter tuning."""
    print(f"DEBUG: Starting Optuna trial {trial.number}") # DEBUG
    
    # Suggest hyperparameters
    dropout_rate = trial.suggest_float("dropout_rate", args.dropout_rate_min, args.dropout_rate_max, step=args.dropout_rate_interval)
    learning_rate = trial.suggest_float("learning_rate", args.learning_rate_min, args.learning_rate_max, log=True)
    l2_regularization = trial.suggest_float("l2_regularization", args.l2_regularization_min, args.l2_regularization_max, log=True)
    
    print(f"DEBUG: Trial {trial.number} parameters - dropout_rate={dropout_rate}, learning_rate={learning_rate}, l2_regularization={l2_regularization}") # DEBUG

    # Run the benchmarking pipeline with the suggested hyperparameters
    roc_auc = run_benchmarking_pipeline(
        active_file=args.active_file,
        inactive_file=args.inactive_file,
        wild_type_file=args.wild_type_file,
        pdb_file=args.pdb_file,
        properties_file=args.properties_file,
        output_dir=os.path.join(args.output_dir, f"trial_{trial.number}"),
        progress_file=args.progress_file,
        seed=args.seed,
        num_cores=args.num_cores,
        hidden_dim=args.hidden_dim,
        dropout_rate=dropout_rate,
        ratio=args.ratio,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        momentum=args.momentum,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        stop_patience=args.stop_patience,
        grad_clip=args.grad_clip,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        device=device,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        trial=trial,
        all_trial_metrics=args.all_trial_metrics,
    )

    print(f"DEBUG: Trial {trial.number} completed with ROC AUC: {roc_auc:.4f}") # DEBUG
    return roc_auc


#################
### MAIN CODE ###
#################

if __name__ == "__main__":
    print("DEBUG: Starting main script execution (__name__ == '__main__')") # DEBUG
    parser = argparse.ArgumentParser(description="StablyzeGraph Benchmarking Pipeline with Optuna Integration")
    parser.add_argument("--active_file", type=str, required=True, help="Path to active sequences CSV file")
    parser.add_argument("--inactive_file", type=str, required=True, help="Path to inactive sequences CSV file")
    parser.add_argument("--wild_type_file", type=str, required=True, help="Path to wild type sequence FASTA file")
    parser.add_argument("--pdb_file", type=str, required=True, help="Path to PDB structure file")
    parser.add_argument("--properties_file", type=str, required=True, help="Path to amino acid properties CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--progress_file", type=str, required=True, help="File path to write progress updates.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_cores", type=int, default=-1, help="Number of CPU cores for parallel processing (-1 uses all available)")
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
    parser.add_argument("--max_epochs", type=int, default=1000, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use for computation (auto, cpu, cuda).")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of splits for K-fold cross-validation")
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of repeats for K-fold cross-validation")

    # Optuna specific arguments
    parser.add_argument("--optuna_tuning", action="store_true", help="Enable Optuna hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials (maximum 100)")

    parser.add_argument("--dropout_rate_min", type=float, default=0.1, help="Min dropout rate for Optuna")
    parser.add_argument("--dropout_rate_max", type=float, default=0.5, help="Max dropout rate for Optuna")
    parser.add_argument("--dropout_rate_interval", type=float, default=0.1, help="Step interval for dropout rate for Optuna")
    parser.add_argument("--learning_rate_min", type=float, default=1e-5, help="Min learning rate for Optuna")
    parser.add_argument("--learning_rate_max", type=float, default=1e-2, help="Max learning rate for Optuna")
    parser.add_argument("--l2_regularization_min", type=float, default=1e-6, help="Min L2 regularization for Optuna")
    parser.add_argument("--l2_regularization_max", type=float, default=1e-3, help="Max L2 regularization for Optuna")

    print("DEBUG: Parsing arguments") # DEBUG
    args = parser.parse_args()
    print(f"DEBUG: Arguments parsed: active_file={args.active_file}, inactive_file={args.inactive_file}") # DEBUG
    print(f"DEBUG: Optuna tuning enabled: {args.optuna_tuning}, n_trials={args.n_trials}") # DEBUG

    # Validate n_trials maximum
    if args.n_trials > 100:
        print("DEBUG: WARNING - n_trials exceeds maximum of 100, setting to 100") # DEBUG
        logging.warning("n_trials exceeds maximum of 100, setting to 100")
        args.n_trials = 100

    # --- Setup based on args ---
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

    total_steps = 10
    current_step = 0
    update_progress(args.progress_file, current_step, total_steps, "Initializing...")
    print("DEBUG: Progress updated: Initializing...") # DEBUG

    try:
        if args.optuna_tuning:
            logging.info("Starting Optuna hyperparameter tuning.")
            print("DEBUG: Starting Optuna hyperparameter tuning") # DEBUG
            study = optuna.create_study(direction="maximize")  # Maximize ROC AUC
            all_trial_metrics = []  # Initialize list to store metrics from all trials
            args.all_trial_metrics = all_trial_metrics  # Attach to args for passing to objective
            print(f"DEBUG: Created Optuna study, will run {args.n_trials} trials") # DEBUG
            study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

            # Save all trial metrics to a single CSV file
            print("DEBUG: Saving all trial metrics") # DEBUG
            all_metrics_df = pd.DataFrame(all_trial_metrics)
            all_metrics_csv_path = os.path.join(args.output_dir, "optuna_all_trial_metrics.csv")
            all_metrics_df.to_csv(all_metrics_csv_path, index=False)
            logging.info(f"All trial metrics saved to {all_metrics_csv_path}")
            print(f"DEBUG: All trial metrics saved to {all_metrics_csv_path}") # DEBUG

            logging.info("Optuna tuning finished.")
            logging.info(f"Best trial: {study.best_trial.value}")
            logging.info(f"Best parameters: {study.best_trial.params}")
            print(f"DEBUG: Optuna tuning finished - Best trial value: {study.best_trial.value}") # DEBUG
            print(f"DEBUG: Best parameters: {study.best_trial.params}") # DEBUG

            # Save best parameters to a JSON file
            best_params_path = os.path.join(args.output_dir, "optuna_best_params.json")
            with open(best_params_path, "w") as f:
                json.dump(study.best_trial.params, f, indent=4)
            print(f"DEBUG: Best parameters saved to {best_params_path}") # DEBUG
            update_progress(args.progress_file, total_steps, total_steps, "Optuna tuning completed successfully.")
            print("DEBUG: Progress updated: Optuna tuning completed successfully.") # DEBUG

        else:
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
                output_dir=args.output_dir,
                progress_file=args.progress_file,
                seed=args.seed,
                num_cores=args.num_cores,
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
                device=device,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
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
