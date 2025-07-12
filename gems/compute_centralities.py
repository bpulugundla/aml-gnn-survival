"""
Script: GEM Gene–Gene Network Centrality Analysis (CLI Version)

Description:
This script loads MATLAB-format personalized genome-scale metabolic models (GEMs),
extracts gene–gene interaction networks using shared metabolite connectivity and AND-complexes
from gene–reaction rules, computes standard network centrality measures, and exports:

- A NetworkX GraphML representation of the gene–gene network
- A JSON file with gene-level centrality metrics
- A pickle file combining results from all models

Parallelized across all `.mat` files in `input_dir`.

Usage:
    python process_gems.py --input_dir personalised_GEMS \
                           --graphml_dir output/graphml \
                           --json_output_dir output/json \
                           --gpr_output_dir output/gpr \
                           --output_pickle_path all_centrality.pkl \
                           --n_processes 8
"""

import os
import re
import json
import pickle
import argparse
import scipy.io
import scipy.sparse
import cobra
import networkx as nx
from scipy.io.matlab import mat_struct
from multiprocessing import Pool, cpu_count
from itertools import combinations


def mat_struct_to_dict(mat_struct_obj):
    """
    Recursively convert a MATLAB mat_struct object (from scipy.io.loadmat)
    into a standard Python dictionary.

    This is necessary because MATLAB structs (loaded with `struct_as_record=True`)
    are returned as nested Python objects with attributes instead of dicts.
    The function preserves the nested structure.

    Args:
        mat_struct_obj (mat_struct): MATLAB struct object loaded via scipy.io.loadmat

    Returns:
        dict: A nested Python dictionary representing the structure and contents
              of the MATLAB object.
    
    Notes:
        - This assumes `mat_struct_obj` is an instance of scipy.io.matlab.mat_struct.
        - Recursion is used to handle arbitrarily deep nesting.
        - Non-struct fields (arrays, scalars, strings) are preserved as-is.
    """
    data_dict = {}
    for field_name in mat_struct_obj._fieldnames:
        field_value = getattr(mat_struct_obj, field_name)
        # Recurse if the field is another nested MATLAB struct
        if isinstance(field_value, mat_struct):
            data_dict[field_name] = mat_struct_to_dict(field_value)
        else:
            data_dict[field_name] = field_value
    return data_dict

def extract_genes_from_rule(rule):
    """
    Extract gene identifiers from a gene–reaction rule string.

    Args:
        rule (str): A gene–reaction rule string (e.g., 'ENSG000001 and ENSG000002')

    Returns:
        list of str: Extracted gene symbols matching the pattern 'ENSG\\d+'
    
    Notes:
        - Assumes all genes follow the Ensembl identifier format 'ENSG' + digits.
        - Case-sensitive: only matches uppercase 'ENSG'.
        - Can be used with COBRApy Reaction.gene_reaction_rule fields.
    """
    # Find all substrings that match Ensembl gene IDs
    genes = re.findall(r'ENSG\d+', rule)
    return genes

def process_model_to_gene_gene_network(args_tuple):
    """
    Convert a personalized GEM (.mat file) to a gene–gene network and compute centrality scores.

    Args:
        args_tuple (tuple): 
            - mat_filename (str): .mat file name
            - input_dir (str): directory where the .mat file lives
            - graphml_dir (str): where to save the GraphML output
            - json_output_dir (str): where to save JSON centrality scores
            - gpr_output_dir (str): where to save GPR rules text files

    Returns:
        tuple or None: (base_name, dict of centrality scores) or None if processing fails
    """
    mat_filename, input_dir, graphml_dir, json_output_dir, gpr_output_dir = args_tuple

    try:
        mat_path = os.path.join(input_dir, mat_filename)
        print(f"Processing: {mat_path}")
        model_data = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

        if isinstance(model_data["model"], mat_struct):
            model_struct = mat_struct_to_dict(model_data["model"])
        else:
            model_struct = model_data["model"]

        cobra_model = cobra.Model(model_struct["description"])
        S = scipy.sparse.csr_matrix(model_struct["S"])

        metabolites = {}
        for i in range(len(model_struct["mets"])):
            met_id = model_struct["mets"][i]
            met_name = model_struct["metNames"][i] if "metNames" in model_struct else met_id
            met_compartment = model_struct["comps"][model_struct["metComps"][i] - 1] if "metComps" in model_struct else "c"
            metabolite = cobra.Metabolite(met_id, name=met_name, compartment=met_compartment)
            cobra_model.add_metabolites(metabolite)
            metabolites[met_id] = metabolite

        rxn_to_genes = {}
        met_to_rxns = {}
        grrules = {}

        base_name = os.path.splitext(os.path.basename(mat_filename))[0]
        gr_file = os.path.join(gpr_output_dir, f"{base_name}_grrules.txt")
        with open(gr_file, "w") as f:
            for i in range(len(model_struct["rxns"])):
                rxn_id = model_struct["rxns"][i]
                rxn_name = model_struct["rxnNames"][i] if "rxnNames" in model_struct else rxn_id
                reaction = cobra.Reaction(rxn_id)
                reaction.name = rxn_name
                reaction.lower_bound = model_struct["lb"][i]
                reaction.upper_bound = model_struct["ub"][i]

                for j in S[:, i].nonzero()[0]:
                    coeff = S[j, i]
                    reaction.add_metabolites({metabolites[model_struct["mets"][j]]: coeff})

                if "grRules" in model_struct and i < len(model_struct["grRules"]):
                    gr_rule = model_struct["grRules"][i]
                    if isinstance(gr_rule, str):
                        reaction.gene_reaction_rule = gr_rule
                        grrules[rxn_id] = gr_rule
                        f.write(f"{rxn_id}: {gr_rule}\n")

                cobra_model.add_reactions([reaction])
                rxn_to_genes[rxn_id] = extract_genes_from_rule(reaction.gene_reaction_rule or "")
                for met in reaction.metabolites:
                    met_to_rxns.setdefault(met.id, set()).add(rxn_id)

        # Build gene-gene network
        G = nx.Graph()

        for rxn_id, rule in grrules.items():
            if 'and' in rule:
                complex_genes = extract_genes_from_rule(rule)
                for g1, g2 in combinations(complex_genes, 2):
                    if g1 != g2:
                        G.add_edge(g1, g2, type='complex')

        for met, rxns in met_to_rxns.items():
            rxns = list(rxns)
            for i in range(len(rxns)):
                for j in range(i + 1, len(rxns)):
                    genes_i = rxn_to_genes.get(rxns[i], [])
                    genes_j = rxn_to_genes.get(rxns[j], [])
                    for g1 in genes_i:
                        for g2 in genes_j:
                            if g1 != g2:
                                G.add_edge(g1, g2, via_metabolite=met)

        graphml_path = os.path.join(graphml_dir, f"{base_name}.graphml")
        nx.write_graphml(G, graphml_path)
        print(f"Saved gene-gene network: {graphml_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

        # Centrality analysis
        degree = nx.degree_centrality(G)
        closeness = nx.closeness_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eigenvector = {node: 0.0 for node in G.nodes()}

        model_stats = {
            node: {
                "degree_centrality": degree.get(node, 0.0),
                "closeness_centrality": closeness.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "eigenvector_centrality": eigenvector.get(node, 0.0)
            } for node in G.nodes()
        }

        json_path = os.path.join(json_output_dir, f"{base_name}.json")
        with open(json_path, "w") as f:
            json.dump(model_stats, f, indent=2)

        return base_name, model_stats

    except Exception as e:
        print(f"Failed to process {mat_filename}: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Process personalized GEMs into gene-gene networks and compute centrality scores.")
    parser.add_argument('--input_dir', required=True, help='Directory containing personalized GEM .mat files')
    parser.add_argument('--graphml_dir', required=True, help='Directory to save GraphML network files')
    parser.add_argument('--json_output_dir', required=True, help='Directory to save per-model JSON centrality scores')
    parser.add_argument('--gpr_output_dir', required=True, help='Directory to save gene-reaction rule text files')
    parser.add_argument('--output_pickle_path', required=True, help='Path to save combined pickle file with all centrality stats')
    parser.add_argument('--n_processes', type=int, default=cpu_count(), help='Number of processes to use (default: all cores)')
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.graphml_dir, exist_ok=True)
    os.makedirs(args.json_output_dir, exist_ok=True)
    os.makedirs(args.gpr_output_dir, exist_ok=True)

    mat_files = [f for f in os.listdir(args.input_dir) if f.endswith(".mat")]

    # Create a list of argument tuples for parallel processing
    # Each tuple: (filename, input_dir, graphml_dir, json_dir, gpr_dir)
    arg_tuples = [
        (f, args.input_dir, args.graphml_dir, args.json_output_dir, args.gpr_output_dir)
        for f in mat_files
    ]

    global_stats = {}

    # Parallel processing using all available CPU cores (or user-defined limit)
    with Pool(args.n_processes) as pool:
        results = pool.map(process_model_to_gene_gene_network, arg_tuples)

    # Collect successful results into a global dictionary
    for result in results:
        if result:
            base_name, stats = result
            global_stats[base_name] = stats

    # Save all centrality scores from all models into a single pickle file
    with open(args.output_pickle_path, "wb") as f:
        pickle.dump(global_stats, f)

    print("All models processed. Centrality scores saved.")

if __name__ == "__main__":
    main()
