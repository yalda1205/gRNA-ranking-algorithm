#%%
import numpy as np
import argparse
import pandas as pd
import ast
from Bio import pairwise2
from Bio.Seq import Seq

# Function to parse command-line arguments
# input cvs should containing two columns "spacer", "PAM" and "target" (PAM seq :NTTN)
# Cas12i2_Type should be integer 2, 3, or 4
def parse_args():
    parser = argparse.ArgumentParser(description="Predict the gRNA ranc score.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file (including .csv).")
    parser.add_argument("--output_csv", required=True, help="Path to the output CSV file (including .csv).")
    parser.add_argument("--Cas12i2_Type", required=True, help="Select between 2, 3, and 4")
    return parser.parse_args()

# Main function
def main():
    # Parse command-line arguments
    args = parse_args()

    # Load the extracted features from GLM model
    features_df = pd.read_csv(f"/mnt/singlefs-4/sandbox_yamidi/QC_Cellecta_data/selected_features1_MV{args.Cas12i2_Type}_offtarget_LR_0_0_3_thre.csv")
    feature_value = features_df['Feature_Value'].tolist()

    # Load the input data
    input_df = pd.read_csv(args.input_csv)
    spacer_seq = input_df['spacer'].tolist()
    pam_seq = input_df['PAM'].tolist()
    target_seq = input_df['target'].tolist()

    # Feature Extraction Step
    # Assuming target_seq is PAM+spacer(20nt)
    def feature_extraction_step (target_seq):
        # Nucleotide identity at each position (120 features)
        nucleotide_features = np.zeros((len(target_seq), 4))
        for i, base in enumerate(target_seq):
            if base == 'A':
                nucleotide_features[i, 0] = 1
            elif base == 'C':
                nucleotide_features[i, 1] = 1
            elif base == 'T':
                nucleotide_features[i, 2] = 1
            elif base == 'G':
                nucleotide_features[i, 3] = 1
        nucleotide_features = nucleotide_features.reshape(1, -1)
        return nucleotide_features
    
    # Ectract the difference between target sequence and gRNA
    # This function takes two sequences and returns the number of differences and a list of positions where differences occur.
    def find_differences(seq1, seq2):
        # Ensure both sequences have the same length
        if len(seq1) != len(seq2):
            raise ValueError("Sequences must have the same length")

        # Initialize a variable to count differences
        num_differences = 0
        differences = []

        # Iterate through the sequences and count differences
        for i, (char1, char2) in enumerate(zip(seq1, seq2)):
            if char1 != char2:
                num_differences += 1
                differences.append((i, char1, char2))

        return num_differences, differences
    
    # Initialize new columns in the DataFrame to store the results
    input_df['deletion'] = 0
    input_df['insertion'] = 0
    input_df['substitution'] = 0
    input_df['pos_indel'] = ''

    # Iterate through spacer and target sequences
    for idx, (spacer_sequence, target_sequence) in enumerate(zip(spacer_seq, target_seq)):
        query = Seq(spacer_sequence)
        target = Seq(target_sequence)
            
        # Perform local alignment
        alignments = pairwise2.align.localms(query, target, 1, -1, -1, -1)
        best_alignment = alignments[0]
        aligned_query, aligned_target, _, _, _ = best_alignment

        trimmed_sequence = aligned_query.strip('-')
        # Find positions of '-' before and after the trimmed sequence
        before_positions = aligned_query.find(next(filter(str.isalpha, aligned_query)))
        reversed_query = aligned_query[::-1]
        after_positions = len(aligned_query) - reversed_query.find(next(filter(str.isalpha, reversed_query)))

        gap_count_del = aligned_target.count('-')
        gap_count_ins = trimmed_sequence.count('-')
        num_differences, differences = find_differences(trimmed_sequence, aligned_target[before_positions:after_positions])
        count_substitution = num_differences - (gap_count_del + gap_count_ins)

        # Update DataFrame with results
            
        input_df.at[idx, 'deletion'] = gap_count_del
        input_df.at[idx, 'insertion'] = gap_count_ins
        input_df.at[idx, 'substitution'] = count_substitution
        input_df.at[idx, 'pos_indel'] = differences

    feature_vec = np.zeros([len(input_df),400])
    indel_pos_RD = input_df['pos_indel'].tolist()
    
    base_pairs = ['T:G', 'T:C', 'T:A', 'A:T', 'A:G', 'A:C', 'G:T', 'G:C', 'G:A', 'C:T', 'C:G', 'C:A',
                '-:G', '-:C', '-:A', '-:T', 'G:-', 'C:-', 'A:-', 'T:-']
    matrix = np.zeros((len(base_pairs), 20))

    for index, poses in enumerate(indel_pos_RD):
        for i in poses:
            pos = i[0]
            if pos>19:
                continue
            pair = f"{i[1]}:{i[2]}"
            matrix[base_pairs.index(pair), pos] += 1
        feature_vec[index,:] = matrix.flatten()
        matrix = np.zeros((len(base_pairs), 20))

    nucleotide = np.zeros([len(input_df),96])
    for idx in range(0,len(input_df)):
        targets = pam_seq[idx]+spacer_seq[idx]
        nucleotide[idx,:] = feature_extraction_step (targets)

    feature_vec = np.concatenate((feature_vec, nucleotide), axis=1)

    # Predict the gRNA rank scor
    coef_final_array = np.array(feature_value[1:]) # Extract coefficients excluding the intercept
    # Calculate the gRNA rank score using the GLM model
    g = feature_value[0]+ np.dot(feature_vec , coef_final_array)  # Apply the GLM model to the feature vector
    rank = 1/(1+np.exp(-g)) # Convert the score to a rank using the sigmoid function
    # Save the output csv
    input_df['gRNA_rank_score'] = rank
    input_df.to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    main()
