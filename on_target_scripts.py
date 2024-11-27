#%%
import numpy as np
import argparse
import pandas as pd
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
    features_df = pd.read_csv(f"/mnt/singlefs-4/sandbox_yamidi/QC_Cellecta_data/selected_features_MV{args.Cas12i2_Type}_ontarget_LR_0_2_thre.csv")
    feature_value = features_df['Feature_Value'].tolist()

    # Load the input data
    input_df = pd.read_csv(args.input_csv)
    spacer_seq = input_df['spacer'].tolist()
    pam_seq = input_df['PAM'].tolist()
    target_seq = input_df['target'].tolist()

    # Feature Extraction Step
    # Assuming target_seq is 2nt+PAM+spacer(20nt)+4nt
    def feature_extraction_step (target_seq, spacer):
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
        rows_to_delete = [3, 4]  # Rows are zero-indexed
        nucleotide_features = np.delete(nucleotide_features, rows_to_delete, axis=0)
        nucleotide_features = nucleotide_features.reshape(1, -1)
        # Di-nucleotide identity at each position (464 features)
        di_nucleotide_features = np.zeros((len(target_seq), 16))
        for i in range(len(target_seq) - 1):
            di_nucleotide = target_seq[i:i+2]
            index = ['AA', 'AC', 'AT', 'AG', 'CA', 'CC', 'CT', 'CG',
                    'TA', 'TC', 'TT', 'TG', 'GA', 'GC', 'GT', 'GG'].index(di_nucleotide)
            di_nucleotide_features[i, index] = 1
        rows_to_delete = [3, 25]  # Rows are zero-indexed
        di_nucleotide_features = np.delete(di_nucleotide_features, rows_to_delete, axis=0)
        di_nucleotide_features = di_nucleotide_features.reshape(1, -1)
        # GC count features
        gc_count_features = np.zeros([1,1])
        for i, base in enumerate(spacer):
            if base == 'G' or base == 'C':
                gc_count_features += 1
    
        feature_vec = np.concatenate((nucleotide_features, di_nucleotide_features, gc_count_features), axis=1)
        return feature_vec
    
    feature_vec = np.zeros([len(input_df),len(features_df)-1])
    for idx, (spacers, pams, targets) in enumerate(zip(spacer_seq, pam_seq, target_seq)):
        query  = Seq(pams+spacers)
        target = Seq(targets)
        
        # Perform local alignment
        alignments = pairwise2.align.localms(query, target, 1, -1, -1, -1)
        best_alignment = alignments[0]
        aligned_query, aligned_target, _, _, _ = best_alignment

        # Find positions of '-' before and after the trimmed sequence
        before_positions = aligned_query.find(next(filter(str.isalpha, aligned_query)))
        reversed_query = aligned_query[::-1]
        after_positions = len(aligned_query) - reversed_query.find(next(filter(str.isalpha, reversed_query)))
        target_site = targets[before_positions-2:after_positions+4]
        
        feature_vec[idx,:] = feature_extraction_step (target_site,spacers)

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
