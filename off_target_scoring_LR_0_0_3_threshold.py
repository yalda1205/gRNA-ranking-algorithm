import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.utils import shuffle
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='threadpoolctl')

#%% Extract the off-target libraries and extract the useful columns for training
library_df = pd.read_csv('/mnt/singlefs-4/sandbox_yamidi/QC_Cellecta_data/cellecta_library_211208.csv')
MV = 2 #2,3,4
library_df_filtered_off_target = library_df[(library_df['On_target'] == False)].copy()
library_df_filtered_on_target = library_df[(library_df['On_target'] == True)].copy()
percent_rank = library_df_filtered_off_target[f'indel_ratio_MV{MV}'].tolist()
spacer_seq = library_df_filtered_off_target['spacer'].tolist()
pam_seq = library_df_filtered_off_target['PAM'].tolist()
indel_pos = library_df_filtered_off_target['target_spacer_differences'].tolist()
target_seq = library_df_filtered_off_target['target_site'].tolist()
gene_name = library_df_filtered_off_target['gene_names'].tolist()

#%% Feature extraction step (nucleotide_feature at each position of spacer and PAM seq)
# Assuming target_seq is PAM+spacer
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

#%% Mode2 : Number of deletion (0, 1, 2, >2), insertion (0, 1, 2, >2), and substitution (0, 1, 2, >2), 
#   has been added to the feature vector

def initialize_feature_vector(mode, spacer_seq_length):
    if mode == 1:
        return np.zeros([spacer_seq_length, 400])
    else:
        return np.zeros([spacer_seq_length, 412])

def update_feature_vector(index, matrix, feature_vec, mode, ins, dels, sub):
    flattened_matrix = matrix.flatten()
    if mode == 1:
        feature_vec[index, :400] = flattened_matrix
    else:
        feature_vec[index, :400] = flattened_matrix
        # Encoding insertion, deletion, and substitution counts
        feature_vec[index, 400 + min(ins, 3)] = 1
        feature_vec[index, 404 + min(dels, 3)] = 1
        feature_vec[index, 408 + min(sub, 3)] = 1

def process_indel_pos(poses, base_pairs, matrix, mode):
    ins, dels, sub = 0, 0, 0
    for i in ast.literal_eval(poses):
        pos = i[0]
        if pos > 19:
            continue
        pair = f"{i[1]}:{i[2]}"
        matrix[base_pairs.index(pair), pos] += 1
        if mode == 2:
            if i[1] == '-':
                ins += 1
            if i[2] == '-':
                dels += 1
            if i[1] != '-' and i[2] != '-':
                sub += 1
    return ins, dels, sub

def feature_extraction(spacer_seq, pam_seq, indel_pos, mode=2):
    base_pairs = ['T:G', 'T:C', 'T:A', 'A:T', 'A:G', 'A:C', 'G:T', 'G:C', 'G:A', 'C:T', 'C:G', 'C:A',
                  '-:G', '-:C', '-:A', '-:T', 'G:-', 'C:-', 'A:-', 'T:-']
    matrix_shape = (len(base_pairs), 20)
    feature_vec = initialize_feature_vector(mode, len(spacer_seq))

    for index, poses in enumerate(indel_pos):
        matrix = np.zeros(matrix_shape)
        ins, dels, sub = process_indel_pos(poses, base_pairs, matrix, mode)
        update_feature_vector(index, matrix, feature_vec, mode, ins, dels, sub)

    nucleotide = np.zeros([len(spacer_seq), 96])
    for idx in range(len(spacer_seq)):
        nucleotide[idx, :] = feature_extraction_step(pam_seq[idx][2:] + spacer_seq[idx])

    feature_vec = np.concatenate((feature_vec, nucleotide), axis=1)
    return feature_vec

mode = 1
feature_vec = feature_extraction(spacer_seq, pam_seq, indel_pos, mode)

#%% Feature names
print('creat the feature names') 
letters = 'ACTG'
numbers = [-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
mutations1 = [letter + str(number) for number in numbers for letter in letters]

numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
pairs = ['T:G', 'T:C', 'T:A', 'A:T', 'A:G', 'A:C', 'G:T', 'G:C', 'G:A', 'C:T', 'C:G', 'C:A',
         '-:G', '-:C', '-:A', '-:T', 'G:-', 'C:-', 'A:-', 'T:-']
mutations = [pair + str(number) for number in numbers for pair in pairs]
if mode == 2:
    diff_spacer_target = ['ins0', 'ins1', 'ins2', 'ins>2',
                        'dels0', 'dels1', 'dels2', 'dels>2',
                        'sub0', 'sub1', 'sub2', 'sub>2']
    features_names = ['int']+mutations+diff_spacer_target+mutations1
else:
    features_names = ['int']+mutations+mutations1

#%% Training step (prepare the data (binarization) and split data to train and test)
    # the validation part is Gene name based
X = feature_vec.copy()
y = percent_rank
z = np.array(gene_name)
y = np.array(y, dtype=np.float64)


# Convert to DataFrame for easier manipulation
df = pd.DataFrame(X)
df['label'] = y
df['gene_name'] = z

# Lists to hold the train and test splits
train_list = []
test_list = []

# Split the data for each gene
for gene, group in df.groupby('gene_name'):
    X_group = group.drop(['label', 'gene_name'], axis=1).values  # Features for the current gene
    y_group = group['label'].values  # Labels for the current gene
    
    if len(group) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X_group, y_group, test_size=0.2, random_state=42)
    else:
        # If the group has only one sample, assign it to the training set
        X_train, X_test, y_train, y_test = X_group, X_group, y_group, y_group
    
    # Combine X_train and y_train, X_test and y_test to form train and test sets
    train_group = np.hstack((X_train, y_train.reshape(-1, 1)))
    test_group = np.hstack((X_test, y_test.reshape(-1, 1)))
    
    train_list.append(train_group)
    test_list.append(test_group)

# Concatenate all groups to form the final train and test sets
train_array = np.vstack(train_list)
test_array = np.vstack(test_list)

# Separate features and labels
X_train = train_array[:, :-1]
y_train = train_array[:, -1]

X_test = test_array[:, :-1]
y_test = test_array[:, -1]

# Shuffle the final train and test sets
X_train, y_train = shuffle(X_train, y_train, random_state=42)
X_test, y_test = shuffle(X_test, y_test, random_state=42)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit scaler on training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert target variables to binary based on threshold
y_train_binary = np.where(y_train > 0.03, 1, 0)
y_test_binary  = np.where(y_test > 0.03, 1, 0) 
#%% training step (LR method)
print('Training step') 
# Add a column of ones to account for the intercept term
X_train_with_intercept = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))
X_test_with_intercept = np.hstack((np.ones((X_test_scaled.shape[0], 1)), X_test_scaled))

# Define Logistic Regression classifier with balanced class weights
lr = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)

# Define parameter grid for hyperparameter tuning
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

# Define nested cross-validation strategy
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform nested cross-validation
best_auc = 0
best_model = None
for train_index, val_index in outer_cv.split(X_train_with_intercept, y_train_binary):
    X_train_fold, X_val_fold = X_train_with_intercept[train_index], X_train_with_intercept[val_index]
    y_train_fold, y_val_fold = y_train_binary[train_index], y_train_binary[val_index]
    
    # Perform GridSearchCV for hyperparameter tuning
    inner_grid_search = GridSearchCV(lr, param_grid, cv=inner_cv, scoring='roc_auc')
    inner_grid_search.fit(X_train_fold, y_train_fold)
    
    # Select the best model based on inner CV
    best_lr = inner_grid_search.best_estimator_
    
    # Train the best model on the entire training fold
    best_lr.fit(X_train_fold, y_train_fold)
    
    # Evaluate the model on the validation fold
    y_pred_val = best_lr.predict_proba(X_val_fold)[:, 1]
    auc_ = roc_auc_score(y_val_fold, y_pred_val)
    
    # Check if this model performs better
    if auc_ > best_auc:
        best_auc = auc_
        best_model = best_lr

# After nested cross-validation, train the best model on the entire training data
best_model.fit(X_train_with_intercept, y_train_binary)

# Select features based on the coefficients
selected_features = np.where(np.abs(best_model.coef_) > 0)[1]

# Use only the selected features
X_train_selected = X_train_with_intercept[:, selected_features]
X_test_selected = X_test_with_intercept[:, selected_features]

# Train the logistic regression classifier on the selected features with balanced class weights
lr_final = LogisticRegression(max_iter=1000)
lr_final.fit(X_train_selected, y_train_binary)

# Evaluate the logistic regression classifier on the test data
y_pred_test = lr_final.predict_proba(X_test_selected)[:, 1]
test_auc = roc_auc_score(y_test_binary, y_pred_test)
print("Train AUC:", best_auc)
print("Test AUC:", test_auc)

#%% Plot ROC figure
# Get predicted probabilities for train and test sets
y_train_pred_prob = best_model.predict_proba(X_train_with_intercept)[:, 1]
y_test_pred_prob = best_model.predict_proba(X_test_with_intercept)[:, 1]

# Compute ROC curve and ROC area for train set
fpr_train, tpr_train, _ = roc_curve(y_train_binary, y_train_pred_prob)
roc_auc_train = auc(fpr_train, tpr_train)

# Compute ROC curve and ROC area for test set
fpr_test, tpr_test, _ = roc_curve(y_test_binary, y_test_pred_prob)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curve
plt.figure()
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='Train ROC curve (AUC = %0.2f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, color='red', lw=2, label='Test ROC curve (AUC = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig('/mnt/singlefs-4/sandbox_yamidi/QC_Cellecta_data/ROC_offtarget_LR_0_0_3_thre_feature1.png', bbox_inches='tight')

#%%
# Save the model to a file
joblib.dump(lr_final, '/mnt/singlefs-4/sandbox_yamidi/QC_Cellecta_data/offtarget_LR_0_0_3_thre_feature_1.joblib')

#%%
coef_final = ((best_model.coef_).tolist()[0])
df_features = pd.DataFrame({'Feature_names': features_names, 'Feature_Value': coef_final})
df_features.to_csv(f'/mnt/singlefs-4/sandbox_yamidi/QC_Cellecta_data/selected_features1_MV{MV}_offtarget_LR_0_0_3_thre.csv', index=False)
