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
library_df_filtered_on_target = library_df[(library_df['On_target'] == True) & (library_df['PAM_Check'] == 'FORMAT_MATCH')].copy()
percent_rank = library_df_filtered_on_target[f'indel_ratio_MV{MV}'].tolist()
spacer_seq = library_df_filtered_on_target['spacer'].tolist()
pam_seq = library_df_filtered_on_target['PAM'].tolist()
indel_pos = library_df_filtered_on_target['target_spacer_differences'].tolist()
target_seqs = library_df_filtered_on_target['target_site'].tolist()
gene_name = library_df_filtered_on_target['gene_names'].tolist()

#%% Feature extraction step 
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

#%% 
# Assuming target_seq is 2nt+PAM+spacer+4nt
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
# Extract the Features
feature_vec = np.zeros([len(spacer_seq),561])
for idx in range(0,len(spacer_seq)):
    index = target_seqs[idx].find(pam_seq[idx]+spacer_seq[idx])
    target_seq = pam_seq[idx]+spacer_seq[idx]+target_seqs[idx][index+26:index+26+4]
    feature_vec[idx,:] = feature_extraction_step (target_seq,spacer_seq[idx])

#%% Feature names
print('creat the feature names') 
letters = 'ACTG'
numbers = [-6,-5,-4,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
mutations1 = [letter + str(number) for number in numbers for letter in letters]
pairs = ['AA', 'AC', 'AT', 'AG', 'CA', 'CC', 'CT', 'CG', 'TA', 'TC', 'TT', 'TG', 'GA', 'GC', 'GT', 'GG']
mutations2 = [pair + str(number) for number in numbers for pair in pairs]
features_names = ['int']+mutations1+mutations2+['GsCs']

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
y_train_binary = np.where(y_train > 0.2, 1, 0)
y_test_binary  = np.where(y_test > 0.2, 1, 0) 
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
plt.savefig(f'/mnt/singlefs-4/sandbox_yamidi/QC_Cellecta_data/ROC_ontarget_MV{MV}_LR_0_2_thre.png', bbox_inches='tight')

#%%
# Save the model to a file
joblib.dump(lr_final, f'/mnt/singlefs-4/sandbox_yamidi/QC_Cellecta_data/ontarget_MV{MV}_LR_0_2_thre.joblib')

#%%
coef_final = ((best_model.coef_).tolist()[0])
df_features = pd.DataFrame({'Feature_names': features_names, 'Feature_Value': coef_final})
df_features.to_csv(f'/mnt/singlefs-4/sandbox_yamidi/QC_Cellecta_data/selected_features_MV{MV}_ontarget_LR_0_2_thre.csv', index=False)
