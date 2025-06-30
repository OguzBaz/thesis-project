import pandas as pd
import rlssm
import warnings


# Load trial-by-trial data
file_path = 'online_data_for_matlab.txt'
df = pd.read_csv(file_path)

# Load summary statistics data
summary_stats_path = 'summary_stats_subs.txt'
df_summary = pd.read_csv(summary_stats_path, sep='\t')
df_summary.tail(5)

print(f"Number of rows in trial-by-trial df: {len(df)}")
print(f"Number of rows in summary stats df: {len(df_summary)}")




















#####################
# Prepare data for a single subject
if 'subject_id' in df.columns:
    subject_data = df[df['subject_id'] == 'sub1'].copy()
else:
    raise KeyError("The column 'subject_id' is missing in the DataFrame.")


# Set block_label and trial_block as required by rlssm
subject_data['block_label'] = 1  # required by rlssm, set to 1 if only one block
subject_data['trial_block'] = range(1, len(subject_data) + 1)  # trial index


# Validate that the 'reward' column contains only binary values (1 or 0)
if not subject_data['reward'].isin([0, 1]).all():
    raise ValueError("The 'reward' column contains values other than 0 and 1.")


# f_cor: 1 if correct, 0 otherwise; f_inc: 1 if incorrect, 0 otherwise
subject_data['f_cor'] = (subject_data['reward'] == 1).astype(int)
subject_data['f_inc'] = (subject_data['reward'] == 0).astype(int)


# cor_option and inc_option: if you have a 'correct_option' column, use it; otherwise, set to NaN
if 'correct_option' in subject_data.columns:
    subject_data['cor_option'] = subject_data['correct_option']
    subject_data['inc_option'] = 1 - subject_data['correct_option']
else:
    warnings.warn("The column 'correct_option' is missing in the DataFrame. Setting 'cor_option' and 'inc_option' to NaN.")
    subject_data['cor_option'] = pd.NA
    subject_data['inc_option'] = pd.NA


# Build the DataFrame for rlssm (only required columns)
data = pd.DataFrame({
    'choice': subject_data['choice_1'],
    'reward': subject_data['reward'],
    'block_label': subject_data['block_label'],
    'trial_block': subject_data['trial_block'],
    'f_cor': subject_data['f_cor'],
    'f_inc': subject_data['f_inc'],
    'cor_option': subject_data['cor_option'],
    'inc_option': subject_data['inc_option'],
})


# Instantiate and fit the model
model = rlssm.RLModel_2A(hierarchical_levels=1)
fit = model.fit(
    data=data,
    K=2,  # number of options per trial
    initial_value_learning=0.5,  # initial Q-value
    n_chains=2,
    n_iter=1000,
    n_warmup=500,
    print_diagnostics=True
)

print(fit)


