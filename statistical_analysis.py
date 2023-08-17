import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import numpy as np
import scikit_posthocs as sp
from itertools import combinations
from statannotations.Annotator import Annotator

with open('/path/to/master_df.pickle', 'rb') as f:
    master_df = pickle.load(f)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def resolve_duplicate_columns(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique(): 
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df


master_df = resolve_duplicate_columns(master_df)

'''Slicing up the dataset using column names - DEPRECATED IN FAVOUR OF DICT, NEEDS UPDATING (see plotly ipynb)'''

# Get the index of the start and end columns
start_idx = list(master_df.columns).index("1-Unstim-Cells-FoP")
end_idx = list(master_df.columns).index("2-PMA-Pop3-PMNs-Hmox1pos-Hmox1-MFI")

# Get all column names between the start and end columns (inclusive) that do not contain the word "Pop"
trimmed_fc_colnames = [col for col in master_df.columns[start_idx : end_idx+1] if 'Pop' not in col]
pop1_fc_colnames = [col for col in master_df.columns[start_idx : end_idx+1] if 'Pop1' in col]
pop2_fc_colnames = [col for col in master_df.columns[start_idx : end_idx+1] if 'Pop2' in col]
pop3_fc_colnames = [col for col in master_df.columns[start_idx : end_idx+1] if 'Pop3' in col]

start_idx_2 = list(master_df.columns).index('1-FoP')
end_idx_2 = list(master_df.columns).index('2-%ROS-lo')

ROS_colnames = [col for col in master_df.columns[start_idx_2 : end_idx_2+1]]

start_idx_3 = list(master_df.columns).index('RBC')
end_idx_3 = list(master_df.columns).index('Gra%')

cell_count_colnames = [col for col in master_df.columns[start_idx_3 : end_idx_3+1]]

start_idx_4 = list(master_df.columns).index('CD163 (BR28) (28)')
end_idx_4 = list(master_df.columns).index('TFR(BR13) (13)')

cytokine_colnames = [col for col in master_df.columns[start_idx_4 : end_idx_4+1]]

start_idx_5 = list(master_df.columns).index('Initial_weight')
end_idx_5 = list(master_df.columns).index('End_haem')

biometric_colnames = [col for col in master_df.columns[start_idx_5 : end_idx_5+1]]

start_idx_6 = list(master_df.columns).index('media_netosis')
end_idx_6 = list(master_df.columns).index('nts_netosis')

netosis_colnames = [col for col in master_df.columns[start_idx_6 : end_idx_6+1]]

def kruskal_wallis(column, master_df, group_column):
    column_df = master_df[[group_column, column]].dropna()
    groups = [column_df[column][column_df[group_column] == group].values for group in column_df[group_column].unique()]
    h_stat, p_value = kruskal(*groups)

    # Dunn's post hoc if Kruskal-Wallis is significant
    post_hoc = None
    if p_value < 0.05:
        post_hoc_res = sp.posthoc_dunn(column_df, val_col=column, group_col=group_column, p_adjust='bonferroni')
        np.fill_diagonal(post_hoc_res.values, np.nan)
        post_hoc = post_hoc_res
    return {
        'group_h_value': h_stat,
        'group_p_value': p_value,
        'test': 'Kruskal-Wallis',
        'post_hoc': post_hoc
    }

def load_dataframe(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def filter_and_process_dataframe(df, group_column):
    df[group_column] = df[group_column].astype('object')
    df = df[df[group_column].isin([1.0, 2.0, 3.0, 4.0, 5.0])]
    df = df.dropna(subset=[group_column])

    return df

def determine_columns_of_interest(df, valid_dtypes, exclude_prefixes, exclude_columns):
    return [col for col in df.columns 
            if df[col].dtype in valid_dtypes 
            and not col.startswith(tuple(exclude_prefixes)) 
            and col not in exclude_columns]

def perform_test_and_save_results(columns, df, group_column, test_function):
    test_results = {column: test_function(column, df, group_column) for column in columns}
    results_df = pd.DataFrame(test_results).T.sort_values("group_p_value")
    
    corrected_p_values = multipletests(results_df['group_p_value'].values, method='bonferroni')[1]
    results_df['group_p_value_corrected'] = corrected_p_values

    return results_df

def prepare_data_for_plotting(filepath, test_function=kruskal_wallis):
    master_df = load_dataframe(filepath)
    master_df = resolve_duplicate_columns(master_df)
    master_df = filter_and_process_dataframe(master_df, 'ph_HDBSCAN')
    master_df = master_df[cytokine_colnames + cell_count_colnames + ['PMA_fc', 'LPS_fc', 'NTS_fc'] + ['ph_HDBSCAN'] + ['ros comparison 1', 'pc_unstim', 'pc_pma']]

    columns_of_interest = determine_columns_of_interest(master_df, ['float64', 'int64'], 
                                ["2-PMA", "2-Unstim", "1-PMA", "1-Unstim", "umap"], 
                                ['ph_HDBSCAN'])
    results_df = perform_test_and_save_results(columns_of_interest, master_df, 'ph_HDBSCAN', test_function)

    sig_features = results_df[results_df["group_p_value_corrected"] <= 0.05].index
    data_long = master_df.melt(id_vars=['ph_HDBSCAN'], value_vars=sig_features)

    return data_long, results_df['post_hoc']

group_column = 'ph_HDBSCAN'

file_path = '/path/to/master_df.pickle'
data_long_for_plot, post_hoc_annotations = prepare_data_for_plotting(file_path)

palette = ['#397fb9', '#5cb55b', '#a05dab', '#fd8a18', '#fdfd46']

name_mapping = {
    "TNFa (BR12) (12)": "TNFα",
    "ros comparison 1": "ROS assay",
    "IFNg (BR29) (29)": "IFNγ",
    "pc_pma": "% Change in in phagocytosed Salmonella (PMA stimulated)",
    "LBP(BR57) (57)":"LBP",
    "IL-12/IL-23p40 (BR67) (67)": "IL-12",
    "IL-4 (BR39) (39)": "IL-4",
    "IL-6 (BR13) (13)": "IL-6",
    "S100A9 (BR46) (46)": "S100A9",
    "IL-1b/IL-1F2 (BR57) (57)": "IL-1β",
    "Mon%": "Monocytes (%)",
    "HCT": "Haematocrit",
    "HGB": "Haemoglobin",
    "MCHC": "Mean Corpuscular Haemoglobin Concentration",
    "RBC": "Red Blood Cells"
}

units_mapping = {
    'TNFα': 'pg/mL',
    'ROS assay': 'PMA MFI/Unstim MFI',
    'IFNγ': 'pg/mL',
    '% Change in in phagocytosed Salmonella (PMA stimulated)': '% change (symlog)',
    'LBP': 'ng/mL',
    'IL-12': 'ng/mL',
    'IL-4': 'pg/mL',
    'IL-6': 'pg/mL',
    'S100A9': 'pg/mL',
    'IL-1β': 'pg/mL',
    'Haematocrit': 'HCT %',
    'Mean Corpuscular Haemoglobin Concentration': 'g/dL', # means 'Mean Corpuscular Haemoglobin Concentration'
    'Haemoglobin': 'g/dL',
    'Monocytes (%)': 'Mon %',
    'Red Blood Cells': '10^12/L'
}

data_long_for_plot['variable'] = data_long_for_plot['variable'].replace(name_mapping)
post_hoc_annotations.index = post_hoc_annotations.index.to_series().replace(name_mapping)

group_labels = [1.0, 2.0, 3.0, 4.0, 5.0]

for variable_name in data_long_for_plot['variable'].unique():

    data_subset = data_long_for_plot[data_long_for_plot['variable'] == variable_name]

    post_hoc_matrix = post_hoc_annotations[variable_name]
    
    pairs = list(combinations(group_labels, 2))
    pvalues = [post_hoc_matrix.loc[pair[0], pair[1]] for pair in pairs]

    significant_pairs = [pair for pair, pvalue in zip(pairs, pvalues) if pvalue < 0.05]
    significant_pvalues = [pvalue for pvalue in pvalues if pvalue < 0.05]
    
    if not significant_pairs:
        plt.figure(figsize=(5, 4)) 
        sns.boxplot(x=group_column, y='value', data=data_subset, palette=palette)
        plt.title(variable_name) 
        plt.show()
        continue
    
    plt.figure(figsize=(5, 4)) 
    ax = sns.boxplot(x=group_column, y='value', data=data_subset, palette=palette)
    plt.ylabel(units_mapping[variable_name])
    plt.xlabel("Immunophenotype Group")
    plt.title(variable_name)  
    
    annot = Annotator(ax=ax, pairs=significant_pairs, data=data_subset, x=group_column, y='value')
    annot.configure(test=None, test_short_name='Post Hoc').set_pvalues(pvalues=significant_pvalues).annotate()
    
    plt.show()
