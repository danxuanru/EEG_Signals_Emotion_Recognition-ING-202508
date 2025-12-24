import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import glob
import os

# Get the absolute path of the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

def perform_rm_anova(data, dv, within, subject):
    """
    Performs a repeated measures ANOVA.
    
    Args:
        data (pd.DataFrame): The dataframe containing the data.
        dv (str): The dependent variable.
        within (list): The within-subject factor(s).
        subject (str): The subject identifier.
        
    Returns:
        AnovaRM results object.
    """
    aov = AnovaRM(data=data, depvar=dv, subject=subject, within=within)
    res = aov.fit()
    return res

def analyze_data(file_path):
    """
    Analyzes data with multiple time points.
    """
    df = pd.read_csv(file_path)
    # Get time columns dynamically by excluding the 'Subject' column
    time_cols = [col for col in df.columns if col != 'Subject']
    
    # Melt the dataframe to long format
    df_melt = df.melt(id_vars=['Subject'], value_vars=time_cols,
                      var_name='Time', value_name='Accuracy')
    
    res = perform_rm_anova(df_melt, 'Accuracy', ['Time'], 'Subject')
    return res.anova_table

def main():
    # Use the script's directory to build absolute paths
    base_path = os.path.join(script_dir, 'merged_by_time')
    output_path = os.path.join(script_dir, 'anova_results_10s_15s.csv')

    files_15s = glob.glob(os.path.join(base_path, '*_15s.csv'))
    files_10s = glob.glob(os.path.join(base_path, '*_10s.csv'))
    
    results = []

    print("Analyzing 15s data...")
    for file in files_15s:
        print(f"  Processing {file}...")
        try:
            anova_table = analyze_data(file)
            feature_set = os.path.basename(file).replace('_01_15s.csv', '')
            f_val = anova_table['F Value'][0]
            p_val = anova_table['Pr > F'][0]
            results.append({'Feature Set': feature_set, 'Time Window': '15s', 'F-Value': f_val, 'P-Value': p_val})
        except Exception as e:
            print(f"    Could not process {file}: {e}")

    print("\nAnalyzing 10s data...")
    for file in files_10s:
        print(f"  Processing {file}...")
        try:
            anova_table = analyze_data(file)
            feature_set = os.path.basename(file).replace('_02_10s.csv', '')
            f_val = anova_table['F Value'][0]
            p_val = anova_table['Pr > F'][0]
            results.append({'Feature Set': feature_set, 'Time Window': '10s', 'F-Value': f_val, 'P-Value': p_val})
        except Exception as e:
            print(f"    Could not process {file}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    print(f"\nAnalysis complete. Results saved to {output_path}")
    print("\nResults Summary:")
    print(results_df)

if __name__ == '__main__':
    main()
