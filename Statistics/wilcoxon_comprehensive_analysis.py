import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

def extract_info_from_filename(filename):
    """從檔案名稱提取特徵類型和時間段信息"""
    parts = filename.replace('.csv', '').split('_')
    
    # 根據不同的命名模式提取信息
    if len(parts) >= 4:
        feature_type = '_'.join(parts[:-2])  # 前面的部分是特徵類型
        time_info = parts[-1]  # 最後一部分是時間信息
    else:
        feature_type = '_'.join(parts[:-1])
        time_info = parts[-1]
    
    return feature_type, time_info

def perform_wilcoxon_test(data1, data2, alternative='two-sided'):
    """執行 Wilcoxon signed-rank test"""
    try:
        # 移除 NaN 值
        mask = ~(np.isnan(data1) | np.isnan(data2))
        data1_clean = data1[mask]
        data2_clean = data2[mask]
        
        if len(data1_clean) < 3:  # 需要至少3個樣本
            return np.nan, np.nan, len(data1_clean)
        
        # 執行 Wilcoxon signed-rank test
        statistic, p_value = wilcoxon(data1_clean, data2_clean, alternative=alternative)
        
        return statistic, p_value, len(data1_clean)
    
    except Exception as e:
        print(f"Error in Wilcoxon test: {e}")
        return np.nan, np.nan, 0

def analyze_merged_by_time_data():
    """分析 merged_by_time 目錄中的所有CSV文件"""
    
    # 設定目錄路徑
    data_dir = 'merged_by_time'
    
    if not os.path.exists(data_dir):
        print(f"目錄 {data_dir} 不存在")
        return
    
    # 獲取所有CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    csv_files.sort()
    
    print(f"找到 {len(csv_files)} 個CSV文件")
    print("文件列表:", csv_files)
    
    # 儲存所有結果
    all_results = []
    
    # 分析每個CSV文件
    for csv_file in csv_files:
        print(f"\n分析文件: {csv_file}")
        
        # 讀取數據
        file_path = os.path.join(data_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            print(f"數據形狀: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            
            # 提取特徵類型和時間信息
            feature_type, time_info = extract_info_from_filename(csv_file)
            
            # 假設第一列是Subject，第二列是準確率數據
            if len(df.columns) >= 2:
                accuracy_data = df.iloc[:, 1].values
                
                # 計算基本統計信息
                mean_acc = np.nanmean(accuracy_data)
                std_acc = np.nanstd(accuracy_data)
                median_acc = np.nanmedian(accuracy_data)
                min_acc = np.nanmin(accuracy_data)
                max_acc = np.nanmax(accuracy_data)
                
                # 與隨機基準（0.5）進行 Wilcoxon test
                baseline = np.full_like(accuracy_data, 0.5)
                w_stat, p_value, n_samples = perform_wilcoxon_test(accuracy_data, baseline, 'greater')
                
                # 效應大小 (r = Z / sqrt(N))
                if not np.isnan(w_stat) and n_samples > 0:
                    # 計算 Z score approximation
                    z_score = (w_stat - n_samples*(n_samples+1)/4) / np.sqrt(n_samples*(n_samples+1)*(2*n_samples+1)/24)
                    effect_size = abs(z_score) / np.sqrt(n_samples)
                else:
                    effect_size = np.nan
                
                # 儲存結果
                result = {
                    'File': csv_file,
                    'Feature_Type': feature_type,
                    'Time_Period': time_info,
                    'N_Subjects': n_samples,
                    'Mean_Accuracy': mean_acc,
                    'Std_Accuracy': std_acc,
                    'Median_Accuracy': median_acc,
                    'Min_Accuracy': min_acc,
                    'Max_Accuracy': max_acc,
                    'Wilcoxon_Statistic': w_stat,
                    'P_Value': p_value,
                    'Effect_Size_r': effect_size,
                    'Significant_05': p_value < 0.05 if not np.isnan(p_value) else False,
                    'Significant_01': p_value < 0.01 if not np.isnan(p_value) else False,
                    'Significant_001': p_value < 0.001 if not np.isnan(p_value) else False
                }
                
                all_results.append(result)
                
                print(f"  平均準確率: {mean_acc:.4f} ± {std_acc:.4f}")
                print(f"  Wilcoxon統計量: {w_stat}")
                print(f"  P值: {p_value}")
                print(f"  效應大小: {effect_size:.4f}" if not np.isnan(effect_size) else "  效應大小: N/A")
                
        except Exception as e:
            print(f"讀取文件 {csv_file} 時發生錯誤: {e}")
            continue
    
    # 創建結果DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 按特徵類型和時間段排序
    results_df = results_df.sort_values(['Feature_Type', 'Time_Period'])
    
    # 輸出結果到CSV
    output_file = 'wilcoxon_signed_rank_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n=== 分析完成 ===")
    print(f"總共分析了 {len(all_results)} 個文件")
    print(f"結果已保存到: {output_file}")
    
    # 顯示摘要統計
    print(f"\n=== 摘要統計 ===")
    print(f"顯著性結果 (p < 0.05): {results_df['Significant_05'].sum()} / {len(results_df)}")
    print(f"顯著性結果 (p < 0.01): {results_df['Significant_01'].sum()} / {len(results_df)}")
    print(f"顯著性結果 (p < 0.001): {results_df['Significant_001'].sum()} / {len(results_df)}")
    
    # 顯示最佳表現
    best_accuracy = results_df.loc[results_df['Mean_Accuracy'].idxmax()]
    print(f"\n最佳平均準確率: {best_accuracy['Mean_Accuracy']:.4f}")
    print(f"文件: {best_accuracy['File']}")
    print(f"特徵類型: {best_accuracy['Feature_Type']}")
    print(f"時間段: {best_accuracy['Time_Period']}")
    
    return results_df

def compare_conditions():
    """比較不同條件之間的差異"""
    print("\n=== 條件間比較 ===")
    
    data_dir = 'merged_by_time'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # 按特徵類型分組
    feature_groups = {}
    for csv_file in csv_files:
        feature_type, time_info = extract_info_from_filename(csv_file)
        if feature_type not in feature_groups:
            feature_groups[feature_type] = []
        feature_groups[feature_type].append((csv_file, time_info))
    
    comparison_results = []
    
    # 對每個特徵類型，比較不同時間段
    for feature_type, files_info in feature_groups.items():
        if len(files_info) > 1:
            print(f"\n比較特徵類型: {feature_type}")
            
            # 讀取所有時間段的數據
            data_dict = {}
            for csv_file, time_info in files_info:
                file_path = os.path.join(data_dir, csv_file)
                df = pd.read_csv(file_path)
                data_dict[time_info] = df.iloc[:, 1].values
            
            # 兩兩比較
            time_periods = list(data_dict.keys())
            for i in range(len(time_periods)):
                for j in range(i+1, len(time_periods)):
                    time1, time2 = time_periods[i], time_periods[j]
                    data1, data2 = data_dict[time1], data_dict[time2]
                    
                    w_stat, p_value, n_samples = perform_wilcoxon_test(data1, data2, 'two-sided')
                    
                    comparison_results.append({
                        'Feature_Type': feature_type,
                        'Condition_1': time1,
                        'Condition_2': time2,
                        'Mean_Diff': np.nanmean(data1) - np.nanmean(data2),
                        'Wilcoxon_Statistic': w_stat,
                        'P_Value': p_value,
                        'N_Samples': n_samples,
                        'Significant_05': p_value < 0.05 if not np.isnan(p_value) else False
                    })
                    
                    print(f"  {time1} vs {time2}: p = {p_value:.6f}")
    
    # 保存比較結果
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv('wilcoxon_pairwise_comparisons.csv', index=False, encoding='utf-8-sig')
        print(f"\n條件間比較結果已保存到: wilcoxon_pairwise_comparisons.csv")

if __name__ == "__main__":
    print("開始 Wilcoxon signed-rank test 分析...")
    
    # 主要分析：與基準比較
    results_df = analyze_merged_by_time_data()
    
    # 條件間比較
    compare_conditions()
    
    print("\n所有分析完成！")
