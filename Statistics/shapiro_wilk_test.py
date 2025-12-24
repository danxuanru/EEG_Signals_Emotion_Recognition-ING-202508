import pandas as pd
import numpy as np
from scipy import stats
import glob
import os

def perform_shapiro_wilk_test():
    """
    對同一Feature組合的前後15s兩組資料進行差值常態性檢定
    """
    
    # 設定資料路徑
    data_folder = "merged_by_time"
    output_file = "shapiro_wilk_results.csv"
    
    # 讀取所有_15s.csv檔案
    csv_files = glob.glob(os.path.join(data_folder, "*_15s.csv"))
    
    results = []
    
    for file_path in csv_files:
        print(f"處理檔案: {file_path}")
        
        # 讀取資料
        df = pd.read_csv(file_path)
        
        # 獲取檔案名稱作為特徵組合識別
        feature_combination = os.path.basename(file_path).replace("_15s.csv", "")
        
        # 假設資料格式有'time_period'欄位標示前後15s，或根據實際資料結構調整
        # 如果沒有明確標示，假設前半部分為前15s，後半部分為後15s
        
        # 移除非數值欄位（如果有的話）
        # numeric_columns = df.select_dtypes(include=[np.number]).columns
        # df_numeric = df[numeric_columns]
        df_numeric = df.iloc[:, 1:]
        
        # 將資料分為前後兩半
        mid_point = len(df_numeric) // 2
        before_15s = df_numeric.iloc[:mid_point]
        after_15s = df_numeric.iloc[mid_point:mid_point*2]
        
        # 確保兩組資料長度相同
        min_length = min(len(before_15s), len(after_15s))
        before_15s = before_15s.iloc[:min_length]
        after_15s = after_15s.iloc[:min_length]
        
        # 對每個特徵計算差值並進行Shapiro-Wilk檢定
        for column in df_numeric.columns:
            if column in before_15s.columns and column in after_15s.columns:
                # 計算差值
                differences = after_15s[column].values - before_15s[column].values
                
                # 移除NaN值
                differences = differences[~np.isnan(differences)]
                
                if len(differences) > 3:  # Shapiro-Wilk需要至少4個樣本
                    # 執行Shapiro-Wilk檢定
                    statistic, p_value = stats.shapiro(differences)
                    
                    # 判斷是否符合常態分布 (α = 0.05)
                    is_normal = p_value > 0.05
                    
                    results.append({
                        'Feature_Combination': feature_combination,
                        'Feature': column,
                        'Shapiro_Statistic': statistic,
                        'P_Value': p_value,
                        'Is_Normal': is_normal,
                        'Sample_Size': len(differences),
                        'Mean_Difference': np.mean(differences),
                        'Std_Difference': np.std(differences)
                    })
                else:
                    print(f"警告: {feature_combination} - {column} 樣本數不足")
    
    # 將結果轉換為DataFrame並輸出
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n常態性檢定完成！結果已輸出至: {output_file}")
    print(f"總共處理了 {len(results)} 個特徵")
    
    # 顯示摘要統計
    normal_count = results_df['Is_Normal'].sum()
    total_count = len(results_df)
    print(f"符合常態分布的特徵數: {normal_count}/{total_count} ({normal_count/total_count*100:.1f}%)")
    
    return results_df

if __name__ == "__main__":
    results = perform_shapiro_wilk_test()
    print("\n前5筆結果預覽:")
    print(results.head())
