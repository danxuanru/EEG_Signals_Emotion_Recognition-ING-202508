import pandas as pd
import numpy as np

def generate_summary_report():
    """生成 Wilcoxon signed-rank test 分析摘要報告"""
    
    print("=" * 60)
    print("Wilcoxon Signed-Rank Test 分析摘要報告")
    print("=" * 60)
    
    # 讀取主要結果
    main_results = pd.read_csv('wilcoxon_signed_rank_results.csv')
    
    print(f"\n【主要分析結果】")
    print(f"總共分析文件數: {len(main_results)}")
    print(f"所有結果都與隨機基準 (0.5) 進行比較")
    
    # 顯著性統計
    sig_05 = main_results['Significant_05'].sum()
    sig_01 = main_results['Significant_01'].sum()
    sig_001 = main_results['Significant_001'].sum()
    
    print(f"\n【顯著性結果統計】")
    print(f"p < 0.05:  {sig_05}/{len(main_results)} ({sig_05/len(main_results)*100:.1f}%)")
    print(f"p < 0.01:  {sig_01}/{len(main_results)} ({sig_01/len(main_results)*100:.1f}%)")
    print(f"p < 0.001: {sig_001}/{len(main_results)} ({sig_001/len(main_results)*100:.1f}%)")
    
    # 準確率排名
    print(f"\n【準確率排名 (Top 5)】")
    top_5 = main_results.nlargest(5, 'Mean_Accuracy')[['Feature_Type', 'Time_Period', 'Mean_Accuracy', 'P_Value', 'Effect_Size_r']]
    for idx, row in top_5.iterrows():
        print(f"{row['Feature_Type']} ({row['Time_Period']}): {row['Mean_Accuracy']:.4f} (p={row['P_Value']:.2e}, r={row['Effect_Size_r']:.3f})")
    
    # 按特徵類型分組分析
    print(f"\n【按特徵類型分組】")
    feature_groups = main_results.groupby('Feature_Type')
    for feature_type, group in feature_groups:
        mean_acc = group['Mean_Accuracy'].mean()
        best_time = group.loc[group['Mean_Accuracy'].idxmax(), 'Time_Period']
        best_acc = group['Mean_Accuracy'].max()
        print(f"{feature_type}:")
        print(f"  平均準確率: {mean_acc:.4f}")
        print(f"  最佳時間段: {best_time} (準確率: {best_acc:.4f})")
    
    # 效應大小分析
    print(f"\n【效應大小分析】")
    large_effect = main_results[main_results['Effect_Size_r'] >= 0.5].shape[0]
    medium_effect = main_results[(main_results['Effect_Size_r'] >= 0.3) & (main_results['Effect_Size_r'] < 0.5)].shape[0]
    small_effect = main_results[(main_results['Effect_Size_r'] >= 0.1) & (main_results['Effect_Size_r'] < 0.3)].shape[0]
    
    print(f"大效應 (r ≥ 0.5): {large_effect}/{len(main_results)}")
    print(f"中效應 (0.3 ≤ r < 0.5): {medium_effect}/{len(main_results)}")
    print(f"小效應 (0.1 ≤ r < 0.3): {small_effect}/{len(main_results)}")
    
    # 讀取配對比較結果
    try:
        pairwise_results = pd.read_csv('wilcoxon_pairwise_comparisons.csv')
        
        print(f"\n【條件間配對比較】")
        print(f"總比較次數: {len(pairwise_results)}")
        
        sig_comparisons = pairwise_results[pairwise_results['Significant_05'] == True]
        print(f"顯著差異 (p < 0.05): {len(sig_comparisons)}/{len(pairwise_results)}")
        
        if len(sig_comparisons) > 0:
            print(f"\n【顯著的配對比較】")
            for idx, row in sig_comparisons.iterrows():
                direction = ">" if row['Mean_Diff'] > 0 else "<"
                print(f"{row['Feature_Type']}: {row['Condition_1']} {direction} {row['Condition_2']} (p={row['P_Value']:.4f})")
        
    except FileNotFoundError:
        print(f"\n【配對比較】文件未找到")
    
    print(f"\n" + "=" * 60)
    print("分析完成！結果已保存到以下文件:")
    print("- wilcoxon_signed_rank_results.csv (主要結果)")
    print("- wilcoxon_pairwise_comparisons.csv (配對比較)")
    print("=" * 60)

if __name__ == "__main__":
    generate_summary_report()
