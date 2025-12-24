"""
AsI (Asymmetry Index) 濾波機制
用於EEG情緒識別中的原始訊號過濾

此模組實現基於額葉不對稱性的情緒激發檢測機制。
通過計算多維定向資訊 (MDI) 來量化左右額葉之間的資訊流動，
並使用 AsI 指數來篩選出高情緒激發的訊號窗口。

電極配置:
- Fp1, Fp2, Fz（資料集沒有 AF3, AF4 通道）

Baseline 設置:
- 使用中性影片（video index 13-16）作為 baseline
- 根據每個受試者的 video order 找到中性影片位置

參考資料: AsI 機制論文
"""

import numpy as np
import os
import pickle
import scipy.io as sio
from pathlib import Path
import argparse
from reorder_vids import video_order_load
import json
from datetime import datetime


def calculate_mdi(X_data, Y_data, Z_data, A=5, B=5):
    """
    計算多維定向資訊 (Multidimensional Directed Information, MDI)
    
    參數:
        X_data: 左額葉電極訊號 (1D array)
        Y_data: 右額葉電極訊號 (1D array)
        Z_data: Fz 電極訊號 (1D array)
        A: 歷史時間點數量 (預設=5)
        B: 未來時間點偏移量 (預設=5)
    
    返回:
        S_XY: X到Y的定向資訊
    """
    R1, R2, R3, R4 = [], [], [], []
    
    # 構建協方差矩陣 R1 和 R2
    for k in range(A, len(X_data)):
        vec_R1, vec_R2 = [], []
        
        # 添加歷史資訊 X_A, Y_A, Z_A
        vec_R1.extend([X_data[k-i] for i in range(A)])
        vec_R2.extend([X_data[k-i] for i in range(A)])
        vec_R1.extend([Y_data[k-i] for i in range(A)])
        vec_R2.extend([Y_data[k-i] for i in range(A)])
        vec_R1.extend([Z_data[k-i] for i in range(A)])
        vec_R2.extend([Z_data[k-i] for i in range(A)])
        
        # R1 包含當前 x_k
        vec_R1.append(X_data[k])
        
        # 兩者都包含當前 y_k 和 z_k
        vec_R1.append(Y_data[k])
        vec_R2.append(Y_data[k])
        vec_R1.append(Z_data[k])
        vec_R2.append(Z_data[k])
        
        R1.append(vec_R1)
        R2.append(vec_R2)
    
    # 構建協方差矩陣 R3 和 R4
    for k in range(A, len(X_data) - B):
        vec_R3, vec_R4 = [], []
        
        # 添加歷史資訊
        vec_R3.extend([X_data[k-i] for i in range(A)])
        vec_R4.extend([X_data[k-i] for i in range(A)])
        vec_R3.extend([Y_data[k-i] for i in range(A)])
        vec_R4.extend([Y_data[k-i] for i in range(A)])
        vec_R3.extend([Z_data[k-i] for i in range(A)])
        vec_R4.extend([Z_data[k-i] for i in range(A)])
        
        # R3 只包含當前 y_k
        vec_R3.append(Y_data[k])
        vec_R4.append(Y_data[k])
        
        # R4 包含當前 x_k
        vec_R4.append(X_data[k])
        
        # 兩者都包含未來 z_{k+B}
        vec_R3.append(Z_data[k + B])
        vec_R4.append(Z_data[k + B])
        
        R3.append(vec_R3)
        R4.append(vec_R4)
    
    # 計算協方差矩陣的行列式
    R1, R2, R3, R4 = np.array(R1), np.array(R2), np.array(R3), np.array(R4)
    
    # 添加正則化以避免奇異矩陣
    epsilon = 1e-10
    det_R1 = np.linalg.det(np.cov(R1, rowvar=False) + epsilon * np.eye(R1.shape[1]))
    det_R2 = np.linalg.det(np.cov(R2, rowvar=False) + epsilon * np.eye(R2.shape[1]))
    det_R3 = np.linalg.det(np.cov(R3, rowvar=False) + epsilon * np.eye(R3.shape[1]))
    det_R4 = np.linalg.det(np.cov(R4, rowvar=False) + epsilon * np.eye(R4.shape[1]))
    
    # 確保行列式為正值
    det_R1, det_R2 = max(det_R1, epsilon), max(det_R2, epsilon)
    det_R3, det_R4 = max(det_R3, epsilon), max(det_R4, epsilon)
    
    # 計算 MDI: S_XY = 0.5 * log(det(R1)*det(R3) / (det(R2)*det(R4)))
    S_XY = 0.5 * np.log((det_R1 * det_R3) / (det_R2 * det_R4))
    
    return S_XY


def calculate_asi_for_window(window_data, baseline_data, channel_indices, fs=250, A=5, B=5):
    """
    計算單個時間窗口的 AsI 值（僅使用 Fp1-Fp2-Fz 一組）
    
    參數:
        window_data: 情緒激發窗口資料 (channels, samples)
        baseline_data: 基線（平靜狀態）資料 (channels, samples)
        channel_indices: 電極索引字典 {'Fp1': idx, 'Fp2': idx, 'Fz': idx}
        fs: 採樣頻率 (預設=250 Hz)
        A, B: MDI 參數
        max_removed_ratio: 最大可移除窗口比例（預設 0.5），會傳遞給 process_subject_data
    
    返回:
        asi_fp: Fp1-Fp2 組的 AsI 值
    """
    
    # 提取電極訊號 - 情緒激發狀態
    Fp1_a = window_data[channel_indices['Fp1'], :]
    Fp2_a = window_data[channel_indices['Fp2'], :]
    Fz_a = window_data[channel_indices['Fz'], :]
    
    # 提取電極訊號 - 平靜狀態（中性影片）
    Fp1_c = baseline_data[channel_indices['Fp1'], :]
    Fp2_c = baseline_data[channel_indices['Fp2'], :]
    Fz_c = baseline_data[channel_indices['Fz'], :]
    
    # 計算 (Fp1-Fp2-Fz) 的 MDI
    # 平靜狀態
    S_XY_c_fp = calculate_mdi(Fp1_c, Fp2_c, Fz_c, A, B)
    S_YX_c_fp = calculate_mdi(Fp2_c, Fp1_c, Fz_c, A, B)
    S_c_fp = S_XY_c_fp + S_YX_c_fp
    
    # 情緒激發狀態
    S_XY_a_fp = calculate_mdi(Fp1_a, Fp2_a, Fz_a, A, B)
    S_YX_a_fp = calculate_mdi(Fp2_a, Fp1_a, Fz_a, A, B)
    S_a_fp = S_XY_a_fp + S_YX_a_fp
    
    # 計算 AsI (Fp1-Fp2)
    asi_fp = np.abs((S_c_fp - S_a_fp) * (np.sqrt(2) / 2))
    
    return asi_fp


def apply_asi_filter_rule(asi_fp):
    """
    應用 AsI 篩選規則判斷是否保留該窗口（僅使用 Fp1-Fp2 組）
    
    簡化規則（只使用一組電極）:
    - AsI[Fp1-Fp2] ∈ [0.5, 1] → 保留
    - AsI[Fp1-Fp2] ∈ [0, 0.5) → 移除
    
    參數:
        asi_fp: Fp1-Fp2 組的 AsI 值
    
    返回:
        bool: True=保留, False=移除
    """
    # 正規化到 [0, 1]
    asi_fp = np.clip(asi_fp, 0, 1)
    
    # 簡化規則：AsI >= 0.5 保留，< 0.5 移除
    if asi_fp >= 0.5:
        return True
    else:
        return False


def normalize_asi_values(asi_values):
    """
    將 AsI 值正規化到 [0, 1] 範圍
    
    參數:
        asi_values: AsI 值陣列
    
    返回:
        正規化後的 AsI 值
    """
    min_val = np.min(asi_values)
    max_val = np.max(asi_values)
    
    if max_val - min_val < 1e-10:
        return np.zeros_like(asi_values)
    
    return (asi_values - min_val) / (max_val - min_val)


def extract_neutral_videos_as_baseline(subject_data, neutral_indices, session_sec=30, fs=250):
    """
    提取中性影片（video 13-16）作為 baseline
    
    參數:
        subject_data: 受試者完整資料 (n_trials, n_channels, n_samples)
        neutral_indices: 中性影片在該受試者資料中的索引列表
        session_sec: 每個影片的長度（秒）
        fs: 採樣頻率
    
    返回:
        baseline_data: 中性影片的平均資料 (n_channels, session_sec*fs)
    """
    # 收集所有中性影片的資料
    neutral_videos = []
    for idx in neutral_indices:
        if idx < subject_data.shape[0]:
            neutral_videos.append(subject_data[idx])
    
    if len(neutral_videos) == 0:
        raise ValueError("找不到中性影片資料")
    
    # 計算平均作為 baseline（也可以選擇拼接）
    baseline_data = np.mean(neutral_videos, axis=0)  # (n_channels, n_samples)
    
    return baseline_data


def process_subject_data(subject_data, video_order, subject_idx, output_file, channel_indices, 
                        session_sec=30, window_sec=1, overlap_ratio=0.5, fs=250, A=5, B=5,
                        max_removed_ratio=0.5):
    """
    處理單個受試者的資料，應用 AsI 濾波（支援重疊視窗）
    
    參數:
        subject_data: 受試者完整資料 (n_trials, n_channels, n_samples)
        video_order: 該受試者的影片順序陣列
        subject_idx: 受試者編號
        output_file: 輸出檔案路徑
        channel_indices: 電極索引字典
        session_sec: 每個影片的長度（秒）
        window_sec: 窗口長度(秒)
        overlap_ratio: 重疊比例 (0.0 = 無重疊, 0.5 = 50%重疊, 0.75 = 75%重疊)
        fs: 採樣頻率
        A, B: MDI 參數
        max_removed_ratio: 最大可移除窗口比例（預設 0.5，對應原始版本的 15/30 規則）
    
    返回:
        filtered_info: 過濾資訊字典
    """
    # 找到中性影片（video 13-16）的索引
    neutral_indices = []
    for i, vid_num in enumerate(video_order):
        if 13 <= vid_num <= 16:
            neutral_indices.append(i)
    
    print(f"  中性影片位置: {neutral_indices}")
    
    # 提取中性影片作為 baseline
    baseline_data = extract_neutral_videos_as_baseline(
        subject_data, neutral_indices, session_sec, fs
    )
    
    # 資料格式: (n_trials, n_channels, n_samples)
    n_trials, n_channels, n_samples = subject_data.shape
    window_samples = window_sec * fs
    
    # 計算重疊視窗的步長
    step_samples = int(window_samples * (1 - overlap_ratio))
    
    # 計算每個 trial 的視窗數量（含重疊）
    n_windows_per_trial = (n_samples - window_samples) // step_samples + 1
    base_windows_per_trial = n_samples // window_samples
    
    print(f"  視窗設定: 長度={window_sec}s, 重疊={overlap_ratio*100}%, 步長={step_samples}樣本")
    print(f"  每個 trial 視窗數: {n_windows_per_trial}")
    
    # 儲存過濾結果
    filtered_data = []
    filtered_info = {
        'subject_idx': subject_idx,
        'total_trials': n_trials,
        'neutral_trials': len(neutral_indices),
        'emotion_trials': n_trials - len(neutral_indices),
        'total_windows': n_trials * n_windows_per_trial,
        'removed_windows': [],
        'removed_trials': [],
        'asi_values': [],
        'window_config': {
            'window_sec': window_sec,
            'overlap_ratio': overlap_ratio,
            'step_samples': step_samples,
            'n_windows_per_trial': n_windows_per_trial,
            'base_windows_per_trial': base_windows_per_trial,
            'max_removed_ratio': max_removed_ratio
        }
    }
    
    # 對每個 trial 進行處理（跳過中性影片本身）
    for trial_idx in range(n_trials):
        # 跳過中性影片
        if trial_idx in neutral_indices:
            continue
        
        trial_data = subject_data[trial_idx]  # (n_channels, n_samples)
        
        # 切分重疊視窗
        windows_to_keep = []
        window_asi_values = []
        
        for win_idx in range(n_windows_per_trial):
            start_idx = win_idx * step_samples
            end_idx = start_idx + window_samples
            
            # 確保不超出範圍
            if end_idx > n_samples:
                break
            
            window = trial_data[:, start_idx:end_idx]
            
            # 對 baseline 也取相同的窗口位置
            baseline_window = baseline_data[:, start_idx:end_idx]
            
            # 計算 AsI
            asi_fp = calculate_asi_for_window(
                window, baseline_window, channel_indices, fs, A, B
            )
            
            window_asi_values.append(asi_fp)
            
            # 應用篩選規則
            if apply_asi_filter_rule(asi_fp):
                windows_to_keep.append(win_idx)
            else:
                filtered_info['removed_windows'].append(
                    f"Trial {trial_idx} (Video {int(video_order[trial_idx])}), Window {win_idx} (AsI_Fp={asi_fp:.3f})"
                )
        
        filtered_info['asi_values'].append({
            'trial_idx': trial_idx,
            'video_num': int(video_order[trial_idx]),
            'asi_values': window_asi_values
        })
        
        # 檢查是否需要移除整個 trial
        removed_windows_count = len(window_asi_values) - len(windows_to_keep)
        allowed_removed = max(1, int(np.ceil(base_windows_per_trial * max_removed_ratio)))

        if removed_windows_count > allowed_removed:
            filtered_info['removed_trials'].append(
                f"Trial {trial_idx} (Video {int(video_order[trial_idx])}, 移除窗口數: {removed_windows_count}/{len(window_asi_values)})"
            )
            filtered_data.append({
                'trial_idx': trial_idx,
                'video_num': int(video_order[trial_idx]),
                'data': trial_data,
                'kept_windows': [],
                'removed': True,
                'step_samples': step_samples,
                'window_samples': window_samples
            })
        else:
            filtered_data.append({
                'trial_idx': trial_idx,
                'video_num': int(video_order[trial_idx]),
                'data': trial_data,
                'kept_windows': windows_to_keep,
                'removed': False,
                'step_samples': step_samples,
                'window_samples': window_samples
            })
    
    # 儲存過濾後的資料
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    save_dict = {
        'filtered_trials': filtered_data,
        'filtered_info': filtered_info,
        'metadata': {
            'n_trials': n_trials,
            'n_channels': n_channels,
            'n_samples': n_samples,
            'window_samples': window_samples,
            'step_samples': step_samples,
            'overlap_ratio': overlap_ratio,
            'n_windows_per_trial': n_windows_per_trial,
            'session_sec': session_sec,
            'window_sec': window_sec,
            'fs': fs
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(save_dict, f)
    
    if len(filtered_data) > 0:
        kept_trials = sum(1 for t in filtered_data if not t['removed'])
        print(f"  已儲存: {kept_trials}/{len(filtered_data)} trials 保留")
    else:
        print(f"  警告: 沒有保留任何情緒 trials")
    
    return filtered_info


def extract_filtered_data(filtered_file, output_format='windows'):
    """
    從過濾後的檔案中提取資料（支援重疊視窗）
    
    參數:
        filtered_file: 過濾後的 pickle 檔案路徑
        output_format: 輸出格式
            - 'windows': 返回窗口級別的資料 (推薦)
            - 'concatenate': 拼接所有保留的窗口
            - 'trials': 返回 trial 級別的完整資料
    
    返回:
        根據 output_format 返回不同格式的資料
    """
    if not os.path.exists(filtered_file):
        raise FileNotFoundError(f"找不到文件: {filtered_file}")
    
    if os.path.getsize(filtered_file) == 0:
        raise ValueError(f"文件為空: {filtered_file}")
    
    try:
        with open(filtered_file, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        raise Exception(f"讀取文件時發生錯誤: {str(e)}")
    
    filtered_trials = data_dict['filtered_trials']
    metadata = data_dict['metadata']
    
    if output_format == 'windows':
        # 返回所有保留的窗口作為獨立樣本（最推薦）
        result = []
        labels = []
        window_indices = []  # 記錄每個視窗的位置資訊
        
        for trial_dict in filtered_trials:
            if trial_dict['removed']:
                continue
            
            trial_data = trial_dict['data']
            kept_windows = trial_dict['kept_windows']
            video_num = trial_dict['video_num']
            trial_idx = trial_dict['trial_idx']
            
            # 從 trial_dict 或 metadata 取得參數
            window_samples = trial_dict.get('window_samples', metadata['window_samples'])
            step_samples = trial_dict.get('step_samples', metadata['step_samples'])
            
            for win_idx in kept_windows:
                start_idx = win_idx * step_samples
                end_idx = start_idx + window_samples
                
                if end_idx > trial_data.shape[1]:
                    continue
                
                window_data = trial_data[:, start_idx:end_idx]
                result.append(window_data)
                labels.append(video_num)
                window_indices.append({
                    'trial_idx': trial_idx,
                    'window_idx': win_idx,
                    'start_sample': start_idx,
                    'end_sample': end_idx
                })
        
        if len(result) > 0:
            result = np.array(result)
            print(f"提取了 {len(result)} 個視窗 (含重疊)")
        else:
            print("警告: 沒有保留任何窗口資料")
            result = np.array([])
        
        return result, labels
    
    elif output_format == 'concatenate':
        # 拼接每個 trial 中保留的窗口（不推薦用於重疊視窗）
        result = []
        for trial_dict in filtered_trials:
            if trial_dict['removed']:
                continue
            
            trial_data = trial_dict['data']
            kept_windows = trial_dict['kept_windows']
            window_samples = trial_dict.get('window_samples', metadata['window_samples'])
            step_samples = trial_dict.get('step_samples', metadata['step_samples'])
            
            if len(kept_windows) > 0:
                kept_data = []
                for win_idx in kept_windows:
                    start_idx = win_idx * step_samples
                    end_idx = start_idx + window_samples
                    if end_idx <= trial_data.shape[1]:
                        kept_data.append(trial_data[:, start_idx:end_idx])
                
                if len(kept_data) > 0:
                    trial_concatenated = np.concatenate(kept_data, axis=1)
                    result.append(trial_concatenated)
        
        return result
    
    elif output_format == 'trials':
        # 返回 trial 級別資料（保持原始結構）
        result = []
        
        for trial_dict in filtered_trials:
            if trial_dict['removed']:
                continue
            
            trial_data = trial_dict['data'].copy()
            kept_windows = trial_dict['kept_windows']
            window_samples = trial_dict.get('window_samples', metadata['window_samples'])
            step_samples = trial_dict.get('step_samples', metadata['step_samples'])
            n_windows = metadata['n_windows_per_trial']
            
            # 將未保留的窗口設為 NaN
            for win_idx in range(n_windows):
                if win_idx not in kept_windows:
                    start_idx = win_idx * step_samples
                    end_idx = start_idx + window_samples
                    if end_idx <= trial_data.shape[1]:
                        trial_data[:, start_idx:end_idx] = np.nan
            
            result.append(trial_data)
        
        if len(result) > 0:
            result = np.array(result)
        else:
            result = np.array([])
        
        return result
    
    else:
        raise ValueError(f"Unknown output_format: {output_format}")


def get_channel_indices():
    """
    返回所需電極的索引
    根據 FACED 資料集的 32 通道配置
    
    FACED 資料集 32 通道順序:
    1	Fp1	9	FC2	17	CP1	25	P8
    2	Fp2	10	FC5	18	CP2	26	PO3
    3	Fz	11	FC6	19	CP5	27	PO4
    4	F3	12	Cz	20	CP6	28	Oz
    5	F4	13	C3	21	Pz	29	O1
    6	F7	14	C4	22	P3	30	O2
    7	F8	15	T7	23	P4	31	 HEOR
    8	FC1	16	T8	24	P7	32	 HEOL
    """
    # FACED 資料集前面的電極通常包含 Fp1, Fp2, Fz
    # 根據標準 10-20 系統，這些是前額葉電極
    channel_map = {
        'Fp1': 0,   # FP1
        'Fp2': 1,   # FP2  
        'Fz': 2,    # FZ
    }
    return channel_map


def generate_filter_report(all_filtered_info, output_dir, dataset, session_sec, window_sec, overlap_ratio, fs, A, B):
    """
    生成詳細的過濾報告
    
    參數:
        all_filtered_info: 所有受試者的過濾資訊列表
        output_dir: 輸出資料夾路徑
        其他參數: 過濾配置參數
        overlap_ratio: 重疊比例
    """
    report_file = Path(output_dir) / 'AsI_Filter_Report.txt'
    json_file = Path(output_dir) / 'AsI_Filter_Report.json'
    csv_file = Path(output_dir) / 'AsI_Filter_Summary.csv'
    
    # 計算統計資訊
    total_subjects = len(all_filtered_info)
    total_trials = sum(info['total_trials'] for info in all_filtered_info)
    total_neutral_trials = sum(info['neutral_trials'] for info in all_filtered_info)
    total_emotion_trials = sum(info['emotion_trials'] for info in all_filtered_info)
    total_removed_trials = sum(len(info['removed_trials']) for info in all_filtered_info)
    total_removed_windows = sum(len(info['removed_windows']) for info in all_filtered_info)
    total_windows = sum(info['total_windows'] for info in all_filtered_info)
    
    kept_trials = total_emotion_trials - total_removed_trials
    kept_windows = total_windows - total_removed_windows
    
    # 生成文字報告
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("AsI (Asymmetry Index) 濾波報告\n")
        f.write("=" * 100 + "\n\n")
        
        # 基本資訊
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"資料集: {dataset}\n")
        f.write(f"輸入資料夾: {output_dir}\n\n")
        
        # 配置參數
        f.write("配置參數:\n")
        f.write("-" * 100 + "\n")
        f.write(f"  影片長度: {session_sec} 秒\n")
        f.write(f"  窗口長度: {window_sec} 秒\n")
        f.write(f"  重疊比例: {overlap_ratio}")
        f.write(f"  採樣頻率: {fs} Hz\n")
        f.write(f"  MDI 參數: A={A}, B={B}\n")
        f.write(f"  使用電極: Fp1, Fp2, Fz\n")
        f.write(f"  Baseline: 中性影片 (Video 13-16)\n\n")
        
        # 過濾規則
        f.write("過濾規則:\n")
        f.write("-" * 100 + "\n")
        f.write("  窗口級別過濾:\n")
        f.write("    - AsI[Fp1-Fp2] >= 0.5 → 保留窗口\n")
        f.write("    - AsI[Fp1-Fp2] < 0.5  → 移除窗口\n\n")
        f.write("  Trial 級別過濾:\n")
        f.write("    - 如果單個 trial 中移除窗口數 > 15 → 移除整個 trial\n")
        f.write("    - 理由: 該受試者在該影片中未能有效激發情緒\n\n")
        
        # 總體統計
        f.write("總體統計:\n")
        f.write("=" * 100 + "\n")
        f.write(f"  處理受試者數: {total_subjects}\n")
        f.write(f"  總 trials 數: {total_trials}\n")
        f.write(f"    - 中性 trials: {total_neutral_trials} (用作 baseline)\n")
        f.write(f"    - 情緒 trials: {total_emotion_trials}\n")
        f.write(f"  總窗口數: {total_windows}\n\n")
        
        f.write(f"  保留情況:\n")
        f.write(f"    - 保留 trials: {kept_trials}/{total_emotion_trials} ({kept_trials/total_emotion_trials*100:.2f}%)\n")
        f.write(f"    - 保留窗口: {kept_windows}/{total_windows} ({kept_windows/total_windows*100:.2f}%)\n\n")
        
        f.write(f"  移除情況:\n")
        f.write(f"    - 移除 trials: {total_removed_trials}/{total_emotion_trials} ({total_removed_trials/total_emotion_trials*100:.2f}%)\n")
        f.write(f"    - 移除窗口: {total_removed_windows}/{total_windows} ({total_removed_windows/total_windows*100:.2f}%)\n\n")
        
        # 每個受試者的詳細資訊
        f.write("\n" + "=" * 100 + "\n")
        f.write("各受試者詳細資訊\n")
        f.write("=" * 100 + "\n\n")
        
        for info in all_filtered_info:
            subject_idx = info['subject_idx']
            f.write(f"\nSubject {subject_idx:03d}:\n")
            f.write("-" * 100 + "\n")
            
            emotion_trials = info['emotion_trials']
            removed_trials_count = len(info['removed_trials'])
            kept_trials_count = emotion_trials - removed_trials_count
            removed_windows_count = len(info['removed_windows'])
            total_windows_sub = info['total_windows']
            kept_windows_count = total_windows_sub - removed_windows_count
            
            f.write(f"  總 trials: {info['total_trials']} (中性: {info['neutral_trials']}, 情緒: {emotion_trials})\n")
            f.write(f"  保留 trials: {kept_trials_count}/{emotion_trials} ({kept_trials_count/emotion_trials*100:.2f}%)\n")
            f.write(f"  保留窗口: {kept_windows_count}/{total_windows_sub} ({kept_windows_count/total_windows_sub*100:.2f}%)\n\n")
            
            # 移除的 trials
            if info['removed_trials']:
                f.write(f"  移除的 Trials ({len(info['removed_trials'])} 個):\n")
                for trial_info in info['removed_trials']:
                    f.write(f"    • {trial_info}\n")
                f.write("\n")
            
            # AsI 值統計
            if info['asi_values']:
                asi_flat = []
                for trial_asi in info['asi_values']:
                    asi_flat.extend(trial_asi['asi_values'])
                
                if len(asi_flat) > 0:
                    asi_array = np.array(asi_flat)
                    f.write(f"  AsI 值統計:\n")
                    f.write(f"    平均值: {np.mean(asi_array):.4f}\n")
                    f.write(f"    標準差: {np.std(asi_array):.4f}\n")
                    f.write(f"    最小值: {np.min(asi_array):.4f}\n")
                    f.write(f"    最大值: {np.max(asi_array):.4f}\n")
                    f.write(f"    中位數: {np.median(asi_array):.4f}\n")
                    f.write(f"    >= 0.5 的窗口比例: {np.sum(asi_array >= 0.5)/len(asi_array)*100:.2f}%\n")
                f.write("\n")
            
            # 移除的窗口（顯示前20個）
            if info['removed_windows']:
                f.write(f"  移除的窗口 (顯示前20個，共 {len(info['removed_windows'])} 個):\n")
                for window_info in info['removed_windows'][:20]:
                    f.write(f"    • {window_info}\n")
                if len(info['removed_windows']) > 20:
                    f.write(f"    ... 還有 {len(info['removed_windows']) - 20} 個窗口\n")
                f.write("\n")
    
    # 生成 JSON 報告（方便程式讀取）
    json_data = {
        'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'dataset': dataset,
            'session_sec': session_sec,
            'window_sec': window_sec,
            'fs': fs,
            'A': A,
            'B': B,
            'electrodes': ['Fp1', 'Fp2', 'Fz'],
            'baseline': 'Neutral videos (13-16)'
        },
        'filtering_rules': {
            'window_level': 'AsI >= 0.5 → keep, AsI < 0.5 → remove',
            'trial_level': 'Remove trial if removed_windows > 15'
        },
        'summary': {
            'total_subjects': total_subjects,
            'total_trials': total_trials,
            'neutral_trials': total_neutral_trials,
            'emotion_trials': total_emotion_trials,
            'kept_trials': kept_trials,
            'removed_trials': total_removed_trials,
            'total_windows': total_windows,
            'kept_windows': kept_windows,
            'removed_windows': total_removed_windows,
            'trial_retention_rate': f"{kept_trials/total_emotion_trials*100:.2f}%",
            'window_retention_rate': f"{kept_windows/total_windows*100:.2f}%"
        },
        'subjects': []
    }
    
    for info in all_filtered_info:
        subject_data = {
            'subject_idx': info['subject_idx'],
            'total_trials': info['total_trials'],
            'neutral_trials': info['neutral_trials'],
            'emotion_trials': info['emotion_trials'],
            'removed_trials': info['removed_trials'],
            'removed_windows_count': len(info['removed_windows']),
            'total_windows': info['total_windows']
        }
        
        # 添加 AsI 統計
        if info['asi_values']:
            asi_flat = []
            for trial_asi in info['asi_values']:
                asi_flat.extend(trial_asi['asi_values'])
            
            if len(asi_flat) > 0:
                asi_array = np.array(asi_flat)
                subject_data['asi_statistics'] = {
                    'mean': float(np.mean(asi_array)),
                    'std': float(np.std(asi_array)),
                    'min': float(np.min(asi_array)),
                    'max': float(np.max(asi_array)),
                    'median': float(np.median(asi_array)),
                    'above_threshold_ratio': float(np.sum(asi_array >= 0.5) / len(asi_array))
                }
        
        json_data['subjects'].append(subject_data)
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # 生成 CSV 摘要（方便 Excel 打開）
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Subject,Total_Trials,Neutral_Trials,Emotion_Trials,Kept_Trials,Removed_Trials,")
        f.write("Total_Windows,Kept_Windows,Removed_Windows,Trial_Retention_Rate,Window_Retention_Rate,")
        f.write("AsI_Mean,AsI_Std,AsI_Min,AsI_Max,AsI_Median\n")
        
        for info in all_filtered_info:
            subject_idx = info['subject_idx']
            emotion_trials = info['emotion_trials']
            removed_trials_count = len(info['removed_trials'])
            kept_trials_count = emotion_trials - removed_trials_count
            removed_windows_count = len(info['removed_windows'])
            total_windows_sub = info['total_windows']
            kept_windows_count = total_windows_sub - removed_windows_count
            
            trial_rate = kept_trials_count / emotion_trials * 100 if emotion_trials > 0 else 0
            window_rate = kept_windows_count / total_windows_sub * 100 if total_windows_sub > 0 else 0
            
            # AsI 統計
            asi_stats = ['', '', '', '', '']
            if info['asi_values']:
                asi_flat = []
                for trial_asi in info['asi_values']:
                    asi_flat.extend(trial_asi['asi_values'])
                
                if len(asi_flat) > 0:
                    asi_array = np.array(asi_flat)
                    asi_stats = [
                        f"{np.mean(asi_array):.4f}",
                        f"{np.std(asi_array):.4f}",
                        f"{np.min(asi_array):.4f}",
                        f"{np.max(asi_array):.4f}",
                        f"{np.median(asi_array):.4f}"
                    ]
            
            f.write(f"{subject_idx},{info['total_trials']},{info['neutral_trials']},{emotion_trials},")
            f.write(f"{kept_trials_count},{removed_trials_count},{total_windows_sub},")
            f.write(f"{kept_windows_count},{removed_windows_count},{trial_rate:.2f},{window_rate:.2f},")
            f.write(f"{','.join(asi_stats)}\n")
    
    print(f"\n已生成過濾報告:")
    print(f"  - 文字報告: {report_file}")
    print(f"  - JSON 報告: {json_file}")
    print(f"  - CSV 摘要: {csv_file}")


def process_all_subjects(input_dir, output_dir, dataset='both', session_sec=30, 
                        window_sec=1, overlap_ratio=0.5, fs=250, A=5, B=5,
                        max_removed_ratio=0.5):
    """
    處理所有受試者的資料（支援重疊視窗）
    
    參數:
        input_dir: 輸入資料夾路徑
        output_dir: 輸出資料夾路徑
        dataset: 資料集類型
        session_sec: 每個影片的長度（秒）
        window_sec: 窗口長度(秒)
        overlap_ratio: 重疊比例 (0.0, 0.5, 0.75)
        fs: 採樣頻率
        A, B: MDI 參數
    """
    channel_indices = get_channel_indices()
    os.makedirs(output_dir, exist_ok=True)
    
    print("載入 video order...")
    vid_orders = video_order_load(dataset, 28)
    print(f"Video order 載入完成，形狀: {vid_orders.shape}")
    
    all_filtered_info = []
    subject_files = sorted(Path(input_dir).glob('sub*.pkl'))
    
    print(f"\n找到 {len(subject_files)} 個受試者資料檔案")
    print(f"視窗設定: 長度={window_sec}s, 重疊率={overlap_ratio*100}%")
    print("=" * 80)
    
    for sub_idx, subject_file in enumerate(subject_files):
        subject_name = subject_file.stem
        print(f"\n處理: {subject_name} (索引: {sub_idx})")
        
        output_file = Path(output_dir) / subject_file.name
        
        try:
            with open(subject_file, 'rb') as f:
                subject_data = pickle.load(f)
            
            print(f"  資料形狀: {subject_data.shape}")
            
            if sub_idx < vid_orders.shape[0]:
                video_order = vid_orders[sub_idx, :]
            else:
                print(f"  警告: video order 中找不到索引 {sub_idx}，跳過")
                continue
            
            # 使用重疊視窗處理資料
            filtered_info = process_subject_data(
                subject_data,
                video_order,
                sub_idx,
                str(output_file),
                channel_indices,
                session_sec,
                window_sec,
                overlap_ratio,  # 新增參數
                fs,
                A,
                B,
                max_removed_ratio
            )
            all_filtered_info.append(filtered_info)
            
        except Exception as e:
            print(f"  錯誤: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成報告
    print("\n正在生成詳細報告...")
    generate_filter_report(all_filtered_info, output_dir, dataset, session_sec, 
                          window_sec, fs, A, B, overlap_ratio)
    
    return all_filtered_info


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='AsI 濾波器 - 支援重疊視窗'
    )
    parser.add_argument('--input-dir', type=str, default='../Processed_Data')
    parser.add_argument('--output-dir', type=str, default='../Processed_Data_after_AsI')
    parser.add_argument('--dataset', type=str, default='both', 
                       choices=['both', 'first_batch', 'second_batch'])
    parser.add_argument('--session-sec', type=int, default=30)
    parser.add_argument('--window-sec', type=int, default=1,
                       help='視窗長度（秒）')
    parser.add_argument('--overlap-ratio', type=float, default=0.5,
                       choices=[0.0, 0.25, 0.5, 0.75],
                       help='視窗重疊比例: 0.0(無重疊), 0.5(50%%), 0.75(75%%)')
    parser.add_argument('--fs', type=int, default=250)
    parser.add_argument('--A', type=int, default=5)
    parser.add_argument('--B', type=int, default=5)
    parser.add_argument('--max-removed-ratio', type=float, default=0.5,
                       help='觸發移除整個 trial 的最大移除窗口比例（對應原始 15/30）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("AsI 濾波器 (支援重疊視窗)")
    print("=" * 80)
    print(f"視窗長度: {args.window_sec} 秒")
    print(f"重疊比例: {args.overlap_ratio * 100}%")
    print(f"步長: {args.window_sec * (1 - args.overlap_ratio)} 秒")
    print("=" * 80)
    
    process_all_subjects(
        args.input_dir,
        args.output_dir,
        args.dataset,
        args.session_sec,
        args.window_sec,
        args.overlap_ratio,  # 新增
        args.fs,
        args.A,
        args.B,
        args.max_removed_ratio  # 新增
    )


if __name__ == '__main__':
    main()


# 使用範例
"""
# 1. 運行 AsI 濾波
python asi_filter.py --input_dir ../Processed_data --output_dir ../Processed_data_after_AsI

# 2. 載入過濾後的資料
from asi_filter import extract_filtered_data

# 方法 A: 拼接每個 trial 保留的窗口（推薦用於後續特徵提取）
trials_data = extract_filtered_data('../Processed_data_after_AsI/sub000.pkl', output_format='concatenate')
# 返回: list of arrays，每個 array 形狀為 (n_channels, variable_length)

# 方法 B: 提取所有窗口作為獨立樣本（推薦用於窗口級別分類）
windows_data, labels = extract_filtered_data('../Processed_data_after_AsI/sub000.pkl', output_format='windows')
# 返回: windows_data 形狀為 (n_windows, n_channels, window_samples)
#       labels 是對應的影片編號列表

# 方法 C: 保持原始 trial 結構，移除的窗口用 NaN 填充
trials_data = extract_filtered_data('../Processed_data_after_AsI/sub000.pkl', output_format='trials')
# 返回: array 形狀為 (n_kept_trials, n_channels, n_samples)

# 3. 直接載入完整資料結構
import pickle
with open('../Processed_data_after_AsI/sub000.pkl', 'rb') as f:
    data_dict = pickle.load(f)

# 資料結構:
# data_dict = {
#     'filtered_trials': [  # 列表，每個元素是一個字典
#         {
#             'trial_idx': int,        # trial 索引
#             'video_num': int,        # 影片編號
#             'data': ndarray,         # 原始資料 (n_channels, n_samples)
#             'kept_windows': list,    # 保留的窗口索引列表
#             'removed': bool          # 是否整個 trial 被移除
#         },
#         ...
#     ],
#     'filtered_info': {  # 過濾資訊
#         'subject_idx': int,
#         'total_trials': int,
#         'neutral_trials': int,
#         'emotion_trials': int,
#         'removed_windows': list,
#         'removed_trials': list,
#         'asi_values': list
#     },
#     'metadata': {  # 元資料
#         'n_trials': int,
#         'n_channels': int,
#         'n_samples': int,
#         'window_samples': int,
#         'n_windows_per_trial': int,
#         'session_sec': int,
#         'window_sec': int,
#         'fs': int
#     }
# }
"""
