# Facebook广告数据异常值专项清洗
# Author: Data Analyst
# Date: 2025-01-30
# Purpose: 清洗基础预处理后发现的异常值
# Input: facebook_ads_clean.csv (来自data_process.py的输出)
# Output: facebook_ads_final_clean.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 配置参数 - 调整为更宽松的清洗标准
# =============================================================================
CLEANING_CONFIG = {
    'ROAS_MAX': 100,       # ROAS最大值（放宽以保留更多数据）
    'ROAS_MIN': 0.01,      # ROAS最小值（允许更低值）
    'CPA_MAX': 1000,       # CPA最大值($)（大幅放宽）
    'CPA_MIN': 0.1,        # CPA最小值($)（降低门槛）
    'CPC_MAX': 50,         # CPC最大值($)（放宽限制）
    'CPC_MIN': 0.01,       # CPC最小值($)（允许更低值）
    'CPM_MAX': 200,        # CPM最大值($)（大幅放宽）
    'CPM_MIN': 0.01,       # CPM最小值($)（降低门槛）
    'MIN_SPEND': 0.01,     # 最小广告支出($)（大幅降低）
    'MIN_CONVERSIONS': 0   # 最小转化数量（允许0转化记录）
}

print("=== Facebook广告数据异常值清洗开始 ===")
print(f"清洗配置: {CLEANING_CONFIG}")

# =============================================================================
# 第1步：加载预处理后的数据
# =============================================================================
print("\n=== 第1步：加载数据 ===")

try:
    # 加载基础预处理后的数据
    df = pd.read_csv('facebook_ads_clean.csv')
    print(f"✅ 成功加载数据: {df.shape}")
except FileNotFoundError:
    print("❌ 未找到facebook_ads_clean.csv，请先运行data_process.py")
    exit(1)

# 显示当前数据概况
print(f"加载数据形状: {df.shape}")
print("主要指标概况:")
key_metrics = ['ROAS_Approved', 'CPA_Approved', 'CPC', 'CPM']
for metric in key_metrics:
    if metric in df.columns:
        print(f"  {metric}: {df[metric].min():.2f} - {df[metric].max():.2f} (均值: {df[metric].mean():.2f})")

# =============================================================================
# 第2步：异常值识别和分析
# =============================================================================
print("\n=== 第2步：异常值识别 ===")

def identify_outliers(df, metric, min_val, max_val):
    """识别指定指标的异常值"""
    if metric not in df.columns:
        return 0, 0
    
    outliers_high = (df[metric] > max_val).sum()
    outliers_low = (df[metric] < min_val).sum()
    outliers_null = df[metric].isnull().sum()
    
    print(f"{metric}:")
    print(f"  - 高异常值 (>{max_val}): {outliers_high} 条")
    print(f"  - 低异常值 (<{min_val}): {outliers_low} 条") 
    print(f"  - 缺失值: {outliers_null} 条")
    
    return outliers_high + outliers_low + outliers_null

# 分析各指标异常值情况
print("异常值统计:")
total_outliers = 0
total_outliers += identify_outliers(df, 'ROAS_Approved', CLEANING_CONFIG['ROAS_MIN'], CLEANING_CONFIG['ROAS_MAX'])
total_outliers += identify_outliers(df, 'CPA_Approved', CLEANING_CONFIG['CPA_MIN'], CLEANING_CONFIG['CPA_MAX'])
total_outliers += identify_outliers(df, 'CPC', CLEANING_CONFIG['CPC_MIN'], CLEANING_CONFIG['CPC_MAX'])
total_outliers += identify_outliers(df, 'CPM', CLEANING_CONFIG['CPM_MIN'], CLEANING_CONFIG['CPM_MAX'])

print(f"\n检测到的潜在异常值总数: {total_outliers}")

# =============================================================================
# 第3步：可视化异常值分布
# =============================================================================
print("\n=== 第3步：异常值可视化分析 ===")

def plot_outlier_analysis(df, metric, min_val, max_val, title):
    """绘制异常值分析图"""
    if metric not in df.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 原始分布
    axes[0].hist(df[metric].dropna(), bins=50, alpha=0.7, color='skyblue')
    axes[0].axvline(min_val, color='red', linestyle='--', label=f'Min: {min_val}')
    axes[0].axvline(max_val, color='red', linestyle='--', label=f'Max: {max_val}')
    axes[0].set_title(f'{title} - 原始分布')
    axes[0].set_xlabel(metric)
    axes[0].legend()
    
    # 清洗后分布预览
    cleaned_data = df[(df[metric] >= min_val) & (df[metric] <= max_val) & (df[metric].notna())][metric]
    axes[1].hist(cleaned_data, bins=30, alpha=0.7, color='lightgreen')
    axes[1].set_title(f'{title} - 清洗后分布预览')
    axes[1].set_xlabel(metric)
    
    plt.tight_layout()
    plt.show()
    
    print(f"{metric} 清洗效果预览:")
    print(f"  原始数据量: {len(df[metric].dropna())}")
    print(f"  清洗后数据量: {len(cleaned_data)}")
    print(f"  将移除: {len(df[metric].dropna()) - len(cleaned_data)} 条记录")

# 可视化主要指标的异常值分布
plot_outlier_analysis(df, 'ROAS_Approved', CLEANING_CONFIG['ROAS_MIN'], CLEANING_CONFIG['ROAS_MAX'], 'ROAS异常值分析')
plot_outlier_analysis(df, 'CPA_Approved', CLEANING_CONFIG['CPA_MIN'], CLEANING_CONFIG['CPA_MAX'], 'CPA异常值分析')

# =============================================================================
# 第4步：应用清洗规则
# =============================================================================
print("\n=== 第4步：应用清洗规则 ===")

def apply_cleaning_rules(df, config):
    """应用异常值清洗规则"""
    original_count = len(df)
    cleaned_df = df.copy()
    
    # 清洗日志
    cleaning_log = {
        'original_count': original_count,
        'steps': [],
        'timestamp': datetime.now()
    }
    
    # 规则1: ROAS范围清洗
    if 'ROAS_Approved' in cleaned_df.columns:
        before = len(cleaned_df)
        cleaned_df = cleaned_df[
            (cleaned_df['ROAS_Approved'] >= config['ROAS_MIN']) & 
            (cleaned_df['ROAS_Approved'] <= config['ROAS_MAX']) &
            (cleaned_df['ROAS_Approved'].notna())
        ]
        removed = before - len(cleaned_df)
        cleaning_log['steps'].append(f"ROAS清洗: 移除 {removed} 条记录")
        print(f"✅ ROAS清洗完成: 移除 {removed} 条异常记录")
    
    # 规则2: CPA范围清洗
    if 'CPA_Approved' in cleaned_df.columns:
        before = len(cleaned_df)
        cleaned_df = cleaned_df[
            (cleaned_df['CPA_Approved'] >= config['CPA_MIN']) & 
            (cleaned_df['CPA_Approved'] <= config['CPA_MAX']) &
            (cleaned_df['CPA_Approved'].notna())
        ]
        removed = before - len(cleaned_df)
        cleaning_log['steps'].append(f"CPA清洗: 移除 {removed} 条记录")
        print(f"✅ CPA清洗完成: 移除 {removed} 条异常记录")
    
    # 规则3: CPC范围清洗
    if 'CPC' in cleaned_df.columns:
        before = len(cleaned_df)
        cleaned_df = cleaned_df[
            (cleaned_df['CPC'] >= config['CPC_MIN']) & 
            (cleaned_df['CPC'] <= config['CPC_MAX']) &
            (cleaned_df['CPC'].notna())
        ]
        removed = before - len(cleaned_df)
        cleaning_log['steps'].append(f"CPC清洗: 移除 {removed} 条记录")
        print(f"✅ CPC清洗完成: 移除 {removed} 条异常记录")
    
    # 规则4: CPM范围清洗
    if 'CPM' in cleaned_df.columns:
        before = len(cleaned_df)
        cleaned_df = cleaned_df[
            (cleaned_df['CPM'] >= config['CPM_MIN']) & 
            (cleaned_df['CPM'] <= config['CPM_MAX']) &
            (cleaned_df['CPM'].notna())
        ]
        removed = before - len(cleaned_df)
        cleaning_log['steps'].append(f"CPM清洗: 移除 {removed} 条记录")
        print(f"✅ CPM清洗完成: 移除 {removed} 条异常记录")
    
    # 规则5: 最小支出和转化要求
    before = len(cleaned_df)
    cleaned_df = cleaned_df[
        (cleaned_df['Spent'] >= config['MIN_SPEND']) &
        (cleaned_df['Approved_Conversion'] >= config['MIN_CONVERSIONS'])
    ]
    removed = before - len(cleaned_df)
    cleaning_log['steps'].append(f"最小阈值清洗: 移除 {removed} 条记录")
    print(f"✅ 最小阈值清洗完成: 移除 {removed} 条记录")
    
    # 汇总清洗结果
    final_count = len(cleaned_df)
    total_removed = original_count - final_count
    removal_rate = (total_removed / original_count) * 100
    
    # 数据量检查和警告
    if final_count < 100:
        print(f"⚠️  警告: 清洗后数据量过少 ({final_count} 条)")
        print("   建议进一步放宽清洗条件或使用基础清洗数据")
    elif final_count < 200:
        print(f"⚠️  注意: 数据量较少 ({final_count} 条)，分析结果可能有限")
    else:
        print(f"✅ 数据量充足 ({final_count} 条)，适合进行全面分析")
    
    cleaning_log.update({
        'final_count': final_count,
        'total_removed': total_removed,
        'removal_rate': removal_rate
    })
    
    print(f"\n📊 清洗汇总:")
    print(f"  原始数据: {original_count} 条")
    print(f"  清洗后数据: {final_count} 条")
    print(f"  移除数据: {total_removed} 条")
    print(f"  清洗率: {removal_rate:.1f}%")
    
    return cleaned_df, cleaning_log

# 应用清洗规则
df_final_clean, log = apply_cleaning_rules(df, CLEANING_CONFIG)

# 如果数据量太少，尝试更宽松的清洗标准
if len(df_final_clean) < 50:
    print("\n🔄 数据量过少，尝试更宽松的清洗标准...")
    
    BACKUP_CONFIG = {
        'ROAS_MAX': 500,       # 进一步放宽
        'ROAS_MIN': 0.001,     # 接近不限制
        'CPA_MAX': 5000,       # 大幅放宽
        'CPA_MIN': 0.01,       # 极低门槛
        'CPC_MAX': 100,        # 放宽限制
        'CPC_MIN': 0.001,      # 接近不限制
        'CPM_MAX': 1000,       # 大幅放宽
        'CPM_MIN': 0.001,      # 接近不限制
        'MIN_SPEND': 0.001,    # 极低门槛
        'MIN_CONVERSIONS': 0   # 允许0转化
    }
    
    print("使用备选清洗配置:")
    for key, value in BACKUP_CONFIG.items():
        print(f"  {key}: {value}")
    
    df_final_clean, log = apply_cleaning_rules(df, BACKUP_CONFIG)
    output_file = 'facebook_ads_final_clean_relaxed.csv'
    print(f"💡 将使用更宽松的清洗标准，输出文件: {output_file}")
else:
    output_file = 'facebook_ads_final_clean.csv'

# =============================================================================
# 第5步：验证清洗效果
# =============================================================================
print("\n=== 第5步：清洗效果验证 ===")

def validate_cleaning_results(df_before, df_after):
    """验证清洗效果"""
    print("清洗前后关键指标对比:")
    
    metrics_to_check = ['ROAS_Approved', 'CPA_Approved', 'CPC', 'CPM']
    
    for metric in metrics_to_check:
        if metric in df_before.columns and metric in df_after.columns:
            before_stats = df_before[metric].describe()
            after_stats = df_after[metric].describe()
            
            print(f"\n{metric}:")
            print(f"  清洗前: 均值={before_stats['mean']:.2f}, 范围=[{before_stats['min']:.2f}, {before_stats['max']:.2f}]")
            print(f"  清洗后: 均值={after_stats['mean']:.2f}, 范围=[{after_stats['min']:.2f}, {after_stats['max']:.2f}]")
            
            # 判断清洗是否有效
            if after_stats['max'] <= CLEANING_CONFIG.get(f"{metric.split('_')[0]}_MAX", float('inf')):
                print(f"  ✅ {metric} 异常值已清除")
            else:
                print(f"  ⚠️ {metric} 仍有异常值需要注意")

validate_cleaning_results(df, df_final_clean)

# =============================================================================
# 第6步：保存清洗结果和日志
# =============================================================================
print("\n=== 第6步：保存清洗结果 ===")

# 保存最终清洗数据
df_final_clean.to_csv(output_file, index=False)
print(f"✅ 最终清洗数据已保存: {output_file}")

# 保存清洗日志
log_file = 'outlier_cleaning_log.txt'
with open(log_file, 'w', encoding='utf-8') as f:
    f.write("Facebook广告数据异常值清洗日志\n")
    f.write("="*50 + "\n")
    f.write(f"清洗时间: {log['timestamp']}\n")
    f.write(f"原始数据量: {log['original_count']} 条\n")
    f.write(f"最终数据量: {log['final_count']} 条\n")
    f.write(f"清洗率: {log['removal_rate']:.1f}%\n\n")
    
    f.write("清洗配置参数:\n")
    for key, value in CLEANING_CONFIG.items():
        f.write(f"  {key}: {value}\n")
    
    f.write("\n清洗步骤详情:\n")
    for step in log['steps']:
        f.write(f"  - {step}\n")
    
    f.write(f"\n最终输出文件: {output_file}\n")

print(f"✅ 清洗日志已保存: {log_file}")

# =============================================================================
# 第7步：为Tableau使用提供建议
# =============================================================================
print("\n=== 第7步：Tableau使用建议 ===")

print("📊 Tableau数据源建议:")
print(f"  推荐使用: {output_file}")
if len(df_final_clean) >= 200:
    print("  ✅ 数据量充足，适合全面分析")
elif len(df_final_clean) >= 100:
    print("  ⚠️  数据量适中，可进行基础分析")
else:
    print("  ❌ 数据量较少，建议结合基础清洗数据使用")
    print("  💡 或考虑直接使用 facebook_ads_clean.csv 并在Tableau中添加筛选器")

print(f"\n最终数据统计:")
print(f"  数据行数: {len(df_final_clean)}")
print(f"  年龄段数: {df_final_clean['age'].nunique()}")
print(f"  性别数: {df_final_clean['gender'].nunique()}")
print(f"  Campaign数: {df_final_clean['fb_campaign_id'].nunique()}")

if len(df_final_clean) >= 100:
    print("\n🎯 建议的Tableau筛选器设置:")
    print("  可添加交互筛选器进一步精细化:")
    roas_max = min(50, df_final_clean['ROAS_Approved'].max())
    cpa_max = min(200, df_final_clean['CPA_Approved'].max())
    print(f"    - ROAS范围: 0.1-{roas_max:.0f}")
    print(f"    - CPA范围: 1-{cpa_max:.0f}")
else:
    print("\n💡 由于数据量有限，建议:")
    print("  1. 使用此数据进行技术验证")
    print("  2. 主要分析使用 facebook_ads_clean.csv")
    print("  3. 在Tableau中设置手动筛选器")

print("\n💡 分析重点:")
print("  专注于相对比较和趋势分析")
print("  数值虽经清洗但仍建议与业务专家验证")

print("\n🎉 异常值清洗完成！数据已准备好用于高质量的Tableau分析！")