# Facebook广告数据预处理 - 付费媒体核心指标分析
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 第1步：数据导入和基本探索
# =============================================================================
print("=== Facebook广告数据预处理开始 ===")
print("=== 第1步：数据导入和基本探索 ===")

# 导入数据
df = pd.read_csv('KAG_conversion_data.csv')

# 基本信息
print("数据集基本信息：")
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print("\n数据类型:")
print(df.dtypes)

# 前5行数据预览
print("\n前5行数据:")
print(df.head())

# 数据概览
print("\n数据概览:")
print(df.info())

# =============================================================================
# 第2步：付费媒体数据质量检查
# =============================================================================
print("\n=== 第2步：付费媒体数据质量检查 ===")

# 2.1 缺失值检查
print("缺失值统计:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing_data,
    '缺失百分比': missing_percent
})
print(missing_df[missing_df['缺失数量'] > 0])

# 2.2 重复值检查
print(f"\n重复行数: {df.duplicated().sum()}")

# 2.3 数值型字段描述性统计
print("\n核心指标描述性统计:")
numeric_cols = ['Impressions', 'Clicks', 'Spent', 'Total_Conversion', 'Approved_Conversion']
print(df[numeric_cols].describe())

# 2.4 分类字段分析
print("\n分类字段分析:")
categorical_cols = ['age', 'gender']

for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  - 唯一值: {df[col].unique()}")
        print(f"  - 分布:\n{df[col].value_counts()}")

# 特殊字段：interest分析
print(f"\ninterest字段:")
print(f"  - 唯一值数量: {df['interest'].nunique()}")
print(f"  - 范围: {df['interest'].min()} - {df['interest'].max()}")
print(f"  - 前10个interest:\n{df['interest'].value_counts().head(10)}")

# =============================================================================
# 第3步：付费媒体业务逻辑验证
# =============================================================================
print("\n=== 第3步：付费媒体业务逻辑验证 ===")

# 3.1 检查广告漏斗逻辑：Impressions >= Clicks >= Conversions
print("广告漏斗逻辑检查:")
logic_issues = []

# Impressions >= Clicks
impressions_clicks_issue = (df['Impressions'] < df['Clicks']).sum()
if impressions_clicks_issue > 0:
    logic_issues.append(f"Impressions < Clicks: {impressions_clicks_issue} 条记录")

# Clicks >= Total_Conversion
clicks_conversion_issue = (df['Clicks'] < df['Total_Conversion']).sum()
if clicks_conversion_issue > 0:
    logic_issues.append(f"Clicks < Total_Conversion: {clicks_conversion_issue} 条记录")

# Total_Conversion >= Approved_Conversion
total_approved_issue = (df['Total_Conversion'] < df['Approved_Conversion']).sum()
if total_approved_issue > 0:
    logic_issues.append(f"Total_Conversion < Approved_Conversion: {total_approved_issue} 条记录")

if logic_issues:
    print("⚠️  发现逻辑问题:")
    for issue in logic_issues:
        print(f"  - {issue}")
else:
    print("✅ 广告漏斗逻辑正常")

# 3.2 检查零值和负值
print("\n零值和负值检查:")
for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    negative_count = (df[col] < 0).sum()
    print(f"{col}: {zero_count} 个零值, {negative_count} 个负值")

# 3.3 检查campaign关联
print("\n活动ID关联检查:")
print(f"不同xyz_campaign_id数量: {df['xyz_campaign_id'].nunique()}")
print(f"不同fb_campaign_id数量: {df['fb_campaign_id'].nunique()}")
print(f"不同ad_id数量: {df['ad_id'].nunique()}")

# =============================================================================
# 第4步：计算付费媒体核心指标
# =============================================================================
print("\n=== 第4步：计算付费媒体核心指标 ===")

# 4.1 创建核心KPI指标
print("计算付费媒体核心KPI...")

# CTR (Click Through Rate) - 点击率
df['CTR'] = df.apply(lambda row: row['Clicks'] / row['Impressions'] if row['Impressions'] > 0 else 0, axis=1)

# CVR (Conversion Rate) - 转化率（基于总转化）
df['CVR_Total'] = df.apply(lambda row: row['Total_Conversion'] / row['Clicks'] if row['Clicks'] > 0 else 0, axis=1)

# CVR (Conversion Rate) - 转化率（基于已确认转化）
df['CVR_Approved'] = df.apply(lambda row: row['Approved_Conversion'] / row['Clicks'] if row['Clicks'] > 0 else 0, axis=1)

# CPC (Cost Per Click) - 每次点击成本
df['CPC'] = df.apply(lambda row: row['Spent'] / row['Clicks'] if row['Clicks'] > 0 else np.nan, axis=1)

# CPM (Cost Per Mille) - 每千次展示成本
df['CPM'] = df.apply(lambda row: (row['Spent'] / row['Impressions']) * 1000 if row['Impressions'] > 0 else np.nan, axis=1)

# CPA (Cost Per Acquisition) - 每次转化成本（总转化）
df['CPA_Total'] = df.apply(lambda row: row['Spent'] / row['Total_Conversion'] if row['Total_Conversion'] > 0 else np.nan, axis=1)

# CPA (Cost Per Acquisition) - 每次转化成本（已确认转化）
df['CPA_Approved'] = df.apply(lambda row: row['Spent'] / row['Approved_Conversion'] if row['Approved_Conversion'] > 0 else np.nan, axis=1)

# 4.2 计算频次相关指标
# Frequency - 平均频次 (假设每个impression对应唯一用户，这里是简化计算)
df['Avg_Frequency'] = df.apply(lambda row: row['Impressions'] / row['Clicks'] if row['Clicks'] > 0 else np.nan, axis=1)

# 4.3 投资回报相关指标（需要假设AOV）
# 假设平均订单价值
AOV = 50  # 可根据业务调整
df['Revenue_Total'] = df['Total_Conversion'] * AOV
df['Revenue_Approved'] = df['Approved_Conversion'] * AOV

# ROAS (Return on Ad Spend)
df['ROAS_Total'] = df.apply(lambda row: row['Revenue_Total'] / row['Spent'] if row['Spent'] > 0 else np.nan, axis=1)
df['ROAS_Approved'] = df.apply(lambda row: row['Revenue_Approved'] / row['Spent'] if row['Spent'] > 0 else np.nan, axis=1)

# =============================================================================
# 第5步：核心指标统计分析
# =============================================================================
print("\n=== 第5步：核心指标统计分析 ===")

# 关键指标汇总
key_metrics = ['CTR', 'CVR_Total', 'CVR_Approved', 'CPC', 'CPM', 'CPA_Total', 'CPA_Approved', 'ROAS_Total', 'ROAS_Approved']

print("付费媒体核心指标统计:")
print("=" * 60)
for metric in key_metrics:
    if metric in df.columns:
        valid_data = df[metric].dropna()
        if len(valid_data) > 0:
            print(f"{metric}:")
            print(f"  有效记录: {len(valid_data)}/{len(df)} ({len(valid_data)/len(df)*100:.1f}%)")
            print(f"  均值: {valid_data.mean():.4f}")
            print(f"  中位数: {valid_data.median():.4f}")
            print(f"  标准差: {valid_data.std():.4f}")
            print(f"  范围: {valid_data.min():.4f} - {valid_data.max():.4f}")
            print()

# =============================================================================
# 第6步：行业基准对比
# =============================================================================
print("=== 第6步：行业基准对比 ===")

# Facebook广告行业基准（参考值）
benchmarks = {
    'CTR': {'excellent': 0.03, 'good': 0.02, 'average': 0.015, 'poor': 0.01},
    'CPC': {'excellent': 0.5, 'good': 1.0, 'average': 1.5, 'poor': 2.0},
    'CPM': {'excellent': 5, 'good': 10, 'average': 15, 'poor': 20},
    'CVR_Total': {'excellent': 0.05, 'good': 0.03, 'average': 0.02, 'poor': 0.01}
}

print("与行业基准对比:")
current_data = {
    'CTR': df['CTR'].mean(),
    'CPC': df['CPC'].mean(),
    'CPM': df['CPM'].mean(),
    'CVR_Total': df['CVR_Total'].mean()
}

for metric, current_value in current_data.items():
    if not np.isnan(current_value) and metric in benchmarks:
        benchmark = benchmarks[metric]
        print(f"\n{metric}: {current_value:.4f}")
        if current_value >= benchmark['excellent']:
            status = "🟢 优秀"
        elif current_value >= benchmark['good']:
            status = "🟡 良好"
        elif current_value >= benchmark['average']:
            status = "🟠 平均"
        else:
            status = "🔴 需要优化"
        print(f"  状态: {status}")

# =============================================================================
# 第7步：数据清洗和异常值处理
# =============================================================================
print("\n=== 第7步：数据清洗和异常值处理 ===")

# 记录原始数据大小
original_size = len(df)

# 创建清洗条件
clean_conditions = [
    df['Spent'] > 0,  # 必须有广告支出
    df['Impressions'] > 0,  # 必须有展示
    df['Impressions'] >= df['Clicks'],  # 展示数 >= 点击数
    df['Clicks'] >= df['Total_Conversion'],  # 点击数 >= 转化数
    df['Total_Conversion'] >= df['Approved_Conversion'],  # 总转化 >= 已确认转化
]

# 应用清洗条件
df_clean = df.copy()
for i, condition in enumerate(clean_conditions):
    before_size = len(df_clean)
    df_clean = df_clean[condition]
    after_size = len(df_clean)
    removed = before_size - after_size
    if removed > 0:
        print(f"清洗条件 {i+1}: 移除 {removed} 条记录")

print(f"\n数据清洗总结:")
print(f"原始数据: {original_size} 条")
print(f"清洗后数据: {len(df_clean)} 条") 
print(f"清洗率: {((original_size - len(df_clean)) / original_size * 100):.1f}%")

# =============================================================================
# 第8步：适用性评估
# =============================================================================
print("\n=== 第8步：付费媒体分析适用性评估 ===")

# 评估标准
evaluation_criteria = {
    "数据完整性": len(df_clean) > 500,  # 足够的样本量
    "指标合理性": df_clean['CTR'].mean() > 0 and df_clean['CTR'].mean() < 1,
    "业务逻辑": len(logic_issues) == 0,
    "平台识别": 'fb_campaign_id' in df.columns,
    "用户分层": df['age'].nunique() > 1 and df['gender'].nunique() > 1,
    "转化追踪": df_clean['Total_Conversion'].sum() > 0
}

print("数据集适用性评估:")
print("=" * 50)
passed_criteria = 0
for criteria, result in evaluation_criteria.items():
    status = "✅ 通过" if result else "❌ 不通过"
    print(f"{criteria}: {status}")
    if result:
        passed_criteria += 1

print(f"\n总体评分: {passed_criteria}/{len(evaluation_criteria)} ({passed_criteria/len(evaluation_criteria)*100:.0f}%)")

# 给出建议
if passed_criteria >= 5:
    recommendation = "🎯 强烈推荐：该数据集非常适合做付费媒体分析"
elif passed_criteria >= 4:
    recommendation = "✅ 推荐：该数据集适合做付费媒体分析"
elif passed_criteria >= 3:
    recommendation = "⚠️  谨慎：数据集可用但有限制"
else:
    recommendation = "❌ 不推荐：数据集不适合付费媒体分析"

print(f"\n推荐建议: {recommendation}")

# =============================================================================
# 第9步：建议的分析方向
# =============================================================================
print("\n=== 第9步：建议的分析方向 ===")

print("基于该数据集，建议进行以下分析:")
print("1. 📊 核心指标分析")
print("   - CTR, CPC, CPM, CPA趋势分析")
print("   - 不同Campaign效果对比")
print("   - 转化漏斗分析")

print("\n2. 👥 用户分群分析")
print("   - 年龄段表现对比")
print("   - 性别差异分析") 
print("   - 兴趣分群效果分析")

print("\n3. 💰 投资回报分析")
print("   - ROAS分析和优化建议")
print("   - 预算分配优化")
print("   - 高价值受众识别")

print("\n4. 🎯 优化建议")
print("   - 低效campaign识别")
print("   - 出价策略优化")
print("   - 受众定向优化")

# =============================================================================
# 第10步：保存处理后数据
# =============================================================================
print("\n=== 第10步：保存处理后数据 ===")

# 保存完整数据（包含所有计算指标）
df.to_csv('facebook_ads_with_metrics.csv', index=False)
print("完整数据已保存: facebook_ads_with_metrics.csv")

# 保存清洗后数据
df_clean.to_csv('facebook_ads_clean.csv', index=False)
print("清洗后数据已保存: facebook_ads_clean.csv")

print("\n🎉 预处理完成！数据已准备好进行付费媒体分析！")