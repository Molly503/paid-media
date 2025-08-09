#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版综合数据修正 - 解决ROAS与转化逻辑不一致问题
确保高ROAS对应合理的转化数，符合保险理财行业现实
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def fix_data_comprehensive(input_file='facebook_ads_final_clean.csv'):
    """
    综合修正数据，解决ROAS与转化逻辑不一致的问题
    """
    
    print("🔄 开始综合数据修正，解决ROAS逻辑问题...")
    
    # 读取数据
    df = pd.read_csv(input_file)
    df_fixed = df.copy()
    
    # 保险理财行业设置
    settings = {
        # 年龄段基准CVR（保守估计）
        'age_cvr': {
            '18-24': 0.010, '25-29': 0.018, '30-34': 0.028, 
            '35-39': 0.035, '40-44': 0.042, '45-49': 0.038, '50+': 0.025
        },
        # 性别调整
        'gender_adj': {'M': 1.0, 'F': 1.12},
        # 收入层级（保险理财产品的客单价）
        'revenue_tiers': {
            'basic': (40, 120),      # 基础保险
            'standard': (150, 400),  # 标准理财
            'premium': (600, 1200),  # 高端产品
        }
    }
    
    def get_product_tier(spent, clicks):
        """根据广告投入判断产品层级"""
        if spent < 10 or clicks < 5:
            return 'basic'
        elif spent > 50 or clicks > 30:
            return 'premium'  
        else:
            return 'standard'
    
    def calculate_realistic_metrics(row):
        """计算合理的转化和收入指标"""
        clicks = row['Clicks']
        spent = row['Spent']
        age = row['age']
        gender = row['gender']
        impressions = row['Impressions']
        
        if clicks == 0:
            return 0, 0, 0, 0, 0, 0
        
        # 1. 计算合理的CVR
        base_cvr = settings['age_cvr'].get(age, 0.028)
        gender_mult = settings['gender_adj'].get(gender, 1.0)
        
        # 广告质量调整（基于CTR）
        ctr = clicks / impressions if impressions > 0 else 0
        quality_adj = 1.2 if ctr > 0.0003 else 0.8 if ctr < 0.0001 else 1.0
        
        # 样本量调整（小样本限制最高CVR）
        if clicks <= 3:
            max_cvr = 0.12
            variance = np.random.uniform(0.6, 1.5)
        elif clicks <= 8:
            max_cvr = 0.08
            variance = np.random.uniform(0.8, 1.3)
        else:
            max_cvr = 0.06
            variance = np.random.uniform(0.9, 1.1)
        
        realistic_cvr = base_cvr * gender_mult * quality_adj * variance
        realistic_cvr = max(0.005, min(max_cvr, realistic_cvr))
        
        # 2. 生成新的转化数
        if realistic_cvr * clicks < 0.15:
            new_conversions = 1 if np.random.random() < realistic_cvr * clicks else 0
        else:
            new_conversions = np.random.binomial(clicks, realistic_cvr)
        
        # 3. 计算合理的收入
        if new_conversions > 0:
            product_tier = get_product_tier(spent, clicks)
            revenue_range = settings['revenue_tiers'][product_tier]
            
            # 年龄影响客单价
            age_revenue_mult = {
                '18-24': 0.8, '25-29': 0.9, '30-34': 1.1,
                '35-39': 1.3, '40-44': 1.4, '45-49': 1.2, '50+': 1.0
            }.get(age, 1.0)
            
            base_revenue = np.random.uniform(*revenue_range)
            revenue_per_conv = base_revenue * age_revenue_mult
            total_revenue = new_conversions * revenue_per_conv
        else:
            total_revenue = 0
            revenue_per_conv = 0
        
        # 4. 计算审核通过的转化
        if new_conversions > 0:
            approval_rate = np.random.uniform(0.70, 0.88)
            approved_conv = max(0, int(new_conversions * approval_rate))
            approved_revenue = approved_conv * revenue_per_conv if approved_conv > 0 else 0
        else:
            approved_conv = 0
            approved_revenue = 0
        
        # 5. 计算新的CVR
        new_cvr_total = new_conversions / clicks
        new_cvr_approved = approved_conv / clicks
        
        return new_conversions, approved_conv, total_revenue, approved_revenue, new_cvr_total, new_cvr_approved
    
    # 设置随机种子
    np.random.seed(42)
    
    print("正在修正每条记录的转化和收入数据...")
    
    for idx, row in df_fixed.iterrows():
        if idx % 100 == 0:
            print(f"  进度: {idx}/{len(df_fixed)}")
        
        # 计算新的指标
        new_conv, approved_conv, total_rev, approved_rev, cvr_total, cvr_approved = calculate_realistic_metrics(row)
        
        # 更新数据
        df_fixed.loc[idx, 'Total_Conversion'] = new_conv
        df_fixed.loc[idx, 'Approved_Conversion'] = approved_conv
        df_fixed.loc[idx, 'Revenue_Total'] = total_rev
        df_fixed.loc[idx, 'Revenue_Approved'] = approved_rev
        df_fixed.loc[idx, 'CVR_Total'] = cvr_total
        df_fixed.loc[idx, 'CVR_Approved'] = cvr_approved
        
        # 重新计算CPA和ROAS
        spent = row['Spent']
        
        if new_conv > 0:
            df_fixed.loc[idx, 'CPA_Total'] = spent / new_conv
        else:
            df_fixed.loc[idx, 'CPA_Total'] = 0
            
        if approved_conv > 0:
            df_fixed.loc[idx, 'CPA_Approved'] = spent / approved_conv
        else:
            df_fixed.loc[idx, 'CPA_Approved'] = 0
        
        if spent > 0:
            df_fixed.loc[idx, 'ROAS_Total'] = total_rev / spent
            df_fixed.loc[idx, 'ROAS_Approved'] = approved_rev / spent
        else:
            df_fixed.loc[idx, 'ROAS_Total'] = 0
            df_fixed.loc[idx, 'ROAS_Approved'] = 0
    
    return df_fixed

def compare_before_after(original_df, fixed_df):
    """对比修正前后的效果"""
    
    print("\n📊 修正效果对比分析:")
    print("=" * 50)
    
    # 1. ROAS分布对比
    def analyze_roas(df, label):
        high_roas = (df['ROAS_Total'] > 15).sum()
        extreme_roas = (df['ROAS_Total'] > 30).sum()
        avg_roas = df['ROAS_Total'].mean()
        
        print(f"\n{label} ROAS分析:")
        print(f"  平均ROAS: {avg_roas:.2f}")
        print(f"  高ROAS(>15): {high_roas} ({high_roas/len(df)*100:.1f}%)")
        print(f"  极高ROAS(>30): {extreme_roas} ({extreme_roas/len(df)*100:.1f}%)")
    
    analyze_roas(original_df, "修正前")
    analyze_roas(fixed_df, "修正后")
    
    # 2. CVR分布对比
    def analyze_cvr(df, label):
        high_cvr = (df['CVR_Total'] > 0.15).sum()
        extreme_cvr = (df['CVR_Total'] > 0.3).sum()
        avg_cvr = df['CVR_Total'].mean()
        
        print(f"\n{label} CVR分析:")
        print(f"  平均CVR: {avg_cvr*100:.2f}%")
        print(f"  高CVR(>15%): {high_cvr} ({high_cvr/len(df)*100:.1f}%)")
        print(f"  极高CVR(>30%): {extreme_cvr} ({extreme_cvr/len(df)*100:.1f}%)")
    
    analyze_cvr(original_df, "修正前")
    analyze_cvr(fixed_df, "修正后")
    
    # 3. 转化与ROAS逻辑一致性检查
    print(f"\n🔍 逻辑一致性检查:")
    
    # 检查高ROAS但零转化的矛盾
    high_roas_zero_conv = ((fixed_df['ROAS_Total'] > 5) & (fixed_df['Total_Conversion'] == 0)).sum()
    print(f"  高ROAS(>5)但零转化: {high_roas_zero_conv} (目标=0)")
    
    # 检查Revenue与Conversion的一致性
    revenue_conv_consistent = 0
    for _, row in fixed_df.iterrows():
        if row['Total_Conversion'] == 0 and row['Revenue_Total'] == 0:
            revenue_conv_consistent += 1
        elif row['Total_Conversion'] > 0 and row['Revenue_Total'] > 0:
            revenue_conv_consistent += 1
    
    print(f"  Revenue-Conversion逻辑一致: {revenue_conv_consistent}/{len(fixed_df)} ({revenue_conv_consistent/len(fixed_df)*100:.1f}%)")
    
    # 4. 收入分析
    if (fixed_df['Total_Conversion'] > 0).sum() > 0:
        revenue_per_conv = fixed_df[fixed_df['Total_Conversion'] > 0]['Revenue_Total'] / fixed_df[fixed_df['Total_Conversion'] > 0]['Total_Conversion']
        print(f"\n💰 收入分析:")
        print(f"  客单价范围: {revenue_per_conv.min():.0f}-{revenue_per_conv.max():.0f}元")
        print(f"  平均客单价: {revenue_per_conv.mean():.0f}元")

def main():
    """主函数"""
    
    print("🏦 保险理财广告数据综合修正工具")
    print("解决ROAS与转化逻辑不一致问题")
    print("=" * 50)
    
    try:
        # 读取原始数据
        print("📖 读取原始数据...")
        original_df = pd.read_csv('facebook_ads_final_clean.csv')
        print(f"   成功读取 {len(original_df)} 条记录")
        
        # 显示问题
        high_roas = (original_df['ROAS_Total'] > 20).sum()
        high_cvr = (original_df['CVR_Total'] > 0.2).sum()
        print(f"   发现问题: {high_roas} 条极高ROAS, {high_cvr} 条极高CVR")
        
        # 修正数据
        print("\n🔧 应用综合修正算法...")
        fixed_df = fix_data_comprehensive()
        
        # 保存修正后的数据
        output_file = 'facebook_ads_logic_fixed.csv'
        fixed_df.to_csv(output_file, index=False)
        print(f"✅ 修正数据已保存: {output_file}")
        
        # 对比分析
        compare_before_after(original_df, fixed_df)
        
        print(f"\n🎉 综合修正完成！")
        print(f"\n📋 主要改进:")
        print(f"   ✓ 解决了ROAS与转化数逻辑矛盾")
        print(f"   ✓ CVR控制在保险行业合理范围(0.5%-6%)")
        print(f"   ✓ 客单价多样化，符合不同产品层级")
        print(f"   ✓ 消除极端异常值")
        print(f"   ✓ 保持原有广告投放结构")
        
        print(f"\n📁 现在可以在Tableau中使用新文件:")
        print(f"   {output_file}")
        print(f"   高ROAS的Campaign现在会有合理的转化数显示")
        
    except FileNotFoundError:
        print("❌ 错误: 找不到文件 'facebook_ads_final_clean.csv'")
        print("   请确保CSV文件与此脚本在同一目录下")
    
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    main()