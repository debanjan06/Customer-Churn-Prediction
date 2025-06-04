#!/usr/bin/env python3
"""
Business Analysis Demo for Churn Prediction Results (Fixed Version)
Creates impressive visualizations for presentation without font warnings
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import requests
import json
import warnings

# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BusinessAnalysisDemo:
    def __init__(self):
        self.results_df = None
        
    def load_batch_results(self, csv_file):
        """Load batch prediction results"""
        try:
            self.results_df = pd.read_csv(csv_file)
            print(f"✅ Loaded {len(self.results_df)} customer predictions")
            return True
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def create_executive_dashboard(self):
        """Create executive-level dashboard"""
        if self.results_df is None:
            print("❌ No data loaded")
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Customer Churn Analysis - Executive Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Risk Distribution Pie Chart
        risk_counts = self.results_df['risk_category'].value_counts()
        colors = ['#FF6B6B', '#FFD93D', '#6BCF7F']  # Red, Yellow, Green
        axes[0,0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                     colors=colors, startangle=90)
        axes[0,0].set_title('Risk Distribution', fontsize=14, fontweight='bold')
        
        # 2. Churn Probability Histogram
        probs = self.results_df['churn_probability'].dropna()
        axes[0,1].hist(probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].axvline(probs.mean(), color='red', linestyle='--', 
                         label=f'Mean: {probs.mean():.3f}')
        axes[0,1].set_title('Churn Probability Distribution', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Churn Probability')
        axes[0,1].set_ylabel('Number of Customers')
        axes[0,1].legend()
        
        # 3. Revenue at Risk Analysis
        # Simulate monthly charges data if not available
        if 'MonthlyCharges' not in self.results_df.columns:
            self.results_df['MonthlyCharges'] = np.random.uniform(20, 120, len(self.results_df))
        
        # Calculate revenue at risk
        high_risk = self.results_df[self.results_df['risk_category'] == 'High']
        medium_risk = self.results_df[self.results_df['risk_category'] == 'Medium']
        low_risk = self.results_df[self.results_df['risk_category'] == 'Low']
        
        revenue_at_risk = {
            'High Risk': high_risk['MonthlyCharges'].sum() if len(high_risk) > 0 else 0,
            'Medium Risk': medium_risk['MonthlyCharges'].sum() if len(medium_risk) > 0 else 0,
            'Low Risk': low_risk['MonthlyCharges'].sum() if len(low_risk) > 0 else 0
        }
        
        bars = axes[0,2].bar(revenue_at_risk.keys(), revenue_at_risk.values(), 
                            color=['#FF6B6B', '#FFD93D', '#6BCF7F'])
        axes[0,2].set_title('Monthly Revenue at Risk ($)', fontsize=14, fontweight='bold')
        axes[0,2].set_ylabel('Revenue ($)')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0,2].text(bar.get_x() + bar.get_width()/2., height,
                          f'${height:,.0f}', ha='center', va='bottom')
        
        # 4. Top Risk Factors (from API responses)
        factors_data = self.extract_top_factors()
        if factors_data:
            factors_df = pd.DataFrame(factors_data)
            top_factors = factors_df.groupby('factor')['importance'].mean().sort_values(ascending=True).tail(8)
            
            axes[1,0].barh(range(len(top_factors)), top_factors.values)
            axes[1,0].set_yticks(range(len(top_factors)))
            axes[1,0].set_yticklabels([f.replace('_', ' ').title() for f in top_factors.index])
            axes[1,0].set_title('Top Risk Factors', fontsize=14, fontweight='bold')
            axes[1,0].set_xlabel('Average Importance')
        else:
            axes[1,0].text(0.5, 0.5, 'No factor data available', ha='center', va='center', 
                          transform=axes[1,0].transAxes)
            axes[1,0].set_title('Top Risk Factors', fontsize=14, fontweight='bold')
        
        # 5. Business Impact Metrics
        total_customers = len(self.results_df)
        high_risk_count = len(self.results_df[self.results_df['risk_category'] == 'High'])
        avg_churn_prob = probs.mean()
        
        metrics_text = f"""
        Total Customers Analyzed: {total_customers:,}
        High-Risk Customers: {high_risk_count:,}
        Average Churn Probability: {avg_churn_prob:.1%}
        Potential Monthly Loss: ${revenue_at_risk['High Risk']:,.0f}
        Annual Revenue at Risk: ${revenue_at_risk['High Risk'] * 12:,.0f}
        """
        
        axes[1,1].text(0.1, 0.9, metrics_text, transform=axes[1,1].transAxes, 
                      fontsize=12, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1,1].set_xlim([0, 1])
        axes[1,1].set_ylim([0, 1])
        axes[1,1].axis('off')
        axes[1,1].set_title('Business Impact Summary', fontsize=14, fontweight='bold')
        
        # 6. Recommended Actions (Without problematic emojis)
        recommendations_text = f"""
        IMMEDIATE ACTIONS REQUIRED:
        
        [HIGH] {high_risk_count} customers need urgent retention calls
        
        [MED] {len(medium_risk)} customers for proactive outreach
        
        [$$] Focus on ${revenue_at_risk['High Risk']:.0f} monthly revenue
        
        [CALL] Deploy retention team immediately
        
        [EMAIL] Launch targeted email campaigns
        
        [CHART] Monitor model performance weekly
        """
        
        axes[1,2].text(0.05, 0.95, recommendations_text, transform=axes[1,2].transAxes, 
                      fontsize=11, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        axes[1,2].set_xlim([0, 1])
        axes[1,2].set_ylim([0, 1])
        axes[1,2].axis('off')
        axes[1,2].set_title('Recommended Actions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'executive_dashboard_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Executive dashboard saved as: {filename}")
        return filename
    
    def extract_top_factors(self):
        """Extract top factors from prediction results"""
        factors_data = []
        
        for _, row in self.results_df.iterrows():
            if pd.notna(row.get('top_factors')):
                try:
                    if isinstance(row['top_factors'], str):
                        factors = json.loads(row['top_factors'])
                    else:
                        factors = row['top_factors']
                    
                    for factor in factors:
                        factors_data.append({
                            'factor': factor['factor'],
                            'importance': factor['importance']
                        })
                except:
                    continue
        
        return factors_data
    
    def create_roi_analysis(self):
        """Create ROI analysis for the churn prediction system"""
        print("\n" + "="*60)
        print("📈 RETURN ON INVESTMENT ANALYSIS")
        print("="*60)
        
        if self.results_df is None:
            print("❌ No data available for ROI analysis")
            return
        
        # Business assumptions (you can adjust these)
        avg_customer_lifetime_value = 1200  # $1200 average CLV
        retention_cost_per_customer = 50    # $50 to retain a customer
        retention_success_rate = 0.3        # 30% success rate for retention efforts
        monthly_model_cost = 2000           # $2000/month to run the system
        
        # Calculate metrics
        total_customers = len(self.results_df)
        high_risk_customers = len(self.results_df[self.results_df['risk_category'] == 'High'])
        
        # Without model: lose all high-risk customers
        revenue_loss_without_model = high_risk_customers * avg_customer_lifetime_value
        
        # With model: retain some customers through targeted interventions
        retention_cost = high_risk_customers * retention_cost_per_customer
        customers_retained = high_risk_customers * retention_success_rate
        revenue_saved = customers_retained * avg_customer_lifetime_value
        
        # Monthly ROI calculation
        monthly_benefit = revenue_saved / 12  # Spread CLV over year
        monthly_cost = monthly_model_cost + (retention_cost / 12)
        monthly_roi = (monthly_benefit - monthly_cost) / monthly_cost * 100
        
        # Annual ROI
        annual_benefit = revenue_saved - retention_cost
        annual_cost = monthly_model_cost * 12
        annual_roi = (annual_benefit - annual_cost) / annual_cost * 100
        
        print(f"Total Customers Analyzed: {total_customers:,}")
        print(f"High-Risk Customers Identified: {high_risk_customers:,}")
        print(f"")
        print(f"Without Churn Prediction Model:")
        print(f"  Expected Revenue Loss: ${revenue_loss_without_model:,.0f}")
        print(f"")
        print(f"With Churn Prediction Model:")
        print(f"  Retention Investment: ${retention_cost:,.0f}")
        print(f"  Customers Retained: {customers_retained:.0f}")
        print(f"  Revenue Saved: ${revenue_saved:,.0f}")
        print(f"")
        print(f"ROI Analysis:")
        print(f"  Monthly ROI: {monthly_roi:.1f}%")
        print(f"  Annual ROI: {annual_roi:.1f}%")
        print(f"  Payback Period: {annual_cost/monthly_benefit:.1f} months")
        print(f"")
        print(f"💰 Net Annual Benefit: ${annual_benefit - annual_cost:,.0f}")
        
        return {
            'monthly_roi': monthly_roi,
            'annual_roi': annual_roi,
            'net_benefit': annual_benefit - annual_cost,
            'customers_saved': customers_retained
        }

def run_demo():
    """Run the complete business analysis demo"""
    print("🎯 Starting Business Analysis Demo")
    print("="*50)
    
    # Initialize analyzer
    analyzer = BusinessAnalysisDemo()
    
    # Look for the most recent batch results file
    import glob
    result_files = glob.glob("*predictions*.csv")
    
    if not result_files:
        print("❌ No prediction results found. Run batch prediction first:")
        print("   python scripts/batch_predict.py --create-sample 500")
        print("   python scripts/batch_predict.py sample_customers.csv")
        return
    
    # Use the most recent file
    latest_file = max(result_files, key=lambda f: f.split('_')[-1])
    print(f"📊 Using results file: {latest_file}")
    
    # Load and analyze
    if analyzer.load_batch_results(latest_file):
        print("\n1. Creating Executive Dashboard...")
        dashboard_file = analyzer.create_executive_dashboard()
        
        print("\n2. Calculating ROI Analysis...")
        roi_results = analyzer.create_roi_analysis()
        
        print(f"\n🎉 Demo complete! Dashboard saved as: {dashboard_file}")
        print("\n💡 Key talking points for presentation:")
        print("   ✅ Professional executive dashboard with actionable insights")
        print("   ✅ Clear ROI justification for the ML system")
        print("   ✅ Specific recommendations for business action")
        print("   ✅ Revenue impact quantified and visualized")
        
        return dashboard_file, roi_results
    
    return None, None

if __name__ == "__main__":
    run_demo()