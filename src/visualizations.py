"""
Visualization module for creating interactive charts and dashboard components
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def create_quality_score_gauge(score: float) -> go.Figure:
    """Create a gauge chart for overall data quality score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Data Quality Score"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def create_issues_by_type_chart(engine) -> go.Figure:
    """Create a bar chart showing different types of data quality issues"""
    issues_data = {
        'Issue Type': ['Expired Licenses', 'Missing NPI', 'Phone Issues', 'Duplicates', 'State Mismatches'],
        'Count': [
            engine.count_expired(),
            len(engine.list_missing_npi()),
            len(engine.list_phone_issues()),
            len(engine.list_duplicates()),
            len(engine.aug[engine.aug.get('license_state_mismatch', False)]) if hasattr(engine, 'aug') and engine.aug is not None else 0
        ]
    }
    
    df = pd.DataFrame(issues_data)
    fig = px.bar(df, x='Issue Type', y='Count', 
                 title="Data Quality Issues by Type",
                 color='Count',
                 color_continuous_scale='Reds')
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_specialties_issues_chart(engine) -> go.Figure:
    """Create a horizontal bar chart showing specialties with most issues"""
    try:
        specialty_data = engine.specialties_with_most_issues()
        if specialty_data.empty:
            # Create empty chart
            fig = go.Figure()
            fig.add_annotation(text="No specialty data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Issues by Medical Specialty", height=400)
            return fig
        
        # Get top 10 specialties with most issues
        top_specialties = specialty_data.head(10)
        
        fig = px.bar(top_specialties, x='issues', y='specialty', 
                     orientation='h',
                     title="Top Specialties with Data Quality Issues",
                     color='issues',
                     color_continuous_scale='Oranges')
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        return fig
    except Exception as e:
        # Fallback empty chart
        fig = go.Figure()
        fig.add_annotation(text=f"Unable to load specialty data: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Issues by Medical Specialty", height=400)
        return fig


def create_state_summary_chart(engine) -> go.Figure:
    """Create a choropleth map showing issues by state"""
    try:
        state_data = engine.state_issue_summary()
        if state_data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No state data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Issues by State", height=400)
            return fig
        
        # Calculate total issues per state
        issue_cols = ['license_state_mismatch', 'license_expired', 'npi_missing', 'phone_issue', 'duplicate_suspect', 'multi_state_single_license']
        available_cols = [col for col in issue_cols if col in state_data.columns]
        state_data['total_issues'] = state_data[available_cols].sum(axis=1)
        
        # Create a simple bar chart instead of map for simplicity
        fig = px.bar(state_data, x='address_state', y='total_issues',
                     title="Data Quality Issues by State",
                     color='total_issues',
                     color_continuous_scale='Blues')
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        # Fallback empty chart
        fig = go.Figure()
        fig.add_annotation(text=f"Unable to load state data: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Issues by State", height=400)
        return fig


def create_license_expiration_timeline(engine) -> go.Figure:
    """Create a timeline chart showing license expirations"""
    try:
        if not hasattr(engine, 'aug') or engine.aug is None:
            fig = go.Figure()
            fig.add_annotation(text="No data loaded", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="License Expiration Timeline", height=400)
            return fig
        
        # Get license expiration data
        df = engine.aug.copy()
        if 'license_expiration_date' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No license expiration data available", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="License Expiration Timeline", height=400)
            return fig
        
        # Filter valid dates and group by month
        valid_dates = df[df['license_expiration_date'].notna()].copy()
        valid_dates['exp_month'] = pd.to_datetime(valid_dates['license_expiration_date']).dt.to_period('M')
        monthly_exp = valid_dates.groupby('exp_month').size().reset_index(name='count')
        monthly_exp['exp_month'] = monthly_exp['exp_month'].astype(str)
        
        fig = px.line(monthly_exp, x='exp_month', y='count',
                      title="License Expirations by Month",
                      markers=True)
        fig.update_layout(height=400, xaxis_title="Month", yaxis_title="Number of Expirations")
        return fig
    except Exception as e:
        # Fallback empty chart
        fig = go.Figure()
        fig.add_annotation(text=f"Unable to create timeline: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="License Expiration Timeline", height=400)
        return fig


def create_quality_metrics_summary(engine) -> dict:
    """Create summary metrics for the dashboard"""
    try:
        if not hasattr(engine, 'aug') or engine.aug is None:
            return {
                'total_providers': 0,
                'expired_licenses': 0,
                'missing_npi': 0,
                'phone_issues': 0,
                'quality_score': 0.0
            }
        
        return {
            'total_providers': len(engine.aug),
            'expired_licenses': engine.count_expired(),
            'missing_npi': len(engine.list_missing_npi()),
            'phone_issues': len(engine.list_phone_issues()),
            'quality_score': round(engine.get_quality_score(), 2)
        }
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return {
            'total_providers': 0,
            'expired_licenses': 0,
            'missing_npi': 0,
            'phone_issues': 0,
            'quality_score': 0.0
        }


def create_duplicate_analysis_chart(engine) -> go.Figure:
    """Create a chart showing duplicate record analysis"""
    try:
        duplicates = engine.list_duplicates()
        if duplicates.empty:
            fig = go.Figure()
            fig.add_annotation(text="No duplicate records found", 
                             xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Duplicate Records Analysis", height=400)
            return fig
        
        # Group by cluster_id and count
        cluster_counts = duplicates.groupby('cluster_id').size().reset_index(name='count')
        
        fig = px.histogram(cluster_counts, x='count', 
                          title="Distribution of Duplicate Cluster Sizes",
                          nbins=10)
        fig.update_layout(height=400, xaxis_title="Records per Cluster", yaxis_title="Number of Clusters")
        return fig
    except Exception as e:
        # Fallback empty chart
        fig = go.Figure()
        fig.add_annotation(text=f"Unable to analyze duplicates: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Duplicate Records Analysis", height=400)
        return fig