import os, sys

# Ensure project root is importable (parent of this file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px
from src.engine import ProviderDQEngine
from src.genai import parse_intent, generate_response, get_follow_up_suggestions
from src.visualizations import (
    create_quality_score_gauge, 
    create_issues_by_type_chart,
    create_specialties_issues_chart,
    create_state_summary_chart,
    create_license_expiration_timeline,
    create_quality_metrics_summary,
    create_duplicate_analysis_chart
)

# Page configuration
try:
    st.set_page_config(
        page_title="Smart Provider Credentialing Analytics Platform", 
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "Smart Provider Credentialing Analytics Platform with AI-powered insights"
        }
    )
except st.errors.StreamlitAPIException:
    # Page config has already been set, skip
    pass

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
    }
    .suggestion-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "engine" not in st.session_state:
    st.session_state.engine = ProviderDQEngine()
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def save_temp(uploaded):
    if not uploaded:
        return None
    os.makedirs("./tmp", exist_ok=True)
    path = os.path.abspath(os.path.join("tmp", uploaded.name))
    with open(path, "wb") as f:
        f.write(uploaded.read())
    return path

# Header
st.markdown('<h1 class="main-header">Smart Provider Credentialing Analytics Platform</h1>', unsafe_allow_html=True)
st.markdown("### Local AI-Powered Data Quality Analytics with Interactive Dashboards")

# Sidebar for data loading
with st.sidebar:
    st.header("Data Management")
    
    # Option A: Auto-load from local datasets folder
    st.subheader("Load Sample Data")
    default_dir = os.path.join(ROOT_DIR, "datasets")
    st.caption(f"Folder: {default_dir}")
    
    if st.button("Load Sample Dataset", type="primary"):
        roster_path = os.path.join(default_dir, "provider_roster_with_errors.csv")
        ny_path = os.path.join(default_dir, "ny_medical_license_database.csv")
        ca_path = os.path.join(default_dir, "ca_medical_license_database.csv")
        npi_path = os.path.join(default_dir, "mock_npi_registry.csv")

        missing = [p for p in [roster_path, ny_path, ca_path, npi_path] if not os.path.exists(p)]
        if missing:
            st.error(" Missing files:\n" + "\n".join(missing))
        else:
            with st.spinner("Loading data..."):
                st.session_state.engine.load_files(roster_path, ny_path, ca_path, npi_path)
                st.session_state.loaded = True
            st.success("Sample data loaded successfully!")
            st.rerun()

    st.markdown("---")

    # Option B: Upload files manually
    st.subheader("Upload Custom Data")
    roster = st.file_uploader("Provider Roster (CSV)", type=["csv"], key="roster_upl")
    ny = st.file_uploader("NY License DB (CSV)", type=["csv"], key="ny_upl")
    ca = st.file_uploader("CA License DB (CSV)", type=["csv"], key="ca_upl")
    npi = st.file_uploader("Mock NPI Registry (CSV)", type=["csv"], key="npi_upl")
    
    if st.button("Load Uploaded Files"):
        if roster is None:
            st.error(" Please upload the roster file.")
        else:
            with st.spinner("Processing uploads..."):
                r_path = save_temp(roster)
                ny_path = save_temp(ny)
                ca_path = save_temp(ca)
                npi_path = save_temp(npi)
                st.session_state.engine.load_files(r_path, ny_path, ca_path, npi_path)
                st.session_state.loaded = True
            st.success("Custom data loaded successfully!")
            st.rerun()

    # AI Configuration
    st.markdown("---")
    st.subheader("Local AI Features")
    st.info(" This platform uses local AI models - no API keys required!")
    
    # Check if transformers is available
    try:
        import transformers
        import sentence_transformers
        st.success("Local AI models available - enhanced natural language processing enabled")
        st.caption(" Semantic query understanding\n🔹 Intelligent response generation\n🔹 No external API dependencies")
    except ImportError:
        st.warning("Local AI models not installed. Install 'transformers' and 'sentence-transformers' for enhanced features.")
        st.caption("The platform will work with basic rule-based processing.")

# Main content area
if not st.session_state.loaded:
    # Landing page when no data is loaded
    st.info(" Please load data from the sidebar to begin analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### Interactive Analytics
        - Real-time data quality metrics
        - Visual charts and dashboards
        - Trend analysis and insights
        """)
    
    with col2:
        st.markdown("""
        ### Local AI-Powered Insights
        - Natural language queries (no API keys!)
        - Intelligent response generation
        - Smart recommendations
        - Local semantic understanding
        """)
    
    with col3:
        st.markdown("""
        ### Quality Management
        - License compliance tracking
        - Duplicate detection
        - Data validation rules
        """)

else:
    # Main dashboard when data is loaded
    
    # Key metrics row
    st.subheader("Key Performance Indicators")
    metrics = create_quality_metrics_summary(st.session_state.engine)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label=" Total Providers", 
            value=metrics['total_providers'],
            help="Total number of providers in the system"
        )
    
    with col2:
        st.metric(
            label="Expired Licenses", 
            value=metrics['expired_licenses'],
            delta=f"{metrics['expired_licenses']/max(metrics['total_providers'], 1)*100:.1f}%",
            help="Number of providers with expired licenses"
        )
    
    with col3:
        st.metric(
            label="Missing NPI", 
            value=metrics['missing_npi'],
            delta=f"{metrics['missing_npi']/max(metrics['total_providers'], 1)*100:.1f}%",
            help="Providers missing NPI numbers"
        )
    
    with col4:
        st.metric(
            label="Phone Issues", 
            value=metrics['phone_issues'],
            delta=f"{metrics['phone_issues']/max(metrics['total_providers'], 1)*100:.1f}%",
            help="Providers with phone formatting issues"
        )
    
    with col5:
        st.metric(
            label="Quality Score", 
            value=f"{metrics['quality_score']}%",
            delta=f"{metrics['quality_score'] - 75:.1f}% vs target",
            help="Overall data quality score"
        )

    # Interactive Charts Dashboard
    st.markdown("---")
    st.subheader("Interactive Analytics Dashboard")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([" Overview", " Quality Analysis", "Clinical Insights", "Detailed Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality Score Gauge
            gauge_fig = create_quality_score_gauge(metrics['quality_score'])
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            # Issues by Type
            issues_fig = create_issues_by_type_chart(st.session_state.engine)
            st.plotly_chart(issues_fig, use_container_width=True)
        
        # License Expiration Timeline
        timeline_fig = create_license_expiration_timeline(st.session_state.engine)
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # State Summary
            state_fig = create_state_summary_chart(st.session_state.engine)
            st.plotly_chart(state_fig, use_container_width=True)
        
        with col2:
            # Duplicate Analysis
            dup_fig = create_duplicate_analysis_chart(st.session_state.engine)
            st.plotly_chart(dup_fig, use_container_width=True)
    
    with tab3:
        # Specialties Analysis
        specialty_fig = create_specialties_issues_chart(st.session_state.engine)
        st.plotly_chart(specialty_fig, use_container_width=True)
    
    with tab4:
        # Detailed data tables
        st.subheader(" Detailed Data Views")
        
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Expired Licenses", "Missing NPI", "Phone Issues", "Duplicate Records", "Multi-State Single License"]
        )
        
        if analysis_type == "Expired Licenses":
            data = st.session_state.engine.compliance_report_expired()
        elif analysis_type == "Missing NPI":
            data = st.session_state.engine.list_missing_npi()
        elif analysis_type == "Phone Issues":
            data = st.session_state.engine.list_phone_issues()
        elif analysis_type == "Duplicate Records":
            data = st.session_state.engine.list_duplicates()
        elif analysis_type == "Multi-State Single License":
            data = st.session_state.engine.multi_state_single_license()
        
        if not data.empty:
            st.dataframe(data, use_container_width=True, height=400)
            
            # Download button
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{analysis_type.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"No records found for {analysis_type}")

    # AI-Powered Chat Interface
    st.markdown("---")
    st.subheader("Local AI-Powered Query Interface")
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            " Ask a question about your provider data:",
            value=st.session_state.get('current_query', ''),
            placeholder="e.g., How many providers have expired licenses?"
        )
    
    with col2:
        run_clicked = st.button(" Analyze", type="primary")

    # Process query
    if run_clicked and query:
        with st.spinner(" Processing your query..."):
            try:
                # Parse intent with AI
                intent, params = parse_intent(query)
                
                # Execute query
                result = st.session_state.engine.run_query(intent, params)
                
                # Get data context for enhanced LLM responses
                data_context = st.session_state.engine.get_data_context_for_query(intent, result, params)
                
                # Generate intelligent response with data context
                ai_response = generate_response(intent, result, query, data_context)
                
                # Display results
                st.markdown('<div class="chat-message">', unsafe_allow_html=True)
                st.markdown(f"** AI Response:** {ai_response}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show data if applicable
                if isinstance(result, pd.DataFrame) and not result.empty:
                    st.subheader("Detailed Results")
                    st.dataframe(result, use_container_width=True)
                    
                    # Download option
                    csv = result.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        " Download Results",
                        data=csv,
                        file_name=f"{intent}_results.csv",
                        mime="text/csv"
                    )
                elif isinstance(result, (int, float)):
                    st.metric("Result", result)
                
                # Follow-up suggestions
                suggestions = get_follow_up_suggestions(intent, result)
                if suggestions:
                    st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                    st.markdown("** Suggested follow-up questions:**")
                    for suggestion in suggestions[:3]:  # Show top 3 suggestions
                        if st.button(f" {suggestion}", key=f"suggestion_{suggestion[:20]}"):
                            st.session_state.current_query = suggestion
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'query': query,
                    'intent': intent,
                    'response': ai_response,
                    'result_type': type(result).__name__
                })
                
            except Exception as e:
                st.error(f" Error processing query: {str(e)}")

    # Clear current query after processing
    if 'current_query' in st.session_state:
        del st.session_state.current_query

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 1rem;'>
         Smart Provider Credentialing Analytics Platform | Powered by AI & Advanced Analytics
    </div>
    """,
    unsafe_allow_html=True
)
