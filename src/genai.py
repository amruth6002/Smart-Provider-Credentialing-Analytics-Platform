"""
Gen-AI module for enhanced natural language processing and intelligent responses
Uses local LLM models via Hugging Face transformers - no API keys required!
"""
import os
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import re
import numpy as np

# Try to import sentence_transformers, but make it optional for compatibility
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# Try to import streamlit, but make it optional for testing
try:
    import streamlit as st
except ImportError:
    # Mock streamlit for testing environments
    class MockStreamlit:
        def warning(self, msg): print(f"WARNING: {msg}")
        def info(self, msg): print(f"INFO: {msg}")
        def spinner(self, msg): 
            class MockSpinner:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return MockSpinner()
    st = MockStreamlit()

# Try to import transformers for local LLM, but make it optional
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .intents import INTENT_PATTERNS, extract_params
from .data_context import DataContext


class GenAIProcessor:
    """Enhanced NLU processor with local LLM capabilities - no API keys needed!"""
    
    def __init__(self):
        self.transformers_available = TRANSFORMERS_AVAILABLE
        self.sentence_transformers_available = SENTENCE_TRANSFORMERS_AVAILABLE
        self.sentence_model = None
        self.text_generator = None
        self.intent_embeddings = None
        
        # Initialize local models if both transformers and sentence_transformers are available
        if self.transformers_available and self.sentence_transformers_available:
            self._initialize_local_models()
    
    def _initialize_local_models(self):
        """Initialize local AI models for intent classification and text generation"""
        try:
            # Load sentence transformer for semantic similarity
            with st.spinner("Loading AI models (first time may take a few minutes)..."):
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Prepare intent embeddings for semantic matching
                self._prepare_intent_embeddings()
                
                # Try to load a small text generation model
                try:
                    self.text_generator = pipeline(
                        "text-generation",
                        model="microsoft/DialoGPT-small",
                        tokenizer="microsoft/DialoGPT-small",
                        max_length=150,
                        do_sample=True,
                        temperature=0.7
                    )
                except Exception as e:
                    st.info("Advanced text generation not available, using template-based responses")
                    self.text_generator = None
                    
        except Exception as e:
            st.warning(f"Local AI models not available: {e}. Using rule-based processing.")
            self.sentence_model = None
            self.text_generator = None
    
    def _prepare_intent_embeddings(self):
        """Prepare embeddings for all available intents for semantic matching"""
        if not self.sentence_model:
            return
            
        intent_descriptions = {
            "expired_license_count": "count expired licenses providers healthcare",
            "phone_format_issues": "phone number format formatting issues problems",
            "missing_npi": "missing NPI number identifier provider",
            "duplicate_records": "duplicate providers records same person",
            "overall_quality_score": "overall quality score data assessment rating",
            "specialties_with_most_issues": "medical specialties most data quality issues problems",
            "state_issue_summary": "state summary issues problems by location geography",
            "compliance_report_expired": "compliance report expired licenses regulatory",
            "filter_by_expiration_window": "filter providers expiring soon within days",
            "multi_state_single_license": "providers multiple states single license",
            "export_update_list": "export list providers needing updates",
            "search_provider_by_name": "search find provider by name person individual"
        }
        
        # Generate embeddings for intent descriptions
        descriptions = list(intent_descriptions.values())
        self.intent_embeddings = {
            intent: embedding for intent, embedding in 
            zip(intent_descriptions.keys(), self.sentence_model.encode(descriptions))
        }
    
    def parse_intent_with_ai(self, text: str) -> Tuple[str, Dict]:
        """Parse intent using local AI if available, fallback to rule-based"""
        # First try rule-based approach
        intent, params = self._rule_based_intent_parsing(text)
        
        # If we have local AI models and we got a fallback intent, try to enhance it
        if self.sentence_model and intent == "overall_quality_score":  # This is our fallback
            try:
                enhanced_intent, enhanced_params = self._ai_intent_parsing(text)
                if enhanced_intent and enhanced_intent != "overall_quality_score":
                    return enhanced_intent, enhanced_params
            except Exception as e:
                st.warning(f"AI parsing failed, using rule-based: {e}")
        
        return intent, params
    
    def _rule_based_intent_parsing(self, text: str) -> Tuple[str, Dict]:
        """Original rule-based intent parsing"""
        for intent, patterns in INTENT_PATTERNS.items():
            for p in patterns:
                if p.search(text):
                    params = extract_params(intent, text)
                    return intent, params
        
        # Fallback logic
        t = text.lower()
        if "expired" in t and "license" in t and "how many" in t:
            return "expired_license_count", {}
        if "duplicate" in t:
            return "duplicate_records", {}
        if "quality score" in t:
            return "overall_quality_score", {}
        if "phone" in t and ("issue" in t or "format" in t):
            return "phone_format_issues", {}
        
        return "overall_quality_score", {}
    
    def _ai_intent_parsing(self, text: str) -> Tuple[str, Dict]:
        """AI-enhanced intent parsing using local sentence transformers"""
        if not self.sentence_model or not self.intent_embeddings:
            return None, {}
        
        try:
            # Get embedding for the user query
            query_embedding = self.sentence_model.encode([text])[0]
            
            # Calculate similarities with all intent embeddings
            similarities = {}
            for intent, intent_embedding in self.intent_embeddings.items():
                similarity = np.dot(query_embedding, intent_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(intent_embedding)
                )
                similarities[intent] = similarity
            
            # Get the best matching intent
            best_intent = max(similarities, key=similarities.get)
            best_score = similarities[best_intent]
            
            # Only use AI prediction if confidence is high enough
            if best_score > 0.5:  # Threshold for semantic similarity
                params = extract_params(best_intent, text)
                return best_intent, params
            
        except Exception as e:
            st.warning(f"Local AI intent parsing error: {e}")
        
        return None, {}
    
    def generate_intelligent_response(self, intent: str, result: Any, query: str, data_context: Optional[DataContext] = None) -> str:
        """Generate intelligent natural language response using local AI and data context"""
        # Use data-aware response generation if context is available
        if data_context:
            return self._generate_data_aware_response(intent, result, query, data_context)
        
        if not self.text_generator:
            return self._generate_enhanced_response(intent, result, query)
        
        try:
            # Prepare context about the result
            if isinstance(result, pd.DataFrame):
                if result.empty:
                    result_context = "No data found"
                else:
                    result_context = f"Found {len(result)} records"
            elif isinstance(result, (int, float)):
                result_context = f"Result: {result}"
            else:
                result_context = f"Result: {str(result)[:100]}"
            
            # Create a simple prompt for local model
            prompt = f"User asked: {query}. Result: {result_context}. Response:"
            
            # Generate response with local model
            response = self.text_generator(prompt, max_length=100, num_return_sequences=1)
            generated_text = response[0]['generated_text']
            
            # Extract just the response part
            if "Response:" in generated_text:
                response_text = generated_text.split("Response:")[-1].strip()
                if response_text and len(response_text) > 10:
                    return response_text
            
        except Exception as e:
            st.warning(f"Local AI response generation failed: {e}")
        
        return self._generate_enhanced_response(intent, result, query)
    
    def _generate_data_aware_response(self, intent: str, result: Any, query: str, data_context: DataContext) -> str:
        """Generate intelligent responses using data context and patterns"""
        # Base result information
        if isinstance(result, pd.DataFrame):
            record_count = len(result) if not result.empty else 0
        elif isinstance(result, (int, float)):
            record_count = result
        else:
            record_count = 0
        
        # Build context-aware response components
        response_parts = []
        
        # 1. Main result with data context
        main_result = self._format_main_result_with_context(intent, result, record_count, data_context)
        response_parts.append(main_result)
        
        # 2. Data insights based on context
        if data_context.key_findings:
            insights = self._format_data_insights(data_context.key_findings)
            response_parts.append(insights)
        
        # 3. Data sample reference (if relevant)
        if data_context.data_sample is not None and not data_context.data_sample.empty:
            sample_info = self._format_sample_reference(intent, data_context)
            if sample_info:
                response_parts.append(sample_info)
        
        # 4. Quality context
        quality_info = self._format_quality_context(data_context)
        if quality_info:
            response_parts.append(quality_info)
        
        return " ".join(response_parts)
    
    def _format_main_result_with_context(self, intent: str, result: Any, record_count: int, data_context: DataContext) -> str:
        """Format the main result with rich data context"""
        total_providers = data_context.total_providers
        
        if intent == "expired_license_count":
            percentage = (record_count / total_providers * 100) if total_providers > 0 else 0
            return f"🚨 Analysis of {total_providers:,} providers shows {record_count:,} have expired licenses ({percentage:.1f}%)."
        
        elif intent == "phone_format_issues":
            percentage = (record_count / total_providers * 100) if total_providers > 0 else 0
            return f"📞 Found {record_count:,} providers with phone formatting issues out of {total_providers:,} total ({percentage:.1f}%)."
        
        elif intent == "missing_npi":
            percentage = (record_count / total_providers * 100) if total_providers > 0 else 0
            return f"🆔 Identified {record_count:,} providers missing NPI numbers from {total_providers:,} total providers ({percentage:.1f}%)."
        
        elif intent == "duplicate_records":
            return f"🔍 Duplicate analysis of {total_providers:,} providers identified {record_count:,} potential duplicate pairs."
        
        elif intent == "overall_quality_score":
            return f"📊 Overall data quality score: {record_count}% across {total_providers:,} providers."
        
        elif intent == "specialties_with_most_issues":
            return f"📈 Quality analysis across medical specialties from {total_providers:,} provider records shows varying issue patterns."
        
        elif intent == "state_issue_summary":
            return f"🌍 Geographic analysis of {total_providers:,} providers reveals state-specific data quality patterns."
        
        elif intent == "compliance_report_expired":
            return f"📋 Compliance report generated: {record_count:,} providers require immediate license renewal attention."
        
        elif intent == "filter_by_expiration_window":
            return f"⏰ Expiration window analysis: {record_count:,} providers have licenses expiring soon from {total_providers:,} total."
        
        elif intent == "multi_state_single_license":
            return f"🏥 Multi-state analysis: {record_count:,} providers practice across state lines with single licenses."
        
        elif intent == "export_update_list":
            return f"📤 Update list generated: {record_count:,} providers require credential updates from {total_providers:,} total."
        
        # Default format
        percentage = (record_count / total_providers * 100) if total_providers > 0 else 0
        return f"✅ Analysis of {total_providers:,} providers completed: {record_count:,} records match your criteria ({percentage:.1f}%)."
    
    def _format_data_insights(self, key_findings: List[str]) -> str:
        """Format key findings into readable insights"""
        if not key_findings:
            return ""
        
        insights = "💡 Key insights: " + "; ".join(key_findings[:2])  # Limit to top 2 insights
        return insights + "."
    
    def _format_sample_reference(self, intent: str, data_context: DataContext) -> str:
        """Reference the actual data when appropriate"""
        if data_context.data_sample is None or data_context.data_sample.empty:
            return ""
        
        sample_size = len(data_context.data_sample)
        
        # Only reference samples for certain intents where it adds value
        if intent in ["expired_license_count", "missing_npi", "phone_format_issues"]:
            if "specialty" in data_context.data_sample.columns:
                specialties = data_context.data_sample["specialty"].value_counts()
                if not specialties.empty:
                    top_specialty = specialties.index[0]
                    return f"🔍 Sample data shows issues affecting specialties including {top_specialty}."
        
        return ""
    
    def _format_quality_context(self, data_context: DataContext) -> str:
        """Add overall quality context when relevant"""
        quality_score = data_context.overall_quality_score
        
        if quality_score > 0:
            if quality_score >= 85:
                quality_status = "excellent"
            elif quality_score >= 75:
                quality_status = "good"
            elif quality_score >= 60:
                quality_status = "fair"
            else:
                quality_status = "needs improvement"
            
            return f"📈 Overall data quality: {quality_score}% ({quality_status})."
        
        return ""
    
    def _generate_enhanced_response(self, intent: str, result: Any, query: str) -> str:
        """Generate enhanced rule-based response with context awareness"""
        # Enhanced responses with more context and insights
        if isinstance(result, pd.DataFrame):
            record_count = len(result) if not result.empty else 0
        elif isinstance(result, (int, float)):
            record_count = result
        else:
            record_count = 0
        
        responses = {
            "expired_license_count": self._format_expired_response(record_count),
            "phone_format_issues": self._format_phone_response(record_count),
            "missing_npi": self._format_npi_response(record_count),
            "duplicate_records": self._format_duplicate_response(record_count),
            "overall_quality_score": self._format_quality_response(result),
            "specialties_with_most_issues": "📊 Analysis complete! Here are the medical specialties with the most data quality issues. Consider focusing quality improvement efforts on these areas.",
            "state_issue_summary": "🌍 Geographic analysis shows data quality patterns by state. This helps identify regional compliance trends and target improvement efforts.",
            "compliance_report_expired": f"📋 Compliance report generated! {record_count} providers require immediate attention for license renewal to maintain regulatory compliance.",
            "filter_by_expiration_window": f"⏰ Found {record_count} providers with licenses expiring soon. Proactive renewal management is crucial for uninterrupted care delivery.",
            "multi_state_single_license": f"🏥 Identified {record_count} providers practicing across state lines with single licenses. This may indicate compliance risks requiring review.",
            "export_update_list": f"📤 Export ready! Generated actionable list of {record_count} providers requiring credential updates for your quality improvement team."
        }
        
        return responses.get(intent, f"✅ Analysis complete! Found {record_count} relevant records for your query.")
    
    def _format_expired_response(self, count: int) -> str:
        if count == 0:
            return "🎉 Excellent! No providers found with expired licenses. Your credentialing team is maintaining strong compliance!"
        elif count < 10:
            return f"⚠️ Found {count} providers with expired licenses. This is manageable - recommend immediate outreach for renewal."
        else:
            return f"🚨 Critical: {count} providers have expired licenses! This represents a significant compliance risk requiring urgent action."
    
    def _format_phone_response(self, count: int) -> str:
        if count == 0:
            return "✅ All provider phone numbers are properly formatted! Your data quality is excellent in this area."
        else:
            return f"📞 Found {count} providers with phone formatting issues. Consider implementing automated phone number validation to improve data quality."
    
    def _format_npi_response(self, count: int) -> str:
        if count == 0:
            return "✅ All providers have valid NPI numbers! This is crucial for billing and compliance."
        else:
            return f"🆔 {count} providers are missing NPI numbers. This is a critical compliance issue that must be resolved for proper billing and identification."
    
    def _format_duplicate_response(self, count: int) -> str:
        if count == 0:
            return "✅ No duplicate records detected! Your provider database maintains excellent data integrity."
        else:
            return f"👥 Found {count} potential duplicate records. Recommend reviewing these for data consolidation to improve accuracy."
    
    def _format_quality_response(self, score) -> str:
        try:
            score_val = float(score)
            if score_val >= 90:
                return f"🌟 Outstanding! Overall data quality score is {score_val:.1f}%. Your credentialing data meets the highest standards."
            elif score_val >= 70:
                return f"👍 Good data quality at {score_val:.1f}%. Some areas for improvement identified - focus on highest-impact issues first."
            else:
                return f"⚠️ Data quality score is {score_val:.1f}%. Significant improvement needed - recommend systematic data cleansing initiative."
        except:
            return f"📊 Overall data quality score: {score}%. Use the detailed analysis tabs to identify specific improvement opportunities."
    
    def _generate_simple_response(self, intent: str, result: Any) -> str:
        """Generate simple rule-based response (fallback)"""
        responses = {
            "expired_license_count": f"Found {result} providers with expired licenses.",
            "phone_format_issues": f"Found {len(result) if hasattr(result, '__len__') else result} providers with phone formatting issues.",
            "missing_npi": f"Found {len(result) if hasattr(result, '__len__') else result} providers missing NPI numbers.",
            "duplicate_records": f"Found {len(result) if hasattr(result, '__len__') else result} potential duplicate records.",
            "overall_quality_score": f"Overall data quality score is {result}%.",
            "specialties_with_most_issues": "Here are the medical specialties with the most data quality issues.",
            "state_issue_summary": "Here's a summary of data quality issues by state.",
            "compliance_report_expired": "Generated compliance report for expired licenses.",
            "filter_by_expiration_window": f"Found {len(result) if hasattr(result, '__len__') else result} providers with licenses expiring soon.",
            "multi_state_single_license": f"Found {len(result) if hasattr(result, '__len__') else result} providers practicing in multiple states with single licenses.",
            "export_update_list": f"Generated list of {len(result) if hasattr(result, '__len__') else result} providers requiring credential updates.",
            "search_provider_by_name": f"Found {len(result) if hasattr(result, '__len__') else 0} provider(s) matching the search criteria." if hasattr(result, '__len__') else "Provider search completed."
        }
        
        return responses.get(intent, "Query completed successfully.")
    
    def suggest_follow_up_questions(self, intent: str, result: Any) -> list:
        """Suggest relevant follow-up questions based on current query"""
        suggestions = {
            "expired_license_count": [
                "Show me the compliance report for expired licenses",
                "Which specialties have the most expired licenses?",
                "What is the trend of license expirations over time?"
            ],
            "phone_format_issues": [
                "Export the list of providers with phone issues",
                "Which states have the most phone formatting problems?",
                "What is our overall data quality score?"
            ],
            "missing_npi": [
                "Show providers missing both NPI and having other issues",
                "Which specialties are missing the most NPI numbers?",
                "Export update list for providers needing NPI numbers"
            ],
            "overall_quality_score": [
                "Show me a breakdown of issues by type",
                "Which specialties have the most data quality issues?",
                "What are the main data quality problems?"
            ],
            "duplicate_records": [
                "Export the duplicate records for manual review",
                "Which specialties have the most duplicates?",
                "Show me the overall data quality score"
            ],
            "specialties_with_most_issues": [
                "What specific issues affect these specialties?",
                "Generate a compliance report",
                "Show the overall quality score trend"
            ],
            "search_provider_by_name": [
                "Show me this provider's data quality issues",
                "What is this provider's license status?",
                "Are there any duplicates for this provider?"
            ]
        }
        
        return suggestions.get(intent, [
            "What is our overall data quality score?",
            "Show me issues by specialty",
            "Generate a compliance report"
        ])


# Global instance
genai_processor = GenAIProcessor()


def parse_intent(text: str) -> Tuple[str, Dict]:
    """Enhanced intent parsing with AI support"""
    return genai_processor.parse_intent_with_ai(text)


def generate_response(intent: str, result: Any, query: str, data_context: Optional[DataContext] = None) -> str:
    """Generate intelligent response"""
    return genai_processor.generate_intelligent_response(intent, result, query, data_context)


def get_follow_up_suggestions(intent: str, result: Any) -> list:
    """Get follow-up question suggestions"""
    return genai_processor.suggest_follow_up_questions(intent, result)