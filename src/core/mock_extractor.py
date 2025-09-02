"""
Mock Patent Concept Extractor for Streamlit Demo
Simulates LLM responses without requiring actual model infrastructure
"""

import json
import datetime
import time
import random
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# Mock data models (same as original)
class NormalizationOutput(BaseModel):
    """Output model for input normalization"""
    problem: str = Field(
        description="Normalized technical problem or objective described in the document."
    )
    technical: str = Field(
        description="Normalized technical content or context of the document."
    )

class ConceptMatrix(BaseModel):
    """Output model for extracting core patent search concepts from technical documents"""
    problem_purpose: str = Field(
        description="The specific technical problem the invention aims to solve or the primary objective described in the document."
    )
    object_system: str = Field(
        description="The main object, device, system, material, or process that is the subject of the invention as stated in the document."
    )
    environment_field: str = Field(
        description="The application domain, industry sector, or operational context where the invention is intended to be used."
    )

class SeedKeywords(BaseModel):
    """Output model for Phase 2 and 3 keyword extraction (patent-specific fields only)"""
    problem_purpose: List[str] = Field(
        description="Distinctive technical keywords describing the technical problem addressed or primary objective."
    )
    object_system: List[str] = Field(
        description="Technical keywords specifying the main object, device, system, material, or process described."
    )
    environment_field: List[str] = Field(
        description="Keywords identifying the application domain, industry sector, or operational context."
    )

class ValidationFeedback(BaseModel):
    """User validation feedback"""
    action: str  # "approve", "edit", "reject"
    edited_keywords: Optional[SeedKeywords] = None
    feedback: Optional[str] = None

class QueriesResponse(BaseModel):
    """Output model for patent search queries"""
    queries: List[str] = Field(
        description="List of queries. Leave empty if none."
    )

class ExtractionState(dict):
    """Mock extraction state"""
    pass

class MockLLM:
    """Mock LLM that simulates realistic responses"""
    
    def __init__(self, model: str = "mock-llm", temperature: float = 0.7, num_ctx: int = 128000):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
    
    def invoke(self, prompt: str) -> str:
        """Simulate LLM response based on prompt content"""
        # Add realistic delay
        time.sleep(random.uniform(1, 3))
        
        # Detect prompt type and return appropriate mock response
        if "normalization" in prompt.lower() or "problem" in prompt.lower() and "technical" in prompt.lower():
            return self._mock_normalization_response()
        elif "concept matrix" in prompt.lower() or "problem_purpose" in prompt.lower():
            return self._mock_concept_matrix_response()
        elif "seed keywords" in prompt.lower() or "keyword" in prompt.lower():
            return self._mock_keywords_response()
        elif "summary" in prompt.lower():
            return self._mock_summary_response()
        elif "queries" in prompt.lower() or "boolean" in prompt.lower():
            return self._mock_queries_response()
        elif "synonyms" in prompt.lower() or "core_synonyms" in prompt.lower():
            return self._mock_synonyms_response()
        else:
            return self._mock_generic_response()
    
    def _mock_normalization_response(self) -> str:
        return '''
        {
            "problem": "Traditional irrigation systems operate on fixed schedules without considering actual soil moisture, weather conditions, or crop-specific needs, leading to water waste, increased costs, and potentially reduced crop yields.",
            "technical": "Smart irrigation system utilizing IoT sensors for real-time soil moisture monitoring, weather data integration, and automated irrigation control based on crop-specific requirements and field location data."
        }
        '''
    
    def _mock_concept_matrix_response(self) -> str:
        return '''
        {
            "problem_purpose": "Optimize water usage in agricultural irrigation while ensuring adequate crop moisture through real-time monitoring and automated adjustment",
            "object_system": "Smart irrigation system with IoT sensors, soil moisture monitors, weather integration, and automated control mechanisms",
            "environment_field": "Agricultural field management, precision farming, smart agriculture, water conservation systems"
        }
        '''
    
    def _mock_keywords_response(self) -> str:
        return '''
        {
            "problem_purpose": ["water optimization", "irrigation control", "moisture monitoring", "automated adjustment"],
            "object_system": ["IoT sensors", "soil monitors", "irrigation system", "control mechanisms"],
            "environment_field": ["agriculture", "farming", "field management", "water conservation"]
        }
        '''
    
    def _mock_summary_response(self) -> str:
        return '''
        {
            "summary": "A smart irrigation system integrating Internet of Things (IoT) sensors for real-time soil moisture monitoring and automated water distribution control. The system employs wireless sensor networks positioned throughout agricultural fields to continuously measure soil moisture levels, ambient temperature, and humidity. Data from multiple sensor nodes is transmitted to a central control unit that processes environmental parameters against crop-specific water requirements. The system features automated valve control mechanisms that adjust water flow rates and irrigation timing based on real-time sensor data and weather forecasting integration. Machine learning algorithms analyze historical irrigation patterns and crop growth stages to optimize water usage efficiency. The technical implementation includes low-power wireless communication protocols, weatherproof sensor housings, and solar-powered sensor nodes for extended field deployment. Control algorithms incorporate predictive analytics to anticipate irrigation needs based on weather patterns and crop development cycles."
        }
        '''
    
    def _mock_queries_response(self) -> str:
        return '''
        {
            "queries": [
                "(irrigation OR watering) AND (IoT OR sensor) AND (agriculture OR farming)",
                "(soil moisture OR water content) AND (monitoring OR detection) AND (automatic OR control)",
                "(smart irrigation OR precision watering) AND (wireless sensor OR remote monitoring)",
                "(agricultural automation OR farm management) AND (water optimization OR conservation)",
                "(crop irrigation OR plant watering) AND (sensor network OR IoT system)",
                "(automated irrigation OR intelligent watering) AND (field monitoring OR agricultural sensor)"
            ]
        }
        '''
    
    def _mock_synonyms_response(self) -> str:
        synonyms_responses = [
            '''
            {
                "core_synonyms": [
                    {"term": "watering", "justification": "direct irrigation synonym", "source": "src 1"},
                    {"term": "sprinkler system", "justification": "irrigation method variant", "source": "src 2"},
                    {"term": "water distribution", "justification": "irrigation process description", "source": "src 1"}
                ],
                "related_terms": [
                    {"term": "drip irrigation", "rationale": "specific irrigation technique", "source": "src 3"},
                    {"term": "fertigation", "rationale": "combined irrigation and fertilization", "source": "src 2"}
                ]
            }
            ''',
            '''
            {
                "core_synonyms": [
                    {"term": "moisture sensor", "justification": "soil monitoring device", "source": "src 1"},
                    {"term": "humidity detector", "justification": "moisture measurement tool", "source": "src 2"},
                    {"term": "water sensor", "justification": "moisture detection equipment", "source": "src 1"}
                ],
                "related_terms": [
                    {"term": "tensiometer", "rationale": "soil water tension measurement", "source": "src 3"},
                    {"term": "capacitance probe", "rationale": "soil moisture sensing technology", "source": "src 2"}
                ]
            }
            '''
        ]
        return random.choice(synonyms_responses)
    
    def _mock_generic_response(self) -> str:
        return "Mock LLM response for generic prompt."

class MockTavilySearch:
    """Mock Tavily search that returns realistic results"""
    
    def __init__(self, max_results=5, **kwargs):
        self.max_results = max_results
    
    def invoke(self, query_dict: dict) -> dict:
        """Mock search results"""
        query = query_dict.get("query", "")
        
        # Simulate search delay
        time.sleep(random.uniform(0.5, 1.5))
        
        mock_results = [
            {
                "content": f"Smart irrigation systems utilize advanced sensor technology to monitor soil conditions and automatically adjust watering schedules based on real-time data from {query}.",
                "url": "https://example.com/smart-irrigation-1"
            },
            {
                "content": f"IoT-based agricultural monitoring solutions provide farmers with precise control over water distribution systems, incorporating {query} for optimal crop management.",
                "url": "https://example.com/iot-agriculture-2"
            },
            {
                "content": f"Precision farming techniques leverage wireless sensor networks and {query} to achieve significant water conservation while maintaining crop yields.",
                "url": "https://example.com/precision-farming-3"
            }
        ]
        
        return {"results": mock_results[:self.max_results]}

class MockCoreConceptExtractor:
    """Mock version of CoreConceptExtractor that simulates the full workflow"""
    
    def __init__(self, model_name: str = None, use_checkpointer: bool = None, custom_evaluation_handler=None):
        self.model_name = model_name or "mock-llm"
        self.use_checkpointer = use_checkpointer or False
        self.custom_evaluation_handler = custom_evaluation_handler
        
        # Mock components
        self.llm = MockLLM(model=self.model_name)
        self.tavily_search = MockTavilySearch()
        
        # Mock IPC classifications
        self.mock_ipcs = [
            {"category": "A01G25/16", "score": 0.95},
            {"category": "G05B15/02", "score": 0.87},
            {"category": "H04L12/28", "score": 0.82}
        ]
        
        # Mock patent URLs
        self.mock_urls = [
            "https://patents.google.com/patent/US10123456B2",
            "https://patents.google.com/patent/US10234567B2", 
            "https://patents.google.com/patent/US10345678B2",
            "https://patents.google.com/patent/US10456789B2",
            "https://patents.google.com/patent/US10567890B2"
        ]
    
    def extract_keywords(self, input_text: str) -> Dict:
        """Run the complete mock extraction workflow"""
        print("🔄 Starting mock extraction workflow...")
        
        # Step 1: Input normalization
        print("📝 Step 1: Input normalization...")
        time.sleep(1)
        normalization_response = self.llm.invoke("normalization prompt")
        normalized_data = json.loads(normalization_response.strip())
        normalized_input = NormalizationOutput(**normalized_data)
        
        state = ExtractionState({
            "input_text": input_text,
            "problem": normalized_input.problem,
            "technical": normalized_input.technical
        })
        
        # Step 2: Concept extraction
        print("🎯 Step 2: Concept extraction...")
        time.sleep(1)
        concept_response = self.llm.invoke("concept matrix prompt")
        concept_data = json.loads(concept_response.strip())
        concept_matrix = ConceptMatrix(**concept_data)
        state["concept_matrix"] = concept_matrix
        
        # Step 3: Keyword generation
        print("🔑 Step 3: Keyword generation...")
        time.sleep(1)
        keyword_response = self.llm.invoke("seed keywords prompt")
        keyword_data = json.loads(keyword_response.strip())
        seed_keywords = SeedKeywords(**keyword_data)
        state["seed_keywords"] = seed_keywords
        
        # Step 4: Human evaluation (use custom handler if provided)
        print("👤 Step 4: Human evaluation...")
        if self.custom_evaluation_handler:
            evaluation_result = self.custom_evaluation_handler(state)
            state.update(evaluation_result)
        else:
            # Mock approval for non-interactive mode
            state["validation_feedback"] = ValidationFeedback(action="approve")
        
        # Check if we need to restart due to rejection
        if state.get("validation_feedback") and state["validation_feedback"].action == "reject":
            print("🔄 Restarting workflow due to rejection...")
            return self.extract_keywords(input_text)  # Recursive call
        
        # Handle manual edits
        if state.get("validation_feedback") and state["validation_feedback"].action == "edit":
            if state["validation_feedback"].edited_keywords:
                state["seed_keywords"] = state["validation_feedback"].edited_keywords
                print("✏️ Using manually edited keywords...")
        
        # Step 5: Generate synonyms
        print("🔍 Step 5: Generating synonyms...")
        final_keywords = {}
        for category, keywords in state["seed_keywords"].dict().items():
            for keyword in keywords:
                time.sleep(0.5)  # Simulate search time
                synonyms_response = self.llm.invoke(f"synonyms for {keyword}")
                try:
                    synonyms_data = json.loads(synonyms_response.strip())
                    synonyms = []
                    if "core_synonyms" in synonyms_data:
                        synonyms.extend([item["term"] for item in synonyms_data["core_synonyms"]])
                    if "related_terms" in synonyms_data:
                        synonyms.extend([item["term"] for item in synonyms_data["related_terms"]])
                    final_keywords[keyword] = synonyms
                except:
                    final_keywords[keyword] = [f"{keyword}_synonym1", f"{keyword}_synonym2"]
        
        state["final_keywords"] = final_keywords
        
        # Step 6: Generate summary and IPC classification
        print("📋 Step 6: Summary and IPC classification...")
        time.sleep(1)
        summary_response = self.llm.invoke("summary prompt")
        summary_data = json.loads(summary_response.strip())
        state["summary_text"] = summary_data["summary"]
        state["ipcs"] = self.mock_ipcs
        
        # Step 7: Generate queries
        print("🔍 Step 7: Generating search queries...")
        time.sleep(1)
        queries_response = self.llm.invoke("queries prompt")
        queries_data = json.loads(queries_response.strip())
        state["queries"] = QueriesResponse(**queries_data)
        
        # Step 8: Generate URLs
        print("🌐 Step 8: Finding patent URLs...")
        time.sleep(2)
        final_urls = []
        for i, url in enumerate(self.mock_urls):
            final_urls.append({
                "url": url,
                "user_scenario": random.uniform(0.6, 0.95),
                "user_problem": random.uniform(0.5, 0.9)
            })
        state["final_url"] = final_urls
        
        print("✅ Mock extraction completed!")
        return dict(state)

# Export the mock extractor for use in Streamlit
__all__ = ['MockCoreConceptExtractor', 'ValidationFeedback', 'SeedKeywords']
