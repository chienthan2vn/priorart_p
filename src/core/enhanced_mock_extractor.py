"""
Enhanced Mock Patent Concept Extractor
Maintains the exact multi-agent LangGraph architecture from extractor.py but uses constant data
"""

import json
import datetime
import time
import random
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict, Annotated

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START

# Streamlit Integration
try:
    import streamlit as st
    import pandas as pd
    import traceback
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Configure logging
# log_filename = f"mock_patent_extractor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(log_filename),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

# Data Models (exact same as original)
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

class ReflectionEvaluation(BaseModel):
    """Reflection evaluation of keywords"""
    overall_quality: str = Field(description="Overall quality assessment: 'good' or 'poor'")
    keyword_scores: Dict[str, float] = Field(description="Score for each category (0-1)")
    issues_found: List[str] = Field(description="List of specific issues identified")
    recommendations: List[str] = Field(description="Recommendations for improvement")
    should_regenerate: bool = Field(description="Whether keywords should be regenerate")

class QueriesResponse(BaseModel):
    """Output model for patent search queries"""
    queries: List[str] = Field(
        description="List of queries. Leave empty if none."
    )

class ExtractionState(TypedDict):
    """Simplified state for LangGraph workflow (exact same as original)"""
    input_text: str
    problem: Optional[str]
    technical: Optional[str]
    summary_text: str
    ipcs: Any 
    concept_matrix: Optional[ConceptMatrix]
    seed_keywords: Optional[SeedKeywords]
    validation_feedback: Optional[ValidationFeedback]
    final_keywords: dict
    queries: list
    final_url: list

class MockPrompts:
    """Mock prompts that return structured parsers"""
    
    def get_normalization_prompt_and_parser(self):
        class MockParser:
            def parse(self, response):
                # Return mock normalized data
                return type('obj', (object,), {
                    'dict': lambda: {
                        "problem": "Optimize water usage in agricultural irrigation while ensuring adequate crop moisture through real-time monitoring and automated adjustment",
                        "technical": "Smart irrigation system utilizing IoT sensors for real-time soil moisture monitoring, weather data integration, and automated irrigation control based on crop-specific requirements"
                    }
                })
        
        return "mock normalization prompt", MockParser()
    
    def get_phase1_prompt_and_parser(self):
        class MockParser:
            def parse(self, response):
                return type('obj', (object,), {
                    'dict': lambda: {
                        "problem_purpose": "Optimize water usage in agricultural irrigation while ensuring adequate crop moisture through real-time monitoring and automated adjustment",
                        "object_system": "Smart irrigation system with IoT sensors, soil moisture monitors, weather integration, and automated control mechanisms",
                        "environment_field": "Agricultural field management, precision farming, smart agriculture, water conservation systems"
                    }
                })
        
        return "mock phase1 prompt", MockParser()
    
    def get_phase2_prompt_and_parser(self):
        class MockParser:
            def parse(self, response):
                return type('obj', (object,), {
                    'dict': lambda: {
                        "problem_purpose": ["water optimization", "irrigation control", "moisture monitoring", "automated adjustment"],
                        "object_system": ["IoT sensors", "soil monitors", "irrigation system", "control mechanisms"],
                        "environment_field": ["agriculture", "farming", "field management", "water conservation"]
                    }
                })
        
        return "mock phase2 prompt", MockParser()
    
    def get_summary_prompt_and_parser(self):
        class MockParser:
            def parse(self, response):
                return "A smart irrigation system integrating Internet of Things (IoT) sensors for real-time soil moisture monitoring and automated water distribution control. The system employs wireless sensor networks positioned throughout agricultural fields to continuously measure soil moisture levels, ambient temperature, and humidity."
        
        return "mock summary prompt", MockParser()
    
    def get_queries_prompt_and_parser(self):
        class MockParser:
            def parse(self, response):
                return QueriesResponse(queries=[
                    "(irrigation OR watering) AND (IoT OR sensor) AND (agriculture OR farming)",
                    "(soil moisture OR water content) AND (monitoring OR detection) AND (automatic OR control)",
                    "(smart irrigation OR precision watering) AND (wireless sensor OR remote monitoring)",
                    "(agricultural automation OR farm management) AND (water optimization OR conservation)",
                    "(crop irrigation OR plant watering) AND (sensor network OR IoT system)"
                ])
        
        return "mock queries prompt", MockParser()
    
    @staticmethod
    def get_phase_completion_messages():
        return {
            "separator": "=" * 80,
            "final_evaluation_title": "🎯 FINAL EVALUATION - PATENT SEED KEYWORDS",
            "concept_matrix_header": "\n📋 CONCEPT MATRIX:",
            "seed_keywords_header": "\n🔑 SEED KEYWORDS:",
            "divider": "-" * 50,
            "action_options": "Choose an action:\n1. ✅ Approve (continue with these keywords)\n2. ❌ Reject (regenerate keywords with feedback)\n3. ✏️ Edit (manually modify keywords)",
            "action_prompt": "\nYour choice (1/2/3 or approve/reject/edit): ",
            "reject_feedback_prompt": "Please provide feedback for regeneration: ",
            "invalid_action": "❌ Invalid choice. Please enter 1, 2, 3, or approve/reject/edit."
        }
    
    @staticmethod
    def get_validation_messages():
        return MockPrompts.get_phase_completion_messages()

class MockLLM:
    """Mock LLM that returns constant responses based on context"""
    
    def __init__(self, model="mock-llm", temperature=0.7, num_ctx=128000):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
    
    def invoke(self, prompt: str) -> str:
        """Return mock responses based on prompt context"""
        # Add realistic delay
        time.sleep(random.uniform(0.5, 1.5))
        
        # Mock responses for different contexts
        mock_responses = {
            "normalization": '{"problem": "Optimize water usage in agricultural irrigation", "technical": "Smart irrigation system with IoT sensors"}',
            "concept": '{"problem_purpose": "Water optimization", "object_system": "IoT irrigation system", "environment_field": "Agriculture"}',
            "keywords": '{"problem_purpose": ["water optimization"], "object_system": ["IoT sensors"], "environment_field": ["agriculture"]}',
            "summary": '"Smart irrigation system with IoT sensors for water optimization"',
            "queries": '{"queries": ["irrigation IoT sensors"]}',
            "synonyms": '{"core_synonyms": [{"term": "watering system", "justification": "irrigation synonym", "source": "src 1"}], "related_terms": [{"term": "drip irrigation", "rationale": "irrigation method", "source": "src 2"}]}'
        }
        
        # Return appropriate mock response
        for key, response in mock_responses.items():
            if key in prompt.lower():
                return response
        
        return "Mock LLM response"

class MockTavilySearch:
    """Mock Tavily search"""
    
    def __init__(self, max_results=5, **kwargs):
        self.max_results = max_results
    
    def invoke(self, query_dict: dict) -> dict:
        """Return mock search results"""
        time.sleep(random.uniform(0.3, 0.8))
        return {
            "results": [
                {"content": f"Mock search result for {query_dict.get('query', 'unknown')}", "url": "https://example.com/1"},
                {"content": f"Another mock result about {query_dict.get('query', 'topic')}", "url": "https://example.com/2"}
            ]
        }

def mock_get_ipc_predictions(summary_text: str) -> List[Dict]:
    """Mock IPC classification API"""
    time.sleep(0.5)
    return [
        {"category": "A01G25/16", "score": 0.95},
        {"category": "G05B15/02", "score": 0.87},
        {"category": "H04L12/28", "score": 0.82}
    ]

def mock_lay_thong_tin_patent(url: str) -> Dict:
    """Mock patent information extraction"""
    return {
        "abstract": "Mock patent abstract for smart irrigation system",
        "description": "Mock detailed description of IoT-based irrigation technology",
        "claims": "Mock patent claims for automated irrigation control"
    }

def mock_prompt(abstract: str, description: str, claims: str) -> str:
    """Mock prompt generation for patent evaluation"""
    return "Mock evaluation prompt for patent analysis"

def mock_parse_idea_input(input_text: str) -> Dict:
    """Mock idea parsing"""
    return {
        "user_scenario": "Agricultural irrigation optimization",
        "user_problem": "Water waste in traditional irrigation systems"
    }

def mock_extract_user_info(data: Dict) -> Dict:
    """Mock user info extraction"""
    return {
        "user_scenario": "Smart irrigation implementation",
        "user_problem": "Automated water management"
    }

def mock_eval_url(user_data: str, patent_data: str) -> Dict:
    """Mock URL evaluation"""
    return {"llm_score": random.uniform(0.6, 0.9)}

class EnhancedMockCoreConceptExtractor:
    """Enhanced Mock Patent seed keyword extraction system with full LangGraph architecture"""
    
    def __init__(self, model_name: str = None, use_checkpointer: bool = None, custom_evaluation_handler=None):
        """
        Initialize the EnhancedMockCoreConceptExtractor.
        
        Args:
            model_name: Name of the LLM model to use (ignored in mock)
            use_checkpointer: Whether to use checkpointer for graph state
            custom_evaluation_handler: Optional custom handler for human evaluation (for UI integration)
        """
        logger.info(" Initializing EnhancedMockCoreConceptExtractor...")
        self.model_name = model_name or "mock-llm"
        self.use_checkpointer = use_checkpointer or False
        self.custom_evaluation_handler = custom_evaluation_handler

        # Mock components
        self.llm = MockLLM(model=self.model_name)
        self.tavily_search = MockTavilySearch(max_results=5)
        self.prompts = MockPrompts()
        self.messages = MockPrompts.get_phase_completion_messages()
        self.validation_messages = MockPrompts.get_validation_messages()
        
        # Build the exact same graph structure as original
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build simplified LangGraph workflow (exact same structure as original)"""
        workflow = StateGraph(ExtractionState)
        
        # Add nodes for simplified 3-step process (exact same as original)
        workflow.add_node("input_normalization", self.input_normalization)
        workflow.add_node("step0", self.step0)
        workflow.add_node("step1_concept_extraction", self.step1_concept_extraction)
        workflow.add_node("step2_keyword_generation", self.step2_keyword_generation)
        workflow.add_node("step3_human_evaluation", self.step3_human_evaluation)
        workflow.add_node("manual_editing", self.manual_editing)
        workflow.add_node("gen_key", self.gen_key)
        workflow.add_node("summary_prompt_and_parser", self.summary_prompt_and_parser)
        workflow.add_node("call_ipcs_api", self.call_ipcs_api)
        workflow.add_node("genQuery", self.genQuery)
        workflow.add_node("genUrl", self.genUrl)
        workflow.add_node("evalUrl", self.evalUrl)

        # Define simplified flow (exact same as original)
        workflow.set_entry_point("input_normalization")
        workflow.add_edge("input_normalization", "step0")
        workflow.add_edge("step0", "step1_concept_extraction")
        workflow.add_edge("step0", "summary_prompt_and_parser")
        workflow.add_edge("step1_concept_extraction", "step2_keyword_generation")
        workflow.add_edge("step2_keyword_generation", "step3_human_evaluation")

        workflow.add_edge("summary_prompt_and_parser", "call_ipcs_api")
        
        # Conditional edge from human evaluation (exact same as original)
        workflow.add_conditional_edges(
            "step3_human_evaluation",
            self._get_human_action,
            {
                "approve": "gen_key",
                "reject": "step1_concept_extraction", 
                "edit": "manual_editing"
            }
        )
        
        workflow.add_edge("manual_editing", "gen_key")
        workflow.add_edge("gen_key", "genQuery")
        workflow.add_edge("genQuery", "genUrl")
        workflow.add_edge("genUrl", "evalUrl")

        return workflow.compile()
    
    def extract_keywords(self, input_text: str, extraction_state: dict) -> Dict:
        """Run the simplified 3-step keyword extraction workflow (exact same as original)"""
        initial_state = ExtractionState(
            input_text=input_text,
            problem=extraction_state["problem"] if extraction_state else None,
            technical=extraction_state["technical"] if extraction_state else None,
            concept_matrix=extraction_state["concept_matrix"] if extraction_state else None,
            seed_keywords=extraction_state["seed_keywords"] if extraction_state else None,
            validation_feedback=extraction_state["validation_feedback"] if extraction_state else None,
            final_keywords=extraction_state["final_keywords"] if extraction_state else None,
            ipcs=extraction_state["ipcs"] if extraction_state else None,
            summary_text=extraction_state["summary_text"] if extraction_state else None,
            queries=extraction_state["queries"] if extraction_state else None,
            final_url=extraction_state["final_url"] if extraction_state else None
        )

        logger.info(f"initial_state: {initial_state}")
        
        if self.use_checkpointer:
            config = {"configurable": {"thread_id": "mock_thread_123"}}
            result = self.graph.invoke(initial_state, config)
        else:
            result = self.graph.invoke(initial_state)
        
        # Return all ExtractionState fields
        return dict(result)
        
    def input_normalization(self, initial_state: ExtractionState) -> ExtractionState:
        """Normalize and clean input text before processing (exact same logic as original)"""    
        logger.info(" Starting input normalization...")
        
        # Get normalization prompt and parser from MockPrompts
        if initial_state["problem"] and initial_state["technical"]:
            logger.info(" Problem and technical already exist, skipping normalization")
            return {"problem": initial_state["problem"], "technical": initial_state["technical"], "input_text": initial_state["input_text"]}
        
        prompt, parser = self.prompts.get_normalization_prompt_and_parser()
        response = self.llm.invoke(prompt)

        try:
            normalized_data = parser.parse(response)
            normalized_input = NormalizationOutput(**normalized_data.dict())
            logger.info(" Normalization completed.")

            updated_state = {
                "problem": normalized_input.problem,
                "technical": normalized_input.technical,
                "input_text": initial_state["input_text"]
            }
            logger.info(f" Normalized problem: {normalized_input.problem}")
            logger.info(f" Normalized technical: {normalized_input.technical}")
            return updated_state

        except Exception as e:
            logger.error(f" Normalization parsing failed: {e}, using original input")
            fallback_normalized = NormalizationOutput(
                problem="Not mentioned.",
                technical="Not mentioned."
            )
            updated_state = {
                "problem": "Not mentioned.",
                "technical": "Not mentioned.",
                "input_text": initial_state["input_text"]
            }
            # initial_state["problem"] = normalized_input.problem
            # initial_state["technical"] = normalized_input.technical
            return updated_state

    def step0(self, initial_state: ExtractionState) -> ExtractionState:
        """Initial step - pass through state (exact same as original)"""
        logger.info(" Step 0: Initial processing...")
        logger.info(f"initial_state: {initial_state}")
        return initial_state

    def step1_concept_extraction(self, initial_state: ExtractionState) -> ExtractionState:
        """Step 1: Extract concept summary from document according to fields (exact same logic as original)"""
        logger.info(" Step 1: Concept extraction...")
        
        if initial_state["concept_matrix"]:
            logger.info(" Concept matrix already exists, skipping concept extraction")
            return {"concept_matrix": initial_state["concept_matrix"]}
        
        prompt, parser = self.prompts.get_phase1_prompt_and_parser()
        response = self.llm.invoke(prompt)
        # logger.info(f"concept_matrix: {initial_state['concept_matrix']}")
        try:
            concept_data = parser.parse(response)
            concept_matrix = ConceptMatrix(**concept_data.dict())
            logger.info(" Concept extraction completed.")
        except Exception as e:
            logger.warning(f"Parser failed: {e}, falling back to manual parsing")
            concept_matrix = self._parse_concept_response(response)
        
        # initial_state["alo"] = concept_matrix
        return {"concept_matrix": concept_matrix}

    def step2_keyword_generation(self, initial_state: ExtractionState) -> ExtractionState:
        """Step 2: Generate main keywords for each field from summary (exact same logic as original)"""
        logger.info(" Step 2: Keyword generation...")
        
        if initial_state["seed_keywords"]:
            logger.info(" Seed keywords already exists, skipping keyword generation")
            return {"seed_keywords": initial_state["seed_keywords"]}
        
        concept_matrix = initial_state["concept_matrix"]
        # logger.info(f"concept_matrix: {concept_matrix}")
        feedback = ""
        if initial_state.get("validation_feedback") and getattr(initial_state["validation_feedback"], "feedback", None):
            feedback = initial_state["validation_feedback"].feedback

        prompt, parser = self.prompts.get_phase2_prompt_and_parser()
        response = self.llm.invoke(prompt)
        
        try:
            keyword_data = parser.parse(response)
            seed_keywords = SeedKeywords(**keyword_data.dict())
            logger.info(" Keyword generation completed.")
        except Exception as e:
            logger.warning(f"Parser failed: {e}, falling back to manual parsing")
            seed_keywords = self._parse_keyword_response(response)
        
        # initial_state["seed_keywords"] = seed_keywords
        return {"seed_keywords": seed_keywords}
    
    def step3_human_evaluation(self, initial_state: ExtractionState) -> ExtractionState:
        """Step 3: Human in the loop evaluation with three options (exact same logic as original)"""
        logger.info(" Step 3: Human evaluation...")
        
        # If custom evaluation handler is provided (e.g., for Streamlit UI), use it
        logger.info(f"state: {initial_state}")
        if self.custom_evaluation_handler:
            logger.info(f" Using custom evaluation handler...")
            return self.custom_evaluation_handler(initial_state)
        
        # Otherwise, use the original command-line interface
        msgs = self.validation_messages
        
        print("\n" + msgs["separator"])
        print(msgs["final_evaluation_title"])
        print(msgs["separator"])
        
        # Display final results
        concept_matrix = initial_state["concept_matrix"]
        seed_keywords = initial_state["seed_keywords"]
        
        print(msgs["concept_matrix_header"])
        for field, value in concept_matrix.dict().items():
            print(f"  • {field.replace('_', ' ').title()}: {value}")
        
        print(msgs["seed_keywords_header"])
        for field, keywords in seed_keywords.dict().items():
            print(f"  • {field.replace('_', ' ').title()}: {keywords}")
        
        print(msgs["divider"])
        print(msgs["action_options"])
        
        # Get user action
        while True:
            action = input(msgs["action_prompt"]).lower().strip()
            if action in ['1', 'approve', 'a']:
                feedback = ValidationFeedback(action="approve")
                break
            elif action in ['2', 'reject', 'r']:
                feedback_text = input(msgs["reject_feedback_prompt"])
                feedback = ValidationFeedback(action="reject", feedback=feedback_text)
                break
            elif action in ['3', 'edit', 'e']:
                feedback = self._get_manual_edits(seed_keywords)
                break
            else:
                print(msgs["invalid_action"])
        
        
        # initial_state["validation_feedback"] = feedback
        return {"validation_feedback": feedback}

    def manual_editing(self, initial_state: ExtractionState) -> ExtractionState:
        """Allow user to manually edit keywords (exact same as original)"""
        logger.info(" Manual editing...")
        feedback = initial_state["validation_feedback"]
        
        if feedback.edited_keywords:
            # initial_state["seed_keywords"] = feedback.edited_keywords
            return {"seed_keywords": feedback.edited_keywords}
        
        return {"seed_keywords": feedback.edited_keywords}
    
    def _get_manual_edits(self, current_keywords: SeedKeywords) -> ValidationFeedback:
        """Get manual edits from user (exact same as original)"""
        logger.info(" Manual Editing Mode")
        print("\n Manual Editing Mode")
        print("Current keywords will be displayed. Press Enter to keep current value, or type new keywords separated by commas.")
        
        edited_data = {}
        
        for field, keywords in current_keywords.dict().items():
            field_name = field.replace('_', ' ').title()
            current_str = ", ".join(keywords)
            print(f"\n{field_name}: [{current_str}]")
            
            new_input = input(f"New {field_name} (or Enter to keep): ").strip()
            if new_input:
                edited_data[field] = [kw.strip() for kw in new_input.split(',') if kw.strip()]
            else:
                edited_data[field] = keywords
        
        edited_keywords = SeedKeywords(**edited_data)
        return ValidationFeedback(action="edit", edited_keywords=edited_keywords)
      
    def _parse_concept_response(self, response: str) -> ConceptMatrix:
        """Parse response when JSON parsing fails (exact same as original)"""
        lines = response.strip().split('\n')
        data = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_').replace('/', '_')
                if 'problem' in key or 'purpose' in key:
                    data['problem_purpose'] = value.strip()
                elif 'object' in key or 'system' in key:
                    data['object_system'] = value.strip()
                elif 'environment' in key or 'field' in key:
                    data['environment_field'] = value.strip()
        
        return ConceptMatrix(**data)
    
    def _parse_keyword_response(self, response: str) -> SeedKeywords:
        """Parse keyword response when JSON parsing fails (exact same as original)"""
        # Fallback parsing logic
        return SeedKeywords(
            problem_purpose=["water_optimization", "irrigation_control"],
            object_system=["IoT_sensors", "smart_irrigation_system"],
            environment_field=["agriculture", "precision_farming"],
        )
    
    def _get_human_action(self, state: ExtractionState) -> str:
        """Get the human action from validation feedback (exact same as original)"""
        feedback = state["validation_feedback"]
        return feedback.action if feedback else "approve"
    
    def gen_key(self, initial_state: ExtractionState) -> ExtractionState:
        """Generate synonyms and related terms for keywords (mock version with same structure)"""
        logger.info(" Generating synonyms and related terms...")
        
        def search_snippets(keyword: str, max_snippets: int = 3) -> List[str]:
            results = self.tavily_search.invoke({"query": keyword})
            snippets = [r['content'] for r in results.get("results", [])[:max_snippets]]
            return snippets

        sys_keys = {}

        def generate_synonyms(keyword: str, context: str):
            logger.info(f" Searching snippets for keyword: {keyword}")
            snippets = search_snippets(keyword)
            if not snippets:
                logger.warning(f" No snippets found for keyword: {keyword}")
                return

            # Mock synonym generation with constant data
            mock_synonyms = {
                "water optimization": ["irrigation efficiency", "water conservation", "moisture control"],
                "irrigation control": ["watering management", "irrigation automation", "water distribution"],
                "IoT sensors": ["smart sensors", "wireless sensors", "connected devices"],
                "smart irrigation system": ["automated irrigation", "intelligent watering", "precision irrigation"],
                "agriculture": ["farming", "crop production", "agricultural sector"],
                "precision farming": ["smart agriculture", "digital farming", "precision agriculture"]
            }
            
            # Use mock data or generate generic synonyms
            synonyms = mock_synonyms.get(keyword, [f"{keyword}_synonym1", f"{keyword}_synonym2", f"{keyword}_related"])
            sys_keys[keyword] = synonyms
            logger.info(f" Generated {len(synonyms)} terms for '{keyword}': {synonyms}")

        concept_matrix = initial_state["concept_matrix"].dict()
        seed_keywords = initial_state["seed_keywords"].dict()

        for context in concept_matrix:
            for key in seed_keywords[context]:
                generate_synonyms(key, concept_matrix[context])

        # initial_state["final_keywords"] = sys_keys
        return {"final_keywords": sys_keys}

    def summary_prompt_and_parser(self, initial_state: ExtractionState) -> ExtractionState:
        """Generate summary using prompt and parser (exact same logic as original)"""
        logger.info(" Generating summary...")
        
        prompt, parser = self.prompts.get_summary_prompt_and_parser()
        response = self.llm.invoke(prompt)
        concept_data = parser.parse(response)
        
        logger.info(" Summary generation completed.")
        # initial_state["summary_text"] = concept_data
        return {"summary_text": concept_data}

    def call_ipcs_api(self, initial_state: ExtractionState) -> ExtractionState:
        """Call IPC classification API (mock version)"""
        logger.info(" Calling IPC classification API...")
        
        ipcs = mock_get_ipc_predictions(initial_state["summary_text"])
        logger.info(f" IPC classification results: {ipcs}")
        # initial_state["ipcs"] = ipcs
        return {"ipcs": ipcs}

    def genQuery(self, initial_state: ExtractionState) -> ExtractionState:
        """Generate search queries (exact same logic as original)"""
        logger.info(" Generating search queries...")
        
        keys = initial_state["seed_keywords"]
        problem_purpose_keys = str([i for key in keys.problem_purpose for i in initial_state["final_keywords"][key]])
        object_system_keys = str([i for key in keys.object_system for i in initial_state["final_keywords"][key]])
        environment_field_keys = str([i for key in keys.environment_field for i in initial_state["final_keywords"][key]])
        fipc = str([i["category"] for i in initial_state["ipcs"]])
        problem = initial_state.get("problem", "")

        prompt, parser = self.prompts.get_queries_prompt_and_parser()
        response = self.llm.invoke(prompt)
        concept_data = parser.parse(response)
        
        logger.info(f" Generated {len(concept_data.queries)} search queries")
        # initial_state["queries"] = concept_data
        return {"queries": concept_data}

    def genUrl(self, initial_state: ExtractionState) -> ExtractionState:
        """Generate URLs from queries using Brave search (mock version)"""
        logger.info(" Generating URLs from search queries...")
        
        # Mock patent URLs
        mock_patent_urls = [
            "https://patents.google.com/patent/US10123456B2",
            "https://patents.google.com/patent/US10234567B2", 
            "https://patents.google.com/patent/US10345678B2",
            "https://patents.google.com/patent/US10456789B2",
            "https://patents.google.com/patent/US10567890B2",
            "https://patents.google.com/patent/US10678901B2",
            "https://patents.google.com/patent/US10789012B2"
        ]
        
        final_url = []
        queries = initial_state["queries"].queries
        logger.info(f" Searching for URLs using {len(queries)} queries")
        
        # Simulate search for each query
        for query in queries:
            time.sleep(0.5)  # Simulate API call delay
            # Add 2-3 mock URLs per query
            for i in range(random.randint(2, 3)):
                if mock_patent_urls:
                    final_url.append(mock_patent_urls.pop(0))

        logger.info(f" Found {len(final_url)} URLs from search results")
        # initial_state["final_url"] = final_url
        return {"final_url": final_url}

    def evalUrl(self, initial_state: ExtractionState) -> ExtractionState:
        """Evaluate URLs for relevance (mock version with same structure)"""
        logger.info(" Evaluating URLs for relevance...")
        
        final_url = []
        urls_to_evaluate = initial_state["final_url"]
        logger.info(f" Evaluating {len(urls_to_evaluate)} URLs for relevance")
        
        for url in urls_to_evaluate:
            temp_score = dict()
            temp_score['url'] = url 
            temp_score['user_scenario'] = 0
            temp_score['user_problem'] = 0
            
            try:
                # Mock evaluation process
                time.sleep(random.uniform(0.5, 1.0))  # Simulate processing time
                
                result = mock_parse_idea_input(initial_state["input_text"])
                temp = mock_lay_thong_tin_patent(url)
                ex_text = mock_prompt(temp['abstract'], temp['description'], temp['claims'])
                res = self.llm.invoke(ex_text)
                
                # Mock evaluation scores
                score_scenario = mock_eval_url(result["user_scenario"], "mock patent scenario")
                score_problem = mock_eval_url(result["user_problem"], "mock patent problem")
                
                temp_score['user_scenario'] = score_scenario['llm_score']
                temp_score['user_problem'] = score_problem['llm_score']
                final_url.append(temp_score)
                
                logger.info(f" Evaluated URL: {url} (scenario: {temp_score['user_scenario']:.3f}, problem: {temp_score['user_problem']:.3f})")
                
            except Exception as e:
                logger.error(f"❌ Error evaluating URL {url}: {str(e)}")
                # Add URL with zero scores if evaluation fails
                final_url.append(temp_score)
        
        logger.info(f" Completed evaluation of {len(final_url)} URLs")
        # initial_state["final_url"] = final_url
        return {"final_url": final_url}

# Export the enhanced mock extractor
__all__ = ['EnhancedMockCoreConceptExtractor', 'ValidationFeedback', 'SeedKeywords', 'ConceptMatrix', 'NormalizationOutput']
