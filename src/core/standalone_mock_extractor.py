"""
Standalone Mock Patent Concept Extractor
Maintains the exact multi-agent architecture logic from extractor.py but uses constant data
This version can run without external dependencies for testing
"""

import json
import datetime
import time
import random
import logging
from typing import Any, Dict, List, Optional, TypedDict

# Configure logging
log_filename = f"standalone_mock_extractor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for environments without pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def Field(**kwargs):
        return None

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

class QueriesResponse(BaseModel):
    """Output model for patent search queries"""
    queries: List[str] = Field(
        description="List of queries. Leave empty if none."
    )

class ExtractionState(dict):
    """Simplified state for workflow (compatible with original)"""
    pass

class MockPrompts:
    """Mock prompts that return structured parsers"""
    
    def get_normalization_prompt_and_parser(self):
        class MockParser:
            def parse(self, response):
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
                return "A smart irrigation system integrating Internet of Things (IoT) sensors for real-time soil moisture monitoring and automated water distribution control."
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
    """Mock LLM that returns constant responses"""
    
    def __init__(self, model="mock-llm", temperature=0.7, num_ctx=128000):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
    
    def invoke(self, prompt: str) -> str:
        """Return mock responses based on prompt context"""
        time.sleep(random.uniform(0.5, 1.5))
        
        mock_responses = {
            "normalization": '{"problem": "Optimize water usage", "technical": "Smart irrigation system"}',
            "concept": '{"problem_purpose": "Water optimization", "object_system": "IoT irrigation system", "environment_field": "Agriculture"}',
            "keywords": '{"problem_purpose": ["water optimization"], "object_system": ["IoT sensors"], "environment_field": ["agriculture"]}',
            "summary": '"Smart irrigation system with IoT sensors"',
            "queries": '{"queries": ["irrigation IoT sensors"]}',
            "synonyms": '{"core_synonyms": [{"term": "watering system", "justification": "irrigation synonym", "source": "src 1"}], "related_terms": [{"term": "drip irrigation", "rationale": "irrigation method", "source": "src 2"}]}'
        }
        
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

class StandaloneMockCoreConceptExtractor:
    """Standalone Mock Patent seed keyword extraction system"""
    
    def __init__(self, model_name: str = None, use_checkpointer: bool = None, custom_evaluation_handler=None):
        """Initialize the StandaloneMockCoreConceptExtractor."""
        self.model_name = model_name or "mock-llm"
        self.use_checkpointer = use_checkpointer or False
        self.custom_evaluation_handler = custom_evaluation_handler

        # Mock components
        self.llm = MockLLM(model=self.model_name)
        self.tavily_search = MockTavilySearch(max_results=5)
        self.prompts = MockPrompts()
        self.messages = MockPrompts.get_phase_completion_messages()
        self.validation_messages = MockPrompts.get_validation_messages()
        
        # Mock data for consistent results
        self.mock_ipcs = [
            {"category": "A01G25/16", "score": 0.95},
            {"category": "G05B15/02", "score": 0.87},
            {"category": "H04L12/28", "score": 0.82}
        ]
        
        self.mock_urls = [
            "https://patents.google.com/patent/US10123456B2",
            "https://patents.google.com/patent/US10234567B2", 
            "https://patents.google.com/patent/US10345678B2",
            "https://patents.google.com/patent/US10456789B2",
            "https://patents.google.com/patent/US10567890B2"
        ]
    
    def extract_keywords(self, input_text: str) -> Dict:
        """Run the complete mock extraction workflow following original architecture"""
        logger.info("🔄 Starting standalone mock extraction workflow...")
        
        # Initialize state
        state = ExtractionState({
            'input_text': input_text,
            'problem': None,
            'technical': None,
            'concept_matrix': None,
            'seed_keywords': None,
            'validation_feedback': None,
            'final_keywords': {},
            'ipcs': None,
            'summary_text': None,
            'queries': None,
            'final_url': []
        })
        
        # Execute workflow steps in order (matching original graph structure)
        logger.info("📝 Step: Input normalization...")
        state.update(self.input_normalization(state))
        
        logger.info("🔄 Step: Step0...")
        state.update(self.step0(state))
        
        # Parallel execution of step1 and summary (as in original)
        logger.info("🎯 Step: Concept extraction...")
        state.update(self.step1_concept_extraction(state))
        
        logger.info("📋 Step: Summary generation...")
        state.update(self.summary_prompt_and_parser(state))
        
        logger.info("🔑 Step: Keyword generation...")
        state.update(self.step2_keyword_generation(state))
        
        logger.info("👤 Step: Human evaluation...")
        state.update(self.step3_human_evaluation(state))
        
        # Handle feedback
        if state.get("validation_feedback"):
            if state["validation_feedback"].action == "reject":
                logger.info("🔄 Restarting due to rejection...")
                return self.extract_keywords(input_text)  # Recursive call
            elif state["validation_feedback"].action == "edit":
                logger.info("✏️ Step: Manual editing...")
                state.update(self.manual_editing(state))
        
        logger.info("📋 Step: IPC classification...")
        state.update(self.call_ipcs_api(state))
        
        logger.info("🔍 Step: Generate synonyms...")
        state.update(self.gen_key(state))
        
        logger.info("🔍 Step: Generate queries...")
        state.update(self.genQuery(state))
        
        logger.info("🌐 Step: Generate URLs...")
        state.update(self.genUrl(state))
        
        logger.info("📊 Step: Evaluate URLs...")
        state.update(self.evalUrl(state))
        
        logger.info("✅ Standalone mock extraction completed!")
        return dict(state)
    
    # All the individual step methods (exact same logic as enhanced version)
    def input_normalization(self, state: ExtractionState) -> ExtractionState:
        """Normalize and clean input text before processing"""    
        prompt, parser = self.prompts.get_normalization_prompt_and_parser()
        response = self.llm.invoke(prompt)

        try:
            normalized_data = parser.parse(response)
            normalized_input = NormalizationOutput(**normalized_data.dict())
            logger.info("✅ Normalization completed.")

            return {
                "problem": normalized_input.problem,
                "technical": normalized_input.technical,
                "input_text": state["input_text"]
            }

        except Exception as e:
            logger.error(f"⚠️ Normalization parsing failed: {e}, using fallback")
            return {
                "problem": "Not mentioned.",
                "technical": "Not mentioned.",
                "input_text": state["input_text"]
            }

    def step0(self, state: ExtractionState) -> ExtractionState:
        """Initial step - pass through state"""
        return {}

    def step1_concept_extraction(self, state: ExtractionState) -> ExtractionState:
        """Step 1: Extract concept summary from document"""
        prompt, parser = self.prompts.get_phase1_prompt_and_parser()
        response = self.llm.invoke(prompt)
        
        try:
            concept_data = parser.parse(response)
            concept_matrix = ConceptMatrix(**concept_data.dict())
            logger.info("✅ Concept extraction completed.")
        except Exception as e:
            logger.warning(f"Parser failed: {e}, using fallback")
            concept_matrix = ConceptMatrix(
                problem_purpose="Water optimization in irrigation",
                object_system="Smart IoT irrigation system",
                environment_field="Agriculture and farming"
            )
        
        return {"concept_matrix": concept_matrix}

    def step2_keyword_generation(self, state: ExtractionState) -> ExtractionState:
        """Step 2: Generate main keywords for each field"""
        prompt, parser = self.prompts.get_phase2_prompt_and_parser()
        response = self.llm.invoke(prompt)
        
        try:
            keyword_data = parser.parse(response)
            seed_keywords = SeedKeywords(**keyword_data.dict())
            logger.info("✅ Keyword generation completed.")
        except Exception as e:
            logger.warning(f"Parser failed: {e}, using fallback")
            seed_keywords = SeedKeywords(
                problem_purpose=["water optimization", "irrigation control"],
                object_system=["IoT sensors", "smart irrigation"],
                environment_field=["agriculture", "farming"]
            )
        
        return {"seed_keywords": seed_keywords}
    
    def step3_human_evaluation(self, state: ExtractionState) -> ExtractionState:
        """Step 3: Human evaluation"""
        if self.custom_evaluation_handler:
            return self.custom_evaluation_handler(state)
        
        # Auto-approve for non-interactive mode
        feedback = ValidationFeedback(action="approve")
        return {"validation_feedback": feedback}

    def manual_editing(self, state: ExtractionState) -> ExtractionState:
        """Allow user to manually edit keywords"""
        feedback = state["validation_feedback"]
        if feedback.edited_keywords:
            return {"seed_keywords": feedback.edited_keywords}
        return {}
    
    def gen_key(self, state: ExtractionState) -> ExtractionState:
        """Generate synonyms and related terms"""
        final_keywords = {}
        
        # Mock synonym generation
        mock_synonyms = {
            "water optimization": ["irrigation efficiency", "water conservation", "moisture control"],
            "irrigation control": ["watering management", "irrigation automation", "water distribution"],
            "IoT sensors": ["smart sensors", "wireless sensors", "connected devices"],
            "smart irrigation": ["automated irrigation", "intelligent watering", "precision irrigation"],
            "agriculture": ["farming", "crop production", "agricultural sector"],
            "farming": ["agriculture", "cultivation", "crop growing"]
        }
        
        if state.get("seed_keywords"):
            for category, keywords in state["seed_keywords"].dict().items():
                for keyword in keywords:
                    synonyms = mock_synonyms.get(keyword, [f"{keyword}_synonym1", f"{keyword}_related"])
                    final_keywords[keyword] = synonyms
                    logger.info(f"✅ Generated {len(synonyms)} terms for '{keyword}': {synonyms}")

        return {"final_keywords": final_keywords}

    def summary_prompt_and_parser(self, state: ExtractionState) -> ExtractionState:
        """Generate summary"""
        prompt, parser = self.prompts.get_summary_prompt_and_parser()
        response = self.llm.invoke(prompt)
        concept_data = parser.parse(response)
        return {"summary_text": concept_data}

    def call_ipcs_api(self, state: ExtractionState) -> ExtractionState:
        """Call IPC classification API"""
        time.sleep(0.5)  # Simulate API call
        return {"ipcs": self.mock_ipcs}

    def genQuery(self, state: ExtractionState) -> ExtractionState:
        """Generate search queries"""
        prompt, parser = self.prompts.get_queries_prompt_and_parser()
        response = self.llm.invoke(prompt)
        concept_data = parser.parse(response)
        logger.info(f"🔍 Generated {len(concept_data.queries)} search queries")
        return {"queries": concept_data}

    def genUrl(self, state: ExtractionState) -> ExtractionState:
        """Generate URLs from queries"""
        time.sleep(1.0)  # Simulate search time
        final_url = self.mock_urls.copy()
        logger.info(f"🔗 Found {len(final_url)} URLs from search results")
        return {"final_url": final_url}

    def evalUrl(self, state: ExtractionState) -> ExtractionState:
        """Evaluate URLs for relevance"""
        final_url = []
        urls_to_evaluate = state["final_url"]
        
        for url in urls_to_evaluate:
            time.sleep(0.3)  # Simulate evaluation time
            temp_score = {
                'url': url,
                'user_scenario': random.uniform(0.6, 0.95),
                'user_problem': random.uniform(0.5, 0.9)
            }
            final_url.append(temp_score)
            logger.info(f"✅ Evaluated URL: {url} (scores: {temp_score['user_scenario']:.3f}, {temp_score['user_problem']:.3f})")
        
        logger.info(f"📊 Completed evaluation of {len(final_url)} URLs")
        return {"final_url": final_url}

# Export the standalone mock extractor
__all__ = ['StandaloneMockCoreConceptExtractor', 'ValidationFeedback', 'SeedKeywords', 'ConceptMatrix', 'NormalizationOutput']
