"""
Main entry point for the Patent AI Agent
Provides a simple interface to run the patent keyword extraction system
"""

import json
import datetime
import os
import logging
from typing import Any, Dict, List, Literal, Optional, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from typing import Dict, List, TypedDict, Annotated, Optional
from langchain_community.llms import Ollama
# from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain_tavily import TavilySearch
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
import requests
import gradio as gr

# Configure logging
log_filename = f"patent_extractor_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import settings from config
from config.settings import settings

# Local imports with updated paths
from src.api.ipc_classifier import get_ipc_predictions
from src.prompts.extraction_prompts import ExtractionPrompts
from src.crawling.patent_crawler import lay_thong_tin_patent
from src.evaluation.similarity_evaluator import (
    eval_url, prompt, parse_idea_text, parse_idea_input, extract_user_info
)

# Set up Tavily API key from settings
os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY

# Data Models
class NormalizationOutput(BaseModel):
    problem: str = Field(description="Normalized technical problem or objective described in the document.")
    technical: str = Field(description="Normalized technical content or context of the document.")

class ConceptMatrix(BaseModel):
    problem_purpose: str = Field(description="The specific technical problem the invention aims to solve or the primary objective described in the document.")
    object_system: str = Field(description="The main object, device, system, material, or process that is the subject of the invention as stated in the document.")
    environment_field: str = Field(description="The application domain, industry sector, or operational context where the invention is intended to be used.")

class SeedKeywords(BaseModel):
    problem_purpose: List[str] = Field(description="Distinctive technical keywords describing the technical problem addressed or primary objective.")
    object_system: List[str] = Field(description="Technical keywords specifying the main object, device, system, material, or process described.")
    environment_field: List[str] = Field(description="Keywords identifying the application domain, industry sector, or operational context.")

class ValidationFeedback(BaseModel):
    action: str  # "approve", "edit", "reject"
    edited_keywords: Optional[SeedKeywords] = None
    feedback: Optional[str] = None

class ReflectionEvaluation(BaseModel):
    overall_quality: str = Field(description="Overall quality assessment: 'good' or 'poor'")
    keyword_scores: Dict[str, float] = Field(description="Score for each category (0-1)")
    issues_found: List[str] = Field(description="List of specific issues identified")
    recommendations: List[str] = Field(description="Recommendations for improvement")
    should_regenerate: bool = Field(description="Whether keywords should be regenerated")

class ExtractionState(TypedDict):
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

class CoreConceptExtractor:
    def __init__(self, model_name: str = None, use_checkpointer: bool = None):
        self.model_name = model_name if model_name is not None else settings.DEFAULT_MODEL_NAME
        self.use_checkpointer = use_checkpointer if use_checkpointer is not None else settings.USE_CHECKPOINTER
        self.llm = Ollama(
            model=self.model_name,
            temperature=settings.MODEL_TEMPERATURE,
            num_ctx=settings.NUM_CTX
        )
        # self.llm = ChatOpenAI(
        #     model_name="qwen/qwen-2.5-72b-instruct:free",
        #     temperature=0.7,
        #     openai_api_key="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",
        #     base_url="https://openrouter.ai/api/v1"
        # )
        self.tavily_search = TavilySearch(
            max_results=settings.MAX_SEARCH_RESULTS,
            topic="general",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
            include_image_descriptions=False,
        )
        self.prompts = ExtractionPrompts()
        self.messages = ExtractionPrompts.get_phase_completion_messages()
        self.validation_messages = ExtractionPrompts.get_validation_messages()
        self.use_checkpointer = use_checkpointer
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ExtractionState)
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

        workflow.set_entry_point("input_normalization")
        workflow.add_edge("input_normalization", "step0")
        workflow.add_edge("step0", "step1_concept_extraction")
        workflow.add_edge("step0", "summary_prompt_and_parser")
        workflow.add_edge("step1_concept_extraction", "step2_keyword_generation")
        workflow.add_edge("step2_keyword_generation", "step3_human_evaluation")
        workflow.add_edge("summary_prompt_and_parser", "call_ipcs_api")
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
    
    def extract_keywords(self, input_text: str) -> Dict:
        initial_state = ExtractionState(
            input_text=input_text,
            problem=None,
            technical=None,
            concept_matrix=None,
            seed_keywords=None,
            validation_feedback=None,
            final_keywords=None,
            ipcs=None,
            summary_text=None,
            queries=None,
            final_url=None
        )
        if self.use_checkpointer:
            config = {"configurable": {"thread_id": settings.THREAD_ID}}
            result = self.graph.invoke(initial_state, config)
        else:
            result = self.graph.invoke(initial_state)
        return dict(result)
        
    def input_normalization(self, state: ExtractionState) -> ExtractionState:
        prompt, parser = self.prompts.get_normalization_prompt_and_parser()
        response = self.llm.invoke(prompt.format(input=state["input_text"]))
        try:
            normalized_data = parser.parse(response)
            normalized_input = NormalizationOutput(**normalized_data.dict())
            logger.info("Normalization completed.")
            updated_state = {
                "problem": normalized_input.problem,
                "technical": normalized_input.technical,
                "input_text": state["input_text"]
            }
            logger.info(f"📝 Normalized problem: {normalized_input.problem}")
            logger.info(f"📝 Normalized technical: {normalized_input.technical}")
            return updated_state
        except Exception as e:
            logger.error(f"⚠️ Normalization parsing failed: {e}, using original input")
            fallback_normalized = NormalizationOutput(
                problem="Not mentioned.",
                technical="Not mentioned."
            )
            return {
                "problem": "Not mentioned.",
                "technical": "Not mentioned.",
                "input_text": state["input_text"]
            }

    def step0(self, state: ExtractionState) -> ExtractionState:
        return state

    def step1_concept_extraction(self, state: ExtractionState) -> ExtractionState:
        prompt, parser = self.prompts.get_phase1_prompt_and_parser()
        response = self.llm.invoke(prompt.format(problem=state["problem"]))
        try:
            concept_data = parser.parse(response)
            concept_matrix = ConceptMatrix(**concept_data.dict())
        except Exception as e:
            logger.warning(f"Parser failed: {e}, falling back to manual parsing")
            concept_matrix = self._parse_concept_response(response)
        return {"concept_matrix": concept_matrix}

    def step2_keyword_generation(self, state: ExtractionState) -> ExtractionState:
        concept_matrix = state["concept_matrix"]
        feedback = ""
        if state.get("validation_feedback") and getattr(state["validation_feedback"], "feedback", None):
            feedback = state["validation_feedback"].feedback
        prompt, parser = self.prompts.get_phase2_prompt_and_parser()
        response = self.llm.invoke(prompt.format(
            problem_purpose=concept_matrix.problem_purpose,
            object_system=concept_matrix.object_system,
            environment_field=concept_matrix.environment_field,
            feedback=feedback
        ))
        try:
            keyword_data = parser.parse(response)
            seed_keywords = SeedKeywords(**keyword_data.dict())
        except Exception as e:
            logger.warning(f"Parser failed: {e}, falling back to manual parsing")
            seed_keywords = self._parse_keyword_response(response)
        return {"seed_keywords": seed_keywords}
    
    def step3_human_evaluation(self, state: ExtractionState, feedback: ValidationFeedback = None) -> ExtractionState:
        # Accept feedback as a parameter from the main Gradio UI, do not launch Gradio here
        if feedback is None:
            feedback = ValidationFeedback(action="approve")
        state["validation_feedback"] = feedback
        return {"validation_feedback": feedback}

    def manual_editing(self, state: ExtractionState) -> ExtractionState:
        feedback = state["validation_feedback"]
        if feedback.edited_keywords:
            state["seed_keywords"] = feedback.edited_keywords
        return {"seed_keywords": feedback.edited_keywords}
    
    def _get_manual_edits(self, current_keywords: SeedKeywords) -> ValidationFeedback:
        return ValidationFeedback(action="edit", edited_keywords=current_keywords)
      
    def _parse_concept_response(self, response: str) -> ConceptMatrix:
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
        return SeedKeywords(
            problem_purpose=["extracted_keyword"],
            object_system=["extracted_keyword"],
            environment_field=["extracted_keyword"],
        )
    
    def _get_human_action(self, state: ExtractionState) -> str:
        feedback = state["validation_feedback"]
        return feedback.action if feedback else "approve"
    
    def gen_key(self, state: ExtractionState) -> ExtractionState:
        def search_snippets(keyword: str, max_snippets: int = 3) -> List[str]:
            try:
                results = self.tavily_search.invoke({"query": keyword})
                snippets = [r['content'] for r in results.get("results", [])[:max_snippets]]
                return snippets
            except Exception as e:
                logger.error(f"Error searching snippets for keyword '{keyword}': {e}")
                return []
        prompt, parser = self.prompts.get_synonym_extraction_prompt_and_parser()
        sys_keys = {}
        def generate_synonyms(keyword: str, context: str):
            logger.info(f"🔍 Searching snippets for keyword: {keyword}")
            snippets = search_snippets(keyword)
            if not snippets:
                logger.warning(f"❌ No snippets found for keyword: {keyword}")
                return
            formatted_snippets = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(snippets)])
            response = self.llm.invoke(prompt.format(
                keyword=keyword,
                context=context,
                snippets=formatted_snippets
            ))
            try:
                parsed_result = parser.parse(response)
                res = []
                if hasattr(parsed_result, "core_synonyms"):
                    res.extend([item.term for item in parsed_result.core_synonyms])
                if hasattr(parsed_result, "related_terms"):
                    res.extend([item.term for item in parsed_result.related_terms])
                sys_keys[keyword] = res
                logger.info(f"✅ Extracted {len(res)} terms for '{keyword}': {res}")
            except Exception as e:
                logger.warning(f"⚠️ Structured parsing failed for '{keyword}': {e}")
                logger.debug(f"Raw result: {response}")
                syn_raw = response
                raws = syn_raw.split("\n")
                st = 0
                et = 0
                for i in range(len(raws)):
                    if "A. Core Synonyms" in raws[i]:
                        st = i+1
                        for j in range(st+1, len(raws)):
                            if "B. Related Terms" in raws[j]:
                                et = j
                                break
                        break
                raws = raws[st:et]
                res = []
                for item in raws:
                    try:
                        res.append(item.split("—")[0].split(".")[1].strip())
                    except:
                        pass
                sys_keys[keyword] = res
        concept_matrix = state["concept_matrix"].dict()
        seed_keywords = state["seed_keywords"].dict()
        for context in concept_matrix:
            for key in seed_keywords[context]:
                generate_synonyms(key, concept_matrix[context])
        return {"final_keywords": sys_keys}

    def summary_prompt_and_parser(self, state: ExtractionState) -> ExtractionState:
        prompt, parser = self.prompts.get_summary_prompt_and_parser()
        response = self.llm.invoke(prompt.format(idea=state["input_text"]))
        concept_data = parser.parse(response)
        return {"summary_text": concept_data}

    def call_ipcs_api(self, state: ExtractionState) -> ExtractionState:
        ipcs = get_ipc_predictions(state["summary_text"])
        logger.info(f"📋 IPC classification results: {ipcs}")
        return {"ipcs": ipcs}

    def genQuery(self, state: ExtractionState) -> ExtractionState:
        keys = state["seed_keywords"]
        problem_purpose_keys = str([i for key in keys.problem_purpose for i in state["final_keywords"][key]])
        object_system_keys = str([i for key in keys.object_system for i in state["final_keywords"][key]])
        environment_field_keys = str([i for key in keys.environment_field for i in state["final_keywords"][key]])
        fipc = str([i["category"] for i in state["ipcs"]])
        problem = state.get("problem", "")
        prompt, parser = self.prompts.get_queries_prompt_and_parser()
        response = self.llm.invoke(prompt.format(
            problem=problem,
            problem_purpose_keys=problem_purpose_keys,
            object_system_keys=object_system_keys,
            environment_field_keys=environment_field_keys,
            CPC_CODES=fipc
        ))
        concept_data = parser.parse(response)
        logger.info(f"🔍 Generated {len(concept_data.queries)} search queries")
        return {"queries": concept_data}

    def genUrl(self, state: ExtractionState) -> ExtractionState:
        final_url = list()
        queries = state["queries"].queries
        logger.info(f"🌐 Searching for URLs using {len(queries)} queries")
        for query in queries:
            url = "https://api.search.brave.com/res/v1/web/search"
            params = {
                "q": query + " site:patents.google.com/"
            }
            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": "BSAQlxb-jIHFbW1mK0_S4zlTqfkuA3Z"
            }
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                try:
                    for i in data["web"]["results"]:
                        url = i.get("url", None)
                        if url:
                            final_url.append(url)
                except:
                    logger.warning(f"❌ No results found for query: {query}")
            else:
                logger.error(f"❌ Search API request failed for query: {query} (status: {response.status_code})")
        logger.info(f"🔗 Found {len(final_url)} URLs from search results")
        return {"final_url": final_url}

    def evalUrl(self, state: ExtractionState) -> ExtractionState:
        final_url = list()
        logger.info(f"📊 Evaluating {len(state['final_url'])} URLs for relevance")
        for url in state["final_url"]:
            temp_score = dict()
            temp_score['url'] = url 
            temp_score['user_scenario'] = 0
            temp_score['user_problem'] = 0
            try:
                result = parse_idea_input(state["input_text"])
                temp = lay_thong_tin_patent(url)
                ex_text = prompt(temp['abstract'], temp['description'], temp['claims'])
                res = self.llm.invoke(ex_text)
                logger.debug(f"📄 LLM evaluation response for {url}: {res}")
                res = res.replace("```json", '')
                res = res.replace("```", '')
                data_res = json.loads(res)
                res_data = extract_user_info(data_res)
                score_scenario = eval_url(result["user_scenario"], res_data['user_scenario'])
                score_problem = eval_url(result["user_problem"], res_data['user_problem'])
                temp_score['user_scenario'] = score_scenario['llm_score']
                temp_score['user_problem'] = score_problem['llm_score']
                final_url.append(temp_score)
                logger.info(f"✅ Evaluated URL: {url} (scenario: {temp_score['user_scenario']}, problem: {temp_score['user_problem']})")
            except Exception as e:
                logger.error(f"❌ Error evaluating URL {url}: {str(e)}")
                final_url.append(temp_score)
        logger.info(f"📊 Completed evaluation of {len(final_url)} URLs")
        return {"final_url": final_url}

extractor = CoreConceptExtractor()

def run_extraction(problem, technical):
    input_text = f"Problem: {problem}\nTechnical: {technical}"
    results = extractor.extract_keywords(input_text)
    output = ""
    for key, value in results.items():
        if value is None:
            continue
        output += f"### {key}\n"
        if hasattr(value, "dict"):
            for subkey, subval in value.dict().items():
                output += f"- **{subkey.replace('_', ' ').title()}**: {subval}\n"
        elif isinstance(value, dict):
            for subkey, subval in value.items():
                output += f"- **{subkey}**: {subval}\n"
        elif isinstance(value, list):
            for i, item in enumerate(value, 1):
                if isinstance(item, dict):
                    output += f"{i}. " + ", ".join([f"{k}: {v}" for k, v in item.items()]) + "\n"
                else:
                    output += f"{i}. {item}\n"
        else:
            output += f"{value}\n"
    return output.strip()

with gr.Blocks(theme="default") as demo:
    gr.Markdown("## Patent Keyword Extraction")
    gr.Markdown("Nhập thông tin vào 2 ô bên dưới. Kết quả sẽ hiển thị đầy đủ, đơn giản và dễ nhìn.")
    with gr.Row():
        problem = gr.Textbox(label="Problem", lines=2, placeholder="Nhập vấn đề kỹ thuật...")
        technical = gr.Textbox(label="Technical", lines=2, placeholder="Nhập nội dung kỹ thuật...")
    output = gr.Markdown()
    submit_btn = gr.Button("Extract")

    submit_btn.click(
        run_extraction,
        inputs=[problem, technical],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
