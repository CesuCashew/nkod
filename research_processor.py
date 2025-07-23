import os
from typing import Dict, List, TypedDict, Any, Optional, Union, Annotated
from langgraph.graph import StateGraph, END, add_messages
import ollama
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tqdm import tqdm
import json
from datetime import datetime

# Load environment variables
load_dotenv()

class ResearchResult(BaseModel):
    """Model for individual research result."""
    original_data: dict = Field(..., description="Original research data")
    analysis: str = Field(..., description="AI-generated analysis of the research")
    title: str = Field(..., description="Title of the research")
    year: int = Field(..., description="Year of the research")
    description: str = Field(..., description="Description of the research")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), 
                         description="When this analysis was performed")

class ResearchState(TypedDict):
    """State for the research processing workflow."""
    research_data: List[dict]
    processed_results: List[ResearchResult]
    current_result: Optional[dict]
    analysis: str
    current_index: int
    messages: Annotated[list, add_messages]  # For LangGraph Studio visualization

class ResearchProcessor:
    def __init__(self, model_name: str = None, max_results: int = None):
        """Initialize the research processor with an Ollama model."""
        self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'llama3')
        self.max_results = max_results  # Limit number of results to process
        self.system_prompt = os.getenv('SYSTEM_PROMPT', """
        You are an AI assistant that analyzes research results. 
        Your task is to process and summarize research findings from the provided data.
        Provide the response in Czech.
        """)
        
        # Initialize the graph
        self.workflow = StateGraph(ResearchState)
        self._setup_workflow()
    
    def _setup_workflow(self):
        """Set up the LangGraph workflow with visualization support."""
        # Add nodes with clear names for visualization
        self.workflow.add_node("initialize", self._initialize)
        self.workflow.add_node("process_result", self._process_result)
        self.workflow.add_node("analyze_results", self._analyze_results)
        
        # Define edges with conditions
        self.workflow.add_edge("initialize", "process_result")
        self.workflow.add_conditional_edges(
            "process_result",
            self._should_continue,
            {
                "continue": "process_result",
                "done": "analyze_results"
            }
        )
        self.workflow.add_edge("analyze_results", END)
        
        # Set entry point
        self.workflow.set_entry_point("initialize")
        
        # Compile the workflow with visualization metadata
        self.app = self.workflow.compile(
            check_pointer=None,  # Disable checkpoints for now
            debug=True  # Enable debug mode for better visualization
        )
    
    def _initialize(self, state: ResearchState) -> Dict:
        """Initialize the processing state."""
        return {
            "research_data": state["research_data"],
            "processed_results": [],
            "current_result": None,
            "analysis": "",
            "current_index": 0,
            "messages": [{"role": "system", "content": "Starting research analysis workflow"}]
        }
    
    def _should_continue(self, state: ResearchState) -> str:
        """Determine if we should continue processing more results."""
        current_idx = state.get("current_index", 0)
        total_results = len(state.get("research_data", []))
        
        if self.max_results and current_idx >= self.max_results:
            return "done"
        return "continue" if current_idx < total_results else "done"
    
    def _process_result(self, state: ResearchState) -> Dict:
        """Process a single research result with LangGraph Studio support."""
        current_idx = state.get("current_index", 0)
        research_data = state["research_data"]
        messages = state.get("messages", [])
        
        if current_idx >= len(research_data):
            return {"current_index": current_idx + 1}
        
        current_result = research_data[current_idx]
        
        # Prepare the prompt with metadata for visualization
        prompt = f"""
        Analyzuj následující výzkumný výsledek a poskytni souhrn v češtině:
        {str(current_result)[:2000]}...
        
        Uveďte:
        1. Stručný souhrn výzkumu
        2. Klíčové zjištění nebo výsledky
        3. Možné aplikace nebo důsledky
        """
        
        try:
            # Log the processing start
            messages.append({
                "role": "system",
                "content": f"Processing research result {current_idx + 1}/{len(research_data)}"
            })
            
            # Call Ollama with streaming for better visualization
            response = ""
            for chunk in ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            ):
                content = chunk.get('message', {}).get('content', '')
                if content:
                    response += content
                    # Update the last message with streaming content
                    if messages and messages[-1]["role"] == "assistant":
                        messages[-1]["content"] = response
                    else:
                        messages.append({"role": "assistant", "content": response})
            
            # Create a research result
            result = ResearchResult(
                original_data=current_result,
                analysis=response,
                title=current_result.get('title', 'Neznámý název'),
                year=current_result.get('year', 2023),
                description=current_result.get('description', 'Bez popisu')
            )
            
            # Update state
            processed_results = state.get("processed_results", []) + [result]
            
            # Add completion message
            messages.append({
                "role": "system",
                "content": f"Completed processing result {current_idx + 1}"
            })
            
            return {
                "current_result": current_result,
                "processed_results": processed_results,
                "current_index": current_idx + 1,
                "messages": messages
            }
            
        except Exception as e:
            error_msg = f"Error processing result {current_idx}: {str(e)}"
            print(error_msg)
            messages.append({"role": "system", "content": error_msg})
            return {
                "current_index": current_idx + 1,
                "messages": messages
            }
    
    def _analyze_results(self, state: ResearchState) -> Dict:
        """Analyze all processed results with LangGraph Studio support."""
        total_processed = len(state.get("processed_results", []))
        messages = state.get("messages", [])
        
        analysis_msg = f"Analýza dokončena. Zpracováno {total_processed} výsledků."
        messages.append({"role": "system", "content": analysis_msg})
        
        return {
            "analysis": analysis_msg,
            "messages": messages
        }
    
    def process(self, research_data: Union[dict, List[dict]]) -> dict:
        """Process research data through the workflow with LangGraph Studio support."""
        # Ensure input is a list
        if not isinstance(research_data, list):
            research_data = [research_data]
            
        # Initialize the state with metadata
        state = {
            "research_data": research_data,
            "processed_results": [],
            "current_result": None,
            "analysis": "",
            "current_index": 0,
            "messages": [{"role": "system", "content": "Starting research analysis workflow"}]
        }
        
        # Run the workflow
        result = self.app.invoke(state)
        return result
    
    def save_results(self, results: List[ResearchResult], output_file: str):
        """Save processed results to a JSON file with proper encoding."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([r.dict() for r in results], f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")

# Example usage for LangGraph Studio
if __name__ == "__main__":
    # This allows the workflow to be imported and used in LangGraph Studio
    processor = ResearchProcessor()
    
    # The workflow can be accessed as processor.app for LangGraph Studio
    # To visualize: 
    # 1. Import this module in LangGraph Studio
    # 2. The workflow will be available as processor.app
    # 3. You can then visualize and debug the workflow in the Studio UI
