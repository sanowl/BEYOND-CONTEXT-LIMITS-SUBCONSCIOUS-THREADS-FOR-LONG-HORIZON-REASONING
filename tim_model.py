#!/usr/bin/env python3
"""
Section 2: Thread Inference Model (TIM) - Core Implementation
From "Beyond Context Limits: Subconscious Threads for Long-Horizon Reasoning"

This module implements the Thread-2 structure as described in Section 2.1 of the paper:
"In our design, the basic unit of reasoning is a task, consisting of a thinking process, 
an optional tool use, an optional subtask list and a conclusion."

⚠️  Research Implementation Notice: This is a conceptual implementation for exploring
the paper's ideas. It uses simulated components rather than actual LLM inference.
"""

from typing import List, Optional, Union, Dict, Any, Literal, Tuple
from pydantic import BaseModel
from dataclasses import dataclass, field
import json
import copy
from collections import deque
import numpy as np


# Exact schema from Figure 2 of the paper
class SearchTool(BaseModel):
    """SearchTool parameters exactly as shown in Figure 2"""
    query: str


class WebReaderTool(BaseModel): 
    """WebReaderTool parameters exactly as shown in Figure 2"""
    goal: str
    url: str


class ToolUse(BaseModel):
    """ToolUse schema exactly matching Figure 2 of the paper"""
    tool_name: Literal["SearchTool", "WebReaderTool"] 
    parameters: Union[SearchTool, WebReaderTool]
    tool_result: dict


class Task(BaseModel):
    """
    Thread-2 Task structure exactly as defined in Section 2.1 of the paper.
    
    From paper: "The roles of these fields are designed as follows:
    • thought: contains a thinking process that catches the mistakes of previous steps, 
      analyzes current progress, and plans the following steps.
    • tooluse: optionally call a specific tool by generating the input of the tool 
      and encode the responses of the tool after receiving them.
    • subtasks: optionally spawns subtasks if the current task needs multi-step reasoning. 
      The reasoning details of the spawned subtasks will be hidden from the next step 
      for efficient and concise context management.
    • conclusion: processes tool results, aggregates the conclusion of the subtask list 
      in the current step, and describes the result of the current task."
    """
    thought: str
    tooluse: Optional[ToolUse] = None
    subtasks: Optional[List['Task']] = None
    conclusion: str
    
    class Config:
        arbitrary_types_allowed = True


class TimResponse(BaseModel):
    """TimResponse schema exactly matching Figure 2 of the paper"""
    reasoning: List[Task]
    answer: str


@dataclass
class KVCacheEntry:
    """Represents a key-value cache entry for attention mechanism"""
    task_id: str
    tokens: List[str]
    hidden_states: np.ndarray
    position_ids: np.ndarray
    is_prunable: bool = True


@dataclass
class WorkingMemory:
    """
    Working memory implementation following Section 2.1 of the paper.
    
    From paper: "Thread-2 fixes this issue by accessing the working memory, containing 
    the system prompt, user input, and all tasks that are not pruned."
    
    This enables "end-to-end inference, finishing the reasoning with only one 
    language model call."
    """
    system_prompt: str
    user_input: str
    active_tasks: List[Task] = field(default_factory=list)
    kv_cache: List[KVCacheEntry] = field(default_factory=list)
    pruning_buffer: deque = field(default_factory=lambda: deque(maxlen=2))
    max_cache_size: int = 4096
    
    def add_task(self, task: Task, task_id: str):
        """Add a task to working memory"""
        self.active_tasks.append(task)
        
        # Create KV cache entry for the task
        task_tokens = self._tokenize_task(task)
        cache_entry = KVCacheEntry(
            task_id=task_id,
            tokens=task_tokens,
            hidden_states=np.random.randn(len(task_tokens), 768),  # Placeholder
            position_ids=np.arange(len(task_tokens)),
            is_prunable=True
        )
        self.kv_cache.append(cache_entry)
    
    def _tokenize_task(self, task: Task) -> List[str]:
        """Convert task to tokens (simplified tokenization)"""
        task_str = f"THOUGHT: {task.thought} "
        if task.tooluse:
            task_str += f"TOOL: {task.tooluse.tool_name} "
        if task.subtasks:
            task_str += f"SUBTASKS: {len(task.subtasks)} "
        task_str += f"CONCLUSION: {task.conclusion}"
        return task_str.split()
    
    def prune_subtasks(self, completed_subtask_list: List[Task]):
        """
        Implement subtask pruning mechanism as described in Section 2.1.
        
        From paper: "When a subtask list is completed, we add this list to the stack. 
        If the stack size is larger than the threshold, we pop the earliest subtask list 
        and prune it from the working memory."
        
        The paper notes: "Ideally, processing the current task only needs to read the 
        thoughts and conclusions of previous tasks at the same or higher level, and can 
        safely ignore previous subtask lists in lower levels."
        """
        # Add completed subtasks to pruning buffer (stack)
        self.pruning_buffer.append(completed_subtask_list)
        
        # Remove prunable KV cache entries for completed subtasks
        prunable_task_ids = set()
        for subtask_list in list(self.pruning_buffer):
            for subtask in subtask_list:
                prunable_task_ids.add(id(subtask))
        
        # Filter out prunable cache entries
        self.kv_cache = [
            entry for entry in self.kv_cache 
            if not (entry.is_prunable and entry.task_id in prunable_task_ids)
        ]
        
        # Update active tasks to remove pruned ones
        self.active_tasks = [
            task for task in self.active_tasks 
            if id(task) not in prunable_task_ids
        ]
    
    def get_context_tokens(self) -> List[str]:
        """Get current working memory context tokens"""
        context = [self.system_prompt, self.user_input]
        
        # Add tokens from non-pruned KV cache entries
        for entry in self.kv_cache:
            context.extend(entry.tokens)
        
        return context
    
    def extend_sequence_after_pruning(self, base_tokens: List[str], new_tokens: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implement Equation (1) from Section 3.1: position embedding reuse after pruning.
        (h′_2, h′_2.1, hk; xk+1) = fextend(t2, t2.1, xk | h1)
        """
        # Reuse position embeddings for base tokens
        base_positions = np.arange(len(base_tokens))
        new_positions = np.arange(len(base_tokens), len(base_tokens) + len(new_tokens))
        
        # Simulate hidden state computation with position reuse
        extended_positions = np.concatenate([base_positions, new_positions])
        extended_hidden_states = np.random.randn(len(base_tokens) + len(new_tokens), 768)
        
        return extended_hidden_states, extended_positions

    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics matching Table 1 format"""
        total_tokens = len(self.get_context_tokens())
        cached_entries = len(self.kv_cache)
        
        # Calculate max cache usage during generation
        max_cache = max(cached_entries, 1)
        output_length = total_tokens
        kv_pruned_percent = max(0, 100 - (max_cache / max(output_length, 1) * 100))
        
        return {
            "max_cache": max_cache,
            "output_len": output_length, 
            "kv_pruned_percent": kv_pruned_percent,
            "buffer_size": len(self.pruning_buffer)
        }


class TIMModel:
    """
    Thread Inference Model - Research Implementation
    
    From paper: "We model reasoning trajectories as recursive subtask trees and train 
    a transformer-based model to learn this structure."
    
    Note: This is a conceptual implementation for research purposes. In practice, 
    this would be integrated with actual transformer models like Qwen3-8b as 
    mentioned in the paper's Section 2.2.
    """
    
    def __init__(self, model_name: str = "TIM-8b", pruning_buffer_size: int = 2):
        self.model_name = model_name
        self.working_memory = None
        self.pruning_buffer_size = pruning_buffer_size  # Paper tests {0, 1, 2}
        self.reasoning_depth = 0
        self.max_reasoning_depth = 10
        
    def initialize_working_memory(self, system_prompt: str, user_input: str):
        """Initialize working memory for a new reasoning session"""
        self.working_memory = WorkingMemory(
            system_prompt=system_prompt,
            user_input=user_input
        )
        self.working_memory.pruning_buffer = deque(maxlen=self.pruning_buffer_size)
    
    def generate_task(self, instruction: str, available_tools: List[str] = None) -> Task:
        """
        Generate a single task with thought, optional tool use, 
        optional subtasks, and conclusion.
        """
        if available_tools is None:
            available_tools = []
        
        # Simulate task generation (in real implementation, this would use the LLM)
        thought = f"Analyzing instruction: {instruction}"
        
        # Determine if tool use is needed
        tooluse = None
        if any(tool in instruction.lower() for tool in ["search", "web", "lookup"]):
            if "SearchTool" in available_tools:
                tooluse = ToolUse(
                    tool_name="SearchTool",
                    parameters=SearchTool(query=instruction),
                    tool_result={"status": "pending"}
                )
        
        # Determine if subtask decomposition is needed
        subtasks = None
        if len(instruction.split()) > 10 and self.reasoning_depth < self.max_reasoning_depth:
            # Decompose into subtasks for complex instructions
            subtasks = self._decompose_into_subtasks(instruction, available_tools)
        
        conclusion = f"Completed analysis of: {instruction}"
        
        return Task(
            thought=thought,
            tooluse=tooluse,
            subtasks=subtasks,
            conclusion=conclusion
        )
    
    def _decompose_into_subtasks(self, instruction: str, available_tools: List[str]) -> List[Task]:
        """Decompose complex instruction into simpler subtasks"""
        self.reasoning_depth += 1
        
        # Simplified decomposition logic
        words = instruction.split()
        mid_point = len(words) // 2
        
        subtask1_instruction = " ".join(words[:mid_point])
        subtask2_instruction = " ".join(words[mid_point:])
        
        subtasks = [
            self.generate_task(subtask1_instruction, available_tools),
            self.generate_task(subtask2_instruction, available_tools)
        ]
        
        self.reasoning_depth -= 1
        return subtasks
    
    def generate_json_with_schema(self, instruction: str, available_tools: List[str] = None) -> str:
        """
        Generate structured JSON following Figure 2 schema with constrained decoding.
        Paper: "the Thread-2 reasoning process can be efficiently decoded as a JSON 
        dictionary with popular inference runtimes with constrained decoding engines"
        """
        response = self.process_reasoning_chain(instruction, available_tools)
        return json.dumps(response.dict(), indent=2)

    def process_reasoning_chain(self, instruction: str, available_tools: List[str] = None) -> TimResponse:
        """
        Process complete reasoning chain implementing Thread-2 structure.
        Returns structured output matching Figure 2 TimResponse schema.
        """
        if available_tools is None:
            available_tools = ["SearchTool", "WebReaderTool"]
        
        # Initialize working memory
        system_prompt = "You are TIM, a recursive reasoning model."
        self.initialize_working_memory(system_prompt, instruction)
        
        # Generate main reasoning task
        main_task = self.generate_task(instruction, available_tools)
        
        # Process the task tree and manage memory
        processed_tasks = self._process_task_recursively(main_task)
        
        # Generate final answer
        answer = self._aggregate_conclusions(processed_tasks)
        
        return TimResponse(
            reasoning=processed_tasks,
            answer=answer
        )
    
    def _process_task_recursively(self, task: Task) -> List[Task]:
        """Process a task and its subtasks recursively with memory management"""
        processed_tasks = []
        
        # Add current task to working memory
        task_id = str(id(task))
        self.working_memory.add_task(task, task_id)
        
        # Process subtasks if they exist
        if task.subtasks:
            completed_subtasks = []
            for subtask in task.subtasks:
                subtask_results = self._process_task_recursively(subtask)
                completed_subtasks.extend(subtask_results)
            
            # Prune completed subtasks from working memory
            self.working_memory.prune_subtasks(task.subtasks)
            
            # Update task conclusion based on subtask results
            task.conclusion = self._aggregate_subtask_conclusions(completed_subtasks)
        
        # Execute tool use if present
        if task.tooluse:
            task.tooluse.tool_result = self._execute_tool(task.tooluse)
            task.conclusion += f" Tool result: {task.tooluse.tool_result}"
        
        processed_tasks.append(task)
        return processed_tasks
    
    def _execute_tool(self, tooluse: ToolUse) -> Dict[str, Any]:
        """
        Execute tool within single inference as described in paper.
        Paper: "TIM waits until receiving tool responses as dumped JSON dictionary 
        strings in the reasoning runtime and extends its KV cache by encoding them 
        as batches of new input tokens."
        """
        # Extract tool parameters
        if tooluse.tool_name == "SearchTool":
            query = tooluse.parameters.query
            # Simulate tool execution and JSON response encoding
            result = {
                "query": query,
                "results": [
                    {"title": f"Result for {query}", "url": "https://example.com"},
                    {"title": f"More on {query}", "url": "https://example2.com"}
                ],
                "status": "success"
            }
        elif tooluse.tool_name == "WebReaderTool":
            goal = tooluse.parameters.goal
            url = tooluse.parameters.url
            result = {
                "url": url,
                "goal": goal, 
                "content": f"Content extracted for goal: {goal}",
                "status": "success"
            }
        else:
            result = {"status": "error", "message": "Unknown tool"}
        
        # Extend KV cache with tool response (as described in paper)
        if self.working_memory:
            tool_tokens = json.dumps(result).split()
            base_tokens = self.working_memory.get_context_tokens()
            self.working_memory.extend_sequence_after_pruning(base_tokens, tool_tokens)
        
        return result
    
    def _aggregate_subtask_conclusions(self, subtasks: List[Task]) -> str:
        """Aggregate conclusions from subtasks"""
        conclusions = [task.conclusion for task in subtasks]
        return f"Aggregated from {len(subtasks)} subtasks: " + " | ".join(conclusions)
    
    def _aggregate_conclusions(self, tasks: List[Task]) -> str:
        """Generate final answer from processed tasks"""
        main_conclusions = [task.conclusion for task in tasks]
        return "Final answer based on reasoning chain: " + " ".join(main_conclusions)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        if self.working_memory:
            return self.working_memory.get_memory_usage()
        return {"status": "No active working memory"}
    
    def export_reasoning_trace(self, response: TimResponse) -> str:
        """Export reasoning trace as formatted JSON"""
        return json.dumps(response.dict(), indent=2, ensure_ascii=False)


def main():
    """Demo function showing TIM usage"""
    print("TIM (Thread Inference Model) Demo")
    print("=" * 50)
    
    # Initialize TIM model
    tim = TIMModel("TIM-8b", pruning_buffer_size=2)
    
    # Example complex reasoning task
    instruction = "Research the impact of climate change on ocean temperatures and analyze how this affects marine ecosystems, then provide recommendations for conservation efforts"
    
    print(f"Input instruction: {instruction}")
    print("\nProcessing with TIM...")
    
    # Process the reasoning chain
    response = tim.process_reasoning_chain(
        instruction, 
        available_tools=["SearchTool", "WebReaderTool"]
    )
    
    print("\nReasoning Chain:")
    print("-" * 30)
    for i, task in enumerate(response.reasoning):
        print(f"Task {i+1}:")
        print(f"  Thought: {task.thought}")
        if task.tooluse:
            print(f"  Tool: {task.tooluse.tool_name}")
        if task.subtasks:
            print(f"  Subtasks: {len(task.subtasks)}")
        print(f"  Conclusion: {task.conclusion}")
        print()
    
    print(f"Final Answer: {response.answer}")
    print()
    
    # Show memory statistics
    memory_stats = tim.get_memory_stats()
    print("Memory Statistics:")
    print(f"  Total tokens: {memory_stats.get('total_tokens', 'N/A')}")
    print(f"  Cached entries: {memory_stats.get('cached_entries', 'N/A')}")
    print(f"  Pruned percentage: {memory_stats.get('pruned_percentage', 'N/A')}%")
    print(f"  Buffer size: {memory_stats.get('buffer_size', 'N/A')}")


if __name__ == "__main__":
    main()