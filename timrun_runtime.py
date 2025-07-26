#!/usr/bin/env python3
"""
Section 3: TIMRUN - TIM Inference Runtime (Research Implementation)
From "Beyond Context Limits: Subconscious Threads for Long-Horizon Reasoning"

This module implements the TIMRUN inference runtime as described in Section 3 of the paper.
From paper: "To fully harness TIM's potential and address the deployment obstacles 
presented by the Thread-2 reasoning framework, we developed TIMRUN, an inference 
runtime system co-designed specifically with the TIM model."

⚠️  Research Implementation Notice: This simulates the concepts from the paper using 
mock components. A production implementation would integrate with actual inference 
engines like vLLM or SGLang.
"""

import asyncio
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import requests
from abc import ABC, abstractmethod

from tim_model import TIMModel, Task, ToolUse, TimResponse, WorkingMemory, KVCacheEntry


@dataclass
class InferenceStats:
    """Statistics tracking for TIMRUN inference"""
    total_tokens_processed: int = 0
    kv_cache_hits: int = 0
    kv_cache_misses: int = 0
    tokens_pruned: int = 0
    tool_calls_made: int = 0
    inference_time_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    max_cache_usage: int = 0
    memory_management_overhead_ms: float = 0.0


@dataclass
class PagedAttentionPage:
    """Represents a page in the paged attention system"""
    page_id: str
    tokens: List[str]
    hidden_states: np.ndarray
    position_embeddings: np.ndarray
    is_active: bool = True
    last_accessed: float = field(default_factory=time.time)


class ToolServer(ABC):
    """Abstract base class for tool servers"""
    
    @abstractmethod
    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        pass


class MCPToolServer(ToolServer):
    """MCP (Model Context Protocol) compatible tool server"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
    
    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool via MCP server"""
        try:
            payload = {
                "tool": tool_name,
                "parameters": parameters
            }
            
            # Simulate HTTP request to MCP server
            await asyncio.sleep(0.1)  # Simulate network latency
            
            if tool_name == "SearchTool":
                return {
                    "results": f"Search results for: {parameters.get('query', '')}",
                    "count": 42,
                    "status": "success"
                }
            elif tool_name == "WebReaderTool":
                return {
                    "content": f"Web content from: {parameters.get('url', '')}",
                    "title": "Example Page",
                    "status": "success"
                }
            else:
                return {"status": "error", "message": f"Unknown tool: {tool_name}"}
                
        except Exception as e:
            return {"status": "error", "message": str(e)}


class PagedAttentionManager:
    """
    Manages paged attention mechanism for efficient KV cache manipulation.
    Implements the memory management described in the paper.
    """
    
    def __init__(self, page_size: int = 1, max_pages: int = 4096):
        # Paper: "we set the page size to 1 since each request in the same batch requires different pruning"
        self.page_size = page_size  
        self.max_pages = max_pages
        self.pages: Dict[str, PagedAttentionPage] = {}
        self.free_pages: deque = deque()
        self.lru_order: deque = deque()
        
    def allocate_page(self, page_id: str, tokens: List[str]) -> PagedAttentionPage:
        """Allocate a new page for tokens"""
        if len(self.pages) >= self.max_pages:
            self._evict_lru_page()
        
        # Create hidden states and position embeddings
        hidden_states = np.random.randn(len(tokens), 768)  # Placeholder
        position_embeddings = np.arange(len(tokens))
        
        page = PagedAttentionPage(
            page_id=page_id,
            tokens=tokens,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings
        )
        
        self.pages[page_id] = page
        self.lru_order.append(page_id)
        return page
    
    def get_page(self, page_id: str) -> Optional[PagedAttentionPage]:
        """Get page and update LRU order"""
        if page_id in self.pages:
            page = self.pages[page_id]
            page.last_accessed = time.time()
            
            # Update LRU order
            if page_id in self.lru_order:
                self.lru_order.remove(page_id)
            self.lru_order.append(page_id)
            
            return page
        return None
    
    def free_page(self, page_id: str):
        """Free a page and add to free list"""
        if page_id in self.pages:
            del self.pages[page_id]
            if page_id in self.lru_order:
                self.lru_order.remove(page_id)
            self.free_pages.append(page_id)
    
    def _evict_lru_page(self):
        """Evict least recently used page"""
        if self.lru_order:
            lru_page_id = self.lru_order.popleft()
            self.free_page(lru_page_id)
    
    def prune_pages(self, page_ids_to_prune: List[str]) -> int:
        """Prune specified pages and return count of pruned tokens"""
        pruned_tokens = 0
        for page_id in page_ids_to_prune:
            if page_id in self.pages:
                pruned_tokens += len(self.pages[page_id].tokens)
                self.free_page(page_id)
        return pruned_tokens
    
    def extend_sequence(self, base_page_id: str, new_tokens: List[str]) -> str:
        """
        Extend a sequence by re-encoding tokens after pruned subtasks.
        Implements the position embedding reuse described in the paper.
        """
        base_page = self.get_page(base_page_id)
        if not base_page:
            return self.allocate_page(f"extended_{base_page_id}", new_tokens).page_id
        
        # Create extended sequence
        extended_tokens = base_page.tokens + new_tokens
        extended_page_id = f"extended_{base_page_id}_{int(time.time())}"
        
        # Reuse position embeddings for efficient encoding
        base_positions = len(base_page.position_embeddings)
        new_positions = np.arange(base_positions, base_positions + len(new_tokens))
        
        extended_page = PagedAttentionPage(
            page_id=extended_page_id,
            tokens=extended_tokens,
            hidden_states=np.random.randn(len(extended_tokens), 768),  # Re-encode
            position_embeddings=np.concatenate([base_page.position_embeddings, new_positions])
        )
        
        self.pages[extended_page_id] = extended_page
        self.lru_order.append(extended_page_id)
        
        return extended_page_id


class TIMRUN:
    """
    TIMRUN - The main inference runtime for TIM.
    Handles memory management, subtask pruning, and tool integration.
    """
    
    def __init__(self, 
                 tim_model: TIMModel,
                 page_size: int = 1,
                 max_cache_pages: int = 4096,
                 tool_server: Optional[ToolServer] = None,
                 batch_size: int = 30):  # Paper evaluates with batch_size=30
        
        self.tim_model = tim_model
        self.page_manager = PagedAttentionManager(page_size, max_cache_pages)
        self.tool_server = tool_server or MCPToolServer()
        self.batch_size = batch_size
        
        # Runtime state
        self.stats = InferenceStats()
        self.active_requests: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Inference configuration
        self.enable_subtask_pruning = True
        self.pruning_buffer_size = 2
        self.max_output_tokens = 8192
        self.memory_management_enabled = True
    
    async def inference(self, 
                       instruction: str, 
                       system_prompt: str = "You are TIM, a recursive reasoning model.",
                       available_tools: List[str] = None,
                       request_id: str = None) -> TimResponse:
        """
        Main inference method that processes a request through the complete TIM pipeline
        with dynamic memory management and tool integration.
        """
        start_time = time.time()
        request_id = request_id or f"req_{int(time.time())}"
        
        if available_tools is None:
            available_tools = ["SearchTool", "WebReaderTool"]
        
        # Initialize request tracking
        self.active_requests[request_id] = {
            "start_time": start_time,
            "instruction": instruction,
            "status": "processing"
        }
        
        try:
            # Initialize working memory for this request
            self.tim_model.initialize_working_memory(system_prompt, instruction)
            
            # Process reasoning with memory management
            response = await self._process_with_memory_management(
                instruction, available_tools, request_id
            )
            
            # Update statistics
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000
            self.stats.inference_time_ms = inference_time
            
            if inference_time > 0:
                self.stats.throughput_tokens_per_sec = (
                    self.stats.total_tokens_processed / (inference_time / 1000)
                )
            
            self.active_requests[request_id]["status"] = "completed"
            return response
            
        except Exception as e:
            self.active_requests[request_id]["status"] = f"error: {str(e)}"
            raise
        finally:
            # Cleanup request tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    async def _process_with_memory_management(self, 
                                            instruction: str, 
                                            available_tools: List[str],
                                            request_id: str) -> TimResponse:
        """Process reasoning with dynamic memory management and subtask pruning"""
        
        # Generate initial task structure
        main_task = self.tim_model.generate_task(instruction, available_tools)
        
        # Process task tree with memory management
        processed_tasks = await self._process_task_tree(main_task, request_id)
        
        # Generate final answer
        answer = self.tim_model._aggregate_conclusions(processed_tasks)
        
        return TimResponse(reasoning=processed_tasks, answer=answer)
    
    async def _process_task_tree(self, task: Task, request_id: str) -> List[Task]:
        """Process task tree with memory management and tool integration"""
        processed_tasks = []
        task_id = f"{request_id}_task_{id(task)}"
        
        # Create page for current task
        task_tokens = self.tim_model.working_memory._tokenize_task(task)
        page_id = self.page_manager.allocate_page(task_id, task_tokens).page_id
        
        self.stats.total_tokens_processed += len(task_tokens)
        
        # Process subtasks if they exist
        if task.subtasks:
            subtask_results = []
            subtask_page_ids = []
            
            # Process each subtask
            for i, subtask in enumerate(task.subtasks):
                subtask_result = await self._process_task_tree(subtask, f"{request_id}_sub_{i}")
                subtask_results.extend(subtask_result)
                
                # Track subtask pages for pruning
                subtask_page_id = f"{request_id}_sub_{i}_task_{id(subtask)}"
                subtask_page_ids.append(subtask_page_id)
            
            # Perform subtask pruning
            if self.enable_subtask_pruning:
                await self._prune_subtasks(subtask_page_ids, task_id)
            
            # Update task conclusion based on subtask results
            task.conclusion = self.tim_model._aggregate_subtask_conclusions(subtask_results)
        
        # Execute tool use if present
        if task.tooluse:
            task.tooluse.tool_result = await self._execute_tool_async(task.tooluse)
            task.conclusion += f" | Tool result: {task.tooluse.tool_result}"
            self.stats.tool_calls_made += 1
        
        processed_tasks.append(task)
        return processed_tasks
    
    async def _prune_subtasks(self, subtask_page_ids: List[str], parent_task_id: str):
        """Implement dynamic subtask pruning with memory management"""
        memory_start = time.time()
        
        # Prune subtask pages from memory
        pruned_tokens = self.page_manager.prune_pages(subtask_page_ids)
        self.stats.tokens_pruned += pruned_tokens
        
        # Update cache usage statistics
        current_cache_size = len(self.page_manager.pages)
        self.stats.max_cache_usage = max(self.stats.max_cache_usage, current_cache_size)
        
        # Extend sequence after pruning (position embedding reuse)
        if subtask_page_ids:
            self.page_manager.extend_sequence(parent_task_id, ["<subtasks_completed>"])
        
        memory_end = time.time()
        self.stats.memory_management_overhead_ms += (memory_end - memory_start) * 1000
    
    async def _execute_tool_async(self, tooluse: ToolUse) -> Dict[str, Any]:
        """Execute tool asynchronously through the tool server"""
        try:
            # Extract parameters based on tool type
            if hasattr(tooluse.parameters, 'dict'):
                params = tooluse.parameters.dict()
            else:
                params = tooluse.parameters.__dict__
            
            # Execute tool via server
            result = await self.tool_server.execute(tooluse.tool_name, params)
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Tool execution failed: {str(e)}"
            }
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics matching Table 1 format from the paper"""
        max_cache = self.stats.max_cache_usage
        output_len = self.stats.total_tokens_processed
        kv_pruned_percent = (self.stats.tokens_pruned / max(output_len, 1)) * 100
        
        # Format matching Table 1: Accuracy, Max Cache, Output Len., KV Pruned (%)
        return {
            "accuracy": "N/A",  # Would be task-specific accuracy
            "max_cache": max_cache,
            "output_len": output_len,
            "kv_pruned_percent": kv_pruned_percent,
            "throughput_tokens_per_sec": self.stats.throughput_tokens_per_sec,
            "inference_time_ms": self.stats.inference_time_ms,
            "tool_calls_made": self.stats.tool_calls_made,
            "memory_management_overhead_ms": self.stats.memory_management_overhead_ms
        }
    
    async def batch_inference(self, 
                            requests: List[Tuple[str, str]], 
                            available_tools: List[str] = None) -> List[TimResponse]:
        """Process multiple requests in batch for improved throughput"""
        batch_start = time.time()
        
        # Process requests concurrently
        tasks = []
        for i, (instruction, system_prompt) in enumerate(requests):
            task = self.inference(
                instruction=instruction,
                system_prompt=system_prompt,
                available_tools=available_tools,
                request_id=f"batch_{int(batch_start)}_{i}"
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                # Create error response
                error_response = TimResponse(
                    reasoning=[],
                    answer=f"Error: {str(result)}"
                )
                processed_results.append(error_response)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def reset_stats(self):
        """Reset inference statistics"""
        self.stats = InferenceStats()
    
    async def shutdown(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)


async def main():
    """Demo function showing TIMRUN usage"""
    print("TIMRUN (TIM Inference Runtime) Demo")
    print("=" * 50)
    
    # Initialize TIM model and TIMRUN
    tim_model = TIMModel("TIM-8b", pruning_buffer_size=2)
    tool_server = MCPToolServer()
    timrun = TIMRUN(tim_model, page_size=1, tool_server=tool_server)
    
    # Example complex reasoning task
    instruction = """
    Analyze the relationship between renewable energy adoption and economic growth in developing countries.
    First, search for recent data on renewable energy investments.
    Then, examine the correlation with GDP growth rates.
    Finally, provide policy recommendations for accelerating clean energy transitions.
    """
    
    print(f"Input instruction: {instruction}")
    print("\nProcessing with TIMRUN...")
    
    # Process single request
    start_time = time.time()
    response = await timrun.inference(
        instruction=instruction,
        available_tools=["SearchTool", "WebReaderTool"]
    )
    end_time = time.time()
    
    print(f"\nProcessing completed in {(end_time - start_time)*1000:.2f}ms")
    
    # Display reasoning chain
    print("\nReasoning Chain:")
    print("-" * 30)
    for i, task in enumerate(response.reasoning):
        print(f"Task {i+1}:")
        print(f"  Thought: {task.thought[:100]}...")
        if task.tooluse:
            print(f"  Tool: {task.tooluse.tool_name}")
            print(f"  Tool Result: {str(task.tooluse.tool_result)[:100]}...")
        if task.subtasks:
            print(f"  Subtasks: {len(task.subtasks)}")
        print(f"  Conclusion: {task.conclusion[:100]}...")
        print()
    
    print(f"Final Answer: {response.answer}")
    print()
    
    # Display comprehensive statistics
    stats = timrun.get_inference_stats()
    print("TIMRUN Performance Statistics:")
    print("-" * 30)
    print(f"Total tokens processed: {stats['total_tokens_processed']}")
    print(f"Tokens pruned: {stats['tokens_pruned']} ({stats['pruning_percentage']:.1f}%)")
    print(f"Tool calls made: {stats['tool_calls_made']}")
    print(f"Inference time: {stats['inference_time_ms']:.2f}ms")
    print(f"Throughput: {stats['throughput_tokens_per_sec']:.1f} tokens/sec")
    print(f"Cache utilization: {stats['cache_utilization_percent']:.1f}%")
    print(f"Memory management overhead: {stats['memory_management_overhead_ms']:.2f}ms")
    print(f"KV cache hit ratio: {stats['kv_cache_efficiency']['hit_ratio']:.2f}")
    
    # Demonstrate batch processing
    print("\n" + "=" * 50)
    print("Batch Processing Demo")
    
    batch_requests = [
        ("What are the benefits of solar energy?", "You are a renewable energy expert."),
        ("How does wind power work?", "You are an engineering consultant."),
        ("Compare nuclear vs renewable energy", "You are an energy policy analyst.")
    ]
    
    print(f"Processing {len(batch_requests)} requests in batch...")
    
    batch_start = time.time()
    batch_results = await timrun.batch_inference(batch_requests)
    batch_end = time.time()
    
    print(f"Batch processing completed in {(batch_end - batch_start)*1000:.2f}ms")
    print(f"Average time per request: {((batch_end - batch_start)*1000)/len(batch_requests):.2f}ms")
    
    for i, result in enumerate(batch_results):
        print(f"\nBatch Request {i+1} Result:")
        print(f"Answer: {result.answer[:100]}...")
    
    # Cleanup
    await timrun.shutdown()


if __name__ == "__main__":
    asyncio.run(main())