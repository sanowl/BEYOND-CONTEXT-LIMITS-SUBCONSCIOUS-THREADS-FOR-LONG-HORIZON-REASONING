#!/usr/bin/env python3
"""
End-to-End Multi-Hop Tool Use for TIM
Implements seamless multi-hop tool calling within a single inference as described in the paper.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import aiohttp
import sqlite3
from pathlib import Path

from tim_model import TIMModel, Task, ToolUse, TimResponse, SearchTool, WebReaderTool


class ToolType(Enum):
    """Types of tools available for multi-hop reasoning"""
    SEARCH = "SearchTool"
    WEB_READER = "WebReaderTool"
    DATABASE = "DatabaseTool"
    CALCULATOR = "CalculatorTool"
    FILE_READER = "FileReaderTool"
    API_CALLER = "APICallerTool"


@dataclass
class ToolCall:
    """Represents a tool call with parameters and results"""
    tool_type: ToolType
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    execution_time_ms: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    call_id: str = field(default_factory=lambda: f"call_{int(time.time()*1000)}")


@dataclass
class MultiHopChain:
    """Represents a chain of tool calls for multi-hop reasoning"""
    chain_id: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    total_execution_time_ms: float = 0.0
    success: bool = True
    final_result: Optional[Dict[str, Any]] = None


class BaseToolExecutor(ABC):
    """Abstract base class for tool executors"""
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool with given parameters"""
        pass
    
    @abstractmethod
    def get_tool_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool's parameters"""
        pass


class SearchToolExecutor(BaseToolExecutor):
    """Executes search operations"""
    
    def __init__(self, search_engine_url: str = "https://api.example-search.com"):
        self.search_engine_url = search_engine_url
        self.search_index = self._build_mock_search_index()
    
    def _build_mock_search_index(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build mock search index for demonstration"""
        return {
            "climate change": [
                {"title": "Climate Change Overview", "url": "https://climate.gov/overview", "snippet": "Comprehensive climate change information"},
                {"title": "Ocean Temperature Rise", "url": "https://ocean-temps.org", "snippet": "Global ocean temperatures increasing"}
            ],
            "renewable energy": [
                {"title": "Solar Power Guide", "url": "https://solar-guide.com", "snippet": "Complete guide to solar energy"},
                {"title": "Wind Energy Statistics", "url": "https://wind-stats.org", "snippet": "Latest wind energy data and trends"}
            ],
            "ocean ecosystems": [
                {"title": "Marine Biodiversity", "url": "https://marine-bio.org", "snippet": "Ocean ecosystem diversity and threats"},
                {"title": "Coral Reef Health", "url": "https://coral-reefs.net", "snippet": "Global coral reef status and conservation"}
            ]
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search with given query"""
        query = parameters.get("query", "").lower()
        
        # Simulate network delay
        await asyncio.sleep(0.2)
        
        # Find matching results
        results = []
        for key, value in self.search_index.items():
            if key in query or any(word in query for word in key.split()):
                results.extend(value)
        
        if not results:
            results = [{"title": "General Search Result", "url": "https://example.com", "snippet": f"Results for: {query}"}]
        
        return {
            "query": parameters.get("query"),
            "results": results[:5],  # Limit to top 5 results
            "total_results": len(results),
            "search_time_ms": 200,
            "status": "success"
        }
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """Get search tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"}
            },
            "required": ["query"]
        }


class WebReaderToolExecutor(BaseToolExecutor):
    """Executes web page reading operations"""
    
    def __init__(self):
        self.page_cache = {}
        self.mock_pages = self._build_mock_pages()
    
    def _build_mock_pages(self) -> Dict[str, Dict[str, Any]]:
        """Build mock web pages for demonstration"""
        return {
            "https://climate.gov/overview": {
                "title": "Climate Change Overview",
                "content": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is natural, scientific evidence shows that human activities have been the main driver of climate change since the 1800s, primarily due to the burning of fossil fuels.",
                "key_points": [
                    "Global temperatures rising due to greenhouse gases",
                    "Human activities primary cause since 1800s",
                    "Impacts include sea level rise and extreme weather"
                ]
            },
            "https://ocean-temps.org": {
                "title": "Ocean Temperature Rise",
                "content": "Ocean temperatures have risen by approximately 0.6°C since 1969. This warming affects marine ecosystems, contributes to sea level rise, and influences weather patterns globally.",
                "key_points": [
                    "0.6°C increase since 1969",
                    "Affects marine ecosystems",
                    "Contributes to sea level rise"
                ]
            },
            "https://marine-bio.org": {
                "title": "Marine Biodiversity",
                "content": "Ocean ecosystems host incredible biodiversity, from microscopic plankton to massive whales. Climate change threatens this diversity through ocean acidification, temperature changes, and habitat loss.",
                "key_points": [
                    "High biodiversity in ocean ecosystems",
                    "Climate change threatens marine life",
                    "Ocean acidification major concern"
                ]
            }
        }
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web reading with given URL and goal"""
        url = parameters.get("url", "")
        goal = parameters.get("goal", "")
        
        # Simulate network delay
        await asyncio.sleep(0.3)
        
        # Get page content
        if url in self.mock_pages:
            page_data = self.mock_pages[url]
            
            # Extract relevant content based on goal
            relevant_content = self._extract_relevant_content(page_data, goal)
            
            return {
                "url": url,
                "goal": goal,
                "title": page_data["title"],
                "content": relevant_content,
                "key_points": page_data["key_points"],
                "extraction_method": "goal-directed",
                "status": "success"
            }
        else:
            return {
                "url": url,
                "goal": goal,
                "error": "Page not found or accessible",
                "status": "error"
            }
    
    def _extract_relevant_content(self, page_data: Dict[str, Any], goal: str) -> str:
        """Extract content relevant to the specified goal"""
        content = page_data["content"]
        
        # Simple relevance extraction based on goal keywords
        goal_keywords = goal.lower().split()
        sentences = content.split('. ')
        
        relevant_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in goal_keywords):
                relevant_sentences.append(sentence)
        
        if relevant_sentences:
            return '. '.join(relevant_sentences) + '.'
        else:
            return content[:200] + "..."  # Fallback to first 200 chars
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """Get web reader tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to read"},
                "goal": {"type": "string", "description": "Specific information to extract"}
            },
            "required": ["url", "goal"]
        }


class DatabaseToolExecutor(BaseToolExecutor):
    """Executes database queries for structured data"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize demo database with sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE climate_data (
                id INTEGER PRIMARY KEY,
                year INTEGER,
                global_temp_anomaly REAL,
                co2_ppm REAL,
                sea_level_mm REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE renewable_energy (
                id INTEGER PRIMARY KEY,
                country TEXT,
                year INTEGER,
                solar_capacity_gw REAL,
                wind_capacity_gw REAL,
                total_renewable_percent REAL
            )
        ''')
        
        # Insert sample data
        climate_data = [
            (2020, 1.02, 414.2, 95.0),
            (2021, 0.85, 416.4, 97.2),
            (2022, 0.89, 418.5, 99.1),
            (2023, 1.15, 421.0, 101.5)
        ]
        
        cursor.executemany(
            'INSERT INTO climate_data (year, global_temp_anomaly, co2_ppm, sea_level_mm) VALUES (?, ?, ?, ?)',
            climate_data
        )
        
        renewable_data = [
            ("USA", 2023, 131.0, 144.0, 21.5),
            ("China", 2023, 261.0, 376.0, 31.2),
            ("Germany", 2023, 67.0, 69.0, 52.1),
            ("India", 2023, 63.0, 87.0, 26.8)
        ]
        
        cursor.executemany(
            'INSERT INTO renewable_energy (country, year, solar_capacity_gw, wind_capacity_gw, total_renewable_percent) VALUES (?, ?, ?, ?, ?)',
            renewable_data
        )
        
        conn.commit()
        conn.close()
    
    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute database query"""
        query = parameters.get("query", "")
        
        # Simulate database processing delay
        await asyncio.sleep(0.1)
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            results = [dict(row) for row in rows]
            
            conn.close()
            
            return {
                "query": query,
                "results": results,
                "row_count": len(results),
                "status": "success"
            }
            
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "status": "error"
            }
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """Get database tool parameter schema"""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query to execute"}
            },
            "required": ["query"]
        }


class MultiHopToolOrchestrator:
    """
    Orchestrates multi-hop tool use for complex reasoning chains.
    Implements the end-to-end tool integration described in the TIM paper.
    """
    
    def __init__(self):
        self.tool_executors = {
            ToolType.SEARCH: SearchToolExecutor(),
            ToolType.WEB_READER: WebReaderToolExecutor(),
            ToolType.DATABASE: DatabaseToolExecutor()
        }
        self.active_chains: Dict[str, MultiHopChain] = {}
        self.execution_stats = {
            "total_chains": 0,
            "successful_chains": 0,
            "failed_chains": 0,
            "total_tool_calls": 0,
            "average_chain_length": 0.0
        }
    
    async def execute_multihop_chain(self, 
                                   chain_specification: List[Dict[str, Any]],
                                   chain_id: Optional[str] = None) -> MultiHopChain:
        """
        Execute a multi-hop tool chain based on specification.
        Each step can depend on results from previous steps.
        """
        chain_id = chain_id or f"chain_{int(time.time()*1000)}"
        chain = MultiHopChain(chain_id=chain_id)
        self.active_chains[chain_id] = chain
        
        start_time = time.time()
        
        try:
            # Execute each step in the chain
            for i, step_spec in enumerate(chain_specification):
                tool_call = await self._execute_chain_step(step_spec, chain, i)
                chain.tool_calls.append(tool_call)
                
                if not tool_call.success:
                    chain.success = False
                    break
            
            # Set final result
            if chain.tool_calls and chain.success:
                chain.final_result = chain.tool_calls[-1].result
            
            # Update statistics
            end_time = time.time()
            chain.total_execution_time_ms = (end_time - start_time) * 1000
            
            self._update_execution_stats(chain)
            
            return chain
            
        except Exception as e:
            chain.success = False
            logging.error(f"Multi-hop chain execution failed: {e}")
            return chain
        finally:
            if chain_id in self.active_chains:
                del self.active_chains[chain_id]
    
    async def _execute_chain_step(self, 
                                step_spec: Dict[str, Any], 
                                chain: MultiHopChain, 
                                step_index: int) -> ToolCall:
        """Execute a single step in the multi-hop chain"""
        
        tool_type_str = step_spec.get("tool_type")
        tool_type = ToolType(tool_type_str)
        
        # Resolve parameters, potentially using results from previous steps
        parameters = self._resolve_parameters(step_spec.get("parameters", {}), chain)
        
        # Create tool call
        tool_call = ToolCall(
            tool_type=tool_type,
            parameters=parameters
        )
        
        # Execute the tool
        start_time = time.time()
        
        try:
            executor = self.tool_executors.get(tool_type)
            if not executor:
                raise ValueError(f"No executor found for tool type: {tool_type}")
            
            result = await executor.execute(parameters)
            
            tool_call.result = result
            tool_call.success = result.get("status") == "success"
            
            if not tool_call.success:
                tool_call.error_message = result.get("error", "Unknown error")
            
        except Exception as e:
            tool_call.success = False
            tool_call.error_message = str(e)
            tool_call.result = {"status": "error", "message": str(e)}
        
        finally:
            end_time = time.time()
            tool_call.execution_time_ms = (end_time - start_time) * 1000
        
        return tool_call
    
    def _resolve_parameters(self, 
                          param_spec: Dict[str, Any], 
                          chain: MultiHopChain) -> Dict[str, Any]:
        """
        Resolve parameters that may reference results from previous tool calls.
        Supports expressions like: ${step_0.results[0].url}
        """
        resolved_params = {}
        
        for key, value in param_spec.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # This is a reference to a previous result
                reference = value[2:-1]  # Remove ${ and }
                resolved_value = self._resolve_reference(reference, chain)
                resolved_params[key] = resolved_value
            else:
                resolved_params[key] = value
        
        return resolved_params
    
    def _resolve_reference(self, reference: str, chain: MultiHopChain) -> Any:
        """Resolve a reference to a previous step's result"""
        
        try:
            # Parse reference like "step_0.results[0].url"
            parts = reference.split('.')
            
            if parts[0].startswith("step_"):
                step_index = int(parts[0].split('_')[1])
                
                if step_index < len(chain.tool_calls):
                    result = chain.tool_calls[step_index].result
                    
                    # Navigate through the remaining parts
                    for part in parts[1:]:
                        if '[' in part and ']' in part:
                            # Array access like "results[0]"
                            field_name = part[:part.index('[')]
                            index = int(part[part.index('[')+1:part.index(']')])
                            result = result[field_name][index]
                        else:
                            # Simple field access
                            result = result[part]
                    
                    return result
            
            return reference  # Return as-is if we can't resolve
            
        except (IndexError, KeyError, ValueError):
            return reference  # Return as-is if resolution fails
    
    def _update_execution_stats(self, chain: MultiHopChain):
        """Update execution statistics"""
        self.execution_stats["total_chains"] += 1
        
        if chain.success:
            self.execution_stats["successful_chains"] += 1
        else:
            self.execution_stats["failed_chains"] += 1
        
        self.execution_stats["total_tool_calls"] += len(chain.tool_calls)
        
        # Update average chain length
        total_calls = self.execution_stats["total_tool_calls"]
        total_chains = self.execution_stats["total_chains"]
        self.execution_stats["average_chain_length"] = total_calls / max(total_chains, 1)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        success_rate = 0.0
        if self.execution_stats["total_chains"] > 0:
            success_rate = (self.execution_stats["successful_chains"] / 
                          self.execution_stats["total_chains"]) * 100
        
        return {
            **self.execution_stats,
            "success_rate_percent": success_rate,
            "active_chains": len(self.active_chains),
            "available_tools": list(self.tool_executors.keys())
        }
    
    async def execute_tim_multihop_task(self, 
                                      task: Task, 
                                      context: Dict[str, Any] = None) -> Task:
        """
        Execute a TIM task that requires multi-hop tool use.
        Automatically extracts tool chain from task structure.
        """
        if not task.subtasks:
            # Simple task with single tool use
            if task.tooluse:
                chain_spec = [self._tool_use_to_chain_step(task.tooluse)]
                chain = await self.execute_multihop_chain(chain_spec)
                
                if chain.success and chain.final_result:
                    task.tooluse.tool_result = chain.final_result
            
            return task
        
        # Complex task with subtasks - build multi-hop chain
        chain_spec = []
        
        for i, subtask in enumerate(task.subtasks):
            if subtask.tooluse:
                chain_spec.append(self._tool_use_to_chain_step(subtask.tooluse, step_index=i))
        
        if chain_spec:
            chain = await self.execute_multihop_chain(chain_spec)
            
            # Update subtasks with results
            for i, subtask in enumerate(task.subtasks):
                if i < len(chain.tool_calls) and subtask.tooluse:
                    subtask.tooluse.tool_result = chain.tool_calls[i].result
                    subtask.conclusion = f"Tool execution completed: {subtask.tooluse.tool_result.get('status', 'unknown')}"
        
        return task
    
    def _tool_use_to_chain_step(self, 
                               tool_use: ToolUse, 
                               step_index: int = 0) -> Dict[str, Any]:
        """Convert a TIM ToolUse to a chain step specification"""
        
        # Extract parameters
        if hasattr(tool_use.parameters, 'dict'):
            params = tool_use.parameters.dict()
        else:
            params = tool_use.parameters.__dict__
        
        return {
            "tool_type": tool_use.tool_name,
            "parameters": params
        }


async def main():
    """Demo function showing multi-hop tool use"""
    print("TIM Multi-Hop Tool Use Demo")
    print("=" * 40)
    
    # Initialize orchestrator
    orchestrator = MultiHopToolOrchestrator()
    
    # Example 1: Simple multi-hop chain
    print("1. Simple Multi-Hop Chain:")
    print("-" * 25)
    
    simple_chain_spec = [
        {
            "tool_type": "SearchTool",
            "parameters": {
                "query": "climate change ocean temperature"
            }
        },
        {
            "tool_type": "WebReaderTool", 
            "parameters": {
                "url": "${step_0.results[0].url}",
                "goal": "extract temperature data"
            }
        }
    ]
    
    chain1 = await orchestrator.execute_multihop_chain(simple_chain_spec, "demo_chain_1")
    
    print(f"Chain success: {chain1.success}")
    print(f"Total execution time: {chain1.total_execution_time_ms:.2f}ms")
    print(f"Number of tool calls: {len(chain1.tool_calls)}")
    
    for i, call in enumerate(chain1.tool_calls):
        print(f"  Step {i}: {call.tool_type.value} - {'Success' if call.success else 'Failed'}")
        if call.result:
            print(f"    Result keys: {list(call.result.keys())}")
    
    print()
    
    # Example 2: Complex research chain
    print("2. Complex Research Chain:")
    print("-" * 26)
    
    research_chain_spec = [
        {
            "tool_type": "SearchTool",
            "parameters": {
                "query": "renewable energy statistics 2023"
            }
        },
        {
            "tool_type": "DatabaseTool",
            "parameters": {
                "query": "SELECT country, total_renewable_percent FROM renewable_energy WHERE year = 2023 ORDER BY total_renewable_percent DESC"
            }
        },
        {
            "tool_type": "WebReaderTool",
            "parameters": {
                "url": "${step_0.results[1].url}",
                "goal": "analyze renewable energy trends"
            }
        }
    ]
    
    chain2 = await orchestrator.execute_multihop_chain(research_chain_spec, "research_chain")
    
    print(f"Research chain success: {chain2.success}")
    print(f"Total execution time: {chain2.total_execution_time_ms:.2f}ms")
    
    if chain2.final_result:
        print("Final result summary:")
        if isinstance(chain2.final_result, dict):
            for key, value in chain2.final_result.items():
                if key != "content":  # Skip long content
                    print(f"  {key}: {value}")
    
    print()
    
    # Example 3: TIM Task Integration
    print("3. TIM Task Integration:")
    print("-" * 23)
    
    # Create a TIM task with multi-hop tool requirements
    from tim_model import SearchTool, WebReaderTool
    
    main_task = Task(
        thought="I need to research climate change impacts on marine ecosystems",
        tooluse=ToolUse(
            tool_name="SearchTool",
            parameters=SearchTool(query="climate change marine ecosystems"),
            tool_result={}
        ),
        subtasks=[
            Task(
                thought="First, I'll search for relevant information",
                tooluse=ToolUse(
                    tool_name="SearchTool", 
                    parameters=SearchTool(query="ocean temperature rise marine life"),
                    tool_result={}
                ),
                conclusion="Search completed"
            ),
            Task(
                thought="Then I'll read detailed information from a source",
                tooluse=ToolUse(
                    tool_name="WebReaderTool",
                    parameters=WebReaderTool(
                        goal="marine ecosystem impacts",
                        url="https://marine-bio.org"
                    ),
                    tool_result={}
                ),
                conclusion="Web reading completed"
            )
        ],
        conclusion="Multi-hop research completed"
    )
    
    # Execute the task with multi-hop orchestration
    processed_task = await orchestrator.execute_tim_multihop_task(main_task)
    
    print("Task processing completed")
    print(f"Main task tool result: {processed_task.tooluse.tool_result.get('status', 'N/A')}")
    
    for i, subtask in enumerate(processed_task.subtasks):
        if subtask.tooluse and subtask.tooluse.tool_result:
            status = subtask.tooluse.tool_result.get('status', 'unknown')
            print(f"  Subtask {i+1}: {status}")
    
    print()
    
    # Show execution statistics
    print("4. Execution Statistics:")
    print("-" * 22)
    
    stats = orchestrator.get_execution_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())