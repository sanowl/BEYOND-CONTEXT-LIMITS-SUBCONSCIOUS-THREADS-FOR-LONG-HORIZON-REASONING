#!/usr/bin/env python3
"""
Complete TIM (Thread Inference Model) System Demo
Demonstrates all features of the TIM implementation as described in the paper:
"Beyond Context Limits: Subconscious Threads for Long-Horizon Reasoning"

This demo showcases:
1. Thread-2 structured reasoning
2. Subtask pruning and memory management  
3. TIMRUN inference runtime
4. Constrained JSON generation
5. End-to-end multi-hop tool use
"""

import asyncio
import json
import time
from typing import Dict, List, Any
import sys
from pathlib import Path

# Import all TIM components
from tim_model import TIMModel, Task, ToolUse, TimResponse, SearchTool, WebReaderTool
from timrun_runtime import TIMRUN, MCPToolServer
from json_constrained_generation import TIMJSONGenerator
from multihop_tools import MultiHopToolOrchestrator, ToolType


class CompleteTIMDemo:
    """Complete demonstration of the TIM system"""
    
    def __init__(self):
        self.tim_model = TIMModel("TIM-8b", pruning_buffer_size=2)
        self.tool_server = MCPToolServer()
        self.timrun = TIMRUN(self.tim_model, tool_server=self.tool_server)
        self.json_generator = TIMJSONGenerator()
        self.multihop_orchestrator = MultiHopToolOrchestrator()
        
    async def run_complete_demo(self):
        """Run the complete TIM system demonstration"""
        
        print("=" * 80)
        print("COMPLETE TIM (THREAD INFERENCE MODEL) SYSTEM DEMONSTRATION")
        print("Implementation of: 'Beyond Context Limits: Subconscious Threads for Long-Horizon Reasoning'")
        print("=" * 80)
        
        # Demo 1: Basic TIM Reasoning
        await self._demo_basic_reasoning()
        
        # Demo 2: Subtask Pruning and Memory Management
        await self._demo_memory_management()
        
        # Demo 3: TIMRUN Inference Runtime
        await self._demo_timrun_runtime()
        
        # Demo 4: Constrained JSON Generation
        await self._demo_json_generation()
        
        # Demo 5: Multi-Hop Tool Use
        await self._demo_multihop_tools()
        
        # Demo 6: Complete End-to-End Scenario
        await self._demo_end_to_end_scenario()
        
        # Demo 7: Performance Benchmarks
        await self._demo_performance_benchmarks()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("All TIM system components successfully demonstrated!")
        print("=" * 80)
    
    async def _demo_basic_reasoning(self):
        """Demonstrate basic TIM reasoning with Thread-2 structure"""
        print("\n" + "=" * 60)
        print("DEMO 1: BASIC TIM REASONING WITH THREAD-2 STRUCTURE")
        print("=" * 60)
        
        instruction = """
        Analyze the relationship between renewable energy adoption and economic growth.
        Consider both direct economic impacts and indirect environmental benefits.
        Provide policy recommendations for developing countries.
        """
        
        print(f"Input: {instruction.strip()}")
        print("\nProcessing with TIM Thread-2 structure...")
        
        start_time = time.time()
        response = self.tim_model.process_reasoning_chain(
            instruction, 
            available_tools=["SearchTool", "WebReaderTool"]
        )
        end_time = time.time()
        
        print(f"\nProcessing completed in {(end_time - start_time)*1000:.2f}ms")
        print(f"Generated {len(response.reasoning)} reasoning tasks")
        
        # Show reasoning structure
        print("\nReasoning Chain Structure:")
        for i, task in enumerate(response.reasoning):
            print(f"  Task {i+1}:")
            print(f"    Thought: {task.thought[:80]}...")
            if task.tooluse:
                print(f"    Tool: {task.tooluse.tool_name}")
            if task.subtasks:
                print(f"    Subtasks: {len(task.subtasks)}")
            print(f"    Conclusion: {task.conclusion[:80]}...")
        
        print(f"\nFinal Answer: {response.answer}")
        
        # Show memory statistics
        memory_stats = self.tim_model.get_memory_stats()
        print(f"\nMemory Usage:")
        print(f"  Total tokens: {memory_stats.get('total_tokens', 'N/A')}")
        print(f"  Pruned percentage: {memory_stats.get('pruned_percentage', 'N/A')}%")
    
    async def _demo_memory_management(self):
        """Demonstrate subtask pruning and working memory management"""
        print("\n" + "=" * 60)
        print("DEMO 2: SUBTASK PRUNING AND MEMORY MANAGEMENT")
        print("=" * 60)
        
        print("Testing memory management with complex nested reasoning...")
        
        # Create complex instruction that will generate many subtasks
        complex_instruction = """
        Conduct a comprehensive analysis of global climate change impacts on agriculture,
        including regional variations, crop-specific effects, adaptation strategies,
        economic implications, policy responses, and future projections for food security.
        """
        
        print(f"Complex instruction: {complex_instruction[:100]}...")
        
        # Process with memory tracking
        initial_memory = self.tim_model.get_memory_stats()
        
        response = self.tim_model.process_reasoning_chain(complex_instruction)
        
        final_memory = self.tim_model.get_memory_stats()
        
        print(f"\nMemory Management Results:")
        print(f"  Tasks generated: {len(response.reasoning)}")
        print(f"  Memory efficiency: {final_memory.get('pruned_percentage', 0):.1f}% pruned")
        print(f"  Buffer utilization: {final_memory.get('buffer_size', 0)} items")
        
        # Demonstrate pruning effectiveness
        print(f"\nPruning Effectiveness:")
        print(f"  Working memory size: {final_memory.get('total_tokens', 0)} tokens")
        print(f"  Cache efficiency: Optimized for long-horizon reasoning")
    
    async def _demo_timrun_runtime(self):
        """Demonstrate TIMRUN inference runtime capabilities"""
        print("\n" + "=" * 60)
        print("DEMO 3: TIMRUN INFERENCE RUNTIME")
        print("=" * 60)
        
        instruction = """
        Research the effectiveness of carbon pricing mechanisms across different countries.
        Search for recent policy implementations, analyze economic impacts, and compare
        different approaches like carbon taxes versus cap-and-trade systems.
        """
        
        print("Testing TIMRUN with dynamic memory management and tool integration...")
        print(f"Instruction: {instruction[:100]}...")
        
        # Single inference
        start_time = time.time()
        response = await self.timrun.inference(
            instruction=instruction,
            available_tools=["SearchTool", "WebReaderTool"]
        )
        end_time = time.time()
        
        single_time = (end_time - start_time) * 1000
        
        print(f"\nSingle Inference Results:")
        print(f"  Processing time: {single_time:.2f}ms")
        print(f"  Reasoning tasks: {len(response.reasoning)}")
        print(f"  Final answer length: {len(response.answer)} characters")
        
        # Batch inference demonstration
        print("\nTesting batch inference capabilities...")
        
        batch_requests = [
            ("What are the benefits of solar energy?", "You are a renewable energy expert."),
            ("How do wind turbines generate electricity?", "You are an engineering consultant."),
            ("Compare nuclear vs renewable energy costs", "You are an energy economist.")
        ]
        
        batch_start = time.time()
        batch_results = await self.timrun.batch_inference(batch_requests)
        batch_end = time.time()
        
        batch_time = (batch_end - batch_start) * 1000
        
        print(f"\nBatch Inference Results:")
        print(f"  Requests processed: {len(batch_results)}")
        print(f"  Total batch time: {batch_time:.2f}ms")
        print(f"  Average per request: {batch_time/len(batch_results):.2f}ms")
        
        # Show detailed statistics
        stats = self.timrun.get_inference_stats()
        print(f"\nTIMRUN Performance Statistics:")
        print(f"  Throughput: {stats['throughput_tokens_per_sec']:.1f} tokens/sec")
        print(f"  Cache utilization: {stats['cache_utilization_percent']:.1f}%")
        print(f"  Memory overhead: {stats['memory_management_overhead_ms']:.2f}ms")
        print(f"  Tool calls: {stats['tool_calls_made']}")
    
    async def _demo_json_generation(self):
        """Demonstrate constrained JSON generation"""
        print("\n" + "=" * 60)
        print("DEMO 4: CONSTRAINED JSON GENERATION")
        print("=" * 60)
        
        print("Demonstrating structured JSON generation following TIM schemas...")
        
        instruction = "Analyze renewable energy trends in developing countries"
        
        # Generate different JSON structures
        print("\n1. Simple Task JSON:")
        simple_task = self.json_generator.generate_task_json(instruction)
        print(simple_task)
        
        print("\n2. Task with Tool Use:")
        task_with_tool = self.json_generator.generate_task_json(instruction, use_tools=True)
        print(task_with_tool)
        
        print("\n3. Task with Subtasks:")
        task_with_subtasks = self.json_generator.generate_task_json(instruction, create_subtasks=True)
        print(task_with_subtasks)
        
        print("\n4. Complete TIM Response:")
        complete_response = self.json_generator.generate_tim_response_json(instruction)
        print(complete_response)
        
        # Validate all generated JSON
        print("\n5. Schema Validation Results:")
        validations = [
            ("Simple Task", simple_task, "task"),
            ("Task with Tool", task_with_tool, "task"),
            ("Task with Subtasks", task_with_subtasks, "task"),
            ("Complete Response", complete_response, "tim_response")
        ]
        
        for name, json_str, schema_name in validations:
            is_valid = self.json_generator.validate_tim_json(json_str, schema_name)
            status = "✓ Valid" if is_valid else "✗ Invalid"
            print(f"  {name}: {status}")
        
        # Demonstrate constrained generation
        print("\n6. Constrained Generation Example:")
        try:
            constrained_json = self.json_generator.generate_with_constraints("task")
            print("Generated with schema constraints:")
            print(constrained_json[:200] + "..." if len(constrained_json) > 200 else constrained_json)
        except Exception as e:
            print(f"Constrained generation: {e}")
    
    async def _demo_multihop_tools(self):
        """Demonstrate multi-hop tool use capabilities"""
        print("\n" + "=" * 60)
        print("DEMO 5: MULTI-HOP TOOL USE")
        print("=" * 60)
        
        print("Demonstrating seamless multi-hop tool calling...")
        
        # Define a multi-hop chain for research
        research_chain = [
            {
                "tool_type": "SearchTool",
                "parameters": {
                    "query": "climate change renewable energy policy"
                }
            },
            {
                "tool_type": "WebReaderTool",
                "parameters": {
                    "url": "${step_0.results[0].url}",
                    "goal": "extract policy recommendations"
                }
            },
            {
                "tool_type": "DatabaseTool",
                "parameters": {
                    "query": "SELECT country, total_renewable_percent FROM renewable_energy WHERE year = 2023 ORDER BY total_renewable_percent DESC LIMIT 5"
                }
            }
        ]
        
        print("Executing 3-step research chain:")
        print("  1. Search for climate/renewable energy policies")
        print("  2. Read detailed information from top result")
        print("  3. Query database for renewable energy statistics")
        
        chain_result = await self.multihop_orchestrator.execute_multihop_chain(
            research_chain, 
            "demo_research_chain"
        )
        
        print(f"\nMulti-Hop Chain Results:")
        print(f"  Chain success: {chain_result.success}")
        print(f"  Total execution time: {chain_result.total_execution_time_ms:.2f}ms")
        print(f"  Tool calls executed: {len(chain_result.tool_calls)}")
        
        for i, call in enumerate(chain_result.tool_calls):
            print(f"    Step {i+1}: {call.tool_type.value} - {'Success' if call.success else 'Failed'}")
            if call.result and call.success:
                result_keys = list(call.result.keys())[:3]  # Show first 3 keys
                print(f"      Result keys: {result_keys}")
        
        if chain_result.final_result:
            print(f"  Final result status: {chain_result.final_result.get('status', 'unknown')}")
        
        # Show orchestrator statistics
        stats = self.multihop_orchestrator.get_execution_stats()
        print(f"\nOrchestrator Statistics:")
        print(f"  Success rate: {stats['success_rate_percent']:.1f}%")
        print(f"  Average chain length: {stats['average_chain_length']:.1f}")
        print(f"  Total tool calls: {stats['total_tool_calls']}")
    
    async def _demo_end_to_end_scenario(self):
        """Demonstrate complete end-to-end TIM scenario"""
        print("\n" + "=" * 60)
        print("DEMO 6: COMPLETE END-TO-END SCENARIO")
        print("=" * 60)
        
        print("Running complete TIM pipeline for complex research task...")
        
        complex_scenario = """
        Conduct a comprehensive analysis of the transition to renewable energy in developing countries.
        
        Your analysis should include:
        1. Current renewable energy adoption rates by region
        2. Economic barriers and opportunities  
        3. Policy frameworks that have proven effective
        4. Technology transfer mechanisms
        5. Financing options including international climate funds
        6. Case studies of successful transitions
        7. Recommendations for accelerating adoption
        
        Use multiple information sources and provide data-driven insights.
        """
        
        print(f"Complex scenario: {complex_scenario[:200]}...")
        
        # Track the complete pipeline
        pipeline_start = time.time()
        
        # Step 1: Process with TIM reasoning
        print("\nStep 1: TIM Thread-2 Reasoning...")
        reasoning_start = time.time()
        response = self.tim_model.process_reasoning_chain(
            complex_scenario,
            available_tools=["SearchTool", "WebReaderTool"]
        )
        reasoning_time = (time.time() - reasoning_start) * 1000
        
        # Step 2: Execute with TIMRUN
        print("Step 2: TIMRUN Inference Runtime...")
        runtime_start = time.time()
        timrun_response = await self.timrun.inference(
            instruction=complex_scenario,
            available_tools=["SearchTool", "WebReaderTool", "DatabaseTool"]
        )
        runtime_time = (time.time() - runtime_start) * 1000
        
        # Step 3: Generate structured JSON
        print("Step 3: JSON Structure Generation...")
        json_start = time.time()
        structured_json = self.json_generator.generate_tim_response_json(complex_scenario)
        json_time = (time.time() - json_start) * 1000
        
        # Step 4: Execute multi-hop tools
        print("Step 4: Multi-Hop Tool Execution...")
        multihop_start = time.time()
        
        multihop_chain = [
            {"tool_type": "SearchTool", "parameters": {"query": "developing countries renewable energy statistics"}},
            {"tool_type": "DatabaseTool", "parameters": {"query": "SELECT * FROM renewable_energy WHERE year = 2023"}},
            {"tool_type": "WebReaderTool", "parameters": {"url": "${step_0.results[0].url}", "goal": "policy analysis"}}
        ]
        
        chain_result = await self.multihop_orchestrator.execute_multihop_chain(multihop_chain)
        multihop_time = (time.time() - multihop_start) * 1000
        
        pipeline_time = (time.time() - pipeline_start) * 1000
        
        # Results summary
        print(f"\nEnd-to-End Pipeline Results:")
        print(f"  Total pipeline time: {pipeline_time:.2f}ms")
        print(f"    - TIM reasoning: {reasoning_time:.2f}ms")
        print(f"    - TIMRUN runtime: {runtime_time:.2f}ms") 
        print(f"    - JSON generation: {json_time:.2f}ms")
        print(f"    - Multi-hop tools: {multihop_time:.2f}ms")
        
        print(f"\nOutput Statistics:")
        print(f"  Reasoning tasks generated: {len(response.reasoning)}")
        print(f"  Final answer length: {len(timrun_response.answer)} characters")
        print(f"  JSON structure size: {len(structured_json)} characters")
        print(f"  Tool chain success: {chain_result.success}")
        
        # Memory efficiency
        memory_stats = self.tim_model.get_memory_stats()
        timrun_stats = self.timrun.get_inference_stats()
        
        print(f"\nSystem Efficiency:")
        print(f"  Memory pruning: {memory_stats.get('pruned_percentage', 0):.1f}%")
        print(f"  Cache utilization: {timrun_stats['cache_utilization_percent']:.1f}%")
        print(f"  Throughput: {timrun_stats['throughput_tokens_per_sec']:.1f} tokens/sec")
    
    async def _demo_performance_benchmarks(self):
        """Demonstrate performance benchmarks and scalability"""
        print("\n" + "=" * 60)
        print("DEMO 7: PERFORMANCE BENCHMARKS")
        print("=" * 60)
        
        print("Running performance benchmarks...")
        
        # Benchmark 1: Reasoning complexity scaling
        print("\n1. Reasoning Complexity Scaling:")
        complexities = [
            ("Simple", "What is renewable energy?"),
            ("Medium", "Compare solar and wind energy technologies including costs and efficiency."),
            ("Complex", "Analyze the comprehensive impact of renewable energy transition on developing economies including policy frameworks, financing mechanisms, technology transfer, and regional variations.")
        ]
        
        for complexity, instruction in complexities:
            start_time = time.time()
            response = self.tim_model.process_reasoning_chain(instruction)
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000
            memory_stats = self.tim_model.get_memory_stats()
            
            print(f"  {complexity}:")
            print(f"    Processing time: {processing_time:.2f}ms")
            print(f"    Tasks generated: {len(response.reasoning)}")
            print(f"    Memory efficiency: {memory_stats.get('pruned_percentage', 0):.1f}% pruned")
        
        # Benchmark 2: Batch processing throughput
        print("\n2. Batch Processing Throughput:")
        batch_sizes = [1, 5, 10]
        
        for batch_size in batch_sizes:
            requests = [
                (f"Analyze renewable energy adoption in region {i}", "You are an energy analyst.")
                for i in range(batch_size)
            ]
            
            batch_start = time.time()
            batch_results = await self.timrun.batch_inference(requests)
            batch_end = time.time()
            
            batch_time = (batch_end - batch_start) * 1000
            throughput = batch_size / (batch_time / 1000)
            
            print(f"  Batch size {batch_size}:")
            print(f"    Total time: {batch_time:.2f}ms")
            print(f"    Throughput: {throughput:.2f} requests/sec")
            print(f"    Average per request: {batch_time/batch_size:.2f}ms")
        
        # Benchmark 3: Memory management efficiency
        print("\n3. Memory Management Efficiency:")
        
        # Test with different pruning buffer sizes
        pruning_sizes = [0, 1, 2, 3]
        
        for size in pruning_sizes:
            test_model = TIMModel("TIM-8b", pruning_buffer_size=size)
            
            start_time = time.time()
            response = test_model.process_reasoning_chain(
                "Conduct detailed analysis of climate change impacts on agriculture with regional breakdowns and policy recommendations"
            )
            end_time = time.time()
            
            processing_time = (end_time - start_time) * 1000
            memory_stats = test_model.get_memory_stats()
            
            print(f"  Pruning buffer size {size}:")
            print(f"    Processing time: {processing_time:.2f}ms")
            print(f"    Memory pruned: {memory_stats.get('pruned_percentage', 0):.1f}%")
            print(f"    Token efficiency: {memory_stats.get('total_tokens', 0)} tokens used")
        
        # Final statistics summary
        print("\n4. Overall System Statistics:")
        timrun_stats = self.timrun.get_inference_stats()
        orchestrator_stats = self.multihop_orchestrator.get_execution_stats()
        
        print(f"  TIMRUN Performance:")
        print(f"    Peak throughput: {timrun_stats['throughput_tokens_per_sec']:.1f} tokens/sec")
        print(f"    Cache efficiency: {timrun_stats['kv_cache_efficiency']['hit_ratio']:.2f}")
        print(f"    Memory overhead: {timrun_stats['memory_management_overhead_ms']:.2f}ms")
        
        print(f"  Multi-Hop Tool Performance:")
        print(f"    Success rate: {orchestrator_stats['success_rate_percent']:.1f}%")
        print(f"    Average chain length: {orchestrator_stats['average_chain_length']:.1f}")
        print(f"    Total tool calls: {orchestrator_stats['total_tool_calls']}")


async def main():
    """Run the complete TIM demonstration"""
    try:
        demo = CompleteTIMDemo()
        await demo.run_complete_demo()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nDemo cleanup completed.")


if __name__ == "__main__":
    # Ensure proper async execution
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())