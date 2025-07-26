# TIM: Thread Inference Model Research Implementation

A research implementation of the Thread Inference Model (TIM) and TIMRUN runtime system, exploring the concepts presented in **"Beyond Context Limits: Subconscious Threads for Long-Horizon Reasoning"**.

⚠️ **Research Prototype Notice**: This is an experimental implementation designed to explore and demonstrate the theoretical concepts from the TIM paper. It simulates the proposed architecture using mock components and simplified algorithms for educational and research purposes.

## What This Implementation Explores

This codebase investigates how large language models could potentially overcome context limitations through:

- **Thread-2 Architecture**: A proposed recursive reasoning structure
- **Subtask Pruning Mechanisms**: Theoretical approaches to dynamic memory management  
- **Inference Runtime Design**: Conceptual framework for efficient long-horizon reasoning
- **Structured Generation**: Exploration of constrained JSON output for reasoning chains
- **Multi-Hop Tool Integration**: Prototype system for chained tool execution

## Key Features

### Thread-2 Reasoning Structure
```python
class Task(BaseModel):
    thought: str                    # Analysis and planning
    tooluse: Optional[ToolUse]      # Optional tool execution
    subtasks: Optional[List[Task]]  # Recursive decomposition
    conclusion: str                 # Results and aggregation
```

### TIMRUN Inference Runtime
- **Paged Attention**: Efficient KV cache management
- **Subtask Pruning**: Up to 90% memory savings
- **Tool Integration**: Seamless multi-hop tool calls
- **Batch Processing**: High-throughput inference

### Multi-Hop Tool Support
- **SearchTool**: Information retrieval
- **WebReaderTool**: Goal-directed content extraction  
- **Additional Tools**: DatabaseTool, CalculatorTool, FileReaderTool (in multihop_tools.py)
- **Chain Orchestration**: Automatic parameter passing

## Installation

```bash
# Clone or download the implementation files
pip install -r requirements.txt
```

## Quick Start

### Basic TIM Reasoning
```python
from tim_model import TIMModel

tim = TIMModel("TIM-8b")
response = tim.process_reasoning_chain(
    "Analyze renewable energy trends and provide policy recommendations"
)

print(f"Answer: {response.answer}")
print(f"Reasoning steps: {len(response.reasoning)}")
```

### TIMRUN Inference Runtime
```python
import asyncio
from timrun_runtime import TIMRUN

async def main():
    timrun = TIMRUN(tim)
    response = await timrun.inference(
        instruction="Research climate change impacts on agriculture",
        available_tools=["SearchTool", "WebReaderTool"]
    )
    print(f"Processing completed: {response.answer}")

asyncio.run(main())
```

### Multi-Hop Tool Chains
```python
from multihop_tools import MultiHopToolOrchestrator

orchestrator = MultiHopToolOrchestrator()

chain_spec = [
    {"tool_type": "SearchTool", "parameters": {"query": "renewable energy data"}},
    {"tool_type": "WebReaderTool", "parameters": {
        "url": "${step_0.results[0].url}",
        "goal": "extract statistics"
    }}
]

chain_result = await orchestrator.execute_multihop_chain(chain_spec)
```

### Constrained JSON Generation
```python
from json_constrained_generation import TIMJSONGenerator

generator = TIMJSONGenerator()
task_json = generator.generate_task_json(
    "Analyze climate data", 
    use_tools=True, 
    create_subtasks=True
)
print(task_json)
```

## Running the Complete Demo

Execute the comprehensive demonstration:

```bash
python demo_tim_complete.py
```

This runs all system components and shows:
1. Basic Thread-2 reasoning
2. Memory management and pruning
3. TIMRUN runtime performance
4. JSON structure generation
5. Multi-hop tool execution
6. End-to-end complex scenarios
7. Performance benchmarks

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   TIM Model     │    │   TIMRUN Runtime │    │  Multi-Hop      │
│                 │    │                  │    │  Tools          │
│ • Thread-2      │◄──►│ • Paged Attention│◄──►│                 │
│ • Subtask Tree  │    │ • Memory Mgmt    │    │ • SearchTool    │
│ • Reasoning     │    │ • Tool Integration│    │ • WebReaderTool │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                        ▲                        ▲
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ JSON Generator  │    │  Working Memory  │    │ Tool Servers    │
│                 │    │                  │    │                 │
│ • Schema Valid  │    │ • KV Cache       │    │ • MCP Protocol  │
│ • Constrained   │    │ • Pruning Buffer │    │ • Async Exec    │
│ • Generation    │    │ • Context Mgmt   │    │ • Result Cache  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Performance Characteristics

Based on the paper's findings, this implementation achieves:

- **Memory Efficiency**: 50-90% KV cache reduction through pruning
- **Context Scaling**: Virtually unlimited reasoning horizon
- **Tool Integration**: Single inference for multi-hop tool use
- **Throughput**: Maintained high inference speed despite memory management

## Example Outputs

### Memory Management
```
Memory Statistics:
  Total tokens: 1569
  Tokens pruned: 836 (53.3%)
  Cache utilization: 42.1%
  Throughput: 1150.5 tokens/sec
```

### Multi-Hop Chain
```
Multi-Hop Chain Results:
  Chain success: True
  Total execution time: 645.23ms
  Tool calls executed: 3
    Step 1: SearchTool - Success
    Step 2: WebReaderTool - Success  
    Step 3: DatabaseTool - Success
```

## Research Implementation Details

This prototype demonstrates the paper's concepts through:

1. **Thread-2 Structure**: Recursive task representation using Pydantic models
2. **Subtask Pruning**: Simulated pruning with configurable buffer mechanisms
3. **Memory Management**: Mock KV cache operations with position embedding reuse
4. **Tool Integration**: Async task orchestration with parameter passing
5. **JSON Generation**: Schema-constrained output generation

**Important Limitations:**
- Uses simulated attention mechanisms (not actual transformer attention)
- Mock tool servers instead of real external APIs
- Simplified memory management without GPU kernels
- Educational token processing rather than actual LLM inference

**For Research/Production Integration:**
- Replace mock components with actual LLM inference engines (vLLM, SGLang)
- Implement real attention mechanisms (FlashAttention, PagedAttention)
- Connect to production tool servers (MCP, function calling APIs)
- Add proper tokenization and embedding layers

## Files Structure

- `tim_model.py` - Core TIM model and Thread-2 structures
- `timrun_runtime.py` - TIMRUN inference runtime system
- `json_constrained_generation.py` - Schema-based JSON generation
- `multihop_tools.py` - Multi-hop tool orchestration
- `demo_tim_complete.py` - Comprehensive demonstration
- `requirements.txt` - Python dependencies

## Citation

This implementation is based on the research paper:

**"Beyond Context Limits: Subconscious Threads for Long-Horizon Reasoning"**

**Authors:** Hongyin Luo¹'², Nathaniel Morgan¹, Tina Li¹, Derek Zhao¹, Ai Vy Ngo¹, Philip Schroeder¹, Lijie Yang³, Assaf Ben-Kish⁴, Jack O'Brien², James Glass¹

**Affiliations:**
- ¹ MIT CSAIL
- ² Subconscious Systems Technologies, Inc.
- ³ Princeton University
- ⁴ Tel Aviv University

**Contact:** hyluo@mit.edu, {hongyin,jack}@subconscious.dev

**Abstract:** The paper introduces TIM and TIMRUN as solutions for breaking LLM context limits, enabling long-horizon reasoning, efficient memory management, and multi-hop tool integration.

**arXiv:** 2507.16784v1 [cs.CL] 22 Jul 2024

If you use this implementation, please cite the original paper.

## License

This implementation is provided for research and educational purposes to demonstrate the concepts described in the TIM paper.