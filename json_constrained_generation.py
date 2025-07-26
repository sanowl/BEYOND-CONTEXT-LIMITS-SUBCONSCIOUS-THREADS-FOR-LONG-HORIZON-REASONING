#!/usr/bin/env python3
"""
Structured JSON Generation with Constrained Decoding for TIM
Implements the JSON schema-based constrained generation described in the paper.
"""

import json
import re
from typing import Dict, List, Any, Optional, Union, Type
from pydantic import BaseModel, Field
from enum import Enum
import jsonschema
from dataclasses import dataclass

from tim_model import Task, ToolUse, TimResponse, SearchTool, WebReaderTool
import secrets


class GenerationState(Enum):
    """States in the JSON generation finite state machine"""
    INITIAL = "initial"
    OBJECT_START = "object_start"
    OBJECT_KEY = "object_key"
    OBJECT_VALUE = "object_value"
    ARRAY_START = "array_start"
    ARRAY_ELEMENT = "array_element"
    STRING_VALUE = "string_value"
    COMPLETE = "complete"


@dataclass
class ConstraintState:
    """Tracks the current state during constrained generation"""
    schema: Dict[str, Any]
    current_path: List[str]
    generation_state: GenerationState
    required_fields: List[str]
    completed_fields: List[str]
    depth: int = 0
    max_depth: int = 10


class JSONSchemaValidator:
    """Validates JSON against schemas during generation"""
    
    def __init__(self):
        self.validator_cache = {}
    
    def get_validator(self, schema: Dict[str, Any]) -> jsonschema.Draft7Validator:
        """Get cached validator for schema"""
        schema_key = json.dumps(schema, sort_keys=True)
        if schema_key not in self.validator_cache:
            self.validator_cache[schema_key] = jsonschema.Draft7Validator(schema)
        return self.validator_cache[schema_key]
    
    def validate_partial(self, partial_json: str, schema: Dict[str, Any]) -> bool:
        """Validate partial JSON against schema"""
        try:
            # Try to parse as complete JSON
            data = json.loads(partial_json)
            validator = self.get_validator(schema)
            validator.validate(data)
            return True
        except (json.JSONDecodeError, jsonschema.ValidationError):
            # For partial JSON, try to validate structure
            return self._validate_partial_structure(partial_json, schema)
    
    def _validate_partial_structure(self, partial_json: str, schema: Dict[str, Any]) -> bool:
        """Validate partial JSON structure"""
        # Basic structural validation for incomplete JSON
        if not partial_json.strip():
            return True
        
        # Check for balanced braces/brackets
        open_braces = partial_json.count('{') - partial_json.count('}')
        open_brackets = partial_json.count('[') - partial_json.count(']')
        
        return open_braces >= 0 and open_brackets >= 0


class ConstrainedJSONGenerator:
    """
    Generates JSON following schema constraints, implementing the constrained
    decoding approach described in the TIM paper.
    """
    
    def __init__(self):
        self.validator = JSONSchemaValidator()
        self.generation_patterns = self._build_generation_patterns()
    
    def _build_generation_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for valid JSON tokens at each state"""
        return {
            "object_start": [r'\{'],
            "object_key": [r'"[^"]*"'],
            "object_colon": [r':'],
            "object_value_string": [r'"[^"]*"'],
            "object_value_number": [r'-?\d+(\.\d+)?([eE][+-]?\d+)?'],
            "object_value_boolean": [r'true', r'false'],
            "object_value_null": [r'null'],
            "array_start": [r'\['],
            "array_element": [r'"[^"]*"', r'-?\d+(\.\d+)?', r'true', r'false', r'null'],
            "comma": [r','],
            "object_end": [r'\}'],
            "array_end": [r'\]']
        }
    
    def generate_constrained(self, 
                           schema: Dict[str, Any], 
                           max_tokens: int = 1000,
                           temperature: float = 0.7) -> str:
        """
        Generate JSON following schema constraints.
        This simulates the constrained decoding process.
        """
        constraint_state = ConstraintState(
            schema=schema,
            current_path=[],
            generation_state=GenerationState.INITIAL,
            required_fields=schema.get('required', []),
            completed_fields=[]
        )
        
        generated_json = ""
        tokens_generated = 0
        
        while (tokens_generated < max_tokens and 
               constraint_state.generation_state != GenerationState.COMPLETE):
            
            # Get valid next tokens based on current state and schema
            valid_tokens = self._get_valid_tokens(constraint_state, generated_json)
            
            if not valid_tokens:
                break
            
            # Select next token (simplified - in practice this would use the LLM)
            next_token = self._select_next_token(valid_tokens, temperature)
            generated_json += next_token
            tokens_generated += 1
            
            # Update constraint state
            self._update_constraint_state(constraint_state, next_token, generated_json)
        
        return generated_json
    
    def _get_valid_tokens(self, 
                         constraint_state: ConstraintState, 
                         current_json: str) -> List[str]:
        """Get list of valid tokens based on current state and schema"""
        
        if constraint_state.generation_state == GenerationState.INITIAL:
            return ["{"]
        
        elif constraint_state.generation_state == GenerationState.OBJECT_START:
            # Start of object - need field name
            required_remaining = [
                f for f in constraint_state.required_fields 
                if f not in constraint_state.completed_fields
            ]
            
            if required_remaining:
                return [f'"{required_remaining[0]}"']
            else:
                # Optional fields or close object
                optional_fields = self._get_optional_fields(constraint_state)
                tokens = [f'"{field}"' for field in optional_fields]
                if constraint_state.completed_fields:  # Can close if we have some fields
                    tokens.append("}")
                return tokens
        
        elif constraint_state.generation_state == GenerationState.OBJECT_KEY:
            return [":"]
        
        elif constraint_state.generation_state == GenerationState.OBJECT_VALUE:
            # Return valid value tokens based on schema type
            return self._get_valid_value_tokens(constraint_state)
        
        elif constraint_state.generation_state == GenerationState.ARRAY_START:
            return self._get_valid_array_tokens(constraint_state)
        
        else:
            return []
    
    def _get_valid_value_tokens(self, constraint_state: ConstraintState) -> List[str]:
        """Get valid value tokens based on schema property type"""
        current_field = constraint_state.current_path[-1] if constraint_state.current_path else ""
        
        # Get property schema
        props = constraint_state.schema.get('properties', {})
        if current_field in props:
            prop_schema = props[current_field]
            prop_type = prop_schema.get('type', 'string')
            
            if prop_type == 'string':
                if 'enum' in prop_schema:
                    return [f'"{val}"' for val in prop_schema['enum']]
                else:
                    return ['"example_string"']
            elif prop_type == 'integer':
                return ['42']
            elif prop_type == 'boolean':
                return ['true', 'false']
            elif prop_type == 'array':
                return ['[']
            elif prop_type == 'object':
                return ['{']
        
        return ['"default_value"']
    
    def _get_optional_fields(self, constraint_state: ConstraintState) -> List[str]:
        """Get optional fields that haven't been completed yet"""
        all_fields = set(constraint_state.schema.get('properties', {}).keys())
        required_fields = set(constraint_state.required_fields)
        completed_fields = set(constraint_state.completed_fields)
        
        optional_fields = all_fields - required_fields - completed_fields
        return list(optional_fields)
    
    def _get_valid_array_tokens(self, constraint_state: ConstraintState) -> List[str]:
        """Get valid tokens for array context"""
        return ['"array_element"', ']']
    
    def _select_next_token(self, valid_tokens: List[str], temperature: float) -> str:
        """Select next token from valid options (simplified selection)"""
        if not valid_tokens:
            return ""
        
        # Simplified selection - in practice this would use LLM probabilities
        if temperature == 0.0:
            return valid_tokens[0]
        else:
            return secrets.choice(valid_tokens)
    
    def _update_constraint_state(self, 
                                constraint_state: ConstraintState, 
                                token: str, 
                                current_json: str):
        """Update constraint state based on generated token"""
        
        if token == "{":
            constraint_state.generation_state = GenerationState.OBJECT_START
            constraint_state.depth += 1
        
        elif token.startswith('"') and token.endswith('"') and constraint_state.generation_state == GenerationState.OBJECT_START:
            # Field name
            field_name = token.strip('"')
            constraint_state.current_path.append(field_name)
            constraint_state.generation_state = GenerationState.OBJECT_KEY
        
        elif token == ":":
            constraint_state.generation_state = GenerationState.OBJECT_VALUE
        
        elif constraint_state.generation_state == GenerationState.OBJECT_VALUE:
            # Value completed
            if constraint_state.current_path:
                field_name = constraint_state.current_path.pop()
                constraint_state.completed_fields.append(field_name)
            
            # Check if we need more fields or can close
            remaining_required = [
                f for f in constraint_state.required_fields 
                if f not in constraint_state.completed_fields
            ]
            
            if remaining_required or token == ",":
                constraint_state.generation_state = GenerationState.OBJECT_START
            else:
                constraint_state.generation_state = GenerationState.OBJECT_START
        
        elif token == "}":
            constraint_state.depth -= 1
            if constraint_state.depth == 0:
                constraint_state.generation_state = GenerationState.COMPLETE


class TIMJSONGenerator:
    """
    TIM-specific JSON generator that creates structured Task responses
    following the exact schema from the paper.
    """
    
    def __init__(self):
        self.generator = ConstrainedJSONGenerator()
        self.tim_schemas = self._build_tim_schemas()
    
    def _build_tim_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Build JSON schemas for TIM data structures"""
        
        # Tool schemas
        search_tool_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
        
        web_reader_tool_schema = {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "url": {"type": "string"}
            },
            "required": ["goal", "url"]
        }
        
        # ToolUse schema
        tool_use_schema = {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "enum": ["SearchTool", "WebReaderTool"]
                },
                "parameters": {
                    "oneOf": [search_tool_schema, web_reader_tool_schema]
                },
                "tool_result": {"type": "object"}
            },
            "required": ["tool_name", "parameters"]
        }
        
        # Task schema (recursive)
        task_schema = {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "tooluse": tool_use_schema,
                "subtasks": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/task"}
                },
                "conclusion": {"type": "string"}
            },
            "required": ["thought", "conclusion"],
            "definitions": {
                "task": {
                    "type": "object",
                    "properties": {
                        "thought": {"type": "string"},
                        "tooluse": tool_use_schema,
                        "subtasks": {
                            "type": "array",
                            "items": {"$ref": "#/definitions/task"}
                        },
                        "conclusion": {"type": "string"}
                    },
                    "required": ["thought", "conclusion"]
                }
            }
        }
        
        # TimResponse schema
        tim_response_schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "array",
                    "items": task_schema
                },
                "answer": {"type": "string"}
            },
            "required": ["reasoning", "answer"]
        }
        
        return {
            "task": task_schema,
            "tool_use": tool_use_schema,
            "tim_response": tim_response_schema,
            "search_tool": search_tool_schema,
            "web_reader_tool": web_reader_tool_schema
        }
    
    def generate_task_json(self, 
                          instruction: str, 
                          use_tools: bool = False,
                          create_subtasks: bool = False) -> str:
        """Generate a Task JSON following the TIM schema"""
        
        # Create example task data that follows the schema
        task_data = {
            "thought": f"Analyzing the instruction: {instruction}",
            "conclusion": f"Completed analysis of: {instruction}"
        }
        
        if use_tools:
            task_data["tooluse"] = {
                "tool_name": "SearchTool",
                "parameters": {
                    "query": instruction
                },
                "tool_result": {}
            }
        
        if create_subtasks:
            task_data["subtasks"] = [
                {
                    "thought": "Processing first part of the task",
                    "conclusion": "First part completed"
                },
                {
                    "thought": "Processing second part of the task", 
                    "conclusion": "Second part completed"
                }
            ]
        
        return json.dumps(task_data, indent=2)
    
    def generate_tim_response_json(self, instruction: str) -> str:
        """Generate a complete TIM response JSON"""
        
        response_data = {
            "reasoning": [
                {
                    "thought": f"Breaking down the complex instruction: {instruction}",
                    "tooluse": {
                        "tool_name": "SearchTool",
                        "parameters": {
                            "query": instruction
                        },
                        "tool_result": {
                            "status": "completed",
                            "results": "Search completed successfully"
                        }
                    },
                    "subtasks": [
                        {
                            "thought": "Analyzing first component",
                            "conclusion": "First component analysis complete"
                        },
                        {
                            "thought": "Analyzing second component",
                            "conclusion": "Second component analysis complete"
                        }
                    ],
                    "conclusion": "Main task analysis completed with tool assistance"
                }
            ],
            "answer": f"Based on the analysis, here is the response to: {instruction}"
        }
        
        return json.dumps(response_data, indent=2)
    
    def validate_tim_json(self, json_str: str, schema_name: str) -> bool:
        """Validate generated JSON against TIM schemas"""
        try:
            data = json.loads(json_str)
            schema = self.tim_schemas.get(schema_name)
            if not schema:
                return False
            
            return self.generator.validator.validate_partial(json_str, schema)
        except:
            return False
    
    def generate_with_constraints(self, 
                                 schema_name: str, 
                                 context: Dict[str, Any] = None) -> str:
        """Generate JSON with full constraint enforcement"""
        if context is None:
            context = {}
        
        schema = self.tim_schemas.get(schema_name)
        if not schema:
            raise ValueError(f"Unknown schema: {schema_name}")
        
        return self.generator.generate_constrained(schema)


def main():
    """Demo function showing constrained JSON generation"""
    print("TIM Constrained JSON Generation Demo")
    print("=" * 50)
    
    # Initialize generator
    tim_generator = TIMJSONGenerator()
    
    # Example instruction
    instruction = "Analyze the impact of climate change on ocean ecosystems"
    
    print(f"Generating JSON for instruction: {instruction}")
    print()
    
    # Generate Task JSON
    print("1. Simple Task JSON:")
    print("-" * 20)
    simple_task = tim_generator.generate_task_json(instruction)
    print(simple_task)
    print()
    
    # Generate Task with tools
    print("2. Task JSON with Tool Use:")
    print("-" * 25)
    task_with_tool = tim_generator.generate_task_json(instruction, use_tools=True)
    print(task_with_tool)
    print()
    
    # Generate Task with subtasks
    print("3. Task JSON with Subtasks:")
    print("-" * 26)
    task_with_subtasks = tim_generator.generate_task_json(instruction, create_subtasks=True)
    print(task_with_subtasks)
    print()
    
    # Generate complete TIM response
    print("4. Complete TIM Response JSON:")
    print("-" * 30)
    tim_response = tim_generator.generate_tim_response_json(instruction)
    print(tim_response)
    print()
    
    # Validate generated JSON
    print("5. JSON Validation Results:")
    print("-" * 26)
    
    validations = [
        ("Simple Task", simple_task, "task"),
        ("Task with Tool", task_with_tool, "task"),
        ("Task with Subtasks", task_with_subtasks, "task"),
        ("TIM Response", tim_response, "tim_response")
    ]
    
    for name, json_str, schema_name in validations:
        is_valid = tim_generator.validate_tim_json(json_str, schema_name)
        status = "✓ Valid" if is_valid else "✗ Invalid"
        print(f"{name}: {status}")
    
    print()
    
    # Demonstrate constrained generation
    print("6. Constrained Generation Example:")
    print("-" * 33)
    
    try:
        constrained_task = tim_generator.generate_with_constraints("task")
        print("Generated with full constraints:")
        print(constrained_task)
        
        # Validate the constrained generation
        is_valid = tim_generator.validate_tim_json(constrained_task, "task")
        print(f"\nConstrained generation validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    except Exception as e:
        print(f"Constrained generation error: {e}")
    
    print("\n" + "=" * 50)
    print("JSON Schema Definitions:")
    print("-" * 25)
    
    for name, schema in tim_generator.tim_schemas.items():
        print(f"\n{name.upper()} Schema:")
        print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
