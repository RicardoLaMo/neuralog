"""
LLM interface with ontology-aware prompt engineering.

Supports multiple LLM providers with consistent interface and
ontology-guided prompt templates.
"""

from typing import Any, Dict, List, Optional, Union
from loguru import logger
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from neuralog.core.config import LLMConfig
from neuralog.core.types import PromptTemplate


class LLMInterface:
    """
    Interface to large language models with ontology-aware prompting.

    Features:
    - Multi-provider support (OpenAI, Anthropic, local models)
    - Ontology schema injection for guided generation
    - Few-shot learning with ontology examples
    - Chain-of-thought prompting with symbolic structure
    - Structured output parsing
    """

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM interface.

        Args:
            config: LLM configuration
        """
        self.config = config
        self.client = None

        self._initialize_client()
        logger.info(f"LLMInterface initialized with provider: {config.provider}")

    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        if self.config.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install with: pip install openai")
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url
            )
        elif self.config.provider == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic not available. Install with: pip install anthropic")
            self.client = Anthropic(api_key=self.config.api_key)
        elif self.config.provider == "ollama":
            # Use OpenAI-compatible client for Ollama
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI client needed for Ollama. Install: pip install openai")
            self.client = OpenAI(
                base_url=self.config.base_url or "http://localhost:11434/v1",
                api_key="ollama"  # Ollama doesn't require API key
            )
        elif self.config.provider == "vllm":
            # vLLM for local GPU inference (H200)
            if not VLLM_AVAILABLE:
                raise ImportError(
                    "vLLM not available. Install with: pip install 'neuralog[production]'"
                )
            self._initialize_vllm()
        elif self.config.provider == "vllm-server":
            # vLLM OpenAI-compatible server
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI client needed for vLLM server")
            self.client = OpenAI(
                base_url=self.config.base_url or "http://localhost:8000/v1",
                api_key="vllm"  # vLLM server doesn't require API key
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.provider}")

    def _initialize_vllm(self):
        """Initialize vLLM engine for local inference."""
        logger.info(f"Initializing vLLM with model: {self.config.model}")

        # Get vLLM-specific config from extra params
        gpu_memory_utilization = getattr(self.config, 'gpu_memory_utilization', 0.9)
        tensor_parallel_size = getattr(self.config, 'tensor_parallel_size', 1)
        max_model_len = getattr(self.config, 'max_model_len', None)
        trust_remote_code = getattr(self.config, 'trust_remote_code', True)
        dtype = getattr(self.config, 'dtype', 'auto')  # auto, float16, bfloat16

        # Initialize vLLM engine
        self.vllm_engine = LLM(
            model=self.config.model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            enforce_eager=False,  # Use CUDA graphs for speed
            disable_log_stats=False,
        )

        logger.info(
            f"vLLM engine initialized: "
            f"GPU mem={gpu_memory_utilization}, "
            f"TP={tensor_parallel_size}, "
            f"dtype={dtype}"
        )

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt
            system_message: System message (optional)
            temperature: Sampling temperature (uses config default if None)
            max_tokens: Maximum tokens to generate (uses config default if None)
            **kwargs: Additional provider-specific arguments

        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_tokens

        if self.config.provider == "vllm":
            # Direct vLLM inference
            return self._generate_vllm(prompt, system_message, temp, max_tok, **kwargs)

        elif self.config.provider in ["openai", "ollama", "vllm-server"]:
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                **kwargs
            )
            return response.choices[0].message.content

        elif self.config.provider == "anthropic":
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=max_tok,
                temperature=temp,
                system=system_message or "",
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.content[0].text

        else:
            raise NotImplementedError(f"Generation not implemented for {self.config.provider}")

    def _generate_vllm(
        self,
        prompt: str,
        system_message: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate using vLLM engine."""
        # Format prompt with system message
        if system_message:
            # Use chat template if available
            try:
                # Try to get tokenizer for chat template
                tokenizer = self.vllm_engine.get_tokenizer()
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})

                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                # Fallback: simple concatenation
                formatted_prompt = f"{system_message}\n\n{prompt}"
        else:
            formatted_prompt = prompt

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.config.top_p,
            **kwargs
        )

        # Generate
        outputs = self.vllm_engine.generate([formatted_prompt], sampling_params)

        # Return first output
        return outputs[0].outputs[0].text

    def extract_entities_relations(
        self,
        text: str,
        ontology_schema: Optional[Dict[str, Any]] = None,
        examples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extract entities and relations from text using LLM.

        Args:
            text: Input text
            ontology_schema: Ontology schema for guided extraction
            examples: Few-shot examples

        Returns:
            Dictionary with extracted entities and relations
        """
        logger.info(f"Extracting entities/relations from text (length: {len(text)})")

        # Build ontology-aware prompt
        prompt = self._build_extraction_prompt(text, ontology_schema, examples)

        # Generate with structured output
        system_message = (
            "You are an expert information extraction system. "
            "Extract entities and relations from the given text, "
            "following the provided ontology schema. "
            "Return results as valid JSON."
        )

        response = self.generate(
            prompt=prompt,
            system_message=system_message,
            temperature=0.1  # Low temperature for consistent extraction
        )

        # Parse JSON response
        try:
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM output as JSON, attempting to extract")
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            else:
                logger.error("Could not extract valid JSON from response")
                return {"entities": [], "relations": []}

    def _build_extraction_prompt(
        self,
        text: str,
        ontology_schema: Optional[Dict[str, Any]],
        examples: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Build ontology-aware extraction prompt."""
        prompt_parts = []

        # Add ontology schema if provided
        if ontology_schema:
            prompt_parts.append("## Ontology Schema\n")
            prompt_parts.append("Extract entities and relations according to this schema:\n\n")

            if "classes" in ontology_schema and ontology_schema["classes"]:
                prompt_parts.append("### Entity Types:\n")
                for cls in ontology_schema["classes"][:20]:  # Limit to top 20
                    label = cls.get("label", "")
                    desc = cls.get("description", "")
                    prompt_parts.append(f"- {label}: {desc}\n")

            if "object_properties" in ontology_schema and ontology_schema["object_properties"]:
                prompt_parts.append("\n### Relation Types:\n")
                for prop in ontology_schema["object_properties"][:20]:  # Limit to top 20
                    label = prop.get("label", "")
                    desc = prop.get("description", "")
                    prompt_parts.append(f"- {label}: {desc}\n")

            prompt_parts.append("\n")

        # Add examples if provided
        if examples:
            prompt_parts.append("## Examples\n\n")
            for ex in examples[:3]:  # Limit to 3 examples
                prompt_parts.append(f"Text: {ex.get('text', '')}\n")
                prompt_parts.append(f"Output: {json.dumps(ex.get('output', {}), indent=2)}\n\n")

        # Add task description
        prompt_parts.append("## Task\n\n")
        prompt_parts.append("Extract entities and relations from the following text:\n\n")
        prompt_parts.append(f"Text: {text}\n\n")
        prompt_parts.append("Return the result as JSON with this structure:\n")
        prompt_parts.append("{\n")
        prompt_parts.append('  "entities": [\n')
        prompt_parts.append('    {"text": "entity mention", "type": "EntityType", "span": [start, end]}\n')
        prompt_parts.append('  ],\n')
        prompt_parts.append('  "relations": [\n')
        prompt_parts.append('    {"subject": "entity1", "predicate": "RelationType", "object": "entity2"}\n')
        prompt_parts.append('  ]\n')
        prompt_parts.append('}\n')

        return "".join(prompt_parts)

    def nl_to_sparql(
        self,
        question: str,
        ontology: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Convert natural language question to SPARQL query.

        Args:
            question: Natural language question
            ontology: Ontology schema

        Returns:
            SPARQL query string
        """
        logger.info(f"Converting question to SPARQL: {question}")

        prompt = f"""Convert the following natural language question to a SPARQL query.

Question: {question}

Return only the SPARQL query, without any explanation.
"""

        if ontology:
            # Add ontology context
            prompt = f"Ontology schema:\n{json.dumps(ontology, indent=2)}\n\n" + prompt

        response = self.generate(
            prompt=prompt,
            temperature=0.1
        )

        # Extract SPARQL query
        if "```sparql" in response:
            query = response.split("```sparql")[1].split("```")[0].strip()
        elif "```" in response:
            query = response.split("```")[1].split("```")[0].strip()
        else:
            query = response.strip()

        return query

    def formalize_statement(
        self,
        statement: str,
        logic_formalism: str = "FOL"
    ) -> str:
        """
        Formalize natural language statement to logic (ARc-inspired).

        Args:
            statement: Natural language statement
            logic_formalism: Target formalism (FOL, OWL, etc.)

        Returns:
            Formalized logical statement
        """
        logger.info(f"Formalizing statement to {logic_formalism}")

        prompt = f"""Convert the following natural language statement to {logic_formalism}.

Statement: {statement}

Provide the formal logical representation.
"""

        return self.generate(prompt=prompt, temperature=0.1)

    def verbalize_axiom(self, axiom: str, ontology_context: Optional[str] = None) -> str:
        """
        Convert logical axiom to natural language.

        Args:
            axiom: Logical axiom
            ontology_context: Ontology context

        Returns:
            Natural language verbalization
        """
        prompt = f"Convert this logical axiom to natural language:\n\n{axiom}"

        if ontology_context:
            prompt = f"Context: {ontology_context}\n\n" + prompt

        return self.generate(prompt=prompt, temperature=0.1)
