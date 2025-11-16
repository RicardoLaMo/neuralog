"""
NeuraLog: Neural Symbolic AI for Information Extraction

A modern framework combining symbolic reasoning with neural learning for
reliable, explainable knowledge extraction from unstructured data.
"""

__version__ = "0.1.0"
__author__ = "NeuraLog Contributors"

from neuralog.core import Engine
from neuralog.symbolic import OntologyManager, GraphEmbedder
from neuralog.neural import LLMInterface, EmbeddingModel
from neuralog.integration import NeurosymbolicReasoner, KGDistiller

__all__ = [
    "__version__",
    "__author__",
    "Engine",
    "OntologyManager",
    "GraphEmbedder",
    "LLMInterface",
    "EmbeddingModel",
    "NeurosymbolicReasoner",
    "KGDistiller",
]
