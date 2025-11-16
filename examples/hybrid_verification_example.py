"""
Hybrid Verification Example

Demonstrates tri-modal verification using Neural + Symbolic + Geometric approaches.

Shows:
1. Individual verification modes (neural, symbolic, geometric)
2. Hybrid confidence scoring from all three sources
3. Comparison of verification approaches
4. When to use each approach
5. Complete workflow integrating all three
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuralog.core.types import Triple, Entity, Relation, ExtractionResult
from neuralog.core.config import Config
from neuralog.integration import HybridGeometricReasoner
from neuralog.symbolic import OntologyManager
from neuralog.neural import LLMInterface


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def example_1_hybrid_triple_validation():
    """Example 1: Validate single triple using tri-modal verification"""
    print_section("Example 1: Tri-Modal Triple Validation")

    # Create mock components (in real usage, these would be fully initialized)
    class MockOntologyManager:
        def get_entity_types(self, uri):
            return ["Person", "Concept"]

        def validate_triple(self, subject_type, predicate, object_type):
            # Simple mock validation
            return True, None

    class MockLLMInterface:
        def generate(self, prompt, **kwargs):
            return "Mock explanation"

    ontology_manager = MockOntologyManager()
    llm_interface = MockLLMInterface()

    # Create hybrid reasoner
    reasoner = HybridGeometricReasoner(
        ontology_manager=ontology_manager,
        llm_interface=llm_interface,
        enable_geometric=True,
        threshold_mode="hard",
    )

    # Create test triple
    subject = Entity(uri="http://example.org/Person1", label="Alice")
    subject.entity_type = "Person"

    predicate = Relation(uri="http://example.org/age", label="hasAge")

    object_entity = Entity(uri="http://example.org/Age70", label="70")
    object_entity.entity_type = "Concept"

    triple = Triple(
        subject=subject,
        predicate=predicate,
        object=object_entity,
        confidence=0.85,  # Neural confidence from LLM
    )

    # Create extraction
    extraction = ExtractionResult(
        entities=[subject, object_entity],
        relations=[predicate],
        triples=[triple],
        confidence=0.85,
    )

    print("Input Triple:")
    print(f"  {triple.to_tuple()}")
    print(f"  Neural confidence: {triple.confidence:.3f}")
    print()

    # Validate with tri-modal verification
    validated = reasoner.validate_extraction(
        extraction,
        use_symbolic=True,
        use_geometric=True,
    )

    print("After Tri-Modal Validation:")
    if validated.triples:
        validated_triple = validated.triples[0]
        print(f"  Hybrid confidence: {validated_triple.confidence:.3f}")
        print(f"  Confidence level: {validated_triple.confidence_level.value}")

        if validated_triple.provenance and 'confidence_breakdown' in validated_triple.provenance:
            breakdown = validated_triple.provenance['confidence_breakdown']
            print(f"\n  Confidence Breakdown:")
            print(f"    Neural:    {breakdown['neural']:.3f}")
            print(f"    Symbolic:  {breakdown['symbolic']:.3f}")
            print(f"    Geometric: {breakdown['geometric']:.3f}")
            print(f"    ─────────────────")
            print(f"    Hybrid:    {breakdown['hybrid']:.3f}")
    else:
        print("  Triple rejected by validation")


def example_2_compare_verification_modes():
    """Example 2: Compare different verification modes"""
    print_section("Example 2: Comparison of Verification Modes")

    class MockOntologyManager:
        def get_entity_types(self, uri):
            return ["Person"]

        def validate_triple(self, subject_type, predicate, object_type):
            return True, None

    class MockLLMInterface:
        def generate(self, prompt, **kwargs):
            return "Mock"

    ontology_manager = MockOntologyManager()
    llm_interface = MockLLMInterface()

    # Test with different configurations
    configs = [
        ("Neural only", False, False),
        ("Neural + Symbolic", True, False),
        ("Neural + Geometric", False, True),
        ("Tri-Modal (All three)", True, True),
    ]

    test_triple = Triple(
        subject=Entity(uri="http://ex.org/P1", label="Person"),
        predicate=Relation(uri="http://ex.org/age", label="age"),
        object=Entity(uri="http://ex.org/70", label="70"),
        confidence=0.75,
    )

    print(f"Test triple: {test_triple.to_tuple()}")
    print(f"Neural baseline: {test_triple.confidence:.3f}\n")

    for name, use_symbolic, use_geometric in configs:
        reasoner = HybridGeometricReasoner(
            ontology_manager=ontology_manager,
            llm_interface=llm_interface,
            enable_geometric=use_geometric,
        )

        extraction = ExtractionResult(
            entities=[],
            relations=[],
            triples=[test_triple],
            confidence=0.75,
        )

        validated = reasoner.validate_extraction(
            extraction,
            use_symbolic=use_symbolic,
            use_geometric=use_geometric,
        )

        if validated.triples:
            result_triple = validated.triples[0]
            print(f"{name:25s}: confidence = {result_triple.confidence:.3f}")
        else:
            print(f"{name:25s}: REJECTED")


def example_3_policy_verification():
    """Example 3: Policy-based verification using geometric algebra"""
    print_section("Example 3: Policy-Based Verification")

    class MockOntologyManager:
        def get_entity_types(self, uri):
            return []

        def validate_triple(self, subject_type, predicate, object_type):
            return True, None

    class MockLLMInterface:
        def generate(self, prompt, **kwargs):
            return "Mock"

    reasoner = HybridGeometricReasoner(
        ontology_manager=MockOntologyManager(),
        llm_interface=MockLLMInterface(),
        enable_geometric=True,
        threshold_mode="hard",
    )

    # Define policy: Senior (age >= 65) in low season with budget >= 22
    policy_spec = {
        "conditions": [
            {"type": "numeric", "value": 70, "threshold": 65},   # age check
            {"type": "binary", "value": True},                    # low season
            {"type": "numeric", "value": 25, "threshold": 22},   # budget check
        ],
        "combination": "and"
    }

    # Test Case 1: LLM correctly identifies eligible user
    print("Test Case 1: Eligible User")
    result1 = reasoner.verify_policy_chain(
        policy_spec=policy_spec,
        llm_answer="The user is eligible for admission"
    )
    print(f"  Verdict: {result1['verdict']}")
    print(f"  Policy truth: {result1['policy_truth']:.3f}")
    print(f"  LLM truth: {result1['llm_truth']:.3f}")

    # Test Case 2: Detect false positive
    print("\nTest Case 2: False Positive Detection")
    policy_spec_fail = {
        "conditions": [
            {"type": "numeric", "value": 60, "threshold": 65},   # age < 65 (fails)
            {"type": "binary", "value": True},
            {"type": "numeric", "value": 25, "threshold": 22},
        ],
        "combination": "and"
    }

    result2 = reasoner.verify_policy_chain(
        policy_spec=policy_spec_fail,
        llm_answer="The user is eligible"  # LLM hallucinates
    )
    print(f"  Verdict: {result2['verdict']}")
    print(f"  Policy truth: {result2['policy_truth']:.3f}")
    print(f"  LLM truth: {result2['llm_truth']:.3f}")


def example_4_counterfactual_analysis():
    """Example 4: Counterfactual "what-if" analysis"""
    print_section("Example 4: Counterfactual Analysis")

    class MockOntologyManager:
        def get_entity_types(self, uri):
            return []

        def validate_triple(self, subject_type, predicate, object_type):
            return True, None

    class MockLLMInterface:
        def generate(self, prompt, **kwargs):
            return "Mock"

    reasoner = HybridGeometricReasoner(
        ontology_manager=MockOntologyManager(),
        llm_interface=MockLLMInterface(),
        enable_geometric=True,
    )

    # Policy with age just below threshold
    policy_spec = {
        "conditions": [
            {"type": "numeric", "value": 63, "threshold": 65},   # age < 65
            {"type": "binary", "value": True},
            {"type": "numeric", "value": 25, "threshold": 22},
        ],
        "combination": "and"
    }

    print("Policy: Senior (age >= 65) with budget >= 22 in low season")
    print("Current: age=63 (below threshold)\n")

    # Analyze counterfactual: "What if age varied from 60 to 70?"
    analysis = reasoner.analyze_counterfactual(
        policy_spec=policy_spec,
        condition_index=0,  # Age condition
        value_range=(60, 70),
        num_steps=11,
    )

    if analysis:
        print("Counterfactual: Age variation from 60 to 70")
        print("-" * 50)
        print(f"Decision boundary: {analysis.get('decision_boundary', 'N/A')}")
        print(f"Robust: {analysis.get('is_robust', 'N/A')}")
        print()

        if 'values' in analysis:
            print("Age | Truth Degree | Eligible")
            print("-" * 40)
            for val, truth in zip(analysis['values'], analysis['truth_degrees']):
                eligible = "YES" if truth > 0.5 else "NO"
                print(f"{val:4.0f} | {truth:12.4f} | {eligible}")


def example_5_confidence_explanation():
    """Example 5: Generate explanations for hybrid confidence"""
    print_section("Example 5: Hybrid Confidence Explanation")

    class MockOntologyManager:
        def get_entity_types(self, uri):
            return ["Person"]

        def validate_triple(self, subject_type, predicate, object_type):
            return True, None

    class MockLLMInterface:
        def generate(self, prompt, **kwargs):
            return "Mock"

    reasoner = HybridGeometricReasoner(
        ontology_manager=MockOntologyManager(),
        llm_interface=MockLLMInterface(),
        enable_geometric=True,
    )

    # Create and validate triple
    triple = Triple(
        subject=Entity(uri="http://ex.org/Alice", label="Alice"),
        predicate=Relation(uri="http://ex.org/worksAt", label="worksAt"),
        object=Entity(uri="http://ex.org/Stanford", label="Stanford"),
        confidence=0.88,  # Neural confidence
    )

    extraction = ExtractionResult(
        entities=[],
        relations=[],
        triples=[triple],
        confidence=0.88,
    )

    validated = reasoner.validate_extraction(extraction)

    if validated.triples:
        validated_triple = validated.triples[0]

        # Generate explanation
        explanation = reasoner.get_confidence_explanation(validated_triple)
        print(explanation)


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("  HYBRID GEOMETRIC-SYMBOLIC-NEURAL VERIFICATION EXAMPLES")
    print("  Tri-Modal Validation for Neurosymbolic AI")
    print("=" * 80)

    try:
        example_1_hybrid_triple_validation()
        example_2_compare_verification_modes()
        example_3_policy_verification()
        example_4_counterfactual_analysis()
        example_5_confidence_explanation()

        print("\n" + "=" * 80)
        print("  ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

        print("Key Takeaways:")
        print("  1. Tri-modal verification combines strengths of all approaches")
        print("  2. Hybrid confidence provides nuanced, multi-source validation")
        print("  3. Geometric verification enables differentiability")
        print("  4. Counterfactual analysis answers 'what-if' questions")
        print("  5. Explainable confidence breakdowns aid debugging")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
