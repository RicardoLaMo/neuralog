"""Example: Complete Geometric Algebra Reasoning Pipeline.

Demonstrates:
1. Policy definition and registration
2. Fact extraction and verification
3. Counterfactual explanation generation
4. Multi-turn LLM refinement
5. Real-time metrics monitoring
"""

from neuralog.symbolic.geometric_algebra import (
    CounterfactualReasoner,
    NeuraLogGASystem,
    Policy,
    PolicyCondition,
    PolicyRule,
    SystemConfig,
    ThresholdMode,
)
from neuralog.symbolic.geometric_algebra.policy_extractor_pipeline import (
    PolicyExtractorPipeline,
)


def example_1_basic_policy_verification():
    """Example 1: Basic policy verification without LLM."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Policy Verification")
    print("=" * 70)

    # Create system
    config = SystemConfig(
        device="cpu",
        batch_size=32,
        enable_metrics=True,
    )
    system = NeuraLogGASystem(config)

    # Define policy: Senior Discount
    # IF age >= 65 AND season="low-season" AND budget >= 22
    # THEN eligible_for_discount = TRUE
    age_condition = PolicyCondition(
        subject="age",
        predicate=">=",
        threshold=65.0,
        threshold_mode=ThresholdMode.SOFT,
    )

    season_condition = PolicyCondition(
        subject="season",
        predicate="=",
        threshold=0.5,  # Encodes "low-season"
        threshold_mode=ThresholdMode.SOFT,
    )

    budget_condition = PolicyCondition(
        subject="budget",
        predicate=">=",
        threshold=22.0,
        threshold_mode=ThresholdMode.SOFT,
    )

    eligible_condition = PolicyCondition(
        subject="eligible",
        predicate="=",
        threshold=0.5,
        threshold_mode=ThresholdMode.HARD,
    )

    rule = PolicyRule(
        name="senior_discount_rule",
        conditions=[age_condition, season_condition, budget_condition],
        conclusion=eligible_condition,
        operator="and",
    )

    policy = Policy(name="senior_discount_policy", rules=[rule])
    system.register_policy("senior_discount", policy.__dict__)

    # Verify facts
    facts = [
        {"age": 72.0, "season": 0.8, "budget": 25.0, "eligible": 1.0},  # Satisfies
        {"age": 55.0, "season": 0.3, "budget": 19.0, "eligible": 0.0},  # Doesn't satisfy
        {"age": 68.0, "season": 0.6, "budget": 23.0, "eligible": 1.0},  # Satisfies
    ]

    print("\nVerifying facts against policy...")
    results = system.verify_facts(facts, policy_names=["senior_discount"])

    for i, result in enumerate(results):
        print(
            f"\nFact {i}: "
            f"Consistent={result.is_consistent}, "
            f"Confidence={result.confidence_score:.2%}"
        )

    # Print metrics
    print("\n" + "-" * 70)
    print(system.log_metrics())


def example_2_counterfactual_generation():
    """Example 2: Generate counterfactual explanations."""
    print("\n" + "=" * 70)
    print("Example 2: Counterfactual Explanations")
    print("=" * 70)

    reasoner = CounterfactualReasoner(
        lr=0.1,
        max_steps=100,
        device="cpu",
        distance_metric="l2",
    )

    policy_spec = {
        "name": "senior_discount",
        "conditions": [
            {"subject": "age", "threshold": 65.0, "mode": "soft"},
            {"subject": "budget", "threshold": 22.0, "mode": "soft"},
        ],
        "operator": "and",
    }

    # Find counterfactual for someone who doesn't satisfy the policy
    original_values = {"age": 60.0, "budget": 20.0}

    print(f"\nOriginal values: {original_values}")
    print("Finding minimal adjustment to satisfy policy...")

    cf = reasoner.find_counterfactual(
        original_values,
        policy_spec,
        value_ranges={"age": (18, 120), "budget": (0, 100)},
    )

    print(f"\nCounterfactual values: {cf.counterfactual_values}")
    print(f"Distance (L2): {cf.distance:.3f}")
    print(f"Steps needed: {cf.steps_to_satisfy}")
    print(f"Policy satisfied: {cf.satisfied}")
    print(f"Constraints satisfied: {cf.constraints}")


def example_3_batch_processing():
    """Example 3: Batch processing with metrics."""
    print("\n" + "=" * 70)
    print("Example 3: Batch Processing with Metrics")
    print("=" * 70)

    config = SystemConfig(
        device="cpu",
        batch_size=64,
        enable_metrics=True,
        enable_counterfactuals=False,
    )
    system = NeuraLogGASystem(config)

    # Register policy
    policy_spec = {
        "name": "adult_policy",
        "conditions": [
            {"subject": "age", "threshold": 18.0, "mode": "soft"},
        ],
        "operator": "and",
    }
    system.register_policy("adult_policy", policy_spec)

    # Generate large batch of facts
    facts = []
    for i in range(1000):
        age = 15 + (i % 70)  # Ages 15-84
        facts.append({"age": float(age)})

    print(f"\nProcessing {len(facts)} facts...")
    results = system.verify_facts(facts)

    print(f"Results: {len(results)} facts evaluated")

    # Get metrics
    metrics = system.get_metrics_summary()
    print(f"\nMetrics:")
    print(f"  Mean truth degree: {metrics.mean_truth_degree:.3f}")
    print(f"  Satisfaction rate: {metrics.satisfaction_rate:.1%}")
    print(f"  Throughput: {metrics.throughput.facts_per_second:.0f} facts/sec")


def example_4_pipeline_extraction():
    """Example 4: Policy-aware extraction pipeline (without LLM)."""
    print("\n" + "=" * 70)
    print("Example 4: Policy-Aware Extraction Pipeline")
    print("=" * 70)

    config = SystemConfig(device="cpu", batch_size=8)
    ga_system = NeuraLogGASystem(config)

    # Register policy
    ga_system.register_policy(
        "senior_discount",
        {
            "name": "senior_discount",
            "conditions": [
                {"subject": "age", "threshold": 65.0, "mode": "soft"},
            ],
            "operator": "and",
        },
    )

    # Create pipeline without LLM (will use mock extraction)
    pipeline = PolicyExtractorPipeline(
        ga_system=ga_system,
        llm_interface=None,
        max_refinement_rounds=1,
        enable_counterfactuals=True,
    )

    text = "A customer aged 72 is seeking senior discount eligibility"

    print(f"\nProcessing: '{text}'")
    print("(Using mock LLM extraction)")

    result = pipeline.extract_and_verify(
        text,
        policy_names=["senior_discount"],
        enable_refinement=False,
    )

    print(f"\nExtraction rounds: {result.extraction_round + 1}")
    print(f"Compliant: {result.is_policy_compliant}")
    print(f"Compliance score: {result.compliance_score:.2%}")

    if result.counterfactuals:
        print(f"\nCounterfactuals:")
        for cf in result.counterfactuals:
            print(f"  - Adjust to: {cf.counterfactual_values}")
            print(f"    Distance: {cf.distance:.3f}")

    # Print compliance report
    print("\n" + pipeline.get_compliance_report(result))


def example_5_multiple_policies():
    """Example 5: Multiple policies and composition."""
    print("\n" + "=" * 70)
    print("Example 5: Multiple Policies")
    print("=" * 70)

    config = SystemConfig(device="cpu")
    system = NeuraLogGASystem(config)

    # Define multiple policies
    system.register_policy(
        "senior_discount",
        {
            "name": "senior_discount",
            "conditions": [
                {"subject": "age", "threshold": 65.0, "mode": "soft"},
            ],
            "operator": "and",
        },
    )

    system.register_policy(
        "young_learner",
        {
            "name": "young_learner",
            "conditions": [
                {"subject": "age", "threshold": 18.0, "mode": "soft"},
                {"subject": "age", "threshold": 25.0, "mode": "soft"},  # and age < 25
            ],
            "operator": "and",
        },
    )

    system.register_policy(
        "loyalty_program",
        {
            "name": "loyalty_program",
            "conditions": [
                {"subject": "orders", "threshold": 10.0, "mode": "soft"},
            ],
            "operator": "and",
        },
    )

    print(f"Registered policies: {list(system.policies.keys())}")

    # Test facts against all policies
    facts = [
        {"age": 72.0, "orders": 20.0},  # Senior + loyalty
        {"age": 22.0, "orders": 5.0},   # Young learner
        {"age": 45.0, "orders": 15.0},  # Loyalty only
    ]

    print(f"\nEvaluating {len(facts)} facts against all policies...")
    results = system.verify_facts(facts)

    for i, result in enumerate(results):
        print(
            f"\nFact {i}: "
            f"Consistent={result.is_consistent}, "
            f"Score={result.confidence_score:.2%}"
        )

    print("\n" + system.summary())


def example_6_streaming_evaluation():
    """Example 6: Streaming evaluation for large datasets."""
    print("\n" + "=" * 70)
    print("Example 6: Streaming Evaluation")
    print("=" * 70)

    config = SystemConfig(device="cpu", batch_size=16)
    system = NeuraLogGASystem(config)

    system.register_policy(
        "age_check",
        {
            "name": "age_check",
            "conditions": [
                {"subject": "age", "threshold": 21.0, "mode": "soft"},
            ],
            "operator": "and",
        },
    )

    # Stream-friendly generator
    def fact_generator():
        print("Generating facts...")
        for i in range(100):
            yield {"age": 18.0 + (i % 60)}

    print("Starting streaming evaluation...")
    satisfied = 0
    unsatisfied = 0

    for i, result in enumerate(system.stream_verify(fact_generator())):
        if result.is_consistent:
            satisfied += 1
        else:
            unsatisfied += 1

        if (i + 1) % 25 == 0:
            print(f"  Processed {i+1} facts...")

    print(f"\nResults:")
    print(f"  Satisfied policy: {satisfied}")
    print(f"  Didn't satisfy: {unsatisfied}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("NeuraLog Geometric Algebra Reasoning System - Examples")
    print("=" * 70)

    try:
        example_1_basic_policy_verification()
    except Exception as e:
        print(f"Example 1 error: {e}")

    try:
        example_2_counterfactual_generation()
    except Exception as e:
        print(f"Example 2 error: {e}")

    try:
        example_3_batch_processing()
    except Exception as e:
        print(f"Example 3 error: {e}")

    try:
        example_4_pipeline_extraction()
    except Exception as e:
        print(f"Example 4 error: {e}")

    try:
        example_5_multiple_policies()
    except Exception as e:
        print(f"Example 5 error: {e}")

    try:
        example_6_streaming_evaluation()
    except Exception as e:
        print(f"Example 6 error: {e}")

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
