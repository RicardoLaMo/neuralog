"""
Geometric Verification Example

Demonstrates the geometric algebra framework for neurosymbolic verification
using the park admission policy example from the reference paper.

Policy: "Seniors (age >= 65) visiting in low season (Jan/Feb/Nov/Dec) with
        budget >= $22 are eligible for admission."

This example shows:
1. Encoding policies as triplet chains in Cl(3,0)
2. Computing truth degrees via geometric alignment
3. Verifying LLM answers against policies
4. Distinguishing false positives from false negatives
5. Counterfactual analysis ("what-if" scenarios)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from neuralog.geometric import (
    TripletEncoder,
    TripletState,
    PolicyChain,
    PolicyValidator,
    VerificationMode,
    CounterfactualReasoner,
    truth_degree,
    logical_and,
    ThresholdMode,
)


def print_section(title: str):
    """Print section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def example_1_basic_triplet_encoding():
    """Example 1: Encoding and evaluating individual triplets"""
    print_section("Example 1: Basic Triplet Encoding")

    # Create encoder with HARD threshold mode
    encoder = TripletEncoder(
        mode="canonical",
        threshold_mode=ThresholdMode.HARD
    )

    # Encode numeric triplet: age >= 65
    age_triplet = encoder.encode_numeric_triplet(
        value=70,          # User's age
        threshold=65,      # Senior threshold
    )

    print(f"Age triplet (age=70, threshold=65):")
    print(f"  State: {age_triplet.state}")
    print(f"  Truth degree: {truth_degree(age_triplet):.4f}")
    print(f"  Decision: {'PASS' if truth_degree(age_triplet) > 0.5 else 'FAIL'}")

    # Encode binary triplet: in low season
    season_triplet = encoder.encode_binary_triplet(
        truth_value=True   # January is low season
    )

    print(f"\nSeason triplet (isLowSeason=True):")
    print(f"  State: {season_triplet.state}")
    print(f"  Truth degree: {truth_degree(season_triplet):.4f}")

    # Encode budget triplet: budget >= 22
    budget_triplet = encoder.encode_numeric_triplet(
        value=25,          # User's budget
        threshold=22,      # Minimum price
    )

    print(f"\nBudget triplet (budget=25, threshold=22):")
    print(f"  State: {budget_triplet.state}")
    print(f"  Truth degree: {truth_degree(budget_triplet):.4f}")


def example_2_policy_chain():
    """Example 2: Building and evaluating policy chains"""
    print_section("Example 2: Policy Chain Evaluation")

    encoder = TripletEncoder(threshold_mode=ThresholdMode.HARD)

    # Build policy: (age >= 65) AND (lowSeason) AND (budget >= 22)
    triplets = [
        encoder.encode_numeric_triplet(value=70, threshold=65),   # age check
        encoder.encode_binary_triplet(truth_value=True),          # season check
        encoder.encode_numeric_triplet(value=25, threshold=22),   # budget check
    ]

    policy = PolicyChain(triplets, combination_mode="and")

    print(f"Policy chain with {len(policy)} triplets (AND combination):")
    for i, t in enumerate(policy.triplets):
        print(f"  Triplet {i+1}: truth = {truth_degree(t):.4f}")

    overall_truth = policy.evaluate()
    print(f"\nOverall policy truth degree: {overall_truth:.4f}")
    print(f"Decision: {'ELIGIBLE' if overall_truth > 0.5 else 'NOT ELIGIBLE'}")


def example_3_threshold_modes():
    """Example 3: Comparing threshold modes at edge cases"""
    print_section("Example 3: Threshold Modes (Edge Case: age=65)")

    # Test at exact threshold (age = 65)
    modes = [ThresholdMode.HARD, ThresholdMode.SOFT, ThresholdMode.MARGIN]

    for mode in modes:
        encoder = TripletEncoder(threshold_mode=mode)

        triplet = encoder.encode_numeric_triplet(
            value=65,       # Exactly at threshold
            threshold=65,
        )

        truth = truth_degree(triplet)
        decision = "PASS" if truth > 0.5 else "FAIL"

        print(f"{mode.value.upper()} mode:")
        print(f"  Truth degree: {truth:.4f}")
        print(f"  Decision: {decision}")
        print()

    print("Key observation:")
    print("  - HARD: Binary decision (1.0 → PASS)")
    print("  - SOFT: Ambiguous (0.5 → boundary)")
    print("  - MARGIN: Clear decision with safety margin (1.0 → PASS)")


def example_4_llm_verification():
    """Example 4: Verifying LLM answers"""
    print_section("Example 4: LLM Answer Verification")

    validator = PolicyValidator(
        threshold_mode="hard",
        verification_mode=VerificationMode.BIDIRECTIONAL,
    )

    # Define policy
    policy_spec = {
        "conditions": [
            {"type": "numeric", "value": 70, "threshold": 65},   # age >= 65
            {"type": "binary", "value": True},                    # low season
            {"type": "numeric", "value": 25, "threshold": 22},   # budget >= 22
        ],
        "combination": "and"
    }

    policy = validator.encode_policy(policy_spec)

    # Test Case 1: LLM correctly identifies eligible user
    print("Test Case 1: Correct Eligibility")
    result = validator.verify(
        llm_answer="The user is eligible for admission",
        policy_chain=policy,
    )
    print(result["explanation"])

    # Test Case 2: LLM false positive (claims eligible when not)
    print("\n" + "-" * 80)
    print("Test Case 2: False Positive Detection")

    policy_spec_fail = {
        "conditions": [
            {"type": "numeric", "value": 60, "threshold": 65},   # age < 65 (fails)
            {"type": "binary", "value": True},
            {"type": "numeric", "value": 25, "threshold": 22},
        ],
        "combination": "and"
    }

    policy_fail = validator.encode_policy(policy_spec_fail)

    result = validator.verify(
        llm_answer="The user is eligible for admission",  # LLM hallucinates
        policy_chain=policy_fail,
    )
    print(result["explanation"])

    # Test Case 3: LLM false negative (denies when eligible)
    print("\n" + "-" * 80)
    print("Test Case 3: False Negative Detection")

    result = validator.verify(
        llm_answer="The user is not eligible",  # LLM is overly conservative
        policy_chain=policy,
    )
    print(result["explanation"])


def example_5_counterfactual_analysis():
    """Example 5: Counterfactual reasoning"""
    print_section("Example 5: Counterfactual Analysis")

    encoder = TripletEncoder(threshold_mode=ThresholdMode.HARD)

    # Build policy
    triplets = [
        encoder.encode_numeric_triplet(value=63, threshold=65),   # age < 65 (fails)
        encoder.encode_binary_triplet(truth_value=True),
        encoder.encode_numeric_triplet(value=25, threshold=22),
    ]

    policy = PolicyChain(triplets)

    print(f"Current policy decision: {policy.evaluate():.4f}")
    print(f"Eligible: {'YES' if policy.evaluate() > 0.5 else 'NO'}")

    # Analyze age counterfactual
    reasoner = CounterfactualReasoner(policy)

    print("\nCounterfactual: What if age ranged from 60 to 70?")

    analysis = reasoner.analyze_numeric_counterfactual(
        triplet_index=0,      # Age triplet
        value_range=(60, 70),
        num_steps=11,
    )

    print(f"\nDecision boundary: {analysis['decision_boundary']}")
    print(f"Robust: {analysis['is_robust']}")

    print("\nAge | Truth Degree | Eligible")
    print("-" * 40)
    for val, truth in zip(analysis["values"], analysis["truth_degrees"]):
        eligible = "YES" if truth > 0.5 else "NO"
        print(f"{val:4.0f} | {truth:12.4f} | {eligible}")

    print("\nKey insight:")
    print("  Decision flips exactly at age=65 (threshold)")
    print("  Sharp boundary due to HARD threshold mode")


def example_6_sensitivity_analysis():
    """Example 6: Policy sensitivity analysis"""
    print_section("Example 6: Sensitivity Analysis")

    encoder = TripletEncoder(threshold_mode=ThresholdMode.HARD)

    triplets = [
        encoder.encode_numeric_triplet(value=70, threshold=65),
        encoder.encode_binary_triplet(truth_value=True),
        encoder.encode_numeric_triplet(value=25, threshold=22),
    ]

    policy = PolicyChain(triplets)
    reasoner = CounterfactualReasoner(policy)

    print("Sensitivity of policy to small perturbations:")
    print()

    for i, triplet in enumerate(policy.triplets):
        sensitivity = reasoner.compute_sensitivity(triplet_index=i, perturbation=1.0)
        print(f"  Triplet {i+1}: sensitivity = {sensitivity:.4f}")

    print("\nInterpretation:")
    print("  Higher sensitivity = policy more affected by changes to this condition")
    print("  Lower sensitivity = policy more robust to changes")


def example_7_complete_validation():
    """Example 7: Complete validation workflow"""
    print_section("Example 7: Complete Validation Workflow")

    # Simulate 7 test cases from the reference paper
    test_cases = [
        {
            "name": "Clear eligible",
            "age": 70,
            "month": "January",
            "budget": 35.40,
            "llm_answer": "User is eligible",
            "expected": "VERIFIED"
        },
        {
            "name": "Not senior",
            "age": 60,
            "month": "January",
            "budget": 35.40,
            "llm_answer": "User is not eligible",
            "expected": "VERIFIED"
        },
        {
            "name": "High season",
            "age": 70,
            "month": "July",
            "budget": 35.40,
            "llm_answer": "User is not eligible",
            "expected": "VERIFIED"
        },
        {
            "name": "Insufficient budget",
            "age": 70,
            "month": "December",
            "budget": 20,
            "llm_answer": "User is not eligible",
            "expected": "VERIFIED"
        },
        {
            "name": "Edge case (exact threshold)",
            "age": 65,
            "month": "February",
            "budget": 22,
            "llm_answer": "User is eligible",
            "expected": "VERIFIED"
        },
        {
            "name": "False positive",
            "age": 60,
            "month": "July",
            "budget": 20,
            "llm_answer": "User is eligible",  # LLM hallucinates
            "expected": "FALSE_POSITIVE"
        },
        {
            "name": "False negative",
            "age": 70,
            "month": "November",
            "budget": 30,
            "llm_answer": "User is not eligible",  # LLM too conservative
            "expected": "FALSE_NEGATIVE"
        },
    ]

    validator = PolicyValidator(threshold_mode="hard")

    correct = 0
    total = len(test_cases)

    print(f"Running {total} test cases...")
    print()

    for i, case in enumerate(test_cases, 1):
        # Encode policy
        low_season_months = ["January", "February", "November", "December"]
        is_low_season = case["month"] in low_season_months

        policy_spec = {
            "conditions": [
                {"type": "numeric", "value": case["age"], "threshold": 65},
                {"type": "binary", "value": is_low_season},
                {"type": "numeric", "value": case["budget"], "threshold": 22},
            ],
            "combination": "and"
        }

        policy = validator.encode_policy(policy_spec)

        # Verify
        result = validator.verify(
            llm_answer=case["llm_answer"],
            policy_chain=policy,
        )

        # Check if verdict matches expected
        match = result["verdict"] == case["expected"]
        if match:
            correct += 1

        status = "✅ PASS" if match else "❌ FAIL"

        print(f"Test {i}: {case['name']}")
        print(f"  Age: {case['age']}, Month: {case['month']}, Budget: ${case['budget']}")
        print(f"  LLM: \"{case['llm_answer']}\"")
        print(f"  Expected: {case['expected']}, Got: {result['verdict']}")
        print(f"  {status}")
        print()

    accuracy = (correct / total) * 100
    print("=" * 80)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print("=" * 80)

    if accuracy == 100:
        print("\n🎉 Perfect accuracy! All test cases passed.")
    else:
        print(f"\n⚠️  {total - correct} test case(s) failed.")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("  GEOMETRIC VERIFICATION EXAMPLES")
    print("  Neurosymbolic AI using Clifford Algebra Cl(3,0)")
    print("=" * 80)

    try:
        example_1_basic_triplet_encoding()
        example_2_policy_chain()
        example_3_threshold_modes()
        example_4_llm_verification()
        example_5_counterfactual_analysis()
        example_6_sensitivity_analysis()
        example_7_complete_validation()

        print("\n" + "=" * 80)
        print("  ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
