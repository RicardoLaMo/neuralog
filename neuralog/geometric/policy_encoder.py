"""
Policy Encoding and Verification

Implements policy chains for complex logical expressions and verification
of LLM answers against geometric policy encodings.

Key concepts:
- PolicyChain: Sequence of triplets forming complex policy
- VerificationMode: Different strategies for checking LLM answers
- Bidirectional verification: Catches both false positives and false negatives
"""

from enum import Enum
from typing import List, Tuple, Union, Optional, Dict, Any

import torch
import torch.nn as nn

from .triplet_logic import (
    TripletState,
    TripletRotor,
    truth_degree,
    logical_and,
    logical_or,
    logical_not,
    implication_loss,
    equivalence_loss,
)


class VerificationMode(Enum):
    """
    Modes for verifying LLM answers against policy.

    IMPLICATION: One-directional check (LLM ⇒ Policy)
        - Catches false positives (LLM claims without support)
        - Allows false negatives (LLM denies when eligible)

    EQUIVALENCE: Symmetric check (LLM ⇔ Policy)
        - Penalizes both false positives and false negatives equally
        - Strict bidirectional enforcement

    BIDIRECTIONAL: Separate checks for both directions
        - Explicitly distinguishes false positives from false negatives
        - Recommended for complete auditing
    """
    IMPLICATION = "implication"
    EQUIVALENCE = "equivalence"
    BIDIRECTIONAL = "bidirectional"


class PolicyChain(nn.Module):
    """
    Chain of triplets forming a complex policy.

    Combines multiple triplets via logical operations to express
    complex eligibility criteria, rules, or constraints.

    Example:
        Policy: "Seniors (age >= 65) in low season with budget >= 22 are eligible"

        Chain:
            τ1: (Person, hasAge, 70)
            τ2: (Age, exceeds, 65)
            τ3: (Person, isSenior, True)
            τ4: (Visit, inSeason, LowSeason)
            τ5: (Budget, exceeds, 22)
            τ6: (Person, isEligible, True)

        Logic: (τ2 ∧ τ4 ∧ τ5) ⇒ τ6

    Args:
        triplets: List of triplets in the chain
        combination_mode: How to combine triplets ("and", "or")
    """

    def __init__(
        self,
        triplets: List[Union[TripletState, TripletRotor]],
        combination_mode: str = "and",
    ):
        super().__init__()

        if not triplets:
            raise ValueError("PolicyChain requires at least one triplet")

        self.triplets = nn.ModuleList(triplets)
        self.combination_mode = combination_mode

    def evaluate(self) -> torch.Tensor:
        """
        Evaluate the policy chain to get overall truth degree.

        Returns:
            Combined truth degree of all triplets
        """
        if self.combination_mode == "and":
            return logical_and(list(self.triplets), method="min")
        elif self.combination_mode == "or":
            return logical_or(list(self.triplets), method="probabilistic")
        else:
            raise ValueError(f"Unknown combination mode: {self.combination_mode}")

    def add_triplet(self, triplet: Union[TripletState, TripletRotor]):
        """Add a triplet to the chain"""
        self.triplets.append(triplet)

    def remove_triplet(self, index: int):
        """Remove triplet at index"""
        del self.triplets[index]

    def get_triplet(self, index: int) -> Union[TripletState, TripletRotor]:
        """Get triplet at index"""
        return self.triplets[index]

    def __len__(self) -> int:
        return len(self.triplets)

    def __repr__(self) -> str:
        truth = self.evaluate().item()
        return f"PolicyChain(triplets={len(self.triplets)}, mode={self.combination_mode}, truth={truth:.3f})"


class PolicyEncoder(nn.Module):
    """
    High-level encoder for complex policies.

    Converts structured policy descriptions into PolicyChain objects
    that can be evaluated and verified against LLM outputs.

    Args:
        threshold_mode: How to evaluate numeric thresholds
        default_combination: Default logical combination ("and" or "or")
    """

    def __init__(
        self,
        threshold_mode: str = "hard",
        default_combination: str = "and",
    ):
        super().__init__()

        from .clifford import ThresholdMode
        from .triplet_logic import TripletEncoder

        # Map string to enum
        threshold_map = {
            "hard": ThresholdMode.HARD,
            "soft": ThresholdMode.SOFT,
            "margin": ThresholdMode.MARGIN,
        }

        self.triplet_encoder = TripletEncoder(
            mode="canonical",
            threshold_mode=threshold_map[threshold_mode],
        )
        self.default_combination = default_combination

    def encode_policy(
        self,
        policy_dict: Dict[str, Any],
        combination_mode: Optional[str] = None,
    ) -> PolicyChain:
        """
        Encode policy from structured dictionary.

        Args:
            policy_dict: Dictionary describing policy conditions
            combination_mode: Override default combination mode

        Returns:
            PolicyChain representing the policy

        Example:
            policy_dict = {
                "conditions": [
                    {"type": "numeric", "value": 70, "threshold": 65},  # age >= 65
                    {"type": "binary", "value": True},  # in low season
                    {"type": "numeric", "value": 25, "threshold": 22},  # budget >= 22
                ],
                "combination": "and"
            }
        """
        conditions = policy_dict.get("conditions", [])
        mode = combination_mode or policy_dict.get("combination", self.default_combination)

        triplets = []
        for cond in conditions:
            if cond["type"] == "numeric":
                triplet = self.triplet_encoder.encode_numeric_triplet(
                    value=cond["value"],
                    threshold=cond["threshold"],
                )
            elif cond["type"] == "binary":
                triplet = self.triplet_encoder.encode_binary_triplet(
                    truth_value=cond["value"]
                )
            else:
                raise ValueError(f"Unknown condition type: {cond['type']}")

            triplets.append(triplet)

        return PolicyChain(triplets, combination_mode=mode)


def verify_llm_answer(
    llm_truth: torch.Tensor,
    policy_truth: torch.Tensor,
    mode: VerificationMode = VerificationMode.BIDIRECTIONAL,
    threshold: float = 0.5,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    """
    Verify LLM answer against policy-derived truth.

    Args:
        llm_truth: Truth degree from LLM claim
        policy_truth: Truth degree from policy evaluation
        mode: Verification mode
        threshold: Decision threshold for binary classification

    Returns:
        Tuple of (loss_fp, loss_fn, verdict) where:
            - loss_fp: False positive loss (None if not computed)
            - loss_fn: False negative loss (None if not computed)
            - verdict: "VERIFIED", "FALSE_POSITIVE", "FALSE_NEGATIVE", or "INCONSISTENT"

    Verification Logic:

    IMPLICATION mode:
        - Computes L_FP = max(0, llm_truth - policy_truth)²
        - Returns "FALSE_POSITIVE" if loss > 0, else "VERIFIED"

    EQUIVALENCE mode:
        - Computes L_equiv = (llm_truth - policy_truth)²
        - Returns "INCONSISTENT" if loss > threshold, else "VERIFIED"

    BIDIRECTIONAL mode:
        - Computes both L_FP and L_FN
        - Returns specific error type or "VERIFIED"
    """
    llm_claim = llm_truth > threshold
    policy_decision = policy_truth > threshold

    if mode == VerificationMode.IMPLICATION:
        # Check LLM ⇒ Policy (catches false positives only)
        loss_fp = torch.max(torch.tensor(0.0), llm_truth - policy_truth) ** 2
        loss_fn = None

        if loss_fp > 0:
            verdict = "FALSE_POSITIVE"
        else:
            verdict = "VERIFIED"

    elif mode == VerificationMode.EQUIVALENCE:
        # Check LLM ⇔ Policy (symmetric)
        loss_equiv = (llm_truth - policy_truth) ** 2
        loss_fp = None
        loss_fn = None

        if loss_equiv > threshold:
            verdict = "INCONSISTENT"
        else:
            verdict = "VERIFIED"

        # Return equiv loss as fp for compatibility
        loss_fp = loss_equiv

    elif mode == VerificationMode.BIDIRECTIONAL:
        # Check both directions
        loss_fp = torch.max(torch.tensor(0.0), llm_truth - policy_truth) ** 2
        loss_fn = torch.max(torch.tensor(0.0), policy_truth - llm_truth) ** 2

        if loss_fp > 0 and loss_fn > 0:
            # Both losses positive means they're close but on opposite sides of threshold
            # Determine which is larger
            if loss_fp > loss_fn:
                verdict = "FALSE_POSITIVE"
            else:
                verdict = "FALSE_NEGATIVE"
        elif loss_fp > 0:
            verdict = "FALSE_POSITIVE"
        elif loss_fn > 0:
            verdict = "FALSE_NEGATIVE"
        else:
            verdict = "VERIFIED"

    else:
        raise ValueError(f"Unknown verification mode: {mode}")

    return loss_fp, loss_fn, verdict


def verify_policy_chain(
    llm_chain: PolicyChain,
    policy_chain: PolicyChain,
    mode: VerificationMode = VerificationMode.BIDIRECTIONAL,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    """
    Verify LLM-generated policy chain against reference policy.

    Args:
        llm_chain: PolicyChain from LLM extraction
        policy_chain: Reference PolicyChain
        mode: Verification mode

    Returns:
        Same as verify_llm_answer
    """
    llm_truth = llm_chain.evaluate()
    policy_truth = policy_chain.evaluate()

    return verify_llm_answer(llm_truth, policy_truth, mode)


def build_policy_from_triplets(
    triplet_specs: List[Dict[str, Any]],
    encoder: PolicyEncoder,
    combination_mode: str = "and",
) -> PolicyChain:
    """
    Build PolicyChain from list of triplet specifications.

    Args:
        triplet_specs: List of triplet specification dictionaries
        encoder: PolicyEncoder to use
        combination_mode: How to combine triplets

    Returns:
        PolicyChain

    Example:
        triplet_specs = [
            {"type": "numeric", "value": 70, "threshold": 65, "name": "age_check"},
            {"type": "binary", "value": True, "name": "season_check"},
            {"type": "numeric", "value": 25, "threshold": 22, "name": "budget_check"},
        ]

        policy = build_policy_from_triplets(triplet_specs, encoder, "and")
    """
    triplets = []

    for spec in triplet_specs:
        if spec["type"] == "numeric":
            triplet = encoder.triplet_encoder.encode_numeric_triplet(
                value=spec["value"],
                threshold=spec["threshold"],
            )
        elif spec["type"] == "binary":
            triplet = encoder.triplet_encoder.encode_binary_triplet(
                truth_value=spec["value"]
            )
        else:
            raise ValueError(f"Unknown triplet type: {spec['type']}")

        triplets.append(triplet)

    return PolicyChain(triplets, combination_mode=combination_mode)


class PolicyValidator:
    """
    Complete validator for LLM outputs using geometric algebra.

    Combines:
        1. Policy encoding from natural language
        2. LLM answer extraction and encoding
        3. Geometric verification
        4. Explainable results

    Usage:
        validator = PolicyValidator(threshold_mode="hard")

        # Define policy
        policy_spec = {
            "conditions": [
                {"type": "numeric", "value": 70, "threshold": 65},
                {"type": "binary", "value": True},
            ],
            "combination": "and"
        }

        # Encode policy
        policy_chain = validator.encode_policy(policy_spec)

        # Verify LLM answer
        result = validator.verify(
            llm_answer="User is eligible",
            policy_chain=policy_chain
        )

        print(result["verdict"])  # "VERIFIED", "FALSE_POSITIVE", or "FALSE_NEGATIVE"
        print(result["explanation"])
    """

    def __init__(
        self,
        threshold_mode: str = "hard",
        verification_mode: VerificationMode = VerificationMode.BIDIRECTIONAL,
        decision_threshold: float = 0.5,
    ):
        self.encoder = PolicyEncoder(threshold_mode=threshold_mode)
        self.verification_mode = verification_mode
        self.decision_threshold = decision_threshold

    def encode_policy(
        self,
        policy_spec: Dict[str, Any],
    ) -> PolicyChain:
        """Encode policy from specification"""
        return self.encoder.encode_policy(policy_spec)

    def encode_llm_claim(
        self,
        claim: str,
        claim_truth: float = 1.0,
    ) -> torch.Tensor:
        """
        Encode LLM claim as truth degree.

        For now, simple mapping: positive claim → claim_truth, negative → 0

        Args:
            claim: LLM answer string
            claim_truth: Truth degree if claim is positive

        Returns:
            Truth degree tensor
        """
        # Simple heuristic: look for negative words
        negative_words = ["not", "no", "ineligible", "denied", "reject"]

        claim_lower = claim.lower()
        is_negative = any(word in claim_lower for word in negative_words)

        if is_negative:
            return torch.tensor(0.0)
        else:
            return torch.tensor(claim_truth)

    def verify(
        self,
        llm_answer: str,
        policy_chain: PolicyChain,
        llm_truth: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Verify LLM answer against policy.

        Args:
            llm_answer: LLM answer string
            policy_chain: Reference policy chain
            llm_truth: Override extracted truth degree

        Returns:
            Dictionary with verification results:
                - verdict: "VERIFIED", "FALSE_POSITIVE", or "FALSE_NEGATIVE"
                - llm_truth: LLM truth degree
                - policy_truth: Policy truth degree
                - loss_fp: False positive loss (if computed)
                - loss_fn: False negative loss (if computed)
                - explanation: Human-readable explanation
        """
        # Get policy truth
        policy_truth = policy_chain.evaluate()

        # Get LLM truth
        if llm_truth is None:
            llm_truth_tensor = self.encode_llm_claim(llm_answer)
        else:
            llm_truth_tensor = torch.tensor(llm_truth)

        # Verify
        loss_fp, loss_fn, verdict = verify_llm_answer(
            llm_truth=llm_truth_tensor,
            policy_truth=policy_truth,
            mode=self.verification_mode,
            threshold=self.decision_threshold,
        )

        # Generate explanation
        explanation = self._generate_explanation(
            verdict=verdict,
            llm_truth=llm_truth_tensor.item(),
            policy_truth=policy_truth.item(),
            loss_fp=loss_fp.item() if loss_fp is not None else None,
            loss_fn=loss_fn.item() if loss_fn is not None else None,
        )

        return {
            "verdict": verdict,
            "llm_truth": llm_truth_tensor.item(),
            "policy_truth": policy_truth.item(),
            "loss_fp": loss_fp.item() if loss_fp is not None else None,
            "loss_fn": loss_fn.item() if loss_fn is not None else None,
            "explanation": explanation,
        }

    def _generate_explanation(
        self,
        verdict: str,
        llm_truth: float,
        policy_truth: float,
        loss_fp: Optional[float],
        loss_fn: Optional[float],
    ) -> str:
        """Generate human-readable explanation of verification result"""
        if verdict == "VERIFIED":
            return (
                f"✅ LLM answer is consistent with policy.\n"
                f"   LLM truth degree: {llm_truth:.3f}\n"
                f"   Policy truth degree: {policy_truth:.3f}\n"
                f"   Both agree on the decision."
            )
        elif verdict == "FALSE_POSITIVE":
            return (
                f"❌ FALSE POSITIVE: LLM claims eligibility without policy support.\n"
                f"   LLM truth degree: {llm_truth:.3f} (claims TRUE)\n"
                f"   Policy truth degree: {policy_truth:.3f} (should be FALSE)\n"
                f"   FP Loss: {loss_fp:.4f}\n"
                f"   This is a hallucination - LLM is making unsupported claims."
            )
        elif verdict == "FALSE_NEGATIVE":
            return (
                f"⚠️ FALSE NEGATIVE: LLM denies when policy supports eligibility.\n"
                f"   LLM truth degree: {llm_truth:.3f} (claims FALSE)\n"
                f"   Policy truth degree: {policy_truth:.3f} (should be TRUE)\n"
                f"   FN Loss: {loss_fn:.4f}\n"
                f"   LLM is being overly conservative."
            )
        else:  # INCONSISTENT
            return (
                f"⚠️ INCONSISTENT: LLM and policy disagree.\n"
                f"   LLM truth degree: {llm_truth:.3f}\n"
                f"   Policy truth degree: {policy_truth:.3f}\n"
                f"   Discrepancy: {abs(llm_truth - policy_truth):.3f}"
            )
