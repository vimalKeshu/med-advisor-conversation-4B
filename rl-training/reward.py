from __future__ import annotations

import json
import asyncio
import os

from typing import Any, Literal
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, create_model
from common import Scenario
from description import *

NULL_REWARD = {
    "valid": False,
    "exclusion_reason": "",
    "quality_reward": None,
    "trajectory_reward": None,
    "hard_failure": False,
    "hard_failure_dimensions": [],
    "dimension_scores": {},
    "active_weight": 0.0,
    "reward_version": "v7",
    "reward_terms": {},
    "length_reward": None,
    "token_score": None,
    "raw_length_score": None,
    "completion_tokens": 0,
    "completion_turns": 0,
}

QUALITY_UTILITY = {
    "A": 1.00,
    "B": 0.85,
    "C": 0.60,
    "D": 0.25,
    "E": 0.00,
}

CATEGORY_WEIGHTS = {
    "correctness": 0.50,
    "conversation_execution": 0.40,
    "presentation": 0.10,
}

CASCADE_ORDER = ("correctness", "conversation_execution", "presentation")

CATEGORY_DIMENSIONS = {
    "correctness": {
        "medical_accuracy": 1.0,
        "medication_and_numerical_integrity": 1.0,
        "evidence_integrity": 1.0,
        "safety_boundary": 1.0,         
    },
    "conversation_execution": {
        "uncertainty_calibration": 1.0,
        "user_question_fulfillment": 1.0,
        "appropriate_completeness": 1.0,        
        "audience_adaptation": 1.0,
        "follow_up_quality": 1.0,
        "multi_turn_coherence": 1.0,
    },    
    "presentation": {
        "answer_format": 1.0,
    },
}

REQUIRED_PRIMARY_DIMENSIONS = {
    "medical_accuracy",

    "uncertainty_calibration",
    "user_question_fulfillment",
    "appropriate_completeness",
    "audience_adaptation",
    "follow_up_quality",

    "answer_format",
}

RUBRIC = {
    dimension: description
    for family in RUBRIC_FAMILIES.values()
    for dimension, description in family["dimensions"].items()
}

DIMENSION_WEIGHTS = {
    name: CATEGORY_WEIGHTS[category] * weight / sum(weights.values())
    for category, weights in CATEGORY_DIMENSIONS.items()
    for name, weight in weights.items()
}

assert RUBRIC.keys() == DIMENSION_WEIGHTS.keys(), (
    f"Rubric/scorer mismatch: {RUBRIC.keys() ^ DIMENSION_WEIGHTS.keys()}"
)

Grade = Literal["A", "B", "C", "D", "E", "NA"]

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class DimensionGrade(StrictModel):
    grade: Grade
    reason: str

Grades = create_model(
    "Grades",
    __base__=StrictModel,
    **{name: (DimensionGrade, ...) for name in RUBRIC},
)

class JudgeResult(StrictModel):
    evaluation_status: Literal["VALID", "INSUFFICIENT_INPUT"]
    evaluation_issue: str
    grades: Grades

def serialize(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)

def compose_reward(result: JudgeResult) -> dict[str, Any]:
    reward = {
        "valid": False,
        "exclusion_reason": "",
        "quality_reward": None,       # Base quality BEFORE the gate/override.
        "trajectory_reward": None,    # The value the trainer must use.
        "hard_failure": False,
        "hard_failure_dimensions": [],
        "dimension_scores": {},
        "active_weight": 0.0,         # Diagnostic only; not a denominator below.
        "reward_version": "v7",
        "category_scores": {},
        "essential_quality": None,
        "limiting_dimensions": [],
        "reward_terms": {},
        "length_reward": None,
        "token_score": None,
        "raw_length_score": None,
        "completion_tokens": 0,
        "completion_turns": 0,
    }

    if result.evaluation_status != "VALID":
        reward["exclusion_reason"] = (
            result.evaluation_issue or "insufficient judge input"
        )
        return reward

    grades = result.grades.model_dump()

    if grades.keys() != DIMENSION_WEIGHTS.keys():
        reward["exclusion_reason"] = (
            "judge dimensions do not match scorer dimensions"
        )
        return reward

    scores = {
        name: None if item["grade"] == "NA" else QUALITY_UTILITY[item["grade"]]
        for name, item in grades.items()
    }

    reward["dimension_scores"] = scores
    reward["active_weight"] = sum(
        DIMENSION_WEIGHTS[name]
        for name, score in scores.items()
        if score is not None
    )

    for name in REQUIRED_PRIMARY_DIMENSIONS:
        if scores[name] is None:
            reward["exclusion_reason"] = (
                f"required primary grade is NA: {name}"
            )
            return reward

    category_scores = {}

    for category, weights in CATEGORY_DIMENSIONS.items():
        active = [
            (scores[name], weight)
            for name, weight in weights.items()
            if scores[name] is not None
        ]

        if not active:
            reward["exclusion_reason"] = (
                f"no applicable grades in {category}"
            )
            return reward

        category_scores[category] = (
            sum(score * weight for score, weight in active)
            / sum(weight for _, weight in active)
        )

    # ("correctness", "conversation_execution", "presentation")
    q1 = category_scores["correctness"]
    q2 = category_scores["conversation_execution"]
    q3 = category_scores["presentation"]
    reward_terms = {
        "correctness": CATEGORY_WEIGHTS["correctness"] * q1,
        "conversation_execution": (
            CATEGORY_WEIGHTS["conversation_execution"] * q1 * q2
        ),
        "presentation": CATEGORY_WEIGHTS["presentation"] * q1 * q2 * q3,
    }
    cascade_reward = sum(reward_terms.values())

    reward.update(
        valid=True,
        quality_reward=cascade_reward,
        trajectory_reward=cascade_reward,
        category_scores=category_scores,
        reward_terms=reward_terms,
    )

    return reward

def add_length_reward(reward: dict[str, Any], trajectory: dict, cfg: dict) -> dict[str, Any]:
    """Scale a bounded token score by judged quality before adding it.

    1 / (average tokens per turn * number of turns) is exactly 1 / total
    completion tokens. Counting completion_ids includes reasoning tokens.
    """
    if not reward["valid"] or reward["quality_reward"] is None:
        return reward

    token_counts = [len(turn.get("completion_ids") or [])
                    for turn in trajectory.get("turns", [])]
    total_tokens = sum(token_counts)
    if not total_tokens:
        return reward

    scale = float(cfg["token_bonus_scale"])
    cap = float(cfg["token_bonus_cap"])
    if scale < 0 or cap < 0:
        raise ValueError("token bonus scale and cap must be nonnegative")

    raw_score = 1.0 / total_tokens
    token_score = min(cap, scale * raw_score)
    bonus = reward["quality_reward"] * token_score
    reward["raw_length_score"] = raw_score
    reward["token_score"] = token_score
    reward["length_reward"] = bonus
    reward["completion_tokens"] = total_tokens
    reward["completion_turns"] = len(token_counts)
    reward["reward_terms"]["token_bonus"] = bonus
    reward["trajectory_reward"] = reward["quality_reward"] + bonus
    return reward

def build_scenario(scenario: Scenario) -> dict:
    return {
        "title": scenario.title,
        "domain": scenario.domain,
        "audience": scenario.simulator["persona"],
        "user_facts_volunteered": scenario.simulator["facts"],
    }    

def build_trajectory(trajectory: dict) -> list[dict]:
    return [
        {"role": m["role"], "content": m["content"]} for m in trajectory["messages"]
    ]

class Judge:

    def __init__(self, cfg: dict, reward_cfg: dict | None = None) -> None:
        self.cfg = cfg
        self.reward_cfg = reward_cfg or {"token_bonus_scale": 0.0, "token_bonus_cap": 0.0}
        self.gate = asyncio.Semaphore(int(cfg["concurrency"]))
        self.client = AsyncOpenAI( api_key=os.environ[cfg["api_key_env"]],
                timeout=float(cfg["timeout"]), max_retries=int(cfg["max_retries"]),)                

    async def __call__(self, scenario: Scenario, trajectory: dict) -> tuple[dict, dict]:
        truncated = (
                trajectory.get("end_reason") == "advisor_truncated"
                or any(
                    turn.get("finish_reason") == "length"
                    for turn in trajectory.get("turns", [])
                )
            )

        if truncated:
            issue = "policy generation truncated"
            return {
                "evaluation_status": "ROLLOUT_FAILURE",
                "evaluation_issue": issue,
            }, {
                **NULL_REWARD,
                "valid": True,                 # participate in GRPO
                "exclusion_reason": "",
                "trajectory_reward": -1.0,
                "hard_failure": True,
                "rollout_failure": "generation_truncation",
            }

        # if not trajectory.get("chat"):
        if not any(m["role"] == "assistant" for m in trajectory.get("chat", [])):
            issue = "trajectory has no assistant turns"
            return {"evaluation_status": "ERROR", "evaluation_issue": issue}, {
                **NULL_REWARD, "exclusion_reason": issue
            }

        try:
            async with self.gate:
                result = await self._call(scenario, trajectory)
            scored = compose_reward(result)
            return result.model_dump(), add_length_reward(scored, trajectory, self.reward_cfg)
        except Exception as error:  # noqa: BLE001
            issue = f"{type(error).__name__}: {error}"
            return {"evaluation_status": "ERROR", "evaluation_issue": issue}, {
                **NULL_REWARD, "exclusion_reason": issue
            }

    async def _call(self, scenario: Scenario, trajectory: dict) -> JudgeResult:
        response = await self.client.responses.parse(
            model=self.cfg["model"],
            instructions=JUDGE_PROMPT,
            input=(
                f"<SCENARIO>\n{serialize(build_scenario(scenario))}\n</SCENARIO>\n\n"
                f"<TRAJECTORY>\n{serialize(trajectory['chat'])}\n</TRAJECTORY>"
            ),
            text_format=JudgeResult,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError(response.output_text or "judge returned no parsed result")
        return response.output_parsed