"""
Benchmark Evaluators

Implements evaluation logic for standard benchmarks:
- GSM8K: Grade school math problems
- MATH: Competition math problems
- HumanEval: Code generation
- MMLU: Massive Multitask Language Understanding
"""

import json
import logging
import math
import re
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

EVAL_DATA_DIR = Path(os.environ.get("EVAL_DATA_DIR", "./eval_data"))


@dataclass
class EvalResult:
    """Result from a single evaluation item."""
    question_id: str
    correct: bool
    predicted: str
    expected: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark result."""
    benchmark: str
    accuracy: float
    total: int
    correct: int
    details: List[EvalResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "accuracy": round(self.accuracy, 4),
            "total": self.total,
            "correct": self.correct,
            "metadata": self.metadata,
        }


class BaseEvaluator(ABC):
    """Base class for benchmark evaluators."""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else EVAL_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def load_dataset(self, split: str = "test") -> List[Dict]:
        pass

    @abstractmethod
    def evaluate_response(self, question: Dict, response: str) -> EvalResult:
        pass

    def evaluate_all(
        self,
        responses: List[Dict[str, str]],
        split: str = "test",
    ) -> BenchmarkResult:
        """
        Evaluate all responses against the benchmark.

        Args:
            responses: List of {"question_id": "...", "response": "..."}
            split: Dataset split to use

        Returns:
            BenchmarkResult with accuracy and details
        """
        dataset = self.load_dataset(split)
        question_map = {str(i): q for i, q in enumerate(dataset)}

        results = []
        for resp in responses:
            qid = resp.get("question_id", "")
            if qid in question_map:
                result = self.evaluate_response(question_map[qid], resp["response"])
                results.append(result)

        correct = sum(1 for r in results if r.correct)
        total = len(results)

        return BenchmarkResult(
            benchmark=self.name(),
            accuracy=correct / total if total > 0 else 0.0,
            total=total,
            correct=correct,
            details=results,
        )

    def get_questions(self, split: str = "test", limit: int = None) -> List[Dict]:
        """Get questions for evaluation."""
        dataset = self.load_dataset(split)
        if limit:
            dataset = dataset[:limit]
        return [
            {
                "question_id": str(i),
                "prompt": q.get("question", q.get("problem", q.get("prompt", ""))),
                "metadata": {k: v for k, v in q.items() if k not in ("question", "problem", "prompt")},
            }
            for i, q in enumerate(dataset)
        ]


# ---- Answer Extraction Utilities ----

def extract_boxed(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format."""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_last_number(text: str) -> Optional[str]:
    """Extract the last number from text."""
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    return None


def extract_answer_after_marker(text: str, marker: str = "answer is") -> Optional[str]:
    """Extract answer after a marker phrase."""
    pattern = rf'{marker}\s*:?\s*(.+?)(?:\.|$)'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def normalize_number(text: str) -> Optional[float]:
    """Normalize a number string to float."""
    if not text:
        return None
    text = text.strip().replace(",", "").replace(" ", "")
    # Handle fractions
    frac_match = re.match(r'^(-?\d+)/(\d+)$', text)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        return num / den if den != 0 else None
    try:
        return float(text)
    except ValueError:
        return None


def answers_equivalent(pred: str, expected: str, tolerance: float = 1e-6) -> bool:
    """Check if two answers are mathematically equivalent."""
    if not pred or not expected:
        return False

    # Exact string match
    if pred.strip() == expected.strip():
        return True

    # Numeric comparison
    pred_num = normalize_number(pred)
    expected_num = normalize_number(expected)
    if pred_num is not None and expected_num is not None:
        return abs(pred_num - expected_num) < tolerance

    return False


# ---- GSM8K Evaluator ----

class GSM8KEvaluator(BaseEvaluator):
    """GSM8K (Grade School Math 8K) evaluator."""

    def name(self) -> str:
        return "GSM8K"

    def load_dataset(self, split: str = "test") -> List[Dict]:
        """Load GSM8K dataset."""
        data_file = self.data_dir / "gsm8k" / f"{split}.jsonl"
        if data_file.exists():
            data = []
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data

        # Generate sample data for testing
        logger.warning("GSM8K dataset not found, using built-in samples")
        return self._get_sample_data()

    def _get_sample_data(self) -> List[Dict]:
        """Built-in sample GSM8K problems."""
        return [
            {
                "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells every duck egg at the farmers' market daily for $2. How much in dollars does she make every day at the farmers' market?",
                "answer": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
            },
            {
                "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "answer": "It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total bolts needed is 2+1=<<2+1=3>>3\n#### 3"
            },
            {
                "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
                "answer": "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\nHe increased the value of the house by 80,000*150%=$<<80000*1.5=120000>>120,000\nSo the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\nSo he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n#### 70000"
            },
            {
                "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
                "answer": "If each chicken eats 3 cups of feed per day, then for 20 chickens they would need 3*20=<<3*20=60>>60 cups of feed per day.\nIf she has already given them 15 cups in the morning and 25 cups in the afternoon, then she has given them 15+25=<<15+25=40>>40 cups so far.\nSo for the final meal, she needs 60-40=<<60-40=20>>20 cups.\n#### 20"
            },
            {
                "question": "Kylar went to the store to get water and found a sale on sports drinks. The normal price for a sport drink is $3.50 but they were on sale for $1.50 less. How much would Kylar spend on 4 drinks?",
                "answer": "The sale price for the drinks is 3.50 - 1.50 = $<<3.50-1.50=2.00>>2.00\n4 drinks would cost 4 * 2.00 = $<<4*2=8.00>>8.00\n#### 8"
            },
        ]

    def _extract_gsm8k_answer(self, answer_text: str) -> str:
        """Extract final answer from GSM8K answer format (after ####)."""
        match = re.search(r'####\s*(.+?)$', answer_text, re.MULTILINE)
        if match:
            return match.group(1).strip().replace(",", "")
        return extract_last_number(answer_text) or ""

    def evaluate_response(self, question: Dict, response: str) -> EvalResult:
        """Evaluate a response against GSM8K question."""
        expected = self._extract_gsm8k_answer(question.get("answer", ""))

        # Try multiple extraction methods
        predicted = (
            extract_boxed(response) or
            self._extract_gsm8k_answer(response) or
            extract_last_number(response) or
            ""
        )

        correct = answers_equivalent(predicted, expected)

        return EvalResult(
            question_id=question.get("question", "")[:50],
            correct=correct,
            predicted=predicted,
            expected=expected,
            score=1.0 if correct else 0.0,
        )


# ---- MATH Evaluator ----

class MATHEvaluator(BaseEvaluator):
    """MATH dataset evaluator (competition math)."""

    def name(self) -> str:
        return "MATH"

    def load_dataset(self, split: str = "test") -> List[Dict]:
        data_file = self.data_dir / "math" / f"{split}.jsonl"
        if data_file.exists():
            data = []
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data

        logger.warning("MATH dataset not found, using built-in samples")
        return self._get_sample_data()

    def _get_sample_data(self) -> List[Dict]:
        return [
            {
                "problem": "Find the value of $x$ such that $\\sqrt{x+7} = 9$.",
                "solution": "Squaring both sides: $x + 7 = 81$, so $x = \\boxed{74}$.",
                "level": "Level 1",
                "type": "Algebra"
            },
            {
                "problem": "What is the sum of all integers from 1 to 100?",
                "solution": "Using the formula $\\frac{n(n+1)}{2} = \\frac{100 \\cdot 101}{2} = \\boxed{5050}$.",
                "level": "Level 1",
                "type": "Number Theory"
            },
            {
                "problem": "If $f(x) = 2x + 3$, what is $f(f(1))$?",
                "solution": "$f(1) = 2(1) + 3 = 5$. Then $f(5) = 2(5) + 3 = \\boxed{13}$.",
                "level": "Level 2",
                "type": "Algebra"
            },
            {
                "problem": "What is the area of a circle with radius 5?",
                "solution": "Area $= \\pi r^2 = \\pi(5)^2 = \\boxed{25\\pi}$.",
                "level": "Level 1",
                "type": "Geometry"
            },
            {
                "problem": "Solve for $x$: $2^x = 32$.",
                "solution": "Since $32 = 2^5$, we have $x = \\boxed{5}$.",
                "level": "Level 1",
                "type": "Algebra"
            },
        ]

    def _normalize_math_answer(self, answer: str) -> str:
        """Normalize math expressions for comparison."""
        answer = answer.strip()
        # Remove common LaTeX formatting
        answer = answer.replace("\\$", "")
        answer = answer.replace("$", "")
        answer = answer.replace("\\text{", "").replace("}", "")
        answer = answer.replace("\\mathrm{", "")
        answer = answer.replace("\\left", "").replace("\\right", "")
        answer = answer.replace("\\,", "")
        answer = answer.replace("\\ ", " ")
        answer = answer.strip()
        return answer

    def evaluate_response(self, question: Dict, response: str) -> EvalResult:
        expected_raw = question.get("solution", "")
        expected = extract_boxed(expected_raw)
        if not expected:
            expected = extract_last_number(expected_raw) or expected_raw

        predicted = extract_boxed(response)
        if not predicted:
            predicted = extract_last_number(response) or ""

        # Normalize and compare
        norm_pred = self._normalize_math_answer(predicted)
        norm_exp = self._normalize_math_answer(expected)

        correct = (norm_pred == norm_exp) or answers_equivalent(norm_pred, norm_exp)

        return EvalResult(
            question_id=question.get("problem", "")[:50],
            correct=correct,
            predicted=predicted,
            expected=expected,
            score=1.0 if correct else 0.0,
            metadata={
                "level": question.get("level", ""),
                "type": question.get("type", ""),
            },
        )


# ---- HumanEval Evaluator ----

class HumanEvalEvaluator(BaseEvaluator):
    """HumanEval code generation evaluator."""

    def name(self) -> str:
        return "HumanEval"

    def load_dataset(self, split: str = "test") -> List[Dict]:
        data_file = self.data_dir / "humaneval" / f"{split}.jsonl"
        if data_file.exists():
            data = []
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data

        logger.warning("HumanEval dataset not found, using built-in samples")
        return self._get_sample_data()

    def _get_sample_data(self) -> List[Dict]:
        return [
            {
                "task_id": "HumanEval/0",
                "prompt": "from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\"Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
                "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
                "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n",
                "entry_point": "has_close_elements"
            },
            {
                "task_id": "HumanEval/1",
                "prompt": "from typing import List\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\"Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those groups into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
                "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string = []\n    return result\n",
                "test": "def check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n    assert candidate('(()(()))') == ['(()(()))']\n",
                "entry_point": "separate_paren_groups"
            },
        ]

    def _run_code_safely(self, code: str, test_code: str, entry_point: str, timeout: int = 5) -> bool:
        """Run code in a restricted environment."""
        import subprocess
        import tempfile

        full_code = f"{code}\n\ncandidate = {entry_point}\n{test_code}\ncheck(candidate)\nprint('ALL_TESTS_PASSED')"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return "ALL_TESTS_PASSED" in result.stdout
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Code execution failed: {e}")
            return False
        finally:
            os.unlink(tmp_path)

    def evaluate_response(self, question: Dict, response: str) -> EvalResult:
        prompt = question.get("prompt", "")
        test = question.get("test", "")
        entry_point = question.get("entry_point", "")

        # Combine prompt with response
        full_code = prompt + response

        # Try to run the code
        passed = self._run_code_safely(full_code, test, entry_point)

        return EvalResult(
            question_id=question.get("task_id", ""),
            correct=passed,
            predicted=response[:200],
            expected=question.get("canonical_solution", "")[:200],
            score=1.0 if passed else 0.0,
        )

    def evaluate_pass_at_k(
        self,
        responses: Dict[str, List[str]],
        k: int = 1,
        split: str = "test",
    ) -> Dict[str, Any]:
        """
        Evaluate pass@k metric.

        Args:
            responses: {question_id: [response1, response2, ...]}
            k: k value for pass@k
            split: dataset split

        Returns:
            {"pass_at_k": float, "details": [...]}
        """
        dataset = self.load_dataset(split)
        question_map = {str(i): q for i, q in enumerate(dataset)}

        pass_counts = []
        for qid, resps in responses.items():
            if qid not in question_map:
                continue
            question = question_map[qid]
            n = len(resps)
            c = sum(
                1 for r in resps
                if self.evaluate_response(question, r).correct
            )
            pass_counts.append((n, c))

        # Calculate pass@k
        if not pass_counts:
            return {"pass_at_k": 0.0, "k": k, "total": 0}

        def pass_at_k_estimator(n, c, k):
            if n - c < k:
                return 1.0
            return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))

        scores = [pass_at_k_estimator(n, c, k) for n, c in pass_counts]
        avg_pass_at_k = sum(scores) / len(scores)

        return {
            "pass_at_k": round(avg_pass_at_k, 4),
            "k": k,
            "total": len(pass_counts),
        }


# ---- MMLU Evaluator ----

class MMLUEvaluator(BaseEvaluator):
    """MMLU (Massive Multitask Language Understanding) evaluator."""

    def name(self) -> str:
        return "MMLU"

    def load_dataset(self, split: str = "test") -> List[Dict]:
        data_file = self.data_dir / "mmlu" / f"{split}.jsonl"
        if data_file.exists():
            data = []
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data

        logger.warning("MMLU dataset not found, using built-in samples")
        return self._get_sample_data()

    def _get_sample_data(self) -> List[Dict]:
        return [
            {
                "question": "Which of the following is the body cavity that contains the pituitary gland?",
                "choices": ["Cranial", "Thoracic", "Abdominal", "Pelvic"],
                "answer": "A",
                "subject": "anatomy"
            },
            {
                "question": "What is the smallest prime number?",
                "choices": ["0", "1", "2", "3"],
                "answer": "C",
                "subject": "math"
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "choices": ["Venus", "Mars", "Jupiter", "Saturn"],
                "answer": "B",
                "subject": "astronomy"
            },
            {
                "question": "What does CPU stand for?",
                "choices": ["Central Process Unit", "Central Processing Unit", "Computer Personal Unit", "Central Processor Utility"],
                "answer": "B",
                "subject": "computer_science"
            },
            {
                "question": "Who wrote 'Romeo and Juliet'?",
                "choices": ["Charles Dickens", "William Shakespeare", "Mark Twain", "Jane Austen"],
                "answer": "B",
                "subject": "literature"
            },
        ]

    def _extract_choice(self, response: str) -> str:
        """Extract choice letter from response."""
        # Look for explicit letter at start
        match = re.match(r'^[^A-Da-d]*([A-Da-d])\b', response.strip())
        if match:
            return match.group(1).upper()

        # Look for "answer is X" pattern
        match = re.search(r'answer\s+is\s+([A-Da-d])', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Look for any standalone letter
        match = re.search(r'\b([A-Da-d])\b', response)
        if match:
            return match.group(1).upper()

        return ""

    def evaluate_response(self, question: Dict, response: str) -> EvalResult:
        expected = question.get("answer", "").upper()
        predicted = self._extract_choice(response)

        correct = predicted == expected

        return EvalResult(
            question_id=question.get("question", "")[:50],
            correct=correct,
            predicted=predicted,
            expected=expected,
            score=1.0 if correct else 0.0,
            metadata={"subject": question.get("subject", "")},
        )

    def evaluate_by_subject(
        self,
        responses: List[Dict[str, str]],
        split: str = "test",
    ) -> Dict[str, Any]:
        """Evaluate with per-subject breakdown."""
        result = self.evaluate_all(responses, split)

        subject_results = {}
        for detail in result.details:
            subject = detail.metadata.get("subject", "unknown")
            if subject not in subject_results:
                subject_results[subject] = {"correct": 0, "total": 0}
            subject_results[subject]["total"] += 1
            if detail.correct:
                subject_results[subject]["correct"] += 1

        for subject, counts in subject_results.items():
            counts["accuracy"] = round(
                counts["correct"] / counts["total"], 4
            ) if counts["total"] > 0 else 0.0

        return {
            **result.to_dict(),
            "by_subject": subject_results,
        }


# ---- Factory Functions ----

_EVALUATORS = {
    "gsm8k": GSM8KEvaluator,
    "math": MATHEvaluator,
    "humaneval": HumanEvalEvaluator,
    "mmlu": MMLUEvaluator,
}


def get_evaluator(benchmark: str, **kwargs) -> BaseEvaluator:
    """Get evaluator by benchmark name."""
    benchmark = benchmark.lower()
    if benchmark not in _EVALUATORS:
        raise ValueError(f"Unknown benchmark: {benchmark}. Available: {list(_EVALUATORS.keys())}")
    return _EVALUATORS[benchmark](**kwargs)


def list_benchmarks() -> List[Dict[str, str]]:
    """List available benchmarks."""
    return [
        {"name": "gsm8k", "description": "GSM8K - Grade School Math (8K problems)", "category": "math"},
        {"name": "math", "description": "MATH - Competition Math", "category": "math"},
        {"name": "humaneval", "description": "HumanEval - Code Generation (164 problems)", "category": "code"},
        {"name": "mmlu", "description": "MMLU - Massive Multitask Language Understanding", "category": "knowledge"},
    ]
