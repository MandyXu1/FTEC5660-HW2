#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once."""
    from langchain_deepseek import ChatDeepSeek
    from langchain.agents import create_agent

    llm = ChatDeepSeek(
        model=MODEL_NAME,
        temperature=0.0,
        max_retries=3,
    )

    system_prompt = """You are an expert KYC and CV verification agent.
Your task is to verify a candidate's CV against their LinkedIn (primary source) and Facebook profiles using the provided tools.

Follow these rules STRICTLY:

1. ENTITY RESOLUTION (FINDING THE RIGHT PERSON):
   - Hundreds of people share the same name! Always call `search_linkedin_people` with BOTH `q=<name>` AND `location=<city from CV>`.
   - Check the `headline`, `location`, `industry`, and `years_experience` in the search results to identify the most likely candidate IDs.
   - Call `get_linkedin_profile` on multiple promising candidate IDs IN PARALLEL (in a single turn) rather than one by one.
   - CRITICAL: A fraudulent CV (Label 0) still matches the real person on MOST fields and only fabricates 1 or 2 details (e.g., a 1-year date shift, an upgraded degree, or an extra skill). If a profile you inspect has COMPLETELY different schools, companies, and skills from the CV, that is a DIFFERENT person with the same name! Keep looking (or use `search_facebook_users` / `get_facebook_profile` with location) until you find the actual person.

2. FIELD COMPARISON RULES:
   - ONLY check these fields: name, city, jobs (company, title, seniority, start_year, end_year), education (degree, school, field, graduation year / end_year), and skills.
   - COMPLETELY IGNORE: job descriptions, headline, summary, and hometown. They are NEVER the source of a discrepancy.

3. WHAT COUNTS AS VALID (Label 1 -> score = 0.95):
   - Wording or abbreviation differences: e.g., "Bachelor of Science" vs "BSc", "Master of Science" vs "MSc", "Bachelor of Arts" vs "BA", "Bachelor of Engineering" vs "BEng", "Master of Business Administration" vs "MBA", "Doctor of Philosophy" vs "PhD", "UI/UX Design" vs "UI/UX".
   - Seniority + Title formatting: LinkedIn stores `title` and `seniority` ("junior", "mid", "senior") separately. Combining them on the CV (e.g., "Senior Engineer" when title="Engineer" and seniority="senior", or "Junior Analyst" when title="Analyst" and seniority="junior") or omitting "mid" is VALID.
   - Current employment dates: On LinkedIn, a current job has `is_current: true` and `end_year: null`. On a CV, this is written as "Present" or "Current". This is VALID.
   - Education dates: If the CV only states the graduation year and it matches LinkedIn's `end_year`, or states the full `start_year - end_year` range matching LinkedIn, it is VALID.
   - Skills subset: Listing FEWER skills on the CV than on LinkedIn is VALID, as long as every skill listed on the CV exists on the profile (allowing minor wording variations).

4. WHAT COUNTS AS A DISCREPANCY (Label 0 -> score = 0.05):
   - Any factual mismatch in the checked fields:
     * Employment start_year or end_year shifted (even by 1 year!).
     * Education graduation year (end_year) shifted (even by 1 year!).
     * Upgraded degree (e.g., claiming MSc when profile says BSc) or wrong field/school.
     * Inflated job title or seniority (e.g., claiming "Senior" when profile seniority is "junior" or "mid", or claiming "Manager" when profile is "Engineer").
     * Fake employer/company or wrong city.
     * Claiming a skill on the CV that the candidate does NOT have on their profile.

5. OUTPUT FORMAT:
   - Conclude your response with a single JSON object in this exact format:
{
  "score": 0.95,
  "reason": "Brief explanation of your finding and matched person_id."
}
"""

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV."""
    sem = asyncio.Semaphore(3)

    async def verify_single_cv(name: str, text: str) -> tuple[str, float | None]:
        async with sem:
            try:
                response = await agent.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Verify this CV strictly according to your instructions:\n\n{text}",
                            }
                        ]
                    },
                    config={"recursion_limit": 60},
                )

                last_content = response["messages"][-1].content
                if isinstance(last_content, list):
                    output_text = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in last_content
                    )
                else:
                    output_text = str(last_content)

                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', output_text)
                reason_str = reason_match.group(1) if reason_match else "No reason parsed"

                matches = re.findall(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', output_text)
                if matches:
                    score = max(0.0, min(1.0, float(matches[-1])))
                    print(f"[{name}] Score: {score} | Reason: {reason_str}")
                    return name, score

                scores = re.findall(r"\b(?:0\.\d+|1\.0|0|1)\b", output_text)
                if scores:
                    score = float(scores[-1])
                    print(f"[{name}] Score (fallback): {score} | Output: {output_text[-120:]}")
                    return name, score

                print(f"[{name}] Defaulting to 0.5 | Output: {output_text[-120:]}")
                return name, 0.5
            except Exception as e:
                print(f"[Error] Failed to process CV {name}: {e}")
                return name, 0.5

    tasks = [verify_single_cv(name, text) for name, text in cvs.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)

# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())