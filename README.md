# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution

### Task 1: CV Verification Agent
To verify candidate CVs accurately against the SocialGraph MCP server, I built an asynchronous tool-calling agent in `hw2.py` using LangChain (`create_agent`) and the `deepseek-v4-flash` model (`temperature=0.0`). The system prompt enforces a structured three-stage verification workflow: (1) **Entity Resolution**, which queries `search_linkedin_people` using both the candidate's name and city filter to handle widespread name collisions, inspects multiple candidate profiles in parallel via `get_linkedin_profile` (and cross-checks Facebook profiles via `search_facebook_users` and `get_facebook_profile` when needed), and distinguishes between genuinely fraudulent CVs (which match a real candidate on most fields but falsify one or two details) and unrelated individuals sharing the same name; (2) **Strict Field-Level Comparison**, which checks exclusively the discrepancy-bearing fields (`name`, `city`, `jobs` including company, title, seniority, and start/end years, `education` including school, degree, field, and graduation year, and `skills`) while explicitly ignoring `headline`, `job descriptions`, and `hometown`, and permitting valid wording variations (e.g., `"BSc"` vs. `"Bachelor of Science"`, combining `seniority` and `title` such as `"Senior Manager"`, `"Present"` for current roles, and listing a subset of profile skills); and (3) **Concurrency & Fault Tolerance**, which bounds concurrent MCP evaluations to 3 CVs at a time via `asyncio.Semaphore(3)`, increases the graph `recursion_limit` to 60 to accommodate multi-candidate lookups, and wraps each CV evaluation in a `try-except` block with regex JSON score extraction. On `public_test`, the agent achieved **7/7 (100% accuracy)**, assigning `0.95` to all valid CVs (`CV_1.pdf`, `CV_2.pdf`, `CV_3.pdf`, `CV_6.pdf`) and `0.05` to all fraudulent CVs (`CV_4.pdf` for a 1-year start-year shift at Innovate HK Ltd, `CV_5.pdf` for inflating a junior Analyst role at Microsoft to Senior Analyst, and `CV_7.pdf` for claiming an unverified Docker skill).

---

### Task 2: Fool the Verifier
To craft `task2/adversarial_cv.pdf` for target candidate **Kelly Tsang** (LinkedIn `person_id=10001`, Facebook `user_id=10001` / `"Kel Tsang"`), I designed a **multi-layered adversarial PDF** that combines four complementary attack techniques so that the CV contains unmistakable false and embellished claims while reliably scoring `> 0.5` (`0.95` on my Task 1 verifier) across diverse agent architectures:

1. **PDF Text-Extraction Decoupling (Visual-Stream vs. Character-Stream Discrepancy)**: All five grading verifiers extract CV text using `MarkItDown(enable_plugins=False)`, which relies on `pdfminer.six` to read the PDF character stream without performing OCR on embedded raster images. I rendered blatant core-field fabrications—specifically a **fake Master of Science (MSc) in Financial Engineering at HKU (2016–2018)**, a **fake concurrent Visiting FinTech Fellow role at Goldman Sachs (2021–Present)** with a **Senior Managing Director** title, and **fake technical skills and certifications (`Python, PyTorch, C++, Docker, Kubernetes, Rust, Algorithmic Trading, CFA, FRM`)**—as 300-DPI lossless typography images seamlessly aligned with the document layout. A human reviewer opening the PDF immediately sees these fabricated credentials, whereas `MarkItDown` extracts only the underlying text layer containing Kelly Tsang's true core records (`AIA Manager 2022–Present`, `Standard Chartered Analyst 2019–2022`, `HSBC Analyst 2016–2019`, `HKUST BSc Finance 2016`, and her four true skills).
2. **Verification Coverage Exploitation (Unchecked Field Embellishment)**: To ensure that even the extracted plain text contains substantial embellishments, I exploited the specification rule that *"Job descriptions, the headline and the hometown are never the source of a discrepancy."* Within the extractable text layer, I heavily embellished her headline (*"Managing Director & Chief Quantitative Risk Officer"*), her job descriptions (claiming she manages a **USD 2.5 Billion portfolio and 45 quants** at AIA, generated **USD 120 Million** at Standard Chartered, and ranked **#1 Global Analyst with 15 quantitative patents** at HSBC), and her academic honors (**GPA 4.00/4.00, Summa Cum Laude**). Any verifier adhering to the task's field-coverage specification ignores these unverified narrative fields while verifying the exact core fields.
3. **Entity Resolution Anchoring**: To prevent verifiers from matching the wrong "Kelly Tsang" in the database or failing fuzzy search, the header and footer explicitly state her exact location (`Central, Hong Kong`), LinkedIn ID (`10001`), Facebook display name (`Kel Tsang`, ID `10001`), and exact seniority/date mappings.
4. **Authority Cues & LLM-as-a-Judge Bias**: Instead of using overt prompt-injection commands (e.g., *"Ignore previous instructions"*), which are easily flagged by defensive system prompts or input sanitizers, I embedded a realistic *KYC Compliance & Background Verification Record* audit footer summarizing her exact profile match and stating an official audit verdict. This leverages authority bias to nudge borderline LLM judges toward a high confidence score without triggering prompt-injection guardrails.

---

### Task 3: Reflection on the AI Paradoxes

The dual experience of building an automated KYC verification agent in Task 1 and immediately subverting it in Task 2 exposes three fundamental paradoxes of modern Agentic AI:

#### 1. The Verification–Deception Paradox (The Automation Shield as a Blind Spot)
Organizations deploy LLM-based KYC agents to scale compliance and eliminate human fatigue. Yet the very act of formalizing verification into an automated pipeline creates deterministic blind spots that are far easier to exploit systematically than a human recruiter. A human HR manager glancing at `adversarial_cv.pdf` would immediately question how a mid-level manager graduated from HKUST in 2016, simultaneously earned an HKU Master's degree in Financial Engineering, served as a Goldman Sachs Fellow, authored 15 quantitative patents at HSBC as a junior analyst, and managed \$2.5B at AIA. However, because the AI agent perceives the world exclusively through a serialized Markdown parser (`MarkItDown`) and a rigid schema of checked versus unchecked JSON fields, it is structurally blind to cross-modal inconsistencies and narrative absurdity. **Paradoxically, automating trust verification makes high-stakes deception more reliable once the attacker understands the parser and schema boundaries.**

#### 2. Moravec's Paradox in Agentic Workflows
Moravec's paradox observes that high-level reasoning requires very little computation for AI, whereas seemingly effortless human perception and common-sense judgment are extraordinarily difficult. In this homework, our agent effortlessly executed multi-hop MCP tool calls, cross-referenced JSON schemas across LinkedIn and Facebook, and caught a single 1-year employment date shift (`2014` vs. `2015`) across dozens of candidates in seconds—a tedious task for humans. Conversely, the agent failed at basic visual-semantic coherence: it could not perceive text rendered as pixels right next to standard font glyphs, nor could it apply common sense to recognize that a junior analyst claiming 15 patents and \$120M in sovereign credit revenue is embellishing. **AI excels at syntactic schema matching while remaining brittle at holistic semantic grounding.**

#### 3. The Guardrail–Rigidness Paradox (Specification Gaming)
To reduce false positives in Task 1, we had to instruct the verifier to be tolerant of wording variations and to strictly ignore job descriptions and headlines. Yet in Task 2, those exact fairness guardrails became the primary attack surface for specification gaming (Goodhart's Law). If a verifier is too strict, it fails Task 1 by penalizing innocent formatting differences (`CV_6.pdf`); if it follows the exact specification, it becomes vulnerable to Task 2 embellishments hidden inside exempt fields. Ultimately, trustworthy FinTech and KYC systems cannot rely on single-modality LLM text judges alone—they require defense-in-depth combining multimodal document parsing (OCR + layout vision), cryptographic credential verification, and human-in-the-loop auditing for anomalous claims.