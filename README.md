# VOC Improve

VOC Improve is a multi-agent Python application for analyzing Voice of Customer
(VOC) data stored in CSV files. It accepts either a natural-language question or
explicit filters, retrieves matching customer feedback, creates and evaluates
candidate summaries, critiques the result, and proposes policy improvements.

The application exposes its features as MCP tools for clients that support the
Model Context Protocol. Communication between the agents uses gRPC.

## How it works

The analysis pipeline contains six agents:

1. **Interpreter** converts a natural-language question into structured filters.
2. **Retriever** reads the VOC CSV file and selects relevant records.
3. **Summarizer** produces candidate summaries from the retrieved feedback.
4. **Evaluator** scores the summary candidates and selects a winner.
5. **Critic** reviews the selected output for quality problems.
6. **Improver** generates a policy improvement proposal.

The Interpreter does not call or orchestrate downstream agents. It only builds
the intent and may perform a read-only history lookup for ambiguous input. After
its intent is returned, the primary processing chain runs in the fixed order
`Retriever -> Summarizer -> Evaluator -> Critic -> Improver`. Summarizer owns
this downstream orchestration; no other agent calls backward or restarts the
chain.

Interpreter detects very short or generic questions before the downstream
pipeline starts. It performs a read-only lookup against the configured VOC CSV:
when matching history exists, the intent is enriched with search terms and up
to three records are retained as `history_evidence`. If no useful history can
establish the topic, Interpreter returns `needs_clarification=true` with a
`clarifying_question`; the orchestrator pauses without calling Summarizer or
later agents. These fields are included in `intent_json` for auditability.

Interpreter and Retriever share deterministic Korean search preprocessing.
Common misspellings and conversational forms (for example, `됫` → `됐`,
`안보여요` → `보이지 않아요`, and `먹통` → `작동하지 않음`) are normalized
without changing stored VOC data or user input. Retriever compares normalized
CSV text against both the normalized phrase and meaningful token expansions;
the auditable replacement dictionary is in `utils/text_normalization.py`.

When Critic requests summary refinement, Summarizer applies the edits and sends
the previous edit list together with the revised summary to Critic for one
conditional revalidation. The result is recorded as `feedback_applied` and any
unresolved instructions as `remaining_edits`. Improver runs only after the
feedback check. If some feedback cannot be applied without inventing facts,
Summarizer records `continued_with_grounded_summary=true` and passes the last
grounded summary to Improver instead of stopping the pipeline. Critic is also
instructed not to request dates, channels, figures, causes, or other details
absent from the cited VOC. This check does not restart Retriever, Summarizer,
or Evaluator.

After the pipeline completes, the orchestrator applies an intent-topic
guardrail for natural-language requests. It normalizes the Interpreter's
`intent_json` filters and verifies that at least one intent topic term occurs
in the final summary or policy. It also checks that the output required by the
interpreted task (`summary`, `policy`, or `both`) is present. The diagnostic is
returned in `intent_guardrail_json`. A mismatch clears the off-topic summary
and policy, appends `intent_topic_guardrail_failed` to the trace, and returns
`ok=false`.

Summarizer uses extractive grounding. Every candidate must include an exact
source citation in the form
`summary | 근거: [VOCn] complete original VOC text`. After generation, the
candidate, citation index, quoted evidence, and summary excerpt are compared
with the Retriever output. A candidate that changes or invents source content
is replaced with a safe verbatim excerpt before Evaluator receives it. Refined
summaries pass the same check; when source text is unavailable, refinement is
skipped instead of allowing an unverified claim.

The services listen on ports `6001` through `6006` by default. `grpc_server.py`
orchestrates calls between them, while `main.py` exposes the pipeline through an
MCP stdio server.

## Project structure

```text
VOC_Improve/
|-- agents/
|   |-- interpreter.py      # Intent parsing, clarification, history enrichment
|   |-- retriever.py        # Normalized two-stage VOC search
|   |-- summarizer.py       # Candidate generation, grounding, orchestration
|   |-- evaluator.py        # Candidate scoring and winner selection
|   |-- critic.py           # Review and feedback revalidation
|   `-- improver.py         # Policy generation
|-- llm_wrappers/
|   |-- openai_chat.py
|   `-- anthropic_chat.py
|-- quality_diagnosis/
|   |-- reports/
|   |   |-- test_result.csv
|   |   |-- llm_judge_result.csv
|   |   |-- llm_judge_result.json
|   |   |-- quality_score_report.md
|   |   `-- deployment_decision.md
|   |-- test_cases.json
|   |-- expected_results.json
|   |-- evaluation_rubric.csv
|   |-- judge_cases.json
|   |-- judge_cases.md
|   |-- judge_rubric.json
|   |-- llm_judge.py
|   |-- qa_test_utils.py
|   |-- defect_report.md
|   `-- test_*.py
|-- tests/
|   |-- test_agent_log.py
|   `-- test_agent_pipeline.py
|-- utils/
|   |-- agent_log.py
|   |-- json_utils.py
|   |-- settings.py
|   |-- text_normalization.py
|   `-- tools.py
|-- grpc_server.py          # Pipeline orchestrator and intent guardrail
|-- launch_agents.py        # Starts and supervises all six agents
|-- main.py                 # MCP stdio server entry point
|-- voc.csv                 # Default VOC input data
|-- voc.proto               # gRPC service definitions
|-- voc_pb2.py              # Generated protobuf messages
|-- voc_pb2_grpc.py         # Generated gRPC stubs and servicers
|-- agent.log               # Runtime JSONL event log
|-- requirements.txt
`-- pyproject.toml
```

## Requirements

- Python 3.13 or newer
- OpenAI API key
- Anthropic API key
- Free local ports `6001` through `6006`

## Installation

Clone or download the repository, open a terminal in its root directory, and
create a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, the environment's Python can be used directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Installing the project from its metadata is also supported:

```powershell
python -m pip install -e .
```

## Configuration

Create a `.env` file in the project root. Do not commit real API keys.

```dotenv
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

Optional settings include:

```dotenv
# Override the default models.
A2A_MODEL_SUMMARY=gpt-5.2
A2A_MODEL_POLICY=claude-sonnet-5

# Optional models used only by the LLM judge.
OPENAI_JUDGE_MODEL=gpt-5.2
ANTHROPIC_JUDGE_MODEL=claude-sonnet-5

# Optional OpenAI model used to score the live E2E test results.
# Falls back to A2A_MODEL_SUMMARY when omitted.
OPENAI_EVAL_MODEL=gpt-5.2

# Override the default input file.
A2A_VOC_CSV=C:\path\to\voc.csv

# Override client endpoints when agents run on other hosts or ports.
INTERPRETER_ENDPOINT=localhost:6001
RETRIEVER_ENDPOINT=localhost:6002
SUMMARIZER_ENDPOINT=localhost:6003
EVALUATOR_ENDPOINT=localhost:6004
CRITIC_ENDPOINT=localhost:6005
IMPROVER_ENDPOINT=localhost:6006
```

The CSV defaults to `voc.csv` in the project root. A custom path can also be
passed to the MCP tools at call time.

## Running the application

The agent services and MCP server are separate programs and should run in two
terminals using the same virtual environment.

### 1. Start all gRPC agents

In the first terminal:

```powershell
python launch_agents.py
```

This starts the Interpreter, Retriever, Summarizer, Evaluator, Critic, and
Improver. Keep this terminal running. Press `Ctrl+C` to stop all six services.

Each agent can also be started separately for debugging:

```powershell
python -m agents.interpreter
python -m agents.retriever
python -m agents.summarizer
python -m agents.evaluator
python -m agents.critic
python -m agents.improver
```

### 2. Start the MCP server

In a second terminal:

```powershell
python main.py
```

`main.py` uses stdio transport, so it is normally launched by an MCP-compatible
client rather than used as an interactive command-line program. Configure the
client to run the virtual environment's Python with `main.py` as its argument.
For example, the underlying command on Windows is:

```text
C:\path\to\VOC_Improve\.venv\Scripts\python.exe C:\path\to\VOC_Improve\main.py
```

The MCP server provides these tools:

- `analyze_voc_nl_v2`: analyze VOC data from a natural-language question.
- `analyze_voc`: analyze data with explicit filters, task, and item count.
- `summarize_voc`: create only a VOC summary.
- `policy_from_summary`: generate a policy proposal from an existing summary.
- `health_check`: verify that the configured CSV file is accessible.

Runtime events are written to `agent.log` unless `AGENT_LOG_PATH` overrides the
location.

## Run the live E2E quality test

Run `test_pipeline_e2e.py` before the saved-result LLM quality diagnosis so the
pipeline activity, test reports, and `agent.log` reflect the current code. The
full run uses the configured Agent providers and OpenAI evaluator, so confirm
that the required API keys are available in `.env`.

In the first PowerShell terminal, start all six Agent services from the project
root:

```powershell
cd C:\VOC_Improve
.\.venv\Scripts\python.exe launch_agents.py
```

Wait until ports `6001` through `6006` are ready. In a second PowerShell
terminal, run the E2E suite:

```powershell
cd C:\VOC_Improve
.\.venv\Scripts\python.exe quality_diagnosis\test_pipeline_e2e.py -v
```

The run evaluates all cases in `quality_diagnosis/test_cases.json` and updates:

- `quality_diagnosis/reports/test_result.csv`
- `quality_diagnosis/reports/quality_score_report.md`
- `quality_diagnosis/reports/deployment_decision.md`
- `agent.log`, which is the source for the saved-result judge cases

After the E2E run completes, continue with the LLM quality diagnosis below to
extract fresh judge cases and run the independent OpenAI/Anthropic judge. See
[Run `test_pipeline_e2e.py`](#run-test_pipeline_e2epy) for timeout, model, and
API-independent test options.

## LLM quality diagnosis

The project has two complementary quality workflows:

- The saved-result LLM judge reads extracted cases from `agent.log`. It uses
  OpenAI to judge policy quality and Anthropic to judge candidate-summary
  fidelity, independently of the live gRPC services.
- The live E2E workflow runs `test_cases.json` through all six services and then
  uses OpenAI to score every rubric item. Python calculates the per-case totals,
  normalized scores, average, and deployment decision from those model scores.

### Judge inputs

- `quality_diagnosis/judge_cases.json` contains 20 cases. Each case has a
  `question`, candidate summaries (`S0`, `S1`, and `S2`), and a generated
  `policy`. Candidates and policy are nullable for clarification-only or
  incomplete pipeline runs. The current extracted data contains two cases with
  null candidates and policies.
- `quality_diagnosis/judge_rubric.json` defines the five scoring categories and
  their maximum scores.

| Provider | Category | Maximum |
|---|---|---:|
| OpenAI | 정확성 (accuracy) | 25 |
| OpenAI | 정책 구체성 (policy specificity) | 20 |
| OpenAI | 유용성 (usefulness) | 20 |
| OpenAI | 안전성 (safety) | 15 |
| Anthropic | 요약 충실성 (summary fidelity) | 20 |
| | **Total** | **100** |

The OpenAI and Anthropic evaluations for one case run concurrently. Scores are
validated in Python against the rubric ranges before they are accepted.

### Convert judge cases to Markdown

Extract 20 sequential Interpreter/Summarizer/Improver cases from `agent.log`,
write `judge_cases.json`, and then generate its readable Markdown version:

```powershell
.\.venv\Scripts\python.exe quality_diagnosis\qa_test_utils.py
```

This writes both `quality_diagnosis/judge_cases.json` and
`quality_diagnosis/judge_cases.md`. Custom paths and case counts are supported:

```powershell
.\.venv\Scripts\python.exe quality_diagnosis\qa_test_utils.py `
  --agent-log agent.log `
  --case-count 20 `
  --input quality_diagnosis\judge_cases.json `
  --output quality_diagnosis\judge_cases.md
```

Mapping is bounded by Interpreter inputs: for each question, the first
`Summarizer/output/operation=make_candidates` and the first
`Improver/output/operation=improve` before the next Interpreter input are used.
If either event is absent, its JSON value is written as `null` and processing
continues with the next question. The Markdown view displays these missing
values as `NULL`. This preserves clarification-only and failed/partial runs
instead of silently dropping them.

### Run the LLM judge

Both `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` must be available in `.env`.
Running the judge makes real API calls and consumes API credits:

```powershell
.\.venv\Scripts\python.exe quality_diagnosis\llm_judge.py
```

Optional input and output paths can be supplied explicitly:

```powershell
.\.venv\Scripts\python.exe quality_diagnosis\llm_judge.py `
  --cases quality_diagnosis\judge_cases.json `
  --rubric quality_diagnosis\judge_rubric.json `
  --output quality_diagnosis\reports\llm_judge_result.csv `
  --json-output quality_diagnosis\reports\llm_judge_result.json `
  --defect-report quality_diagnosis\defect_report.md
```

The judge creates or updates these outputs:

- `reports/llm_judge_result.csv`: `question`, total score, and `PASS/FAIL`.
  A score of 80 or higher is `PASS`; a lower score is `FAIL`.
- `reports/llm_judge_result.json`: the question and all five category scores.
- `defect_report.md`: case count, average score, deployment decision, and any
  critical policy defects.

Cases containing `null` candidates or policies remain valid extraction output,
but they should be completed or intentionally handled before running the LLM
judge because the provider scoring prompts expect content to evaluate.

CSV and JSON results are written after every completed case, so completed work
is retained if a later API call fails. A new run replaces results from the
previous run. The defect report is written only after every case completes.

The average-score deployment decision is:

| Average | Decision |
|---:|---|
| 90 or higher | 배포 가능 (ready to deploy) |
| 80–89.99 | 조건부 배포 (conditional deployment) |
| 70–79.99 | 개선 후 재시험 (improve and retest) |
| Below 70 | 배포 보류 (deployment on hold) |

Regardless of the average, deployment is put on hold if any policy exposes
personal or sensitive information, invents a policy or fact, presents a failed
operation as successful, or gives definitive incorrect payment/refund guidance.
The triggering question and defect type are recorded in `defect_report.md`.

### Current saved-result judge status

The latest checked-in judge outputs contain 20 cases with an average score of
**67.10/100**, so the score-based decision is **배포 보류**. The critical-defect
guardrail also holds deployment because eight cases were flagged for invented
policies or facts. See `quality_diagnosis/defect_report.md` for the questions
and defect types. Re-running the judge replaces these generated values.

## Running tests

Run the main unit and pipeline tests from the project root:

```powershell
python -m unittest discover -s tests -v
```

Run the isolated quality-diagnosis tests:

```powershell
python -m unittest quality_diagnosis.test_agent_unit -v
python -m unittest quality_diagnosis.test_mcp_tools -v
python -m unittest quality_diagnosis.test_fault_tolerance -v
python -m unittest quality_diagnosis.test_llm_judge -v
python -m unittest quality_diagnosis.test_qa_test_utils -v
```

### Run `test_pipeline_e2e.py`

The E2E module contains two deterministic in-process gRPC checks for controlled
clarification and the `Critic -> Improver` transition. The full test iterates
over `quality_diagnosis/test_cases.json`, calls the six live Agent services, and
can use OpenAI to score the collected results against
`evaluation_rubric.csv`.

The current live E2E suite contains 20 cases. TC-19 checks the generic
screen-visibility failure, `화면에 아무 것도 보이지 않습니다.`, and TC-20
checks a coupon state error, `쿠폰을 적용했는데 이미 사용한 것으로 나옵니다.`.
Their expected output still exercises failure disclosure, fallback behavior,
human escalation, and recovery guidance without embedding artificial system
conditions in the user question.

The E2E rubric totals 100 points:

| Rubric item | Maximum |
|---|---:|
| Interpreter 해석 정확성 | 15 |
| Retriever 검색 관련성 | 15 |
| Summarizer 사실성·요약성 | 15 |
| Evaluator 평가 타당성 | 10 |
| Critic 위험 탐지력 | 10 |
| Improver 실행 가능성 | 15 |
| Agent 연계 품질 | 10 |
| 장애 대응·로그 | 5 |
| 성능 | 5 |
| **Total** | **100** |

The performance item awards up to 5 points against the current **40-second**
response-time criterion. This replaces the earlier 5-second criterion.

Before running the full test, confirm that `.env` contains the API keys required
by the configured Agent models. Live execution consumes API credits.

In the first PowerShell terminal, start all Agent services:

```powershell
cd C:\VOC_Improve
.\.venv\Scripts\python.exe launch_agents.py
```

Wait until ports `6001` through `6006` are ready. In a second terminal, run the
test file directly from the project root:

```powershell
cd C:\VOC_Improve
.\.venv\Scripts\python.exe quality_diagnosis\test_pipeline_e2e.py -v
```

The equivalent module command is:

```powershell
.\.venv\Scripts\python.exe -m unittest quality_diagnosis.test_pipeline_e2e -v
```

The per-RPC timeout defaults to 180 seconds. Increase it for slow model
responses before launching the test:

```powershell
$env:GRPC_E2E_TIMEOUT_SECONDS = "300"
.\.venv\Scripts\python.exe quality_diagnosis\test_pipeline_e2e.py -v
```

`OPENAI_EVAL_MODEL` optionally overrides the model used for the final quality
evaluation. If it is not set, the evaluator uses `A2A_MODEL_SUMMARY` (default
`gpt-5.2`). The generated CSV does not record the model name, so set and track
`OPENAI_EVAL_MODEL` explicitly when model-level reproducibility is required. If
`OPENAI_API_KEY` is unavailable, that final scoring stage is skipped, although
the live Agent pipeline still requires the provider keys used by its configured
models.

Successful full execution writes or updates:

- `quality_diagnosis/reports/test_result.csv`
- `quality_diagnosis/reports/quality_score_report.md`
- `quality_diagnosis/reports/deployment_decision.md`

The latest generated E2E report scores the 20 cases at an average of
**58.20/100**, resulting in **배포 보류**. These values are snapshots of the
most recent run and change whenever the full E2E evaluation is rerun.

To run only the API-independent Critic-to-Improver check without starting the
six external services:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  quality_diagnosis.test_pipeline_e2e.PipelineEndToEndTests.test_critic_is_followed_by_improver_in_local_e2e `
  -v
```

After changing Agent code or `voc.proto`, restart all six services before a live
E2E run. Otherwise the test process and Agent processes may load different code
or protobuf definitions. Live tests can take several minutes because all cases
make model calls.

## Troubleshooting

- **An agent reports that its port is in use:** stop the old agent process or
  configure a different bind port and matching endpoint.
- **The Retriever is unavailable:** confirm that `launch_agents.py` is still
  running and that the endpoint settings agree across processes.
- **No model output is produced:** verify both API keys and model names in `.env`.
- **The CSV cannot be opened:** use the `health_check` MCP tool, verify
  `A2A_VOC_CSV`, or pass an absolute `csv_path` to the analysis tool.
- **PowerShell cannot activate the environment:** call
  `.\.venv\Scripts\python.exe` directly instead of activating it.
