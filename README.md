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

The Interpreter is parsing-only and does not call downstream agents. After its
intent is returned, the processing chain runs once in the fixed order
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
|-- agents/                 # The six gRPC agent services
|-- llm_wrappers/           # OpenAI and Anthropic client wrappers
|-- quality_diagnosis/      # Test cases, LLM judge, reports, and QA tests
|-- tests/                  # Main automated test suite
|-- utils/                  # Settings, MCP tools, logging, and helpers
|-- grpc_server.py          # Pipeline orchestrator
|-- launch_agents.py        # Starts and supervises all agent services
|-- main.py                 # MCP stdio server entry point
|-- voc.csv                 # Default VOC input data
|-- voc.proto               # gRPC service definitions
|-- requirements.txt        # pip dependencies
`-- pyproject.toml          # Python project metadata
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

## LLM quality diagnosis

The `quality_diagnosis` workflow evaluates saved pipeline results independently
of the live gRPC services. It uses OpenAI to judge policy quality and Anthropic
to judge candidate-summary fidelity.

### Judge inputs

- `quality_diagnosis/judge_cases.json` contains 20 cases. Each case has a
  `question`, candidate summaries (`S0`, `S1`, and `S2`), and a generated
  `policy`.
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

## Forwarding VOC statements to the Interpreter

Start the six agents before forwarding a statement. The Interpreter listens at
`localhost:6001` by default and converts the statement into a structured intent
containing a task, filters, item limit, and CSV path.

### Through the MCP server (recommended)

Call `analyze_voc_nl_v2` from the connected MCP client and put the raw customer
statement in the `question` field:

```json
{
  "question": "My payment was completed, but the order status has not changed.",
  "csv_path": "C:\\path\\to\\VOC_Improve\\voc.csv"
}
```

The request follows this route:

```text
MCP client -> analyze_voc_nl_v2 -> Interpreter -> remaining agent pipeline
```

The `csv_path` field is optional. When omitted, the application uses
`A2A_VOC_CSV` or the project-root `voc.csv`. The returned object includes the
generated `summary`, `policy`, and `intent_json` produced from the Interpreter's
classification.

### Directly through gRPC (development and debugging)

To inspect only the Interpreter's structured response, save the following as a
temporary script in the project root and run it while the agents are active:

```python
import asyncio

import grpc

import voc_pb2
import voc_pb2_grpc


async def main() -> None:
    statement = "My payment was completed, but the order status has not changed."

    async with grpc.aio.insecure_channel("localhost:6001") as channel:
        interpreter = voc_pb2_grpc.InterpreterStub(channel)
        response = await interpreter.ParseQuestion(
            voc_pb2.ParseQuestionReq(
                question=statement,
                default_csv="voc.csv",
            ),
            timeout=30,
        )

    print("task:", response.task)
    print("filters:", list(response.filters))
    print("max_items:", response.max_items)
    print("csv_path:", response.csv_path)


asyncio.run(main())
```

Run the script with the project's virtual environment:

```powershell
python forward_voc.py
```

For an Interpreter running elsewhere, replace `localhost:6001` with its address.
When using the complete application, set `INTERPRETER_ENDPOINT` in `.env` to the
same address.

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
```

The live end-to-end quality test makes real model calls and expects all six
agents to already be running in another terminal:

```powershell
python -m unittest quality_diagnosis.test_pipeline_e2e -v
```

Quality reports are generated under `quality_diagnosis/reports/`. Live tests can
consume API credits and may take several minutes depending on model response
times.

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
