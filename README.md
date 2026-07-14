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

The services listen on ports `6001` through `6006` by default. `grpc_server.py`
orchestrates calls between them, while `main.py` exposes the pipeline through an
MCP stdio server.

## Project structure

```text
VOC_Improve/
|-- agents/                 # The six gRPC agent services
|-- llm_wrappers/           # OpenAI and Anthropic client wrappers
|-- quality_diagnosis/      # Quality, fault-tolerance, and end-to-end tests
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
