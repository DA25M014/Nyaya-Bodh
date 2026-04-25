# Nyaya-Bodh

A multilingual legal Q&A assistant for the Bharatiya Nyaya Sanhita (BNS) 2023, with IPC-to-BNS clause comparison and government scheme discovery. Built on Databricks (Delta Lake, Spark, MLflow) with OpenAI for embeddings and generation, served as a Databricks App via Gradio.

Track: Nyaya-Sahayak (Governance & Access to Justice)

## What it does

A user types a question in Hindi or English (e.g., "What is the punishment for theft under BNS?" or "BNS me chori ki saza kya hai?") and the system:

1. Retrieves the relevant BNS sections using semantic search over a Delta-backed corpus.
2. Generates a grounded answer in the user's language with section citations.
3. Surfaces matching government schemes from `gov_myscheme`.
4. Provides a side-by-side IPC to BNS clause comparison for legacy queries.

## Architecture

```
                         User Query (Hindi / English)
                                     |
                         +---------- v ----------+
                         |   Gradio UI (App)     |
                         +---------- + ----------+
                                     |
                  +------------------+------------------+
                  |                  |                  |
          +-------v-------+  +-------v-------+  +-------v-------+
          |  Q&A Pipeline |  |  IPC vs BNS   |  | Scheme Finder |
          |     (RAG)     |  |   Compare     |  |  (Spark SQL)  |
          +-------+-------+  +-------+-------+  +-------+-------+
                  |                  |                  |
        +---------+--------+         |                  |
        |                  |         |                  |
+-------v------+   +-------v------+  |                  |
|  OpenAI      |   |  FAISS index |  |                  |
|  Embeddings  |   |   (DBFS)     |  |                  |
+--------------+   +-------+------+  |                  |
                           |         |                  |
                  +--------v---------v------------------v--------+
                  |              Delta Lake Tables               |
                  |  legal.bns_chunks | legal.ipc_bns_map |      |
                  |  legal.schemes                               |
                  +-------------------+--------------------------+
                                      |
                              +-------v--------+
                              |   MLflow       |
                              |  (queries,     |
                              |   retrievals,  |
                              |   responses)   |
                              +----------------+
```

A renderable mermaid version is in `docs/architecture.md`.

## Repository layout

```
nyaya-bodh/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── config.py
├── src/
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── llm.py
│   ├── ipc_bns_compare.py
│   ├── scheme_finder.py
│   ├── prompts.py
│   └── pipeline.py
├── app/
│   └── app.py
├── notebooks/
│   ├── 01_data_ingestion.py
│   ├── 02_build_embeddings.py
│   └── 03_evaluation.py
├── data/
│   ├── bns_sections.txt
│   ├── ipc_bns_mapping.csv
│   ├── schemes.csv
│   └── eval_questions.json
├── tests/
│   └── test_pipeline.py
├── scripts/
│   ├── setup.sh
│   └── run_local.sh
└── docs/
    └── architecture.md
```

## How to run (local, for development)

Tested on Python 3.10 and 3.11.

```bash
git clone <your-fork-url> nyaya-bodh
cd nyaya-bodh

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set OPENAI_API_KEY

bash scripts/setup.sh

python -m app.app
```

The Gradio UI starts at `http://127.0.0.1:7860`.

`scripts/setup.sh` ingests the bundled sample BNS text, builds the FAISS index, and loads the IPC-to-BNS mapping and scheme tables. It writes everything under `./artifacts/` so the local run does not require a Spark cluster.

## How to run (on Databricks Free Edition)

1. Create a workspace on Databricks Free Edition and upload the repo (or clone it from GitHub via Repos).
2. Add `OPENAI_API_KEY` as a secret:
   ```bash
   databricks secrets create-scope nyaya-bodh
   databricks secrets put --scope nyaya-bodh --key openai_api_key
   ```
3. Run the notebooks in order:
   - `notebooks/01_data_ingestion.py` writes `legal.bns_chunks`, `legal.ipc_bns_map`, `legal.schemes` Delta tables.
   - `notebooks/02_build_embeddings.py` computes embeddings and persists the FAISS index to `/dbfs/FileStore/nyaya_bodh/faiss.index`.
   - `notebooks/03_evaluation.py` runs the eval set and logs to MLflow.
4. Deploy the app:
   ```bash
   databricks apps deploy nyaya-bodh --source-code-path /Workspace/Users/<you>/nyaya-bodh
   ```
   The app entrypoint is `app/app.py`.

## Demo steps

The judges should be able to reproduce these in under two minutes.

1. Open the running app URL.
2. In the "Ask a Question" tab, paste: `What is the punishment for theft under the new BNS?`
   Expected: a grounded answer citing BNS Section 303, with the source chunk shown below the answer.
3. Switch the language toggle to Hindi and paste: `BNS me dhokhadhadi ki saza kya hai?`
   Expected: a Hindi answer citing BNS Section 318 (cheating).
4. In the "IPC vs BNS" tab, enter `420` in the IPC section box.
   Expected: side-by-side panel showing IPC 420 (cheating) mapped to BNS Section 318, with a short summary of changes.
5. Below every Q&A answer, the "Related Government Schemes" panel surfaces matching schemes from `legal.schemes` (e.g., legal aid schemes for queries about access to justice).

## Mandatory requirements checklist

| Requirement | How it is met |
| --- | --- |
| Databricks as core | Delta Lake tables for chunks, mappings, schemes; Spark UDF for chunking; FAISS persisted to DBFS; MLflow tracking |
| AI is central | RAG-driven legal Q&A is the core value; the app is unusable without the model |
| Working demo | Gradio UI deployed as a Databricks App, full reproduction script in `scripts/setup.sh` |
| Indian context | BNS 2023 transition, government schemes, Hindi support |
| User-facing component | Gradio app served via Databricks Apps |

## Models used

- `gpt-4o-mini` for grounded answer generation (configurable to `gpt-4o`).
- `text-embedding-3-small` for chunk and query embeddings.

Both are configured via `config.py` and can be swapped for any OpenAI-compatible endpoint, including locally hosted Param-1 / Sarvam-m via an OpenAI-compatible server such as `vllm` or `llama-cpp-python`. See `src/llm.py` and `src/embeddings.py` for the abstraction points.

## Datasets

- BNS 2023 (`data/bns_sections.txt`): a curated subset of the Bharatiya Nyaya Sanhita 2023 covering offences against the body, property, public tranquillity, and women. Replace with the full Gazette PDF for the production version (drop it into `data/bns.pdf` and re-run `scripts/setup.sh`).
- IPC-to-BNS mapping (`data/ipc_bns_mapping.csv`): clause-level mapping of legacy IPC sections to their BNS equivalents.
- Government schemes (`data/schemes.csv`): a sample drawn from the `gov_myscheme` open dataset.

## Evaluation

`notebooks/03_evaluation.py` runs the system on `data/eval_questions.json` and reports retrieval hit-rate at top-5 and a simple LLM-as-judge correctness score, both logged to MLflow. Run locally with:

```bash
python -m notebooks.03_evaluation
```

## Configuration

All configuration lives in `config.py` and is overridable via environment variables. Key knobs:

- `OPENAI_API_KEY`: required.
- `OPENAI_CHAT_MODEL`: default `gpt-4o-mini`.
- `OPENAI_EMBED_MODEL`: default `text-embedding-3-small`.
- `CHUNK_TOKENS`: default `512`.
- `CHUNK_OVERLAP`: default `64`.
- `RETRIEVAL_TOP_K`: default `5`.
- `ARTIFACTS_DIR`: default `./artifacts` locally, `/dbfs/FileStore/nyaya_bodh` on Databricks.

## License

MIT. Datasets retain their original licenses (Open Government Data License for `data.gov.in`, BNS 2023 is in the public domain as a Government of India enactment).
