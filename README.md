# GPTKB 2.0 with disambiguation on the fly

A knowledge-base construction pipeline driven by LLMs. Starting from a seed entity, it repeatedly asks an LLM for facts
about entities, then recognises and disambiguates the subjects, predicates and objects of those facts against what is
already in the knowledge base. Newly found entities are explored in turn.

The state lives in a SQLite database. Entities, predicates and concepts additionally have embeddings in memory-mapped
files with a faiss index each, which is used to retrieve disambiguation candidates.

Requests can be sent in two modes:

- **Batch mode** (default) uses the OpenAI Batch API: cheaper (batch pricing), but results can take up to 24 h.
  Per role, it can instead use any service implementing the OpenAI Batch API on the chat completions endpoint
  (e.g. Together, Groq).
- **Single mode** sends one request per item, concurrently, and processes each result right away. It works with the
  OpenAI Responses API or with any OpenAI-compatible Chat Completions server (vLLM, Ollama, SGLang, a hosted
  endpoint run by someone else, ...), configurable per role.

## Pipeline

Each round of the main loop (`Constructor.loop`) runs the stages below. Every stage picks up the items whose status
says they are ready for it, so items move through the pipeline over several rounds.

| Stage | Input | What the LLM does | Result |
|---|---|---|---|
| Elicitation | unexplored entity | lists facts about the entity | new triples |
| PD (predicate disambiguation) | triple with a new predicate label | picks a matching existing predicate, or none | predicate id, or → PDG |
| PDG (predicate description generation) | predicate without a match | writes a description | new predicate |
| NER | triple whose predicate is done | decides whether the object is a named entity | NE → NED1, literal → done |
| NED1 | named-entity object | picks a matching existing entity from candidates, or none | entity id, or → NEDG |
| NEDG | object without a match | writes a description | → NED2 |
| NED2 | object with description | picks a matching entity, comparing descriptions | entity id, or new entity |
| CD (concept disambiguation) | `instanceOf` triple | picks a matching existing concept, or none | concept id, or → CDG |
| CDG | concept without a match | writes a description | new concept |

New entities created by NED2 start as unexplored and are later sent to elicitation.

Before calling the LLM, several stages first try cheaper shortcuts: reusing an object id that a (predicate, object
label) pair has consistently resolved to, reusing the predicate id of an identical predicate label, and grouping
similar object labels so only one per group is disambiguated per round.

The LLMs are used in three roles, each with its own model:

| Role | Stages |
|---|---|
| `elicitation` | Elicitation |
| `disambiguation` | NER, NED1, NED2, PD, CD |
| `description` | NEDG, PDG, CDG |

## Setup

Python 3.10+ with:

```bash
pip install sqlmodel sqlalchemy openai loguru faiss-cpu numpy scikit-learn sentence-transformers tqdm fire
```

The embedding model (`Qwen/Qwen3-Embedding-4B`) is downloaded by sentence-transformers on first use.

Set `OPENAI_API_KEY` whenever a role of the mode you run has no backend config, since such a role uses OpenAI. When
every role is configured to use another service, it is not needed; the API keys of those services are configured per
role instead (see below).

## Running

`main.py` takes the database path, a log path and the paths of the embedding/index files:

```bash
python main.py \
    --db_path=data/kb.db --log_path=logs/run.log \
    --inode_embeddings_mmap_path=data/inode.mmap --inode_embeddings_mmap_metadata_path=data/inode_meta.json --inode_Index_path=data/inode.index \
    --predicate_embeddings_mmap_path=data/predicate.mmap --predicate_embeddings_mmap_metadata_path=data/predicate_meta.json --predicate_Index_path=data/predicate.index \
    --concept_embeddings_mmap_path=data/concept.mmap --concept_embeddings_mmap_metadata_path=data/concept_meta.json --concept_Index_path=data/concept.index
```

If the database is empty, it is seeded with the seed entity, predicate and concept (defaults in `Constructor.__init__`).

### Sizing a new KB

Pass the number of entities you expect the KB to end up with, elicited or not:

```bash
python main.py ... --expected_entities=3000000
```

| What | How it is sized |
|---|---|
| Embedding mmaps | entities × 1.5 inode rows; predicates and concepts by the ratios of an existing KB (2.30M entities: 207,633 predicates ≈ 9.0%, 66,523 concepts ≈ 2.9%), × 1.5. Without `--expected_entities`: 20M / 2M / 1M rows as before |
| SQLite page cache (`cache_size`) | a quarter of the physical memory, split over the few connections that can be open at once, and at most the expected database size. `--sqlite_cache_size_mb` sets it directly (per connection) |
| SQLite `mmap_size` | fixed upper bound; SQLite clamps it to its compile-time maximum (often ~2 GB). The effective value is logged at startup |
| faiss indexes | not presized, they grow as vectors are added. They are held in memory: about 4.3 KB per vector (1024 dims, HNSW M=32), i.e. ~10 GB for 2.3M entities |

The embedding mmaps grow on their own (doubling) when they fill up, so the estimate does not have to be exact. The
capacity is stored in the mmap metadata files, and an existing KB is always reopened with that capacity;
`--expected_entities` then only affects the page cache.
The loop stops on its own after a few consecutive rounds with nothing to do (`max_idle_rounds`, default 3), e.g. once
`max_nes_explored` entities were explored and everything downstream is processed.

### Batch mode

The default. Queue sizes and per-round limits are set in `main.py` (`max_batch_size`, `max_queue_size`,
`max_*_triples_iter`). OpenAI models are set where `PromptSchema` is created in `main.py`.

To send a role's batches to another service, pass `--batch_backends_config` with a JSON file mapping roles to
backends; roles left out keep using the OpenAI Batch API. See
[`batch_backends.example.json`](batch_backends.example.json):

```json
{
    "disambiguation": {"base_url": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY",
                       "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "completion_window": "24h"}
}
```

The service must implement the OpenAI Batch API (file upload, `/v1/batches`, result download) on
`/v1/chat/completions`. Requests are translated to chat completions when the batch file is written and the results
are translated back when downloaded, so the rest of the pipeline is unchanged. The backend options are the same as for
single mode (below), plus `completion_window` (default `"24h"`; check which values the service accepts). Which backend
checks a submitted batch follows from its job type, so keep the config unchanged while batches of a run are pending.

Most self-hosted servers (vLLM, Ollama, ...) and hosted chat endpoints have no batch API; use single mode for them.

### Single mode

```bash
python main.py ... --requests_in_batch=False
```

| Option | Default | Meaning |
|---|---|---|
| `--single_chunk_size` | 20 | items sent concurrently per chunk |
| `--single_fetch_size` | 500 | items fetched per round for NED1/PD/CD, which keep only one item per group before sending |
| `--single_request_workers` | 16 | max concurrent requests; keep it low on shared or rate-limited endpoints |
| `--single_backends_config` | none | JSON file routing roles to Chat Completions servers (see below); without it all roles use OpenAI |

Only the API calls run concurrently; parsing and database/index writes stay in the main thread.

### Using other models in single mode

Write a JSON file mapping roles to backends; roles left out keep using OpenAI. See
[`single_backends.example.json`](single_backends.example.json):

```json
{
    "elicitation":    {"base_url": "https://llm.example.com/v1", "api_key_env": "LLM_API_KEY", "model": "your-org/your-model"},
    "disambiguation": {"base_url": "https://llm.example.com/v1", "api_key_env": "LLM_API_KEY", "model": "your-org/your-model"},
    "description":    {"base_url": "https://llm.example.com/v1", "api_key_env": "LLM_API_KEY", "model": "your-org/your-model"}
}
```

```bash
export LLM_API_KEY=...
python main.py ... --requests_in_batch=False --single_backends_config=single_backends.json --single_request_workers=4
```

Backend options:

| Key | Required | Meaning |
|---|---|---|
| `base_url` | yes | e.g. `http://localhost:8000/v1` |
| `model` | yes | model name as the server knows it |
| `api_key_env` | | environment variable holding the API key; preferred over `api_key` so keys stay out of files |
| `api_key` | | the key itself; defaults to `"EMPTY"`, which most local servers accept |
| `merge_system_prompt` | | `true` puts the system prompt at the start of the user message, for models whose chat template has no system role |
| `structured_output` | | `false` stops sending `response_format`, for servers without guided decoding (default `true`) |
| `max_tokens` | | overrides the prompt's output limit; reasoning models need more than the 128 tokens the disambiguation prompts allow |
| `temperature` | | overrides the prompt's temperature |
| `extra_body` | | passed to the server as is, e.g. `{"chat_template_kwargs": {"enable_thinking": false}}` for Qwen3 on vLLM |

The prompts were written for GPT models. The disambiguation parsers expect exactly one option letter (A–G) in the
answer, so check the failure rate per stage in the log on a small run before a large one.

## Notes

- **Existing databases and single mode.** Single mode stores triples without a batch id. Databases created before
  `triple.creating_batch_id` became nullable still have it as `NOT NULL`, so use single mode only on new databases.
  Batch mode works on both.
- **Failed items are retried.** A request or parsing failure leaves the item in its previous status, so a later round
  picks it up again. In batch mode, NED1, PD and CD are not started while a batch of their group (NED1/NEDG/NED2,
  PD/PDG, CD/CDG) is still running, so new entities, predicates and concepts are in the index before they are searched.
- **Errors in one stage do not stop the run.** They are logged with a traceback (`... stage failed in this round`) and
  the loop continues with the next stage.
- **Switching modes.** Batches submitted in batch mode are still collected and processed after switching to single mode.

## Files

| Path | Contents |
|---|---|
| `main.py` | command-line entry point |
| `construction.py` | `Constructor`: the main loop and all stages |
| `llm_backends.py` | LLM backends: OpenAI Responses and Chat Completions (single mode), OpenAI Batch API and compatible batch services (batch mode) |
| `prompter_parser/` | request templates and response parsers (`PromptSchema`) |
| `prompts/` | system prompts per stage |
| `db/db_models.py` | database schema |

## Citation

If you use this code, please cite:

> Yujia Hu, Tuan-Phong Nguyen, Simon Razniewski. *Constructing Disambiguated Knowledge Bases from Large Language
> Models at Scale.* arXiv:2608.03729, 2026. https://arxiv.org/abs/2608.03729

```bibtex
@misc{hu2026constructing,
  title         = {Constructing Disambiguated Knowledge Bases from Large Language Models at Scale},
  author        = {Hu, Yujia and Nguyen, Tuan-Phong and Razniewski, Simon},
  year          = {2026},
  eprint        = {2608.03729},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2608.03729}
}
```