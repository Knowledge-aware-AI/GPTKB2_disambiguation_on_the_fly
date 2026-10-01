import json

import fire

from prompter_parser import PromptSchema
from construction import Constructor


def load_json(path: str):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(
        db_path: str,
        log_path:str,
        inode_embeddings_mmap_path:str,
        inode_embeddings_mmap_metadata_path:str,
        inode_Index_path:str,
        predicate_embeddings_mmap_path: str,
        predicate_embeddings_mmap_metadata_path:str,
        predicate_Index_path:str,
        concept_embeddings_mmap_path: str,
        concept_embeddings_mmap_metadata_path:str,
        concept_Index_path:str,
        requests_in_batch: bool = True,
        single_chunk_size: int = 20,
        single_fetch_size: int = 500,
        single_request_workers: int = 16,
        single_backends_config: str = None,
        batch_backends_config: str = None,
        expected_entities: int = None,
        sqlite_cache_size_mb: int = None,
):
    single_backends = load_json(single_backends_config)
    batch_backends = load_json(batch_backends_config)

    prompter_parser_module = PromptSchema(
        elicitation_gpt_model = "gpt-5.1",
        disambiguation_gpt_model="gpt-5-mini",
        description_gen_gpt_model="gpt-5.1",
    )

    constructor = Constructor(
        db_path=db_path,
        log_path=log_path,
        inode_embeddings_mmap_path=inode_embeddings_mmap_path,
        inode_embeddings_mmap_metadata_path=inode_embeddings_mmap_metadata_path,
        inode_Index_path=inode_Index_path,
        predicate_embeddings_mmap_path=predicate_embeddings_mmap_path,
        predicate_embeddings_mmap_metadata_path=predicate_embeddings_mmap_metadata_path,
        predicate_Index_path=predicate_Index_path,
        concept_embeddings_mmap_path=concept_embeddings_mmap_path,
        concept_embeddings_mmap_metadata_path=concept_embeddings_mmap_metadata_path,
        concept_Index_path=concept_Index_path,
        prompter_parser_module=prompter_parser_module,
        single_request_workers=single_request_workers,
        single_backends=single_backends,
        batch_backends=batch_backends,
        expected_entities=expected_entities,
        sqlite_cache_size_mb=sqlite_cache_size_mb,
    )

    if requests_in_batch:
        constructor.loop(
            max_nes_explored=80000,
            batchwise=True,
            max_batch_size=400,
            max_queue_size=600,
            max_ned_triples_iter=80000,
            max_pd_triples_iter=150000,
            max_cd_triples_iter=50000,
            max_nedg_triples_iter=10000,
            max_pdg_triples_iter=10000,
            max_cdg_triples_iter=10000,
        )
    else:
        constructor.loop(
            max_nes_explored=80000,
            batchwise=False,
            single_chunk_size=single_chunk_size,
            single_fetch_size=single_fetch_size,
        )


if __name__ == "__main__":
    fire.Fire(main)