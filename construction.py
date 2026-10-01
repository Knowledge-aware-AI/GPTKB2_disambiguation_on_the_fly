from db.db_models import SQLModel, Batch, InstanceNode, Triple, StatusType, JobType, ObjectStatusType, ObjectType, Predicate, PredicateStatusType, Concept, ConceptStatusType, InstanceTriple, TripleInstanceNodeLink, TriplePredicateLink, InstanceTripleConceptLink, TripleInstanceNodeDescriptionLink    
from prompter_parser import AbstractPrompterParser
from llm_backends import build_single_backends, build_batch_backends

import sys
import time
import csv
import os
import re
import queue
from loguru import logger
from pathlib import Path
import pickle
import json
import threading
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from types import SimpleNamespace
from tqdm import tqdm

from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import case, event, delete, or_, and_, inspect, text
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select, create_engine, update, and_, tuple_, text, func, cast, Integer
import openai
from openai import OpenAI
from openai.types import Batch as OpenAIBatch
from datetime import datetime, timedelta
import faiss
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer


SINGLE_CONTROL_GROUP = {
    JobType.NAMED_ENTITY_DISAMBIGUATION_SOURCE_TRIPLE.value: "NED",
    JobType.NAMED_ENTITY_DESCRIPTION_GEN.value: "NED",
    JobType.NAMED_ENTITY_DISAMBIGUATION_DESCRIPTION.value: "NED",
    JobType.PREDICATE_DISAMBIGUATION.value: "PD",
    JobType.PREDICATE_DESCRIPTION_GEN.value: "PD",
    JobType.CONCEPT_DISAMBIGUATION.value: "CD",
    JobType.CONCEPT_DESCRIPTION_GEN.value: "CD",
}

JOB_TYPE_ROLE = {
    JobType.ELICITATION.value: "elicitation",
    JobType.NAMED_ENTITY_RECOGNITION.value: "disambiguation",
    JobType.NAMED_ENTITY_DISAMBIGUATION_SOURCE_TRIPLE.value: "disambiguation",
    JobType.NAMED_ENTITY_DISAMBIGUATION_DESCRIPTION.value: "disambiguation",
    JobType.PREDICATE_DISAMBIGUATION.value: "disambiguation",
    JobType.CONCEPT_DISAMBIGUATION.value: "disambiguation",
    JobType.NAMED_ENTITY_DESCRIPTION_GEN.value: "description",
    JobType.PREDICATE_DESCRIPTION_GEN.value: "description",
    JobType.CONCEPT_DESCRIPTION_GEN.value: "description",
}

OPENAI_BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

PREDICATES_PER_ENTITY = 207_633 / 2_297_995  
CONCEPTS_PER_ENTITY = 66_523 / 2_297_995       
TRIPLES_PER_ENTITY = 38_450_135 / 2_297_995    
CAPACITY_HEADROOM = 1.5
MIN_MMAP_CAPACITY = {"inode": 100_000, "predicate": 10_000, "concept": 10_000}
DEFAULT_MMAP_CAPACITY = {"inode": 20_000_000, "predicate": 2_000_000, "concept": 1_000_000}
DB_BYTES_PER_TRIPLE = 1024
SQLITE_MMAP_SIZE = 25_769_803_776
SQLITE_CONNECTIONS_ESTIMATE = 4


def initial_mmap_capacities(expected_entities: int = None) -> dict:
    if not expected_entities:
        return dict(DEFAULT_MMAP_CAPACITY)
    expected = {
        "inode": expected_entities,
        "predicate": expected_entities * PREDICATES_PER_ENTITY,
        "concept": expected_entities * CONCEPTS_PER_ENTITY,
    }
    return {kind: max(MIN_MMAP_CAPACITY[kind], math.ceil(n * CAPACITY_HEADROOM)) for kind, n in expected.items()}


def sqlite_cache_size_kib(expected_entities: int = None, cache_size_mb: int = None) -> int:
    if cache_size_mb:
        return cache_size_mb * 1024
    try:
        physical_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        physical_bytes = 16 * 1024 ** 3
    cache_bytes = physical_bytes * 0.25 / SQLITE_CONNECTIONS_ESTIMATE
    if expected_entities:
        cache_bytes = min(cache_bytes, expected_entities * TRIPLES_PER_ENTITY * DB_BYTES_PER_TRIPLE * CAPACITY_HEADROOM)
    return max(64 * 1024, int(cache_bytes // 1024))


class Constructor():
    
    def __init__(
            self,
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
            prompter_parser_module: AbstractPrompterParser = None,
            seed_subject_label: str = "Vannevar Bush",
            seed_subject_description: str = "American electrical engineer and science administrator (1890~1974)",
            seed_predicate_label: str = "instanceOf",
            seed_predicate_description: str = "relation of type constraints",
            seed_concept_label: str = "human",
            seed_concept_description: str = "A human is a highly intelligent, social, and self-aware primate species capable of complex language, abstract reasoning, and cultural development.",
            job_description: str = "Knowledge Base Construction, disambiguation implemented",
            single_request_workers: int = 16,
            openai_timeout: float = 600,
            openai_max_retries: int = 5,
            single_backends: dict = None,
            batch_backends: dict = None,
            expected_entities: int = None,
            sqlite_cache_size_mb: int = None,
    ):
        logger.add(log_path, rotation="10 MB")
        logger.info("Initialize the GPT-KBC runner")
        if prompter_parser_module is None:
            raise ValueError("Prompter Parser module is not provided.")

        self.db_path = Path(db_path).resolve()
        self.sqlite_url = f"sqlite:///{self.db_path}"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.tmp_folder = self.db_path.parent / f"tmp_{self.db_path.stem}"
        self.tmp_folder.mkdir(exist_ok=True)

        logger.info(f"Create DB engine with SQLite url: `{self.sqlite_url}`")
        self.db_engine = create_engine(self.sqlite_url, echo=False)
        cache_size_kib = sqlite_cache_size_kib(expected_entities, sqlite_cache_size_mb)

        @event.listens_for(self.db_engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute(f"PRAGMA mmap_size={SQLITE_MMAP_SIZE}")
            cursor.execute(f"PRAGMA cache_size=-{cache_size_kib}")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()

        logger.info("Create DB tables if none exists")
        SQLModel.metadata.create_all(self.db_engine)
        with self.db_engine.connect() as connection:
            effective_mmap_size = connection.exec_driver_sql("PRAGMA mmap_size").scalar()
        logger.info(f"SQLite page cache {cache_size_kib / 1024 ** 2:.1f} GiB per connection, "
                    f"mmap_size {effective_mmap_size / 1024 ** 3:.1f} GiB (requested {SQLITE_MMAP_SIZE / 1024 ** 3:.1f} GiB)")

        self.inode_embeddings_mmap_path = inode_embeddings_mmap_path
        self.inode_embeddings_mmap_metadata_path = inode_embeddings_mmap_metadata_path
        self.inode_Index_path = inode_Index_path

        self.predicate_embeddings_mmap_path = predicate_embeddings_mmap_path
        self.predicate_embeddings_mmap_metadata_path = predicate_embeddings_mmap_metadata_path
        self.predicate_Index_path = predicate_Index_path

        self.concept_embeddings_mmap_path = concept_embeddings_mmap_path
        self.concept_embeddings_mmap_metadata_path = concept_embeddings_mmap_metadata_path
        self.concept_Index_path = concept_Index_path
        
        try:
            self.client = OpenAI(timeout=openai_timeout, max_retries=openai_max_retries)
        except openai.OpenAIError as e:
            logger.warning(f"No OpenAI client ({e}). Batch mode and single-mode roles without a backend config are unavailable.")
            self.client = None
        self.single_backends = build_single_backends(self.client, single_backends, timeout=openai_timeout, max_retries=openai_max_retries)
        for role, backend in self.single_backends.items():
            logger.info(f"Single-mode backend for {role}: {backend.describe()}")
        self.batch_backends = build_batch_backends(self.client, batch_backends, timeout=openai_timeout, max_retries=openai_max_retries)
        for role, backend in self.batch_backends.items():
            logger.info(f"Batch-mode backend for {role}: {backend.describe()}")
        self.single_request_workers = single_request_workers
        self.job_description = job_description
        self.prompter_parser_module = prompter_parser_module
        self.openai_batch_status_in_process = ["created", "validating", "in_progress", "finalizing", "cancelling", "parsing"]
        self.stransformer_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=1024)
        self.embedding_dim = self.stransformer_model.get_sentence_embedding_dimension()
        self.embedding_dtype = 'float32'

        if self._is_db_empty():
            logger.info(f"Seed the database with the seed subject: '{seed_subject_label}'")

            capacities = initial_mmap_capacities(expected_entities)
            self.mmap_inode_max = capacities["inode"]
            self.mmap_predicate_max = capacities["predicate"]
            self.mmap_concept_max = capacities["concept"]
            size_info = f"for {expected_entities:,} expected entities" if expected_entities else "(default, no expected_entities given)"
            logger.info(f"Embedding mmap capacity {size_info}: {self.mmap_inode_max:,} inodes, "
                        f"{self.mmap_predicate_max:,} predicates, {self.mmap_concept_max:,} concepts.")

            self.inode_embeddings_mmap = np.memmap(self.inode_embeddings_mmap_path, dtype=self.embedding_dtype, mode='w+', shape=(self.mmap_inode_max, self.embedding_dim))
            base_inode_index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            base_inode_index.hnsw.efConstruction = 200
            base_inode_index.hnsw.efSearch = 64
            self.inode_index = faiss.IndexIDMap2(base_inode_index)
            self.inode_embeddings_mmap_metadata = {
                'vector_count': 0,
                'total_capacity': self.mmap_inode_max,
                'embedding_dim': self.embedding_dim,
                'dtype': self.embedding_dtype
            }
            
            self.predicate_embeddings_mmap = np.memmap(self.predicate_embeddings_mmap_path, dtype=self.embedding_dtype, mode='w+', shape=(self.mmap_predicate_max, self.embedding_dim))
            base_predicate_index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            base_predicate_index.hnsw.efConstruction = 200
            base_predicate_index.hnsw.efSearch = 64
            self.predicate_index = faiss.IndexIDMap2(base_predicate_index)
            self.predicate_embeddings_mmap_metadata = {
                'vector_count': 0,
                'total_capacity': self.mmap_predicate_max,
                'embedding_dim': self.embedding_dim,
                'dtype': self.embedding_dtype
            }

            self.concept_embeddings_mmap = np.memmap(self.concept_embeddings_mmap_path, dtype=self.embedding_dtype, mode='w+', shape=(self.mmap_concept_max, self.embedding_dim))
            base_concept_index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
            base_concept_index.hnsw.efConstruction = 200
            base_concept_index.hnsw.efSearch = 64
            self.concept_index = faiss.IndexIDMap2(base_concept_index)
            self.concept_embeddings_mmap_metadata = {
                'vector_count': 0,
                'total_capacity': self.mmap_concept_max,
                'embedding_dim': self.embedding_dim,
                'dtype': self.embedding_dtype
            }
            
            self.id_counter_triple = 0
            self._seed_db(seed_subject_label, seed_subject_description, seed_predicate_label, seed_predicate_description, seed_concept_label, seed_concept_description)

        else:
            self.load_inode_data()
            self.mmap_inode_max = self.inode_embeddings_mmap_metadata.get('total_capacity', DEFAULT_MMAP_CAPACITY["inode"])
            self.inode_embeddings_mmap = np.memmap(self.inode_embeddings_mmap_path, dtype=self.embedding_dtype, mode='r+', shape=(self.mmap_inode_max, self.embedding_dim))
            self.load_predicate_data()
            self.mmap_predicate_max = self.predicate_embeddings_mmap_metadata.get('total_capacity', DEFAULT_MMAP_CAPACITY["predicate"])
            self.predicate_embeddings_mmap = np.memmap(self.predicate_embeddings_mmap_path, dtype=self.embedding_dtype, mode='r+', shape=(self.mmap_predicate_max, self.embedding_dim))
            self.load_concept_data()
            self.mmap_concept_max = self.concept_embeddings_mmap_metadata.get('total_capacity', DEFAULT_MMAP_CAPACITY["concept"])
            self.concept_embeddings_mmap = np.memmap(self.concept_embeddings_mmap_path, dtype=self.embedding_dtype, mode='r+', shape=(self.mmap_concept_max, self.embedding_dim))
            if expected_entities:
                logger.info("Existing KB: expected_entities only sizes the SQLite page cache, the embedding mmaps keep their capacity.")

            with Session(self.db_engine) as session:
                max_id_query_triple = select(func.max(cast(func.substr(Triple.triple_id, 2), Integer)))
                max_id = session.execute(max_id_query_triple).scalar()
                self.id_counter_triple = (max_id or 0) + 1
            
    def update_inode_data(self):
        self.inode_embeddings_mmap_metadata['total_capacity'] = self.mmap_inode_max
        faiss.write_index(self.inode_index, self.inode_Index_path)
        with open(self.inode_embeddings_mmap_metadata_path, 'w') as finode:
            json.dump(self.inode_embeddings_mmap_metadata, finode, indent=4)
    
    def update_predicate_data(self):
        self.predicate_embeddings_mmap_metadata['total_capacity'] = self.mmap_predicate_max
        faiss.write_index(self.predicate_index, self.predicate_Index_path)
        with open(self.predicate_embeddings_mmap_metadata_path, 'w') as fpredicate:
            json.dump(self.predicate_embeddings_mmap_metadata, fpredicate, indent=4)
    
    def update_concept_data(self):
        self.concept_embeddings_mmap_metadata['total_capacity'] = self.mmap_concept_max
        faiss.write_index(self.concept_index, self.concept_Index_path)
        with open(self.concept_embeddings_mmap_metadata_path, 'w') as fconcept:
            json.dump(self.concept_embeddings_mmap_metadata, fconcept, indent=4)
    
    def load_inode_data(self):
        self.inode_index = faiss.read_index(self.inode_Index_path)
        with open(self.inode_embeddings_mmap_metadata_path, 'r') as finodemeta:
            self.inode_embeddings_mmap_metadata = json.load(finodemeta)
    
    def load_predicate_data(self):
        self.predicate_index = faiss.read_index(self.predicate_Index_path)
        with open(self.predicate_embeddings_mmap_metadata_path, 'r') as fpredicatemeta:
            self.predicate_embeddings_mmap_metadata = json.load(fpredicatemeta)

    def load_concept_data(self):
        self.concept_index = faiss.read_index(self.concept_Index_path)
        with open(self.concept_embeddings_mmap_metadata_path, 'r') as fconceptmeta:
            self.concept_embeddings_mmap_metadata = json.load(fconceptmeta)

    def _ensure_mmap_capacity(self, kind: str, required_rows: int):
        metadata = getattr(self, f"{kind}_embeddings_mmap_metadata")
        capacity = getattr(self, f"mmap_{kind}_max")
        if required_rows <= capacity:
            return
        new_capacity = max(required_rows, capacity * 2)
        path = getattr(self, f"{kind}_embeddings_mmap_path")
        getattr(self, f"{kind}_embeddings_mmap").flush()
        os.truncate(path, new_capacity * self.embedding_dim * np.dtype(self.embedding_dtype).itemsize)
        setattr(self, f"{kind}_embeddings_mmap",
                np.memmap(path, dtype=self.embedding_dtype, mode='r+', shape=(new_capacity, self.embedding_dim)))
        setattr(self, f"mmap_{kind}_max", new_capacity)
        metadata['total_capacity'] = new_capacity
        logger.info(f"Grew the {kind} embedding mmap from {capacity:,} to {new_capacity:,} rows.")
        
    def _is_db_empty(self):

        with Session(self.db_engine) as session:
            inode = session.exec(
                select(InstanceNode)
            ).first()
            return inode is None
    
    def _seed_db(self, seed_subject_label, seed_subject_description, seed_predicate_label, seed_predicate_description, seed_concept_label, seed_concept_description):
        try:
            seed_label_embedding = self.stransformer_model.encode(seed_subject_label, convert_to_tensor=False, normalize_embeddings=True).astype(self.embedding_dtype).reshape(1, -1)
            seed_predicate_embedding = self.stransformer_model.encode(seed_predicate_label, convert_to_tensor=False, normalize_embeddings=True).astype(self.embedding_dtype).reshape(1, -1)
            seed_concept_embedding = self.stransformer_model.encode(seed_concept_label, convert_to_tensor=False, normalize_embeddings=True).astype(self.embedding_dtype).reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to generate embedding for seed entity/predicate/concept: {e}")
            raise

        mmap_inode_index = self.inode_embeddings_mmap_metadata.get('vector_count', 0)
        if mmap_inode_index != 0:
            logger.error("Non-empty entity mmap.")
            raise Exception("KB not empty.")
        new_inode_id_str = f"E{mmap_inode_index}"

        mmap_predicate_index = self.predicate_embeddings_mmap_metadata.get('vector_count', 0)
        if mmap_predicate_index != 0:
            logger.error("Non-empty predicate mmap.")
            raise Exception("KB not empty.")
        new_predicate_id_str = f"P{mmap_predicate_index}"

        mmap_concept_index = self.concept_embeddings_mmap_metadata.get('vector_count', 0)
        if mmap_concept_index != 0:
            logger.error("Non-empty concept mmap.")
            raise Exception("KB not empty.")
        new_concept_id_str = f"C{mmap_concept_index}"

        with Session(self.db_engine) as session:
            try:
                inode = InstanceNode(
                    id=new_inode_id_str,
                    label=seed_subject_label,
                    description=seed_subject_description,
                    embedding_index=mmap_inode_index,
                    first_appeared="Where the exploration starts",
                    status=StatusType.UNEXPLORED,
                    )
                session.add(inode)
                self.inode_embeddings_mmap[mmap_inode_index] = seed_label_embedding
                self.inode_embeddings_mmap.flush()
                self.inode_index.add_with_ids(seed_label_embedding, np.array([mmap_inode_index]))
                session.commit()

                self.inode_embeddings_mmap_metadata['vector_count'] += 1
                self.update_inode_data()
                logger.info(f"Seed entity added to database successfully with ID: {new_inode_id_str}.")

            except Exception as e:
                logger.error(f"Failed to add seed entity: {e}")
                self.load_inode_data()
                raise

            try:
                predicate = Predicate(
                    id=new_predicate_id_str,
                    label=seed_predicate_label,
                    description=seed_predicate_description,
                    embedding_index=mmap_predicate_index,
                    first_appeared="Where the exploration starts",
                    )
                session.add(predicate)
                self.predicate_embeddings_mmap[mmap_predicate_index] = seed_predicate_embedding
                self.predicate_embeddings_mmap.flush()
                self.predicate_index.add_with_ids(seed_predicate_embedding, np.array([mmap_predicate_index]))
                session.commit()

                self.predicate_embeddings_mmap_metadata['vector_count'] += 1
                self.update_predicate_data()
                logger.info(f"Seed predicate added to database successfully with ID: {new_predicate_id_str}.")

            except Exception as e:
                logger.error(f"Failed to add seed predicate: {e}")
                self.load_predicate_data()
                raise

            try:
                concept = Concept(
                    id=new_concept_id_str,
                    label=seed_concept_label,
                    description=seed_concept_description,
                    embedding_index=mmap_concept_index,
                    first_appeared="Where the exploration starts",
                    )
                session.add(concept)
                self.concept_embeddings_mmap[mmap_concept_index] = seed_concept_embedding
                self.concept_embeddings_mmap.flush()
                self.concept_index.add_with_ids(seed_concept_embedding, np.array([mmap_concept_index]))
                session.commit()

                self.concept_embeddings_mmap_metadata['vector_count'] += 1
                self.update_concept_data()
                logger.info(f"Seed concept added to database successfully with ID: {new_concept_id_str}.")

            except Exception as e:
                logger.error(f"Failed to add seed concept: {e}")
                self.load_concept_data()
                raise

    def _seed_entity(self, seed_subject_label, seed_subject_description):
        try:
            seed_label_embedding = self.stransformer_model.encode(seed_subject_label, convert_to_tensor=False, normalize_embeddings=True).astype(self.embedding_dtype).reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to generate embedding for seed entity: {e}")
            raise

        mmap_inode_index = self.inode_embeddings_mmap_metadata.get('vector_count', 0)
        new_inode_id_str = f"E{mmap_inode_index}"

        with Session(self.db_engine) as session:
            try:
                inode = InstanceNode(
                    id=new_inode_id_str,
                    label=seed_subject_label,
                    description=seed_subject_description,
                    embedding_index=mmap_inode_index,
                    first_appeared="Where the exploration starts",
                    status=StatusType.UNEXPLORED,
                    )
                session.add(inode)
                self.inode_embeddings_mmap[mmap_inode_index] = seed_label_embedding
                self.inode_embeddings_mmap.flush()
                self.inode_index.add_with_ids(seed_label_embedding, np.array([mmap_inode_index]))
                session.commit()

                self.inode_embeddings_mmap_metadata['vector_count'] += 1
                self.update_inode_data()
                logger.info(f"Seed entity added to database successfully with ID: {new_inode_id_str}.")

            except Exception as e:
                logger.error(f"Failed to add seed entity: {e}")
                self.load_inode_data()
                raise

    def _seed_predicate(self, seed_predicate_label, seed_predicate_description):
        try:
            seed_predicate_embedding = self.stransformer_model.encode(seed_predicate_label, convert_to_tensor=False, normalize_embeddings=True).astype(self.embedding_dtype).reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to generate embedding for seed predicate: {e}")
            raise

        mmap_predicate_index = self.predicate_embeddings_mmap_metadata.get('vector_count', 0)
        new_predicate_id_str = f"P{mmap_predicate_index}"

        with Session(self.db_engine) as session:
            try:
                predicate = Predicate(
                    id=new_predicate_id_str,
                    label=seed_predicate_label,
                    description=seed_predicate_description,
                    embedding_index=mmap_predicate_index,
                    first_appeared="Where the exploration starts",
                    )
                session.add(predicate)
                self.predicate_embeddings_mmap[mmap_predicate_index] = seed_predicate_embedding
                self.predicate_embeddings_mmap.flush()
                self.predicate_index.add_with_ids(seed_predicate_embedding, np.array([mmap_predicate_index]))
                session.commit()

                self.predicate_embeddings_mmap_metadata['vector_count'] += 1
                self.update_predicate_data()
                logger.info(f"Seed predicate added to database successfully with ID: {new_predicate_id_str}.")

            except Exception as e:
                logger.error(f"Failed to add seed predicate: {e}")
                self.load_predicate_data()
                raise

    def _seed_concept(self, seed_concept_label, seed_concept_description):
        try:
            seed_concept_embedding = self.stransformer_model.encode(seed_concept_label, convert_to_tensor=False, normalize_embeddings=True).astype(self.embedding_dtype).reshape(1, -1)
        except Exception as e:
            logger.error(f"Failed to generate embedding for seed concept: {e}")
            raise

        mmap_concept_index = self.concept_embeddings_mmap_metadata.get('vector_count', 0)
        new_concept_id_str = f"C{mmap_concept_index}"

        with Session(self.db_engine) as session:

            try:
                concept = Concept(
                    id=new_concept_id_str,
                    label=seed_concept_label,
                    description=seed_concept_description,
                    embedding_index=mmap_concept_index,
                    first_appeared="Where the exploration starts",
                    )
                session.add(concept)
                self.concept_embeddings_mmap[mmap_concept_index] = seed_concept_embedding
                self.concept_embeddings_mmap.flush()
                self.concept_index.add_with_ids(seed_concept_embedding, np.array([mmap_concept_index]))
                session.commit()

                self.concept_embeddings_mmap_metadata['vector_count'] += 1
                self.update_concept_data()
                logger.info(f"Seed concept added to database successfully with ID: {new_concept_id_str}.")

            except Exception as e:
                logger.error(f"Failed to add seed concept: {e}")
                self.load_concept_data()
                raise


    def _check_batch_queue(self):
        outstanding_batch_ids = []
        completed = defaultdict(list) 
        allow = {"NED": True, "PD": True, "CD": True}

        try:
            with Session(self.db_engine) as session:
                batches = session.exec(
                    select(Batch)
                    .where(Batch.status.in_(self.openai_batch_status_in_process))
                ).all()
                for batch in batches:
                    backend = self.batch_backends[JOB_TYPE_ROLE[batch.job_type]]
                    if backend.client is None:
                        continue
                    try:
                        openai_batch = backend.retrieve(batch.id)
                    except Exception as e:
                        logger.error(f"Failed to retrieve batch {batch.id} due to: {e}")
                        allow = dict.fromkeys(allow, False)
                        outstanding_batch_ids.append(batch.id)
                        continue

                    status = str(openai_batch.status or "").lower()
                    openai_batch.status = status
                    batch.input_file_id = openai_batch.input_file_id
                    batch.output_file_id = openai_batch.output_file_id

                    is_active = status in self.openai_batch_status_in_process
                    is_terminal = status in OPENAI_BATCH_TERMINAL_STATUSES
                    if not (is_active or is_terminal):
                        logger.warning(f"Batch {batch.id} has unknown status `{status}`, treating it as still running.")
                        is_active = True
                    else:
                        batch.status = status

                    if is_active or is_terminal:
                        group = SINGLE_CONTROL_GROUP.get(batch.job_type)
                        if group:
                            allow[group] = False

                    if is_active:
                        outstanding_batch_ids.append(openai_batch.id)
                    elif is_terminal:
                        batch.status = "parsing"
                        completed[batch.job_type].append(openai_batch)

                    session.add(batch)
                session.commit()

        except Exception as e:
            allow = dict.fromkeys(allow, False)
            logger.error(f"Failed to retrieve batches due to: {e}")

        return outstanding_batch_ids, completed, allow["NED"], allow["PD"], allow["CD"]

    
    def _send_single_request(self, req: dict, role: str, poll_interval: int, max_tries: int = 1) -> dict:
        for num_tries in range(max_tries):
            try:
                body = self.single_backends[role].send(req["body"])
                return {"custom_id": req["custom_id"], "response": {"body": body}}
            except openai.RateLimitError as e:
                if num_tries == max_tries - 1:
                    raise
                logger.error(f"Rate limit error: {e}")
                logger.info(f"Waiting for {poll_interval} seconds before retrying.")
                time.sleep(poll_interval)

    def _send_single_requests(self, reqs: list[dict], role: str, poll_interval: int, max_tries: int = 1):
        executor = ThreadPoolExecutor(max_workers=self.single_request_workers)
        try:
            futures = {executor.submit(self._send_single_request, req, role, poll_interval, max_tries): req for req in reqs}
            for future in as_completed(futures):
                req = futures[future]
                try:
                    yield req, future.result(), None
                except Exception as e:
                    yield req, None, e
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    def _create_elicitation_batch(self, nes_to_explore: list[InstanceNode], poll_interval:int, batchwise=True, max_tries: int = 1):
        if not batchwise:
            num_explored = 0
            reqs = [self.prompter_parser_module.get_elicitation_prompt(ne.id, ne.label, ne.description) for ne in nes_to_explore]
            for req, line, error in self._send_single_requests(reqs, "elicitation", poll_interval, max_tries):
                inode_id = req["custom_id"]
                with Session(self.db_engine) as session:
                    try:
                        if error:
                            raise error
                        triples = self.prompter_parser_module.parse_elicitation_response(line)
                        if len(triples) > 0:
                            self._commit_new_triples(raw_triples=triples, batch_id=None, session=session)
                        new_status = StatusType.EXPLORED
                        num_explored += 1
                    except Exception as e:
                        logger.error(f"Failed elicitation for entity {inode_id}: {e}")
                        session.rollback()
                        new_status = StatusType.UNEXPLORED
                    session.execute(
                        update(InstanceNode)
                        .where(InstanceNode.id == inode_id)
                        .values(status=new_status, explored_batch_id=None)
                    )
                    session.commit()
            return num_explored

        batch_requests = []
        for ne in nes_to_explore:
            req = self.prompter_parser_module.get_elicitation_prompt(ne.id, ne.label, ne.description)
            batch_requests.append(req)

        batch_input_file = self.batch_backends["elicitation"].upload(batch_requests, self.tmp_folder / "elicitation_batch_requests.jsonl")

        batch_input_file_id = batch_input_file.id
        openai_batch = None
        for num_tries in range(max_tries):
            try:
                openai_batch = self.batch_backends["elicitation"].create_batch(batch_input_file_id, metadata={
                        "description": self.job_description,
                        "type": "elicitation",
                    })
                logger.info(f"Elicitation batch file created successfully. Batch ID: `{openai_batch.id}`.")
                with Session(self.db_engine) as session:
                    batch = Batch(
                        id=openai_batch.id,
                        input_file_id=batch_input_file_id,
                        status="created",
                        job_type=JobType.ELICITATION.value,
                    )
                    session.add(batch)

                    inode_ids = [ne.id for ne in nes_to_explore]
                    statement = (
                        update(InstanceNode)
                        .where(InstanceNode.id.in_(inode_ids))
                        .values(status=StatusType.EXPLORING, explored_batch_id=openai_batch.id)
                    )
                    session.execute(statement)
                    session.commit()
                    return 0
            except openai.RateLimitError as e:
                logger.error(f"Rate limit error: {e}")
                logger.info(f"Waiting for {poll_interval} seconds before retrying.")
                time.sleep(poll_interval)
                continue

        if openai_batch is None:
            raise Exception(f"Failed to create elicitation batch file after {max_tries} attempts.")


    def _create_ner_batch(self, triples_ner: list[Triple], poll_interval:int, batchwise=True, max_tries: int = 1):
        if not batchwise:
            num_ne, num_literal = 0, 0
            reqs = [self.prompter_parser_module.get_ner_prompt(t.triple_id, t.subject_label, t.predicate_label, t.object_label) for t in triples_ner]
            for req, line, error in self._send_single_requests(reqs, "disambiguation", poll_interval, max_tries):
                triple_id = req["custom_id"]
                try:
                    if error:
                        raise error
                    ner = self.prompter_parser_module.parse_ner_response(json.dumps(line))
                except Exception as e:
                    logger.error(f"Failed NER for triple {triple_id}: {e}")
                    continue
                if ner is None:
                    logger.error(f"Failed NER for triple {triple_id}: unparsable response")
                    continue
                if ner[1]:
                    self._update_triple_after_ner([triple_id], [])
                    num_ne += 1
                else:
                    self._update_triple_after_ner([], [triple_id])
                    num_literal += 1
            logger.info(f"Found {num_ne} nes and {num_literal} literals in single NER requests.")
            return

        batch_requests = []
        for triple in triples_ner:
            req = self.prompter_parser_module.get_ner_prompt(triple.triple_id, triple.subject_label, triple.predicate_label, triple.object_label)
            batch_requests.append(req)

        batch_input_file = self.batch_backends["disambiguation"].upload(batch_requests, self.tmp_folder / "ner_batch_requests.jsonl")

        batch_input_file_id = batch_input_file.id
        openai_batch = None
        for num_tries in range(max_tries):
            try:
                openai_batch = self.batch_backends["disambiguation"].create_batch(batch_input_file_id, metadata={
                        "description": "Named entity recognition",
                        "type": "NER",
                    })
                logger.info(f"NER batch file created successfully. Batch ID: `{openai_batch.id}`.")
                with Session(self.db_engine) as session:
                    batch = Batch(
                        id=openai_batch.id,
                        input_file_id=batch_input_file_id,
                        status="created",
                        job_type=JobType.NAMED_ENTITY_RECOGNITION.value,
                    )
                    session.add(batch)
                    triple_ids = [triple.triple_id for triple in triples_ner]
                    statement = (
                        update(Triple)
                        .where(Triple.triple_id.in_(triple_ids))
                        .values(object_status=ObjectStatusType.ONNER, ner_batch_id=openai_batch.id)
                    )
                    session.execute(statement)
                    session.commit()
                    return
            except openai.RateLimitError as e:
                logger.error(f"Rate limit error: {e}")
                logger.info(f"Waiting for {poll_interval} seconds before retrying.")
                time.sleep(poll_interval)
                continue

        if openai_batch is None:
            raise Exception(f"Failed to create NER batch file after {max_tries} attempts.")

    def _create_ned1_batches(self, triples_ned1: list[Triple], poll_interval: int, batchwise=True, max_tries: int = 2):

        num_neighbor = min(5, self.inode_index.ntotal)
        object_labels = [triple.object_label for triple in triples_ned1]
        all_query_embeddings = self.stransformer_model.encode(
            object_labels,
            batch_size=64,
            convert_to_tensor=False,
            show_progress_bar=True,
            normalize_embeddings=True,
            ).astype("float32")
        _distances, all_indices = self.inode_index.search(all_query_embeddings, k=num_neighbor)
        search_results_indices = {triple.triple_id: all_indices[i] for i, triple in enumerate(triples_ned1)}

        batch_requests = []
        links_ready = False
        for num_tries in range(max_tries):
            try:
                with Session(self.db_engine) as session:
                    batch_requests = []
                    all_required_inode_indices = {idx for indices in all_indices for idx in indices if idx != -1}
                    required_candidate_inodes = {f"E{idx}" for idx in all_required_inode_indices}
                    labels_map = {}
                    desc_map = {}
                    ids_in_db = set()
                    if required_candidate_inodes:
                        results = session.exec(
                            select(InstanceNode.id, InstanceNode.label, InstanceNode.description)
                            .where(InstanceNode.id.in_(required_candidate_inodes),
                                   InstanceNode.label.is_not(None),
                                   InstanceNode.description.is_not(None))
                        ).all()
                        for cid, label, desc in results:
                            labels_map[str(cid)] = str(label)
                            desc_map[str(cid)] = str(desc)
                            ids_in_db.add(cid)
                    
                    triple_to_candidate_ids = {}
                    for triple in triples_ned1:
                        indices = search_results_indices[triple.triple_id]
                        valid_indices = [idx for idx in indices if idx != -1]
                        ids = [f"E{idx}" for idx in valid_indices]
                        ids = [cid for cid in ids if cid in ids_in_db]
                        triple_to_candidate_ids[triple.triple_id] = ids
                        labels = [labels_map.get(str(cid), " ") for cid in ids]
                        descs = [desc_map.get(str(cid), " ") for cid in ids]
                        req = self.prompter_parser_module.get_ned1_prompt(triple.triple_id, triple.subject_label, triple.predicate_label, triple.object_label, labels, descs)
                        batch_requests.append(req)
                    
                    if not batch_requests:
                        logger.info("No valid NED1 requests, try to prepare again.")
                        continue

                    triple_ids = [triple.triple_id for triple in triples_ned1]
                    if batchwise:
                        batch_input_file = self.batch_backends["disambiguation"].upload(batch_requests, self.tmp_folder / "ned1_batch_requests.jsonl")
                        openai_batch = self.batch_backends["disambiguation"].create_batch(batch_input_file.id, metadata={"description": "Named entity disambiguation 1", "type": "NED1"})
                        logger.info(f"NED1 batch file created successfully. Batch ID: `{openai_batch.id}`.")
                        new_ned1_batch = Batch(
                            id=openai_batch.id, input_file_id=batch_input_file.id, status="created",
                            job_type=JobType.NAMED_ENTITY_DISAMBIGUATION_SOURCE_TRIPLE.value,
                            )
                        session.add(new_ned1_batch)

                        update_statement = (
                            update(Triple)
                            .where(Triple.triple_id.in_(triple_ids))
                            .values(object_status=ObjectStatusType.ONNED1, ned1_batch_id=openai_batch.id)
                            )
                        session.execute(update_statement)

                    session.execute(
                        delete(TripleInstanceNodeLink)
                        .where(TripleInstanceNodeLink.triple_id.in_(triple_ids))
                        )
                    links_to_insert = []
                    for triple in triples_ned1:
                        ids_to_link = triple_to_candidate_ids.get(triple.triple_id, [])
                        for position, inode_id in enumerate(ids_to_link):
                            links_to_insert.append({
                                "triple_id": triple.triple_id,
                                "instance_node_id": inode_id,
                                "position": position
                            })
                    if links_to_insert:
                        link_statement = insert(TripleInstanceNodeLink).values(links_to_insert)
                        session.execute(link_statement)

                    session.commit()
                    links_ready = True
                    break
            except Exception as e:
                if num_tries < max_tries -1:
                    logger.info(f"Retry updating database for NED1 batch creation in {poll_interval} seconds.")
                    time.sleep(poll_interval)
                else:
                    logger.error(f"Failed to update database for NED1 batch creation after {max_tries} attempts due to {e}")

        if batchwise or not links_ready:
            return

        for req, line, error in self._send_single_requests(batch_requests, "disambiguation", poll_interval, max_tries):
            triple_id = req["custom_id"]
            try:
                if error:
                    raise error
                existence = self.prompter_parser_module.parse_ned1_response(json.dumps(line))
            except Exception as e:
                logger.error(f"Failed NED1 for triple {triple_id}: {e}")
                continue
            if existence:
                if isinstance(existence, int) and existence <= 5:
                    self._update_triple_after_ned1({triple_id: existence - 1}, [], [])
                elif existence == "unknown":
                    self._update_triple_after_ned1({}, [], [triple_id])
            else:
                self._update_triple_after_ned1({}, [triple_id], [])
    
    def _create_ned2_batches(self, triples_ned2: list[Triple], poll_interval: int, batchwise=True, max_tries: int = 2):
        batch_requests = []
        links_ready = False
        for num_tries in range(max_tries):
            try:
                with Session(self.db_engine) as session:
                    batch_requests = []
                    triple_object_label_map = {t.triple_id: t.object_label for t in triples_ned2}
                    triple_object_desc_map = {t.triple_id: t.object_description for t in triples_ned2}

                    object_labels = {triple.object_label for triple in triples_ned2}
                    entity_label_string_matching_query = (
                        select(InstanceNode.id, InstanceNode.label, InstanceNode.description)
                        .where(
                            InstanceNode.label.in_(object_labels),
                            InstanceNode.description.is_not(None)
                        )
                    )
                    over_5_results = session.exec(entity_label_string_matching_query).all()
                    grouped_by_label = defaultdict(list)
                    for row in over_5_results:
                        grouped_by_label[row.label].append(row)
                    labels_freq_over_5 = [t for t in triples_ned2 if len(grouped_by_label.get(t.object_label, [])) > 5]
                    labels_freq_less_5 = [t for t in triples_ned2 if len(grouped_by_label.get(t.object_label, [])) <= 5]
                    triple_to_candidate_ids = {}
                    labels_map = {}
                    desc_map = {}

                    if len(labels_freq_over_5) > 0:
                        object_descriptions = [t.object_description for t in labels_freq_over_5]
                        object_descriptions_embedding = self.stransformer_model.encode(object_descriptions, normalize_embeddings=True, convert_to_tensor=True)
                        for idx, triple in enumerate(labels_freq_over_5):
                            target_desc_embedding = object_descriptions_embedding[idx].unsqueeze(0)
                            entities_with_same_string = grouped_by_label[triple.object_label]
                            descriptions = [e.description for e in entities_with_same_string]
                            list_desc_embeddings = self.stransformer_model.encode(descriptions, normalize_embeddings=True, convert_to_tensor=True)
                            desc_similarity = self.stransformer_model.similarity(target_desc_embedding, list_desc_embeddings)
                            scores, indices = desc_similarity.topk(k=5, dim=1)
                            indices = indices[0].tolist()
                            triple_to_candidate_ids[triple.triple_id] = [entities_with_same_string[i].id for i in indices]
                            for i in indices:
                                labels_map[entities_with_same_string[i].id] = entities_with_same_string[i].label
                                desc_map[entities_with_same_string[i].id] = entities_with_same_string[i].description
                        
                    if len(labels_freq_less_5) > 0:
                        object_labels_less = [t.object_label for t in labels_freq_less_5]
                        num_neighbor = min(5, self.inode_index.ntotal)
                        all_query_embeddings = self.stransformer_model.encode(
                            object_labels_less,
                            batch_size=64,
                            convert_to_tensor=False,
                            normalize_embeddings=True,
                            show_progress_bar=False
                            ).astype("float32")
                        _distances, all_indices = self.inode_index.search(all_query_embeddings, k=num_neighbor)
                        search_results_indices = {triple.triple_id: all_indices[i] for i, triple in enumerate(labels_freq_less_5)}
                        all_required_inode_indices = {idx for indices in all_indices for idx in indices if idx != -1}
                        required_candidate_inodes = {f"E{idx}" for idx in all_required_inode_indices}
                        ids_in_db = set()
                        if required_candidate_inodes:
                            less_5_results = session.exec(
                                select(InstanceNode.id, InstanceNode.label, InstanceNode.description)
                                .where(InstanceNode.id.in_(required_candidate_inodes),
                                    InstanceNode.label.is_not(None),
                                    InstanceNode.description.is_not(None))
                            ).all()
                            for cid, label, desc in less_5_results:
                                labels_map[cid] = label
                                desc_map[cid] = desc
                                ids_in_db.add(cid)
                        for triple in labels_freq_less_5:
                            triple_to_candidate_ids[triple.triple_id] = [e.id for e in grouped_by_label.get(triple.object_label, [])]
                            for e in grouped_by_label.get(triple.object_label, []):
                                labels_map[e.id] = e.label
                                desc_map[e.id] = e.description
                            indices = search_results_indices[triple.triple_id]
                            valid_indices = [idx for idx in indices if idx != -1]
                            ids = [f"E{idx}" for idx in valid_indices]
                            ids = [cid for cid in ids if cid in ids_in_db]
                            for id in ids:
                                if id not in triple_to_candidate_ids[triple.triple_id] and len(triple_to_candidate_ids[triple.triple_id]) < 5:
                                    triple_to_candidate_ids[triple.triple_id].append(id)
                    
                    for triple_id, entity_candidates in tqdm(triple_to_candidate_ids.items()):
                        labels = [labels_map[id] for id in entity_candidates]
                        descs = [desc_map[id] for id in entity_candidates]
                        req = self.prompter_parser_module.get_ned2_prompt(triple_id, triple_object_label_map[triple_id], triple_object_desc_map[triple_id], labels, descs)
                        batch_requests.append(req)
                    
                    if not batch_requests:
                        logger.info("No valid NED2 requests, try to prepare again.")
                        continue

                    if batchwise:
                        batch_input_file = self.batch_backends["disambiguation"].upload(batch_requests, self.tmp_folder / "ned2_batch_requests.jsonl")
                        openai_batch = self.batch_backends["disambiguation"].create_batch(batch_input_file.id, metadata={"description": "Named entity disambiguation 2", "type": "NED2"})
                        logger.info(f"NED2 batch file created successfully. Batch ID: `{openai_batch.id}`.")
                        new_ned2_batch = Batch(
                            id=openai_batch.id, input_file_id=batch_input_file.id, status="created",
                            job_type=JobType.NAMED_ENTITY_DISAMBIGUATION_DESCRIPTION.value,
                            )
                        session.add(new_ned2_batch)

                        update_statement = (
                            update(Triple)
                            .where(Triple.triple_id.in_(triple_to_candidate_ids.keys()))
                            .values(object_status=ObjectStatusType.ONNED2, ned2_batch_id=openai_batch.id)
                            )
                        session.execute(update_statement)

                    session.execute(
                        delete(TripleInstanceNodeDescriptionLink)
                        .where(TripleInstanceNodeDescriptionLink.triple_id.in_(triple_to_candidate_ids.keys()))
                        )
                    links_to_insert = []
                    for triple_id, ids_to_link in triple_to_candidate_ids.items():
                        
                        for position, inode_id in enumerate(ids_to_link):
                            links_to_insert.append({
                                "triple_id": triple_id,
                                "instance_node_id": inode_id,
                                "position": position
                            })
                    if links_to_insert:
                        link_statement = insert(TripleInstanceNodeDescriptionLink).values(links_to_insert)
                        session.execute(link_statement)

                    session.commit()
                    links_ready = True
                    break

            except Exception as e:
                if num_tries < max_tries -1:
                    logger.info(f"Retry updating database for NED2 batch creation in {poll_interval} seconds.")
                    time.sleep(poll_interval)
                else:
                    logger.error(f"Failed to update database for NED2 batch creation after {max_tries} attempts due to {e}")

        if batchwise or not links_ready:
            return

        entity_new_list = []
        for req, line, error in self._send_single_requests(batch_requests, "disambiguation", poll_interval, max_tries):
            triple_id = req["custom_id"]
            try:
                if error:
                    raise error
                existence = self.prompter_parser_module.parse_ned2_response(json.dumps(line))
            except Exception as e:
                logger.error(f"Failed NED2 for triple {triple_id}: {e}")
                continue
            if existence:
                if isinstance(existence, int) and existence <= 5:
                    self._update_triple_after_ned2({triple_id: existence - 1}, None)
                else:
                    logger.error(f"Failed NED2 for triple {triple_id}: unexpected answer {existence}")
            else:
                entity_new_list.append(triple_id)

        if entity_new_list:
            with Session(self.db_engine) as session:
                triple_object_ids = self._commit_new_inodes(entity_new_list, session)
            self._update_triple_after_ned2({}, triple_object_ids)
        
    
    def _create_pd_batches(self, triples_pd: list[Triple], poll_interval: int, batchwise=True, max_tries: int = 1):

        num_neighbor = min(5, self.predicate_index.ntotal)
        predicate_labels = [triple.predicate_label for triple in triples_pd]
        all_query_embeddings = self.stransformer_model.encode(
            predicate_labels,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_tensor=False,
            show_progress_bar=True
            ).astype(self.embedding_dtype)
        _distances, all_indices = self.predicate_index.search(all_query_embeddings, k=num_neighbor)
        search_results_indices = {triple.triple_id: all_indices[i] for i, triple in enumerate(triples_pd)}

        batch_requests = []
        links_ready = False
        for num_tries in range(max_tries):
            try:
                with Session(self.db_engine) as session:
                    batch_requests = []
                    all_required_predicate_indices = {idx for indices in all_indices for idx in indices if idx != -1}
                    required_candidate_predicates = {f"P{idx}" for idx in all_required_predicate_indices}
                    labels_map = {}
                    desc_map = {}
                    if required_candidate_predicates:
                        results = session.exec(
                            select(Predicate.id, Predicate.label, Predicate.description)
                            .where(Predicate.id.in_(required_candidate_predicates))
                        ).all()
                        for id, label, desc in results:
                            labels_map[str(id)] = str(label)
                            desc_map[str(id)] = str(desc)
                    
                    batch_requests.clear()
                    triple_to_candidate_ids = {}
                    for triple in triples_pd:
                        indices = search_results_indices[triple.triple_id]
                        valid_indices = [idx for idx in indices if idx != -1]
                        ids = [f"P{idx}" for idx in valid_indices]
                        triple_to_candidate_ids[triple.triple_id] = ids
                        labels = [labels_map.get(str(id), " ") for id in ids]
                        descs = [desc_map.get(str(id), " ") for id in ids]
                        req = self.prompter_parser_module.get_pd_prompt(triple.triple_id, triple.subject_label, triple.predicate_label, triple.object_label, labels, descs)
                        batch_requests.append(req)
                    
                    if not batch_requests:
                        logger.info("No valid PD requests, try to prepare again.")
                        continue

                    if batchwise:
                        batch_input_file = self.batch_backends["disambiguation"].upload(batch_requests, self.tmp_folder / "pd_batch_requests.jsonl")
                        openai_batch = self.batch_backends["disambiguation"].create_batch(batch_input_file.id, metadata={"description": "Predicate disambiguation", "type": "PD"})
                        logger.info(f"PD batch file created successfully. Batch ID: `{openai_batch.id}`.")

                        new_pd_batch = Batch(
                            id=openai_batch.id, input_file_id=batch_input_file.id, status="created",
                            job_type=JobType.PREDICATE_DISAMBIGUATION.value,
                            )
                        session.add(new_pd_batch)

                        triple_ids = [triple.triple_id for triple in triples_pd]
                        update_statement = (
                            update(Triple)
                            .where(Triple.triple_id.in_(triple_ids))
                            .values(predicate_status=PredicateStatusType.ONPD, pd_batch_id=openai_batch.id)
                            )
                        session.execute(update_statement)

                    session.execute(
                        delete(TriplePredicateLink)
                        .where(TriplePredicateLink.triple_id.in_([triple.triple_id for triple in triples_pd]))
                        )
                    links_to_insert = []
                    for triple in triples_pd:
                        ids_to_link = triple_to_candidate_ids.get(triple.triple_id, [])
                        for position, predicate_id in enumerate(ids_to_link):
                            links_to_insert.append({
                                "triple_id": triple.triple_id,
                                "predicate_id": predicate_id,
                                "position": position
                            })
                    if links_to_insert:
                        link_statement = insert(TriplePredicateLink).values(links_to_insert)
                        final_statement = link_statement.on_conflict_do_nothing(
                            index_elements=['triple_id', 'predicate_id']
                        )
                        session.execute(final_statement)

                    session.commit()
                    links_ready = True
                    break
            except Exception as e:
                if num_tries < max_tries -1:
                    logger.info(f"Retry creating PD batch in {poll_interval} seconds.")
                    time.sleep(poll_interval)
                else:
                    logger.error(f"Failed to create PD batch after {max_tries} attempts due to {e}")

        if batchwise or not links_ready:
            return

        for req, line, error in self._send_single_requests(batch_requests, "disambiguation", poll_interval, max_tries):
            triple_id = req["custom_id"]
            try:
                if error:
                    raise error
                existence = self.prompter_parser_module.parse_pd_response(json.dumps(line))
            except Exception as e:
                logger.error(f"Failed PD for triple {triple_id}: {e}")
                continue
            if existence:
                self._update_triple_after_pd({triple_id: existence - 1}, [])
            else:
                self._update_triple_after_pd({}, [triple_id])
    
    def _create_cd_batches(self, triples_cd: list[InstanceTriple], poll_interval: int, batchwise=True, max_tries: int = 1):
        corresponding_triple_ids = {itriple.original_triple_id for itriple in triples_cd}
        with Session(self.db_engine) as pre_session:
            triples_map = {
                t.triple_id: t for t in pre_session.exec(
                    select(Triple).where(Triple.triple_id.in_(corresponding_triple_ids))
                ).all()
            }
        if not triples_map:
            logger.warning("Corresponding original triples for CD not found.")
            return

        valid_triple_ids = triples_map.keys()
        concept_labels = [triples_map[tid].object_label for tid in valid_triple_ids]
        if self.concept_index.ntotal == 0:
            logger.warning("Empty concept Index.")
            return
        num_neighbor = min(5, self.concept_index.ntotal)
        all_query_embeddings = self.stransformer_model.encode(concept_labels, 
                                                              batch_size=64,
                                                              normalize_embeddings=True,
                                                              convert_to_tensor=False, 
                                                              show_progress_bar=True).astype(self.embedding_dtype)
        _distances, all_indices = self.concept_index.search(all_query_embeddings, k=num_neighbor)
        search_results_indices = {tid: all_indices[i] for i, tid in enumerate(valid_triple_ids)}

        batch_requests = []
        links_ready = False
        for num_tries in range(max_tries):
            try:
                with Session(self.db_engine) as session:
                    all_required_concept_indices = {idx for indices in all_indices for idx in indices if idx != -1}
                    required_candidate_concepts = {f"C{idx}" for idx in all_required_concept_indices}
                    labels_map = {}
                    desc_map = {}
                    if required_candidate_concepts:
                        results = session.exec(
                            select(Concept.id, Concept.label, Concept.description)
                            .where(Concept.id.in_(required_candidate_concepts))
                        ).all()
                        for id, label, desc in results:
                            labels_map[str(id)] = str(label)
                            desc_map[str(id)] = str(desc)
                    
                    batch_requests = []
                    triple_to_candidate_ids = {}
                    for triple_id in valid_triple_ids:
                        triple = triples_map[triple_id]
                        indices = search_results_indices.get(triple_id, [])
                        ids = [f"C{idx}" for idx in indices if idx != -1]
                        triple_to_candidate_ids[triple_id] = ids
                        
                        labels = [labels_map.get(str(id), " ") for id in ids]
                        descs = [desc_map.get(str(id), " ") for id in ids]
                        req = self.prompter_parser_module.get_cd_prompt(triple.triple_id, triple.subject_label, triple.predicate_label, triple.object_label, labels, descs)
                        batch_requests.append(req)
                    
                    if not batch_requests:
                        logger.info("No valid CD requests, try to prepare again.")
                        continue

                    if batchwise:
                        batch_input_file = self.batch_backends["disambiguation"].upload(batch_requests, self.tmp_folder / "cd_batch_requests.jsonl")
                        openai_batch = self.batch_backends["disambiguation"].create_batch(batch_input_file.id, metadata={"description": "Class disambiguation", "type": "CD"})
                        logger.info(f"CD batch file created successfully. Batch ID: `{openai_batch.id}`.")

                        new_cd_batch = Batch(
                            id=openai_batch.id, input_file_id=batch_input_file.id, status="created",
                            job_type=JobType.CONCEPT_DISAMBIGUATION.value,
                            )
                        session.add(new_cd_batch)

                        update_statement = (
                            update(InstanceTriple)
                            .where(InstanceTriple.original_triple_id.in_(valid_triple_ids))
                            .values(status=ConceptStatusType.ONCD, cd_batch_id=openai_batch.id)
                            )
                        session.execute(update_statement)

                    session.execute(
                        delete(InstanceTripleConceptLink)
                        .where(InstanceTripleConceptLink.original_triple_id.in_(valid_triple_ids))
                        )
                    links_to_insert = []
                    for itriple in triples_cd:
                        if itriple.original_triple_id in valid_triple_ids:
                            ids_to_link = triple_to_candidate_ids.get(itriple.original_triple_id, [])
                            for position, concept_id in enumerate(ids_to_link):
                                links_to_insert.append({
                                    "original_triple_id": itriple.original_triple_id,
                                    "concept_id": concept_id,
                                    "position": position
                                })
                    if links_to_insert:
                        link_statement = insert(InstanceTripleConceptLink).values(links_to_insert)
                        final_statement = link_statement.on_conflict_do_nothing(
                            index_elements=['original_triple_id', 'concept_id']
                        )
                        session.execute(final_statement)

                    session.commit()
                    links_ready = True
                    break
            except Exception as e:
                if num_tries < max_tries -1:
                    logger.info(f"Retry creating CD batch in {poll_interval} seconds.")
                    time.sleep(poll_interval)
                else:
                    logger.error(f"Failed to create CD batch after {max_tries} attempts due to {e}")

        if batchwise or not links_ready:
            return
        for req, line, error in self._send_single_requests(batch_requests, "disambiguation", poll_interval, max_tries):
            triple_id = req["custom_id"]
            try:
                if error:
                    raise error
                existence = self.prompter_parser_module.parse_cd_response(json.dumps(line))
            except Exception as e:
                logger.error(f"Failed CD for triple {triple_id}: {e}")
                continue
            if existence:
                self._update_instance_triple_after_cd({triple_id: existence - 1}, [])
            else:
                self._update_instance_triple_after_cd({}, [triple_id])
    
    def _create_nedg_batches(self, triples_nedg: list[Triple], poll_interval: int, batchwise=True, max_tries: int = 2):
        if not batchwise:
            reqs = [self.prompter_parser_module.get_nedg_prompt(t.triple_id, t.subject_label, t.predicate_label, t.object_label) for t in triples_nedg]
            for req, line, error in self._send_single_requests(reqs, "description", poll_interval, max_tries):
                triple_id = req["custom_id"]
                try:
                    if error:
                        raise error
                    desc = self.prompter_parser_module.parse_dg_response(json.dumps(line))
                except Exception as e:
                    logger.error(f"Failed NEDG for triple {triple_id}: {e}")
                    continue
                self._update_triple_after_nedg({triple_id: desc})
            return

        for num_tries in range(max_tries):
            try:
                with Session(self.db_engine) as session:
                    batch_requests = []
                    for triple in triples_nedg:
                        req = self.prompter_parser_module.get_nedg_prompt(triple.triple_id, triple.subject_label, triple.predicate_label, triple.object_label)
                        batch_requests.append(req)

                    batch_input_file = self.batch_backends["description"].upload(batch_requests, self.tmp_folder / "nedg_batch_requests.jsonl")
                    openai_batch = self.batch_backends["description"].create_batch(batch_input_file.id, metadata={"description": "Named entity description generation", "type": "NEDG"})
                    logger.info(f"NEDG batch file created successfully. Batch ID: `{openai_batch.id}`.")
                    new_nedg_batch = Batch(
                        id=openai_batch.id, input_file_id=batch_input_file.id, status="created",
                        job_type=JobType.NAMED_ENTITY_DESCRIPTION_GEN.value,
                        )
                    session.add(new_nedg_batch)

                    triple_ids = [triple.triple_id for triple in triples_nedg]
                    update_statement = (
                        update(Triple)
                        .where(Triple.triple_id.in_(triple_ids))
                        .values(object_status=ObjectStatusType.ONDG, nedg_batch_id=openai_batch.id)
                        )
                    session.execute(update_statement)
                    session.commit()
                    return 
            except Exception as e:
                if num_tries < max_tries -1:
                    logger.info(f"Retry updating database for NEDG batch creation in {poll_interval} seconds.")
                    time.sleep(poll_interval)
                else:
                    logger.error(f"Failed to update database for NEDG batch creation after {max_tries} attempts due to {e}")
    
    def _create_pdg_batches(self, triples_pdg: list[Triple], poll_interval: int, batchwise=True, max_tries: int = 2):
        if not batchwise:
            predicate_new_map = {}
            reqs = []
            for triple in triples_pdg:
                if len(triple.predicate_label) > 1:
                    reqs.append(self.prompter_parser_module.get_pdg_prompt(triple.triple_id, triple.predicate_label))
                else:
                    reqs.append(self.prompter_parser_module.get_pdg_triple_prompt(triple.triple_id, triple.subject_label, triple.predicate_label, triple.object_label))
            for req, line, error in self._send_single_requests(reqs, "description", poll_interval, max_tries):
                triple_id = req["custom_id"]
                try:
                    if error:
                        raise error
                    predicate_new_map[triple_id] = self.prompter_parser_module.parse_dg_response(json.dumps(line))
                except Exception as e:
                    logger.error(f"Failed PDG for triple {triple_id}: {e}")

            if predicate_new_map:
                triple_predicate_ids = self._commit_new_predicates(predicate_new_map)
                self._update_triple_after_pdg(triple_predicate_ids)
            return

        for num_tries in range(max_tries):
            try:
                with Session(self.db_engine) as session:
                    batch_requests = []
                    for triple in triples_pdg:
                        if len(triple.predicate_label) > 1:
                            req = self.prompter_parser_module.get_pdg_prompt(triple.triple_id, triple.predicate_label)
                        else:
                            req = self.prompter_parser_module.get_pdg_triple_prompt(triple.triple_id, triple.subject_label, triple.predicate_label, triple.object_label)
                        batch_requests.append(req)

                    batch_input_file = self.batch_backends["description"].upload(batch_requests, self.tmp_folder / "pdg_batch_requests.jsonl")
                    openai_batch = self.batch_backends["description"].create_batch(batch_input_file.id, metadata={"description": "Predicate description generation", "type": "PDG"})
                    logger.info(f"PDG batch file created successfully. Batch ID: `{openai_batch.id}`.")
                    new_pdg_batch = Batch(
                        id=openai_batch.id, input_file_id=batch_input_file.id, status="created",
                        job_type=JobType.PREDICATE_DESCRIPTION_GEN.value,
                        )
                    session.add(new_pdg_batch)

                    triple_ids = [triple.triple_id for triple in triples_pdg]
                    update_statement = (
                        update(Triple)
                        .where(Triple.triple_id.in_(triple_ids))
                        .values(predicate_status=PredicateStatusType.ONDG, pdg_batch_id=openai_batch.id)
                        )
                    session.execute(update_statement)
                    session.commit()
                    return 
            except Exception as e:
                if num_tries < max_tries -1:
                    logger.info(f"Retry updating database for PDG batch creation in {poll_interval} seconds.")
                    time.sleep(poll_interval)
                else:
                    logger.error(f"Failed to update database for PDG batch creation after {max_tries} attempts due to {e}")
    
    def _create_cdg_batches(self, triples_cdg: list[InstanceTriple], poll_interval: int, batchwise=True, max_tries: int = 1):
        if not batchwise:
            concept_new_map = {}
            reqs = [self.prompter_parser_module.get_cdg_prompt(t.original_triple_id, t.concept_label) for t in triples_cdg]
            for req, line, error in self._send_single_requests(reqs, "description", poll_interval, max_tries):
                triple_id = req["custom_id"]
                try:
                    if error:
                        raise error
                    concept_new_map[triple_id] = self.prompter_parser_module.parse_dg_response(json.dumps(line))
                except Exception as e:
                    logger.error(f"Failed CDG for triple {triple_id}: {e}")

            if concept_new_map:
                triple_concept_ids = self._commit_new_concepts(concept_new_map)
                self._update_instance_triple_after_cdg(triple_concept_ids)
            return

        for num_tries in range(max_tries):
            try:
                with Session(self.db_engine) as session:
                    batch_requests = []
                    for triple in triples_cdg:
                        req = self.prompter_parser_module.get_cdg_prompt(triple.original_triple_id, triple.concept_label)
                        batch_requests.append(req)

                    if not batch_requests:
                        logger.info("No valid CDG requests, try to prepare again.")
                        continue

                    batch_input_file = self.batch_backends["description"].upload(batch_requests, self.tmp_folder / "cdg_batch_requests.jsonl")
                    openai_batch = self.batch_backends["description"].create_batch(batch_input_file.id, metadata={"description": "Class description generation", "type": "CDG"})
                    logger.info(f"CDG batch file created successfully. Batch ID: `{openai_batch.id}`.")

                    new_cdg_batch = Batch(
                        id=openai_batch.id, input_file_id=batch_input_file.id, status="created",
                        job_type=JobType.CONCEPT_DESCRIPTION_GEN.value,
                        )
                    session.add(new_cdg_batch)

                    triple_ids = [itriple.original_triple_id for itriple in triples_cdg]
                    update_statement = (
                        update(InstanceTriple)
                        .where(InstanceTriple.original_triple_id.in_(triple_ids))
                        .values(status=ConceptStatusType.ONDG, cdg_batch_id=openai_batch.id)
                        )
                    session.execute(update_statement)
                    session.commit()
                    return 
            except Exception as e:
                if num_tries < max_tries -1:
                    logger.info(f"Retry creating CDG batch in {poll_interval} seconds.")
                    time.sleep(poll_interval)
                else:
                    logger.error(f"Failed to create CDG batch after {max_tries} attempts due to {e}")
            
    def _process_one_completed_elicitation_batch(self, openai_batch):

        with Session(self.db_engine) as session:
            if openai_batch.status == "completed":
                try:
                    logger.info(f"Processing a newly completed elicitation batch: `{openai_batch.id}`. Downloading results.")
                    batch_result = self.batch_backends["elicitation"].download_output(openai_batch)

                    def process_lines(lines, raw_triples, explored_inodes, failed_inodes):
                        for line in lines:
                            line = json.loads(line.strip())
                            if not line:
                                continue
                            subject_id = line.get("custom_id", "UNKNOWN_SUBJECT")

                            try:
                                triples = self.prompter_parser_module.parse_elicitation_response(line)
                                raw_triples.extend(triples)
                                explored_inodes.append(subject_id)
                            except Exception as e:
                                failed_inodes.append(subject_id)
                                logger.error(f"Failed elicitation for entity {subject_id}: {e}")

                    raw_triples = []
                    explored_inodes = []
                    failed_inodes = []
                    content = batch_result.decode("utf-8")
                    lines = content.splitlines()
                    process_lines(lines, raw_triples, explored_inodes, failed_inodes)
                    
                    if len(raw_triples) > 0:
                        self._commit_new_triples(raw_triples=raw_triples, batch_id=openai_batch.id, session=session)
                    if len(explored_inodes) > 0:
                        explored_inodes = set(explored_inodes)
                        session.execute(
                            update(InstanceNode)
                            .where(InstanceNode.id.in_(explored_inodes))
                            .values(status=StatusType.EXPLORED,
                                    explored_batch_id=openai_batch.id)
                        )
                    if len(failed_inodes) > 0:
                        failed_inodes = set(failed_inodes)
                        session.execute(
                            update(InstanceNode)
                            .where(InstanceNode.id.in_(failed_inodes))
                            .values(status=StatusType.UNEXPLORED)
                        )
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "completed"
                    else:
                        logger.warning(f"Elicitation batch {openai_batch.id} not found.")
                    
                    session.commit()
                    return len(explored_inodes)
                
                except Exception as e:
                    logger.error(f"Failed to process batch {openai_batch.id}: {str(e)}")
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "parsing_failed"
                        statement = (
                                update(InstanceNode)
                                .where(InstanceNode.explored_batch_id == batch.id)
                                .values(status=StatusType.UNEXPLORED, explored_batch_id=None)
                            )
                        session.execute(statement)
                        session.commit()
                    else:
                        logger.warning(f"Elicitation batch {openai_batch.id} not found.")

                    return 0
                
            else:
                logger.error(f"Elicitation batch `{openai_batch.id}` is not completed. Status: `{openai_batch.status}`")
                batch = session.get(Batch, openai_batch.id)
                if batch:
                    batch.status = openai_batch.status
                    statement = (
                            update(InstanceNode)
                            .where(InstanceNode.explored_batch_id == batch.id)
                            .values(status=StatusType.UNEXPLORED, explored_batch_id=None)
                        )
                    session.execute(statement)
                    session.commit()
                else:
                    logger.error(f"Failed elicitation batch {openai_batch.id} not found.")

                return 0
    
    def _process_one_completed_ner_batch(self, openai_batch: OpenAIBatch):

        with Session(self.db_engine) as session:
            if openai_batch.status == "completed":
                try:
                    logger.info(f"Processing a newly completed NER batch: `{openai_batch.id}`. Downloading results.")
                    batch_result = self.batch_backends["disambiguation"].download_output(openai_batch)

                    def process_lines(lines, ne_list, literal_list, failed_ner):
                        for line in lines:
                            obj = json.loads(line)
                            triple_id = obj.get("custom_id", "UNKNOWN_TRIPLE")
                            if not line:
                                continue
                            try:
                                ner = self.prompter_parser_module.parse_ner_response(line)
                                if ner[1]:
                                    ne_list.append(triple_id)
                                else:
                                    literal_list.append(triple_id)
                            except Exception as e:
                                failed_ner.append(triple_id)
                                logger.error(f"Failed NER for triple {triple_id}: {e}")

                    ne_list, literal_list, failed_ner = [], [], []
                    content = batch_result.decode("utf-8")
                    lines = content.splitlines()
                    process_lines(lines, ne_list, literal_list, failed_ner)

                    self._update_triple_after_ner(ne_list, literal_list)
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "completed"
                    else:
                        logger.warning(f"NER batch {openai_batch.id} not found.")
                    
                    if len(failed_ner) > 0:
                        failed_ner = set(failed_ner)
                        session.execute(
                            update(Triple)
                            .where(Triple.triple_id.in_(failed_ner))
                            .values(object_status=ObjectStatusType.GENERATED)
                        )
                    session.commit()
                    logger.info(f"Found {len(ne_list)} nes and {len(literal_list)} literals in this NER batch.")

                except Exception as e:
                    logger.error(f"Failed to process batch {openai_batch.id}: {str(e)}")
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "parsing_failed"
                        statement = (
                            update(Triple)
                            .where(Triple.ner_batch_id == batch.id)
                            .values(object_status=ObjectStatusType.GENERATED, ner_batch_id=None)
                        )
                        session.execute(statement)
                        session.commit()
                    else:
                        logger.warning(f"NER batch {openai_batch.id} not found.")
    
            else:
                logger.error(f"NER batch `{openai_batch.id}` is not completed. Status: `{openai_batch.status}`")
                batch = session.get(Batch, openai_batch.id)
                if batch:
                    batch.status = openai_batch.status
                    statement = (
                        update(Triple)
                        .where(Triple.ner_batch_id == batch.id)
                        .values(object_status=ObjectStatusType.GENERATED, ner_batch_id=None)
                    )
                    session.execute(statement)
                    session.commit()
                else:
                    logger.error(f"Failed NER batch {openai_batch.id} not found.")


    def _process_one_completed_ned1_batch(self, openai_batch: OpenAIBatch):

        with Session(self.db_engine) as session:
            if openai_batch.status == "completed":
                try:
                    logger.info(f"Processing a newly completed NED1 batch: `{openai_batch.id}`. Downloading results.")
                    batch_result = self.batch_backends["disambiguation"].download_output(openai_batch)

                    def process_lines(lines, entity_existing_map, entity_unknown_list, entity_new_list, failed_ned):
                        for line in lines:
                            obj = json.loads(line)
                            triple_id = obj.get("custom_id", "UNKNOWN_TRIPLE")
                            if not line:
                                continue
                            try:
                                existence = self.prompter_parser_module.parse_ned1_response(line)
                                if existence:
                                    if isinstance(existence, int) and existence <= 5:
                                        entity_existing_map[triple_id] = existence-1
                                    elif existence == "unknown":
                                        entity_unknown_list.append(triple_id)
                                else:
                                    entity_new_list.append(triple_id)
                            except Exception as e:
                                failed_ned.append(triple_id)
                                logger.error(f"Failed NED1 for triple {triple_id}: {e}")

                    entity_existing_map = {}
                    entity_unknown_list = []
                    entity_new_list = []
                    failed_ned = []
                    content = batch_result.decode("utf-8")
                    lines = content.splitlines()
                    process_lines(lines, entity_existing_map, entity_unknown_list, entity_new_list, failed_ned)
                    self._update_triple_after_ned1(entity_existing_map, entity_new_list, entity_unknown_list)
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "completed"
                    else:
                        logger.warning(f"NED1 batch {openai_batch.id} not found.")
                    
                    if len(failed_ned) > 0:
                        failed_ned = set(failed_ned)
                        session.execute(
                            update(Triple)
                            .where(Triple.triple_id.in_(failed_ned))
                            .values(object_status=ObjectStatusType.NERFINISHED)
                        )

                    session.commit()

                except Exception as e:
                    logger.error(f"Failed to process NED1 batch {openai_batch.id}: {str(e)}")
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "parsing_failed"
                        statement = (
                            update(Triple)
                            .where(Triple.ned1_batch_id == batch.id)
                            .values(object_status=ObjectStatusType.NERFINISHED, ned1_batch_id=None)
                        )
                        session.execute(statement)
                        session.commit()
                    else:
                        logger.warning(f"NED1 batch {openai_batch.id} not found.")
                
            else:
                logger.error(f"NED1 batch `{openai_batch.id}` is not completed. Status: `{openai_batch.status}`")
                batch = session.get(Batch, openai_batch.id)
                if batch:
                    batch.status = openai_batch.status
                    statement = (
                            update(Triple)
                            .where(Triple.ned1_batch_id == batch.id)
                            .values(object_status=ObjectStatusType.NERFINISHED, ned1_batch_id=None)
                        )
                    session.execute(statement)
                    session.commit()
                else:
                    logger.error(f"Failed NED1 batch {openai_batch.id} not found.")

    def _process_one_completed_nedg_batch(self, openai_batch: OpenAIBatch):
        with Session(self.db_engine) as session:
            if openai_batch.status == "completed":
                try:
                    logger.info(f"Processing a newly completed NEDG batch: `{openai_batch.id}`. Downloading results.")
                    batch_result = self.batch_backends["description"].download_output(openai_batch)

                    def process_lines(lines, entity_new_map, failed_nedg):
                        for line in lines:
                            obj = json.loads(line)
                            triple_id = obj.get("custom_id", "UNKNOWN_TRIPLE")
                            if not line:
                                continue
                            try:
                                desc = self.prompter_parser_module.parse_dg_response(line)
                                entity_new_map[triple_id] = desc
                            except Exception as e:
                                failed_nedg.append(triple_id)
                                logger.error(f"Failed NEDG for triple {triple_id}: {e}")

                    entity_new_map = {}
                    failed_nedg = []
                    content = batch_result.decode("utf-8")
                    lines = content.splitlines()
                    process_lines(lines, entity_new_map, failed_nedg)
                    self._update_triple_after_nedg(entity_new_map)
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "completed"
                    else:
                        logger.warning(f"NEDG batch {openai_batch.id} not found.")

                    if len(failed_nedg) > 0:
                        failed_nedg = set(failed_nedg)
                        session.execute(
                            update(Triple)
                            .where(Triple.triple_id.in_(failed_nedg))
                            .values(object_status=ObjectStatusType.NEDFINISHED1)
                        )

                    session.commit()

                except Exception as e:
                    logger.error(f"Failed to process NEDG batch {openai_batch.id}: {str(e)}")
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "parsing_failed"
                        statement = (
                            update(Triple)
                            .where(Triple.nedg_batch_id == batch.id)
                            .values(object_status=ObjectStatusType.NEDFINISHED1, nedg_batch_id=None)
                        )
                        session.execute(statement)
                        session.commit()
                    else:
                        logger.warning(f"NEDG batch {openai_batch.id} not found.")
                
            else:
                logger.error(f"NEDG batch `{openai_batch.id}` is not completed. Status: `{openai_batch.status}`")
                batch = session.get(Batch, openai_batch.id)
                if batch:
                    batch.status = openai_batch.status
                    statement = (
                            update(Triple)
                            .where(Triple.nedg_batch_id == batch.id)
                            .values(object_status=ObjectStatusType.NEDFINISHED1, nedg_batch_id=None)
                        )
                    session.execute(statement)
                    session.commit()
                else:
                    logger.error(f"Failed NEDG, batch {openai_batch.id} not found.")
    
    def _process_one_completed_ned2_batch(self, openai_batch: OpenAIBatch):

        with Session(self.db_engine) as session:
            if openai_batch.status == "completed":
                try:
                    logger.info(f"Processing a newly completed NED2 batch: `{openai_batch.id}`. Downloading results.")
                    batch_result = self.batch_backends["disambiguation"].download_output(openai_batch)

                    def process_lines(lines, entity_existing_map, entity_new_list, failed_ned):
                        for line in lines:
                            obj = json.loads(line)
                            triple_id = obj.get("custom_id", "UNKNOWN_TRIPLE")
                            if not line:
                                continue
                            try:
                                existence = self.prompter_parser_module.parse_ned2_response(line)
                                if existence:
                                    if isinstance(existence, int) and existence <= 5:
                                        entity_existing_map[triple_id] = existence-1
                                    else:
                                        failed_ned.append(triple_id)
                                else:
                                    entity_new_list.append(triple_id)
                            except Exception as e:
                                failed_ned.append(triple_id)
                                logger.error(f"Failed NED2 for triple {triple_id}: {e}")

                    entity_existing_map = {}
                    entity_new_list = []
                    failed_ned = []
                    content = batch_result.decode("utf-8")
                    lines = content.splitlines()
                    process_lines(lines, entity_existing_map, entity_new_list, failed_ned)
                    triple_object_ids = self._commit_new_inodes(entity_new_list, session)
                    self._update_triple_after_ned2(entity_existing_map, triple_object_ids)
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "completed"
                    else:
                        logger.warning(f"NED2 batch {openai_batch.id} not found.")
                    
                    if len(failed_ned) > 0:
                        failed_ned = set(failed_ned)
                        session.execute(
                            update(Triple)
                            .where(Triple.triple_id.in_(failed_ned))
                            .values(object_status=ObjectStatusType.DGFINISHED)
                        )
                    session.commit()

                except Exception as e:
                    logger.error(f"Failed to process NED2 batch {openai_batch.id}: {str(e)}")
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "parsing_failed"
                        statement = (
                            update(Triple)
                            .where(Triple.ned2_batch_id == batch.id)
                            .values(object_status=ObjectStatusType.DGFINISHED, ned2_batch_id=None)
                        )
                        session.execute(statement)
                        session.commit()
                    else:
                        logger.warning(f"NED2 batch {openai_batch.id} not found.")
                
            else:
                logger.error(f"NED2 batch `{openai_batch.id}` is not completed. Status: `{openai_batch.status}`")
                batch = session.get(Batch, openai_batch.id)
                if batch:
                    batch.status = openai_batch.status
                    statement = (
                            update(Triple)
                            .where(Triple.ned2_batch_id == batch.id)
                            .values(object_status=ObjectStatusType.DGFINISHED, ned2_batch_id=None)
                        )
                    session.execute(statement)
                    session.commit()
                else:
                    logger.error(f"Failed NED2 batch {openai_batch.id} not found.")
    
    def _process_one_completed_pd_batch(self, openai_batch: OpenAIBatch):
        if openai_batch.status == "completed":
            try:
                logger.info(f"Processing a newly completed PD batch: `{openai_batch.id}`. Downloading results.")
                batch_result = self.batch_backends["disambiguation"].download_output(openai_batch)

                def process_lines(lines, predicate_existing_map, predicate_new_list, failed_pd):
                    for line in lines:
                        obj = json.loads(line)
                        triple_id = obj.get("custom_id", "UNKNOWN_TRIPLE")
                        if not line:
                            continue
                        
                        try:
                            existence = self.prompter_parser_module.parse_pd_response(line)
                            if existence:
                                predicate_existing_map[triple_id] = existence-1
                            else:
                                predicate_new_list.append(triple_id)
                        except Exception as e:
                            failed_pd.append(triple_id)
                            logger.error(f"Failed PD for triple {triple_id}: {e}")

                predicate_existing_map = {}
                predicate_new_list = []
                failed_pd = []
                content = batch_result.decode("utf-8")
                lines = content.splitlines()
                process_lines(lines, predicate_existing_map, predicate_new_list, failed_pd)
                self._update_triple_after_pd(predicate_existing_map, predicate_new_list)
                with Session(self.db_engine) as session:
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "completed"
                    else:
                        logger.warning(f"PD batch {openai_batch.id} not found.")
                    
                    if len(failed_pd) > 0:
                        failed_pd = set(failed_pd)
                        session.execute(
                            update(Triple)
                            .where(Triple.triple_id.in_(failed_pd))
                            .values(predicate_status=PredicateStatusType.GENERATED)
                        )

                    session.commit()

            except Exception as e:
                logger.error(f"Failed to process PD batch {openai_batch.id}: {str(e)}")
                with Session(self.db_engine) as session:
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "parsing_failed"
                        statement = (
                                update(Triple)
                                .where(Triple.pd_batch_id == batch.id)
                                .values(predicate_status=PredicateStatusType.GENERATED, pd_batch_id=None)
                            )
                        session.execute(statement)
                        session.commit()
                    else:
                        logger.error(f"PD batch {openai_batch.id} not found.")
                
        else:
            logger.error(f"PD batch `{openai_batch.id}` is not completed. Status: `{openai_batch.status}`")
            with Session(self.db_engine) as session:
                batch = session.get(Batch, openai_batch.id)
                if batch:
                    batch.status = openai_batch.status
                    statement = (
                            update(Triple)
                            .where(Triple.pd_batch_id == batch.id)
                            .values(predicate_status=PredicateStatusType.GENERATED, pd_batch_id=None)
                        )
                    session.execute(statement)
                    session.commit()
                else:
                    logger.error(f"Failed PD batch {openai_batch.id} not found.")
    
    def _process_one_completed_pdg_batch(self, openai_batch: OpenAIBatch):
        if openai_batch.status == "completed":
            try:
                logger.info(f"Processing a newly completed PDG batch: `{openai_batch.id}`. Downloading results.")
                batch_result = self.batch_backends["description"].download_output(openai_batch)

                def process_lines(lines, predicate_new_map, failed_pdg):
                    for line in lines:
                        obj = json.loads(line)
                        triple_id = obj.get("custom_id", "UNKNOWN_TRIPLE")
                        if not line:
                            continue
                        try:
                            description = self.prompter_parser_module.parse_dg_response(line)
                            predicate_new_map[triple_id] = description
                        except Exception as e:
                            failed_pdg.append(triple_id)
                            logger.error(f"Failed PDG for triple {triple_id}: {e}")

                predicate_new_map = {}
                failed_pdg = []
                content = batch_result.decode("utf-8")
                lines = content.splitlines()
                process_lines(lines, predicate_new_map, failed_pdg)

                triple_predicate_ids = self._commit_new_predicates(predicate_new_map)
                self._update_triple_after_pdg(triple_predicate_ids)
                with Session(self.db_engine) as session:
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "completed"
                    else:
                        logger.error(f"PDG batch {openai_batch.id} not found.")
                    
                    if len(failed_pdg) > 0:
                        failed_pdg = set(failed_pdg)
                        session.execute(
                            update(Triple)
                            .where(Triple.triple_id.in_(failed_pdg))
                            .values(predicate_status=PredicateStatusType.PDFINISHED)
                        )
                    session.commit()

            except Exception as e:
                logger.error(f"Failed to process PDG batch {openai_batch.id}: {str(e)}")
                with Session(self.db_engine) as session:
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "parsing_failed"
                        statement = (
                                update(Triple)
                                .where(Triple.pdg_batch_id == batch.id)
                                .values(predicate_status=PredicateStatusType.PDFINISHED, pdg_batch_id=None)
                            )
                        session.execute(statement)
                        session.commit()
                    else:
                        logger.warning(f"PDG batch {openai_batch.id} not found.")
                
        else:
            logger.error(f"PDG batch `{openai_batch.id}` is not completed. Status: `{openai_batch.status}`")
            with Session(self.db_engine) as session:
                batch = session.get(Batch, openai_batch.id)
                if batch:
                    batch.status = openai_batch.status
                    statement = (
                            update(Triple)
                            .where(Triple.pdg_batch_id == batch.id)
                            .values(predicate_status=PredicateStatusType.PDFINISHED, pdg_batch_id=None)
                        )
                    session.execute(statement)
                    session.commit()
                else:
                    logger.error(f"Failed PDG batch {openai_batch.id} not found.")
    
    def _process_one_completed_cd_batch(self, openai_batch: OpenAIBatch):
        if openai_batch.status == "completed":
            try:
                logger.info(f"Processing a newly completed CD batch: `{openai_batch.id}`. Downloading results.")
                batch_result = self.batch_backends["disambiguation"].download_output(openai_batch)

                def process_lines(lines, concept_existing_map, concept_new_list, failed_cd):
                    for line in lines:
                        obj = json.loads(line)
                        triple_id = obj.get("custom_id", "UNKNOWN_TRIPLE")
                        if not line:
                            continue
                        try:
                            existence = self.prompter_parser_module.parse_cd_response(line)
                            if existence:
                                concept_existing_map[triple_id] = existence-1
                            else:
                                concept_new_list.append(triple_id)
                        except Exception as e:
                            failed_cd.append(triple_id)
                            logger.error(f"Failed CD for triple {triple_id}: {e}")
                
                concept_new_list = []
                concept_existing_map = {}
                failed_cd = []
                content = batch_result.decode("utf-8")
                lines = content.splitlines()
                process_lines(lines, concept_existing_map, concept_new_list, failed_cd)
                self._update_instance_triple_after_cd(concept_existing_map, concept_new_list)

                with Session(self.db_engine) as session:
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "completed"
                    else:
                        logger.warning(f"CD batch {openai_batch.id} not found.")
                    
                    if len(failed_cd) > 0:
                        failed_cd = set(failed_cd)
                        session.execute(
                            update(InstanceTriple)
                            .where(InstanceTriple.original_triple_id.in_(failed_cd))
                            .values(status=ConceptStatusType.GENERATED)
                        )
                    
                    session.commit()

            except Exception as e:
                logger.error(f"Failed to process CD batch {openai_batch.id}: {str(e)}")
                with Session(self.db_engine) as session:
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "parsing_failed"
                        statement = (
                                update(InstanceTriple)
                                .where(InstanceTriple.cd_batch_id == batch.id)
                                .values(status=ConceptStatusType.GENERATED, cd_batch_id=None)
                            )
                        session.execute(statement)
                        session.commit()
                    else:
                        logger.error(f"CD batch {openai_batch.id} not found.")
                
        else:
            logger.error(f"CD batch `{openai_batch.id}` is not completed. Status: `{openai_batch.status}`")
            with Session(self.db_engine) as session:
                batch = session.get(Batch, openai_batch.id)
                if batch:
                    batch.status = openai_batch.status
                    statement = (
                            update(InstanceTriple)
                            .where(InstanceTriple.cd_batch_id == batch.id)
                            .values(status=ConceptStatusType.GENERATED, cd_batch_id=None)
                        )
                    session.execute(statement)
                    session.commit()
                else:
                    logger.error(f"Failed CD batch {openai_batch.id} not found.")
    
    def _process_one_completed_cdg_batch(self, openai_batch: OpenAIBatch):
        if openai_batch.status == "completed":
            try:
                logger.info(f"Processing a newly completed CDG batch: `{openai_batch.id}`. Downloading results.")
                batch_result = self.batch_backends["description"].download_output(openai_batch)

                def process_lines(lines, concept_new_map, failed_cdg):
                    for line in lines:
                        obj = json.loads(line)
                        triple_id = obj.get("custom_id", "UNKNOWN_TRIPLE")
                        if not line:
                            continue
                        try:
                            description = self.prompter_parser_module.parse_dg_response(line)
                            concept_new_map[triple_id] = description
                        except Exception as e:
                            failed_cdg.append(triple_id)
                            logger.error(f"Failed CDG for triple {triple_id}: {e}")


                concept_new_map = {}
                failed_cdg = []
                content = batch_result.decode("utf-8")
                lines = content.splitlines()
                process_lines(lines, concept_new_map, failed_cdg)

                triple_concept_ids = self._commit_new_concepts(concept_new_map)
                self._update_instance_triple_after_cdg(triple_concept_ids)
                with Session(self.db_engine) as session:
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "completed"
                    else:
                        logger.warning(f"CDG batch {openai_batch.id} not found.")
                    
                    if len(failed_cdg) > 0:
                        failed_cdg = set(failed_cdg)
                        session.execute(
                            update(InstanceTriple)
                            .where(InstanceTriple.original_triple_id.in_(failed_cdg))
                            .values(status=ConceptStatusType.CDFINISHED)
                        )
                    
                    session.commit()

            except Exception as e:
                logger.error(f"Failed to process CDG batch {openai_batch.id}: {str(e)}")
                with Session(self.db_engine) as session:
                    batch = session.get(Batch, openai_batch.id)
                    if batch:
                        batch.status = "parsing_failed"
                        statement = (
                                update(InstanceTriple)
                                .where(InstanceTriple.cdg_batch_id == batch.id)
                                .values(status=ConceptStatusType.CDFINISHED, cdg_batch_id=None)
                            )
                        session.execute(statement)
                        session.commit()
                    else:
                        logger.error(f"CDG batch {openai_batch.id} not found.")
                
        else:
            logger.error(f"CDG batch `{openai_batch.id}` is not completed. Status: `{openai_batch.status}`")
            with Session(self.db_engine) as session:
                batch = session.get(Batch, openai_batch.id)
                if batch:
                    batch.status = openai_batch.status
                    statement = (
                            update(InstanceTriple)
                            .where(InstanceTriple.cdg_batch_id == batch.id)
                            .values(status=ConceptStatusType.CDFINISHED, cdg_batch_id=None)
                        )
                    session.execute(statement)
                    session.commit()
                else:
                    logger.error(f"Failed CDG batch {openai_batch.id} not found.")

    @staticmethod
    def _triple_signature(triple: Triple, object_id: str) -> tuple:
        return (triple.subject_id, triple.subject_label, triple.predicate_id, triple.predicate_label, object_id, triple.object_label)

    def _find_existing_triple_signatures(self, session: Session, signatures) -> set:
        columns = (Triple.subject_id, Triple.subject_label, Triple.predicate_id, Triple.predicate_label, Triple.object_id, Triple.object_label)
        signatures = list(signatures)
        existing = set()
        for i in range(0, len(signatures), 500):
            rows = session.exec(select(*columns).where(tuple_(*columns).in_(signatures[i:i + 500]))).all()
            existing.update(tuple(row) for row in rows)
        return existing

    def _commit_new_triples(self, raw_triples: list[dict], batch_id: str, processing_batch_size: int = 500, session: Session = None):
    
        num_new_triples = 0

        for i in tqdm(range(0, len(raw_triples), processing_batch_size), desc=f"Commiting triples of Batch `{batch_id}`"):
            raw_triple_batch = raw_triples[i:i + processing_batch_size]

            unique_pairs_to_check = {(triple["predicate"], triple["object"]) for triple in raw_triple_batch}
            mapping_cache = {}
            if unique_pairs_to_check:
                
                mapping_query = (
                        select(
                            Triple.predicate_label,
                            Triple.object_label,
                            func.min(Triple.object_id).label("object_id"),
                        )
                        .where(
                            Triple.object_type == ObjectType.NE,
                            Triple.object_id.is_not(None),
                            tuple_(Triple.predicate_label, Triple.object_label).in_(unique_pairs_to_check)
                        )
                        .group_by(Triple.predicate_label, Triple.object_label)
                        .having(and_(
                            func.count(Triple.triple_id) > 50,
                            func.min(Triple.object_id) == func.max(Triple.object_id)
                        ))
                    )
                    
                for row in session.exec(mapping_query).all():
                    mapping_cache[(row.predicate_label, row.object_label)] = row.object_id
                    
            triples_to_insert = []
            instance_triples = []
            for triple in raw_triple_batch:
                triple_id = f"T{self.id_counter_triple}"
                self.id_counter_triple += 1

                p_label = triple["predicate"]
                o_label = triple["object"]
                key = (p_label, o_label)

                if key in mapping_cache:
                    triples_to_insert.append({
                        "triple_id": triple_id,
                        "subject_id": triple["subject_id"], 
                        "subject_label": triple["subject"],
                        "predicate_label": p_label, 
                        "predicate_id": None,
                        "object_label": o_label,
                        "object_id": str(mapping_cache[key]), 
                        "object_type": ObjectType.NE,
                        "creating_batch_id": batch_id, 
                        "object_status": ObjectStatusType.FINISHED,
                    })
                    continue

                if p_label == "instanceOf":
                    instance_triples.append({
                        "original_triple_id": triple_id,
                        "entity_id": triple["subject_id"],
                        "concept_label": o_label,
                        "status": ConceptStatusType.GENERATED,
                    })
                    triples_to_insert.append({
                        "triple_id": triple_id,
                        "subject_id": triple["subject_id"], 
                        "subject_label": triple["subject"],
                        "predicate_label": p_label, 
                        "predicate_id": "P0",
                        "object_label": o_label,
                        "object_id": None,
                        "object_type": ObjectType.CONCEPT,
                        "creating_batch_id": batch_id, 
                        "object_status": ObjectStatusType.FINISHED,
                        "predicate_status": PredicateStatusType.FINISHED,
                    })
                    continue

                triples_to_insert.append({
                    "triple_id": triple_id,
                    "subject_id": triple["subject_id"], 
                    "subject_label": triple["subject"],
                    "predicate_label": p_label, 
                    "predicate_id": None,
                    "object_label": o_label,
                    "object_id": None,
                    "object_type": ObjectType.UNRECOGNIZED,
                    "creating_batch_id": batch_id, 
                    "object_status": ObjectStatusType.GENERATED,
                })

            
            if triples_to_insert:
                insert_statement = insert(Triple).values(triples_to_insert).on_conflict_do_nothing(
                    index_elements=["subject_id", "subject_label", "predicate_id", "predicate_label", "object_id", "object_label"]
                )
                session.execute(insert_statement)
                existing_ids = set(
                    session.exec(
                        select(Triple.triple_id)
                        .where(Triple.triple_id.in_([t["triple_id"] for t in triples_to_insert]))
                        ).all()
                    )
                num_new_triples += len(existing_ids)
                valid_instance_triples = [t for t in instance_triples if t["original_triple_id"] in existing_ids]
                if valid_instance_triples:
                    instance_insert_statement = insert(InstanceTriple).values(valid_instance_triples)
                    session.execute(instance_insert_statement)
                
                session.commit()


        logger.info(f"{num_new_triples} new triples from batch `{batch_id}` inserted, {self.id_counter_triple} triples in total.")
        return num_new_triples
    
    def _commit_new_inodes(self, entity_new_list, session: Session = None):

        if not entity_new_list:
            return None

        try:
            triples = session.exec(
                select(Triple)
                .where(Triple.triple_id.in_(entity_new_list))
                ).all()
            if not triples:
                logger.warning("No matching triples found after NED2")
                return None
            
            triple_data_list = []
            unique_keys = set()

            for triple in triples:
                desc = triple.object_description
                label = triple.object_label
                triple_data_list.append({
                    "triple_id": triple.triple_id,
                    "label": label,
                    "description": desc
                })
                unique_keys.add((label, desc))

            existing_inodes = session.exec(
                select(InstanceNode.id, InstanceNode.label, InstanceNode.description)
                .where(tuple_(InstanceNode.label, InstanceNode.description).in_(unique_keys))
            ).all()

            key_to_id_map = {
                (node.label, node.description): node.id
                for node in existing_inodes
            }

            to_insert_values = []
            triple_id_to_inode_id = {}
            newly_created_in_batch = {}

            for item in triple_data_list:
                key = (item["label"], item["description"])

                if key in key_to_id_map:
                    triple_id_to_inode_id[item["triple_id"]] = key_to_id_map[key]
                elif key in newly_created_in_batch:
                    new_node = newly_created_in_batch[key]
                    triple_id_to_inode_id[item["triple_id"]] = new_node["id"]
                else:
                    inode_mmap_index = self.inode_embeddings_mmap_metadata.get('vector_count', 0)
                    new_inode_id_str = f"E{inode_mmap_index}"
                    self.inode_embeddings_mmap_metadata['vector_count'] += 1
                    new_node_value = {
                        "id": new_inode_id_str,
                        "label": item["label"],
                        "description": item["description"],
                        "embedding_index": inode_mmap_index,
                        "first_appeared": item["triple_id"],
                        "status": StatusType.UNEXPLORED,
                    }
                    to_insert_values.append(new_node_value)
                    newly_created_in_batch[key] = new_node_value
                    triple_id_to_inode_id[item["triple_id"]] = new_inode_id_str

            valid_values = [] 
            if to_insert_values:
                session.execute(
                    insert(InstanceNode).values(to_insert_values)
                )
                session.commit()
                valid_values = to_insert_values
                inode_count = session.execute(select(func.count(InstanceNode.id))).scalar_one()
                logger.info(f"{len(valid_values):,} new inodes added. Total: {inode_count}.")
            else:
                logger.info("No new inodes added (all matched existing) after NED2.")

            if valid_values:
                try:
                    labels_to_encode = [v["label"] for v in valid_values]
                    all_new_embeddings = self.stransformer_model.encode(
                        labels_to_encode, 
                        batch_size=64,
                        normalize_embeddings=True,
                        convert_to_tensor=False, 
                        show_progress_bar=True
                    ).astype(self.embedding_dtype)
                    self._ensure_mmap_capacity("inode", max(v["embedding_index"] for v in valid_values) + 1)
                    for v, emb_l in zip(valid_values, all_new_embeddings):
                        self.inode_embeddings_mmap[v["embedding_index"]] = emb_l
                    new_faiss_ids = np.array([v["embedding_index"] for v in valid_values])
                    self.inode_index.add_with_ids(all_new_embeddings, new_faiss_ids)
                    self.inode_embeddings_mmap.flush()
                    self.update_inode_data()
                except Exception as e:
                    logger.error(f"Failed to generate embeddings: {e}")
                    raise

            return triple_id_to_inode_id

        except Exception as e:
            logger.error(f"Failed to commit new inodes after NED2: {e}")
            session.rollback()
            self.load_inode_data() 
            return {}


    def _commit_new_predicates(self, predicate_new_map: dict):
        if not predicate_new_map:
            return {}
        
        with Session(self.db_engine) as session:
            try:
                triples = session.exec(
                    select(Triple)
                    .where(Triple.triple_id.in_(predicate_new_map.keys()))
                    ).all()
                if not triples:
                    logger.warning("No matching triples found for the provided predicate_new_map keys")
                    return {}

                signature_to_triple_ids = {}
                for triple in triples:
                    description = predicate_new_map[triple.triple_id]
                    signature = (triple.predicate_label, description)
                    if signature not in signature_to_triple_ids:
                        signature_to_triple_ids[signature] = []
                    signature_to_triple_ids[signature].append(triple.triple_id)
                
                existing_predicates_map = {}
                if signature_to_triple_ids:
                    results = session.exec(select(Predicate.label, Predicate.description, Predicate.id)
                                           .where(tuple_(Predicate.label, Predicate.description).in_(signature_to_triple_ids.keys()))
                                           ).all()
                    if results:
                        for label, desc, pid in results:
                            existing_predicates_map[(label, desc)] = pid
                
                values_to_insert = []
                final_triple_predicate_ids = {}

                for signature, tids in signature_to_triple_ids.items():
                    label, description = signature
                    
                    if signature in existing_predicates_map:
                        existing_id = existing_predicates_map[signature]
                        for tid in tids:
                            final_triple_predicate_ids[tid] = existing_id
                    else:
                        predicate_mmap_index = self.predicate_embeddings_mmap_metadata.get('vector_count', 0)
                        new_predicate_id_str = f"P{predicate_mmap_index}"
                        self.predicate_embeddings_mmap_metadata['vector_count'] += 1
                        values_to_insert.append({
                            "id": new_predicate_id_str,
                            "label": label,
                            "description": description,
                            "embedding_index": predicate_mmap_index,
                            "first_appeared": tids[0],
                        })
                        for tid in tids:
                            final_triple_predicate_ids[tid] = new_predicate_id_str

                insert_statement = (
                    insert(Predicate)
                    .values(values_to_insert)
                    .on_conflict_do_nothing(
                        index_elements=["label", "description"]
                    )
                )
                session.execute(insert_statement)
                session.commit()

                existing_ids = set(
                    session.exec(
                        select(Predicate.id)
                        .where(Predicate.id.in_([v["id"] for v in values_to_insert]))
                    ).all()
                )
                valid_values = [v for v in values_to_insert if v["id"] in existing_ids]
                if valid_values:
                    try:
                        labels_to_encode = [v["label"] for v in valid_values]
                        all_new_embeddings = self.stransformer_model.encode(labels_to_encode, 
                                                                            batch_size=64,
                                                                            normalize_embeddings=True,
                                                                            convert_to_tensor=False, 
                                                                            show_progress_bar=True).astype(self.embedding_dtype)
                        logger.info(f"{len(all_new_embeddings)} predicate embeddings generated.")
                    except Exception as e:
                        logger.error(f"Failed to generate embeddings during commiting new predicates : {e}")
                        raise

                    self._ensure_mmap_capacity("predicate", max(v["embedding_index"] for v in valid_values) + 1)
                    for v, emb in zip(valid_values, all_new_embeddings):
                        self.predicate_embeddings_mmap[v["embedding_index"]] = emb
                    new_faiss_ids = np.array([v["embedding_index"] for v in valid_values])
                    self.predicate_index.add_with_ids(all_new_embeddings, new_faiss_ids)
                    self.predicate_embeddings_mmap.flush()
                    self.update_predicate_data()

                    predicate_count = session.execute(select(func.count(Predicate.id))).scalar_one()
                    logger.info(f"{len(valid_values):,} new predicates to the database, total predicates: {predicate_count}.")
                    return final_triple_predicate_ids
                else:
                    return {}
    
            except Exception as e:
                logger.error(f"Failed to commit new predicates: {e}")
                session.rollback()
                self.load_predicate_data()
                return {}
    
    def _commit_new_concepts(self, concept_new_map: dict):
        
        if not concept_new_map:
            return None
        
        with Session(self.db_engine) as session:
            try:
                statement = (
                    select(InstanceTriple.original_triple_id, InstanceTriple.concept_label)
                    .where(InstanceTriple.original_triple_id.in_(concept_new_map.keys()))
                )
                required_data = session.exec(statement).all()

                if not required_data:
                    logger.warning("No matching triples found for new concepts.")
                    return None
                
                keys = {(concept_label, concept_new_map[triple_id]) for triple_id, concept_label in required_data}
                existing_concepts = {
                    (label, desc): cid for label, desc, cid in session.exec(
                        select(Concept.label, Concept.description, Concept.id)
                        .where(tuple_(Concept.label, Concept.description).in_(keys))
                    ).all()
                }

                values = []
                triple_concept_ids = {}
                new_concept_ids_by_key = {}
                for triple_id, concept_label in required_data:
                    key = (concept_label, concept_new_map[triple_id])
                    if key in existing_concepts:
                        triple_concept_ids[triple_id] = existing_concepts[key]
                        continue
                    if key in new_concept_ids_by_key:
                        triple_concept_ids[triple_id] = new_concept_ids_by_key[key]
                        continue
                    concept_mmap_index = self.concept_embeddings_mmap_metadata.get('vector_count', 0)
                    new_concept_id_str = f"C{concept_mmap_index}"
                    self.concept_embeddings_mmap_metadata['vector_count'] += 1
                    values.append({
                        "id": new_concept_id_str,
                        "label": concept_label,
                        "description": concept_new_map[triple_id],
                        "embedding_index": concept_mmap_index,
                        "first_appeared": triple_id,
                    })
                    new_concept_ids_by_key[key] = new_concept_id_str
                    triple_concept_ids[triple_id] = new_concept_id_str

                if not values:
                    logger.info(f"All {len(triple_concept_ids)} concepts after CDG already exist.")
                    return triple_concept_ids

                insert_statement = (
                    insert(Concept)
                    .values(values)
                    .on_conflict_do_nothing(
                        index_elements=["label", "description"]
                    )
                )
                session.execute(insert_statement)
                session.commit()

                existing_ids = set(
                    session.exec(
                        select(Concept.id)
                        .where(Concept.id.in_([v["id"] for v in values]))
                    ).all()
                )
                valid_values = [v for v in values if v["id"] in existing_ids]
                triple_concept_ids = {
                    tid: cid for tid, cid in triple_concept_ids.items()
                    if cid in existing_ids or cid in existing_concepts.values()
                }
                if valid_values:
                    try:
                        labels_to_encode = [v["label"] for v in valid_values]
                        all_new_embeddings = self.stransformer_model.encode(labels_to_encode, 
                                                                            batch_size=64,
                                                                            normalize_embeddings=True,
                                                                            convert_to_tensor=False, 
                                                                            show_progress_bar=True).astype(self.embedding_dtype)
                        logger.info(f"{len(all_new_embeddings)} concept embeddings generated.")
                    except Exception as e:
                        logger.error(f"Failed to generate embeddings during commiting new concepts : {e}")
                        raise

                    self._ensure_mmap_capacity("concept", max(v["embedding_index"] for v in valid_values) + 1)
                    for v, emb in zip(valid_values, all_new_embeddings):
                        self.concept_embeddings_mmap[v["embedding_index"]] = emb
                    new_faiss_ids = np.array([v["embedding_index"] for v in valid_values])
                    self.concept_index.add_with_ids(all_new_embeddings, new_faiss_ids)
                    self.concept_embeddings_mmap.flush()
                    self.update_concept_data()

                    concept_count = session.execute(select(func.count(Concept.id))).scalar_one()
                    logger.info(f"{len(valid_values):,} new concepts to the database, total concepts: {concept_count}.")
                return triple_concept_ids

            except Exception as e:
                logger.error(f"Failed to commit new concepts: {e}")
                session.rollback()
                self.load_concept_data()
                return {}
    
    def _update_triple_after_ner(self, nes, literals):
        with Session(self.db_engine) as session:
            if nes:
                try:
                    nes = set(nes)
                    session.execute(
                        update(Triple)
                        .where(Triple.triple_id.in_(nes))
                        .values(object_type=ObjectType.NE,
                                object_status=ObjectStatusType.NERFINISHED)
                    )
                except Exception as e:
                    logger.error(f"Failed to update triple objects with NEs: {e}")
            if literals:
                try:
                    literals = set(literals)
                    session.execute(
                        update(Triple)
                        .where(Triple.triple_id.in_(literals))
                        .values(object_type=ObjectType.LITERAL, 
                                object_status=ObjectStatusType.FINISHED)
                    )
                except Exception as e:
                    logger.error(f"Failed to update triple objects with literals: {e}")
            session.commit()
    
    def _update_triple_after_ned1(self, entity_existing_map, entity_new_list, entity_unknown_list):
        with Session(self.db_engine) as session:
            try:
                if entity_new_list:
                    update_statement_new = (
                        update(Triple)
                        .where(Triple.triple_id.in_(entity_new_list))
                        .values(
                            object_status=ObjectStatusType.NEDFINISHED1,
                        )
                    )
                    session.execute(update_statement_new)
                    logger.info(f"{len(entity_new_list)} triple objects are new entities after NED1.")
                
                if entity_unknown_list:
                    update_statement_unknown = (
                        update(Triple)
                        .where(Triple.triple_id.in_(entity_unknown_list))
                        .values(
                            object_status=ObjectStatusType.FINISHED,
                            object_id="unclear NED1"
                        )
                    )
                    session.execute(update_statement_unknown)
                    logger.info(f"{len(entity_unknown_list)} triple objects are unknown after NED1.")

                final_update_map = {}


                if entity_existing_map:
                    triples_for_existing = session.exec(
                        select(Triple)
                        .where(Triple.triple_id.in_(entity_existing_map.keys()))
                        .options(selectinload(Triple.ned_candidates))
                    ).all()
                    
                    for triple in triples_for_existing:
                        idx = entity_existing_map.get(triple.triple_id)
                        if idx is not None and 0 <= idx < len(triple.ned_candidates):
                            final_update_map[triple.triple_id] = triple.ned_candidates[idx].id
                
                all_triples_to_process = session.exec(
                    select(Triple).where(Triple.triple_id.in_(final_update_map.keys()))
                ).all()
                triples_map = {t.triple_id: t for t in all_triples_to_process}

                potential_new_signatures = set()
                for triple_id, new_object_id in final_update_map.items():
                    triple_obj = triples_map.get(triple_id)
                    if not triple_obj: continue
                    signature = (
                        triple_obj.subject_id, triple_obj.subject_label,
                        triple_obj.predicate_id, triple_obj.predicate_label,
                        new_object_id, triple_obj.object_label
                    )
                    potential_new_signatures.add(signature)
                
                conflict_conditions = []
                for sig_tuple in potential_new_signatures:
                    conflict_conditions.append(and_(
                        Triple.subject_id == sig_tuple[0],
                        Triple.subject_label == sig_tuple[1],
                        Triple.predicate_id == sig_tuple[2],
                        Triple.predicate_label == sig_tuple[3],
                        Triple.object_id == sig_tuple[4],
                        Triple.object_label == sig_tuple[5]
                    ))
                
                if conflict_conditions:
                    existing_signatures_query = select(
                        Triple.subject_id, Triple.subject_label,
                        Triple.predicate_id, Triple.predicate_label,
                        Triple.object_id, Triple.object_label
                    ).where(or_(*conflict_conditions))
                    existing_signatures_in_db = {
                        (r[0], r[1], r[2], r[3], r[4], r[5]) 
                        for r in session.exec(existing_signatures_query).all()
                    }
                else:
                    existing_signatures_in_db = set()
                
                signatures_seen_in_batch = {}
                triples_to_actually_update = {}
                triples_to_mark_as_duplicate = []

                sorted_triple_ids = sorted(final_update_map.keys())
                for triple_id in sorted_triple_ids:
                    triple_obj = triples_map.get(triple_id)
                    if not triple_obj: continue
                    new_object_id = final_update_map[triple_id]

                    signature = (
                        triple_obj.subject_id, triple_obj.subject_label,
                        triple_obj.predicate_id, triple_obj.predicate_label,
                        new_object_id, triple_obj.object_label
                    )
                    if signature in existing_signatures_in_db:
                        triples_to_mark_as_duplicate.append(triple_id)
                        continue
                    if signature in signatures_seen_in_batch:
                        triples_to_mark_as_duplicate.append(triple_id)
                    else:
                        signatures_seen_in_batch[signature] = triple_id
                        triples_to_actually_update[triple_id] = new_object_id


                if triples_to_actually_update:
                    update_statement = (
                        update(Triple)
                        .where(Triple.triple_id.in_(triples_to_actually_update.keys()))
                        .values(
                            object_status=ObjectStatusType.FINISHED,
                            object_id=case(
                                triples_to_actually_update,
                                value=Triple.triple_id
                            )
                        )
                    )
                    session.execute(update_statement)
                    logger.info(f"Updated {len(triples_to_actually_update)} triple to existing entities.")
                
                if triples_to_mark_as_duplicate:
                    session.execute(
                        delete(Triple).where(Triple.triple_id.in_(triples_to_mark_as_duplicate))
                    )
                    logger.info(f"Deleted {len(triples_to_mark_as_duplicate)} triple duplication.")


                session.commit()
            except Exception as e:
                logger.warning(f"Failed to update triple objects to new/existing entities: {e}")
                session.rollback()
    
    def _update_triple_after_nedg(self, entity_new_map):
        if not entity_new_map:
            return None

        with Session(self.db_engine) as session:
            try:
                triples_update = session.exec(
                        select(Triple)
                        .where(Triple.triple_id.in_(entity_new_map.keys()))
                    ).all()

                for t in triples_update:
                    try:
                        t.object_description = entity_new_map[t.triple_id]
                        t.object_status = ObjectStatusType.DGFINISHED
                        session.add(t)
                    except IntegrityError:
                        session.rollback()
                        session.delete(t)
                session.commit()
                logger.info(f"Updated {len(entity_new_map)} triple with object description after NEDG.")
            except Exception as e:
                logger.warning(f"Failed to update triple object description after NEDG: {e}")
                session.rollback()
    
    def _update_triple_after_ned2(self, entity_existing_map, triple_object_ids):

        with Session(self.db_engine) as session:
            try:
                if triple_object_ids:
                    triples_update = session.exec(
                        select(Triple)
                        .where(Triple.triple_id.in_(triple_object_ids.keys()))
                        ).all()

                    taken_signatures = self._find_existing_triple_signatures(
                        session, {self._triple_signature(t, triple_object_ids[t.triple_id]) for t in triples_update}
                    )
                    num_duplicates = 0
                    for t in sorted(triples_update, key=lambda t: t.triple_id):
                        signature = self._triple_signature(t, triple_object_ids[t.triple_id])
                        if signature in taken_signatures:
                            session.delete(t)
                            num_duplicates += 1
                        else:
                            t.object_id = triple_object_ids[t.triple_id]
                            t.object_status = ObjectStatusType.FINISHED
                            session.add(t)
                            taken_signatures.add(signature)
                    logger.info(f"Updated {len(triples_update) - num_duplicates} triple to new entities, {num_duplicates} duplicates deleted.")
                
                final_update_map = {}
                if entity_existing_map:
                    triples_for_existing = session.exec(
                        select(Triple)
                        .where(Triple.triple_id.in_(entity_existing_map.keys()))
                        .options(selectinload(Triple.ned2_candidates))
                    ).all()
                    
                    for triple in triples_for_existing:
                        idx = entity_existing_map.get(triple.triple_id)
                        if idx is not None and 0 <= idx < len(triple.ned2_candidates):
                            final_update_map[triple.triple_id] = triple.ned2_candidates[idx].id
                
                all_triples_to_process = session.exec(
                    select(Triple).where(Triple.triple_id.in_(final_update_map.keys()))
                ).all()
                triples_map = {t.triple_id: t for t in all_triples_to_process}

                potential_new_signatures = set()
                for triple_id, new_object_id in final_update_map.items():
                    triple_obj = triples_map.get(triple_id)
                    if not triple_obj: continue
                    signature = (
                        triple_obj.subject_id, triple_obj.subject_label,
                        triple_obj.predicate_id, triple_obj.predicate_label,
                        new_object_id, triple_obj.object_label
                    )
                    potential_new_signatures.add(signature)
                
                conflict_conditions = []
                for sig_tuple in potential_new_signatures:
                    conflict_conditions.append(and_(
                        Triple.subject_id == sig_tuple[0],
                        Triple.subject_label == sig_tuple[1],
                        Triple.predicate_id == sig_tuple[2],
                        Triple.predicate_label == sig_tuple[3],
                        Triple.object_id == sig_tuple[4],
                        Triple.object_label == sig_tuple[5]
                    ))
                
                if conflict_conditions:
                    existing_signatures_query = select(
                        Triple.subject_id, Triple.subject_label,
                        Triple.predicate_id, Triple.predicate_label,
                        Triple.object_id, Triple.object_label
                    ).where(or_(*conflict_conditions))
                    existing_signatures_in_db = {
                        (r[0], r[1], r[2], r[3], r[4], r[5]) 
                        for r in session.exec(existing_signatures_query).all()
                    }
                else:
                    existing_signatures_in_db = set()
                
                signatures_seen_in_batch = {}
                triples_to_actually_update = {}
                triples_to_mark_as_duplicate = []

                sorted_triple_ids = sorted(final_update_map.keys())
                for triple_id in sorted_triple_ids:
                    triple_obj = triples_map.get(triple_id)
                    if not triple_obj: continue
                    new_object_id = final_update_map[triple_id]

                    signature = (
                        triple_obj.subject_id, triple_obj.subject_label,
                        triple_obj.predicate_id, triple_obj.predicate_label,
                        new_object_id, triple_obj.object_label
                    )
                    if signature in existing_signatures_in_db:
                        triples_to_mark_as_duplicate.append(triple_id)
                        continue
                    if signature in signatures_seen_in_batch:
                        triples_to_mark_as_duplicate.append(triple_id)
                    else:
                        signatures_seen_in_batch[signature] = triple_id
                        triples_to_actually_update[triple_id] = new_object_id

                if triples_to_actually_update:
                    update_statement = (
                        update(Triple)
                        .where(Triple.triple_id.in_(triples_to_actually_update.keys()))
                        .values(
                            object_status=ObjectStatusType.FINISHED,
                            object_id=case(
                                triples_to_actually_update,
                                value=Triple.triple_id
                            )
                        )
                    )
                    session.execute(update_statement)
                    logger.info(f"Updated {len(triples_to_actually_update)} triple to existing entities.")
                
                if triples_to_mark_as_duplicate:
                    session.execute(
                        delete(Triple).where(Triple.triple_id.in_(triples_to_mark_as_duplicate))
                    )
                    logger.info(f"Deleted {len(triples_to_mark_as_duplicate)} triple duplication.")

                session.commit()
            except Exception as e:
                logger.warning(f"Failed to update triple objects to new/existing entities after NED2: {e}")
                session.rollback()

    
    def _update_triple_after_pd(self, predicate_existing_map, predicate_new_list):
        with Session(self.db_engine) as session:
            try:
                if predicate_new_list:
                    statement_new = (
                        update(Triple)
                        .where(Triple.triple_id.in_(predicate_new_list))
                        .values(predicate_status=PredicateStatusType.PDFINISHED)
                        )
                    session.execute(statement_new)

                if predicate_existing_map:
                    triples_to_update = session.exec(
                        select(Triple).where(Triple.triple_id.in_(predicate_existing_map.keys()))
                        .options(selectinload(Triple.pd_candidates))
                    ).all()
                    instanceof_id = session.exec(select(Predicate.id).where(Predicate.label == "instanceOf")).first()
                    update_map = {}
                    new_instance_triples = []
                    invalid_choice_ids = []
                    for triple in triples_to_update:
                        idx = predicate_existing_map.get(triple.triple_id)
                        if idx is None or not 0 <= idx < len(triple.pd_candidates):
                            invalid_choice_ids.append(triple.triple_id)
                            continue
                        predicate_id = triple.pd_candidates[idx].id
                        update_map[triple.triple_id] = predicate_id
                        if predicate_id == instanceof_id:
                            new_instance_triples.append(triple)

                    if invalid_choice_ids:
                        session.execute(
                            update(Triple)
                            .where(Triple.triple_id.in_(invalid_choice_ids))
                            .values(predicate_status=PredicateStatusType.GENERATED)
                        )
                        logger.warning(f"{len(invalid_choice_ids)} triples got an out-of-range PD answer, reset for retry.")

                    if update_map:
                        statement_existing = (
                            update(Triple)
                            .where(Triple.triple_id.in_(update_map.keys()))
                            .values(
                                predicate_status=PredicateStatusType.FINISHED,
                                predicate_id=case(
                                    update_map,
                                    value=Triple.triple_id
                                )
                            )
                        )
                        session.execute(statement_existing)
                        session.commit()
                        logger.info(f"Updated {len(update_map)} triples' predicate id to existing ones after PD.")

                        if new_instance_triples:
                            values = []
                            new_instance_triple_ids = [triple.triple_id for triple in new_instance_triples]
                            for triple in new_instance_triples:
                                values.append({
                                    "original_triple_id": triple.triple_id,
                                    "entity_id": triple.subject_id,
                                    "concept_label": triple.object_label,
                                    "status": ConceptStatusType.GENERATED,
                                })
                            insert_statement = (
                                insert(InstanceTriple)
                                .values(values)
                                .on_conflict_do_nothing(
                                    index_elements=["original_triple_id"]
                                    )
                            )
                            session.execute(insert_statement)

                            instance_statement_existing = (
                                update(Triple)
                                .where(Triple.triple_id.in_(new_instance_triple_ids))
                                .values(
                                    object_status=ObjectStatusType.FINISHED,
                                    object_type=ObjectType.CONCEPT,
                                )
                            )
                            session.execute(instance_statement_existing)
                            instance_triple_count = session.execute(select(func.count(InstanceTriple.original_triple_id))).scalar_one()
                            logger.info(f"{len(new_instance_triple_ids)} new instance triples to database. Total instance triples: {instance_triple_count}.")
                        
                session.commit()
            except Exception as e:
                logger.warning(f"Failed to update triple predicates to new/existing predicates: {e}")
                session.rollback()
    
    def _update_triple_after_pdg(self, triple_predicate_ids):

        with Session(self.db_engine) as session:
            try:
                update_statement = (
                    update(Triple)
                    .where(Triple.triple_id.in_(triple_predicate_ids.keys()))
                    .values(
                        predicate_status=PredicateStatusType.FINISHED,
                        predicate_id=case(
                            triple_predicate_ids,
                            value=Triple.triple_id
                        )
                    )
                )
                session.execute(update_statement)
                logger.info(f"Updated {len(triple_predicate_ids)} triple with new predicates after PDG.")
                session.commit()
            except Exception as e:
                logger.warning(f"Failed to update triple with new predicates after PDG: {e}")
                session.rollback()
    

    def _update_instance_triple_after_cd(self, concept_existing_map, concept_new_list):

        with Session(self.db_engine) as session:
            try:
                if concept_new_list:
                    statement_new = (
                        update(InstanceTriple)
                        .where(InstanceTriple.original_triple_id.in_(concept_new_list))
                        .values(status=ConceptStatusType.CDFINISHED)
                        )
                    session.execute(statement_new)

                if concept_existing_map:
                    triples_to_update = session.exec(
                        select(InstanceTriple).where(InstanceTriple.original_triple_id.in_(concept_existing_map.keys()))
                        .options(selectinload(InstanceTriple.cd_candidates))
                    ).all()
                    update_map = {}
                    invalid_choice_ids = []
                    for triple in triples_to_update:
                        idx = concept_existing_map.get(triple.original_triple_id)
                        if idx is None or not 0 <= idx < len(triple.cd_candidates):
                            invalid_choice_ids.append(triple.original_triple_id)
                            continue
                        update_map[triple.original_triple_id] = triple.cd_candidates[idx].id

                    if invalid_choice_ids:
                        session.execute(
                            update(InstanceTriple)
                            .where(InstanceTriple.original_triple_id.in_(invalid_choice_ids))
                            .values(status=ConceptStatusType.GENERATED)
                        )
                        logger.warning(f"{len(invalid_choice_ids)} instance triples got an out-of-range CD answer, reset for retry.")
                    if update_map:
                        statement_existing = (
                            update(InstanceTriple)
                            .where(InstanceTriple.original_triple_id.in_(update_map.keys()))
                            .values(
                                status=ConceptStatusType.FINISHED,
                                concept_id=case(
                                    update_map,
                                    value=InstanceTriple.original_triple_id
                                )
                            )
                        )
                        session.execute(statement_existing)
                        logger.info(f"Updated {len(update_map)} instance triples' concept id to existing ones.")
                session.commit()
            except Exception as e:
                logger.warning(f"Failed to update triple concepts to existing concepts after CD: {e}")
                session.rollback()
    
    def _update_instance_triple_after_cdg(self, triple_concept_ids):

        with Session(self.db_engine) as session:
            try:
                update_statement = (
                    update(InstanceTriple)
                    .where(InstanceTriple.original_triple_id.in_(triple_concept_ids.keys()))
                    .values(
                        status=ConceptStatusType.FINISHED,
                        concept_id=case(
                            triple_concept_ids,
                            value=InstanceTriple.original_triple_id
                        )
                    )
                )
                session.execute(update_statement)
                logger.info(f"Updated {len(triple_concept_ids)} instance triple with new concepts after CDG.")
                session.commit()
            except Exception as e:
                logger.warning(f"Failed to update instance triple with new concepts after CDG: {e}")
                session.rollback()

    def _get_unexplored_inodes_batch(self, max_batch_size: int, number_of_batches:int = 3):
        max_number = max_batch_size * number_of_batches
        with Session(self.db_engine) as session:
            unexplored_inodes = session.exec(
                select(InstanceNode)
                .where(
                    InstanceNode.status == StatusType.UNEXPLORED,
                )
                .limit(max_number)
            ).all()
            if len(unexplored_inodes) == 0:
                return []

            batches = [unexplored_inodes[i:i + max_batch_size] for i in
                       range(0, len(unexplored_inodes), max_batch_size)]

            logger.info(
                f"Prepared {len(batches)} batches for {sum(len(b) for b in batches)} named entities to explore.")
        return batches
    
    def _get_ner_triples_batch(self, max_batch_size: int, number_of_batches: int = 3):
        max_number = max_batch_size * number_of_batches
        with Session(self.db_engine) as session:
            after_pd_status = [PredicateStatusType.PDFINISHED, PredicateStatusType.ONDG, PredicateStatusType.FINISHED]
            triples_ner = session.exec(
                select(Triple)
                .where(
                    Triple.object_type == ObjectType.UNRECOGNIZED,
                    Triple.object_status == ObjectStatusType.GENERATED,
                    Triple.predicate_status.in_(after_pd_status),
                )
                .limit(max_number)
            ).all()

            if len(triples_ner) == 0:
                return []

            batches = [triples_ner[i:i + max_batch_size] for i in
                       range(0, len(triples_ner), max_batch_size)]

            logger.info(
                f"Prepared {len(batches)} batches for {sum(len(b) for b in batches)} triples for NER.")
        return batches
    
    def _get_ned2_triples_batch(self, max_number, max_batch_size: int, number_of_batches: int = 3):
        with Session(self.db_engine) as session:
            triples_ned2 = session.exec(
                select(Triple)
                .where(
                    Triple.object_type == ObjectType.NE,
                    Triple.object_status == ObjectStatusType.DGFINISHED,
                    Triple.object_description.is_not(None)
                )
                .limit(max_number)
            ).all()
            if len(triples_ned2) == 0:
                return []
            
            batches = [triples_ned2[i:i + max_batch_size] for i in
                       range(0, len(triples_ned2), max_batch_size)]
            logger.info(f"Prepared {len(batches)} batches of {len(triples_ned2)} triples for NED2.")
        
        return batches

    def _get_ned1_triples_batch(self, max_number: int, max_batch_size:int, number_of_batches: int = 3, batchwise=True):
        with Session(self.db_engine) as session:
            triples_ned = session.exec(
                select(Triple)
                .where(
                    Triple.object_type == ObjectType.NE,
                    Triple.object_status == ObjectStatusType.NERFINISHED,
                )
                .limit(max_number)
            ).all()
            if len(triples_ned) == 0:
                return []
            

            unique_pairs_to_check = list({(triple.predicate_label, triple.object_label) for triple in triples_ned})
            mapping_cache = {}
            
            if unique_pairs_to_check:
                chunk_size = 500
                for i in tqdm(range(0, len(unique_pairs_to_check), chunk_size)):
                    chunk = unique_pairs_to_check[i:i + chunk_size]
                    
                    mapping_query = (
                        select(
                            Triple.predicate_label,
                            Triple.object_label,
                            func.count(Triple.triple_id).label("count"),
                            func.min(Triple.object_id).label("object_id"),
                        )
                        .where(
                            Triple.object_type == ObjectType.NE,
                            Triple.object_id.is_not(None),
                            tuple_(Triple.predicate_label, Triple.object_label).in_(chunk)
                        )
                        .group_by(Triple.predicate_label, Triple.object_label)
                        .having(and_(
                            func.count(Triple.triple_id) > 50,
                            func.min(Triple.object_id) == func.max(Triple.object_id) 
                        ))
                    )
                    
                    results = session.exec(mapping_query).all()
                    for row in results:
                        mapping_cache[(row.predicate_label, row.object_label)] = row.object_id

            triples_ned_cached = 0
            triples_not_cached = []
            triples_to_cache = []

            for triple in triples_ned:
                cached_object_id = mapping_cache.get((triple.predicate_label, triple.object_label))
                if cached_object_id:
                    triples_to_cache.append((triple, str(cached_object_id)))
                else:
                    triples_not_cached.append(triple)

            taken_signatures = self._find_existing_triple_signatures(
                session, {self._triple_signature(t, object_id) for t, object_id in triples_to_cache}
            )
            num_duplicates = 0
            for triple, object_id in triples_to_cache:
                signature = self._triple_signature(triple, object_id)
                if signature in taken_signatures:
                    session.delete(triple)
                    num_duplicates += 1
                else:
                    triple.object_id = object_id
                    triple.object_status = ObjectStatusType.FINISHED
                    session.add(triple)
                    taken_signatures.add(signature)
                    triples_ned_cached += 1

            session.commit()
            logger.info(f"{triples_ned_cached} triples cached before NED1, {num_duplicates} duplicates deleted.")

            triples_list = []
            labels_to_check = {
                triple.object_label
                for triple in triples_not_cached
            }
            ned_count_query = (
                select(
                    Triple.object_label,
                    func.count(Triple.triple_id).label("ned_count")
                )
                .where(

                    Triple.ned1_batch_id.is_not(None) if batchwise
                    else and_(Triple.object_type == ObjectType.NE, Triple.object_id.is_not(None)),
                    Triple.object_label.in_(labels_to_check)
                )
                .group_by(Triple.object_label)
            )
            ned_count_results = session.exec(ned_count_query).all()
            ned_count_by_object_label = {
                row.object_label: row.ned_count
                for row in ned_count_results
            }

            inode_count = session.execute(select(func.count(InstanceNode.id))).scalar_one()
            object_label_to_triple_ids = defaultdict(list)
            for t in triples_not_cached:
                object_label_to_triple_ids[t.object_label].append(t)

            if not object_label_to_triple_ids:
                logger.info("All found triples were cached. No NED1.")
                return []

            labels_to_embedd = list(object_label_to_triple_ids.keys())
            object_embeddings = self.stransformer_model.encode(labels_to_embedd,
                                                               batch_size=64,
                                                               convert_to_tensor=False,
                                                               normalize_embeddings=True,)

            if len(labels_to_embedd) == 1:
                cluster_assignment = [0]
            else:
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    metric='cosine',
                    linkage='average',
                    distance_threshold=0.15
                )
                clustering.fit(object_embeddings)
                cluster_assignment = clustering.labels_
            cluster_map = defaultdict(list)
            for label, cluster_id in zip(labels_to_embedd, cluster_assignment):
                cluster_map[cluster_id].append(label)
            for grouped_object_labels in cluster_map.values():
                triples_to_insert = object_label_to_triple_ids.get(grouped_object_labels[0], [])
                if ned_count_by_object_label.get(grouped_object_labels[0], 0) > 50:
                    subject_id_set = set()
                    for t in triples_to_insert:
                        if t.subject_id not in subject_id_set:
                            triples_list.append(t)
                            subject_id_set.add(t.subject_id)
                else:
                    triples_list.append(triples_to_insert[0])
            
            batches = [triples_list[i:i + max_batch_size] for i in
                       range(0, len(triples_list), max_batch_size)]

            logger.info(f"Prepared {len(batches)} batches of {len(triples_list)} triples for NED1.")
        return batches
    
    def _get_pd_triples_batch(self, max_number: int, max_batch_size:int, number_of_batches: int = 3):
        with Session(self.db_engine) as session:
            triples_without_pid = session.exec(
                select(Triple)
                .where(
                    Triple.predicate_status == PredicateStatusType.GENERATED,
                    )
                .limit(max_number)
            ).all()
            if len(triples_without_pid) == 0:
                return []
            
            unique_labels = {t.predicate_label for t in triples_without_pid}

            predicate_stats = session.exec(
                select(Triple.predicate_label,
                       Triple.predicate_id)
                .where(
                    Triple.predicate_label.in_(unique_labels),
                    Triple.predicate_id.is_not(None)
                    )
                ).all()
            
            auto_map = {
                row.predicate_label: row.predicate_id 
                for row in predicate_stats 
                if len(row.predicate_label) > 1
                }

            if auto_map:
                triples_to_map = [
                    t for t in triples_without_pid
                    if t.predicate_label in auto_map
                ]

                check_conditions = [
                    tuple_(t.subject_id, auto_map[t.predicate_label], t.object_label) 
                    for t in triples_to_map
                    ]

                collision_set = set()
                if check_conditions:
                    for i in tqdm(range(0, len(check_conditions), 500)):
                        chunk = check_conditions[i:i+500]
                        existing = session.exec(
                            select(Triple.subject_id, Triple.predicate_id, Triple.object_label)
                            .where(tuple_(Triple.subject_id, Triple.predicate_id, Triple.object_label).in_(chunk))
                        ).all()
                        collision_set.update(existing)
                

                triples_pd_cached_count = 0
                for t in triples_to_map:
                    new_pid = auto_map[t.predicate_label]
                    future_state = (t.subject_id, new_pid, t.object_label)
                    
                    if future_state in collision_set:
                        session.delete(t)
                    else:
                        t.predicate_id = new_pid
                        t.predicate_status = PredicateStatusType.FINISHED
                        session.add(t)
                        collision_set.add(future_state)
                        triples_pd_cached_count += 1

                session.commit()
                logger.info(f"Predicate Cache: {triples_pd_cached_count} mapped, {len(triples_to_map) - triples_pd_cached_count} deleted due to collision.")
            else:
                logger.info(f"No PD cache.")

            triples_pd = [
                t for t in triples_without_pid
                if t.predicate_label not in auto_map
            ]
            if not triples_pd:
                logger.info("All found triples' predicates were cached. No predicate to disambiguate.")
                return []
            triples_list = []
            triples_set = set()
            predicate_count = session.execute(select(func.count(Predicate.id))).scalar_one()
            for triple in triples_pd:
                if predicate_count > 300:
                    label_to_check = str(triple.predicate_label).lower().strip()
                else:
                    label_to_check = str(triple.predicate_label)[:5].lower().strip()
                if label_to_check not in triples_set:
                    triples_list.append(triple)
                    triples_set.add(label_to_check)
            batches = [triples_list[i:i + max_batch_size] for i in
                       range(0, len(triples_list), max_batch_size)]
            logger.info(f"Prepared {len(batches)} batches of {len(triples_list)} triples for PD.")
        
        return batches
    
    def _get_cd_instance_triples_batch(self, max_number: int, max_batch_size:int, number_of_batches: int = 3):
        with Session(self.db_engine) as session:
            instance_triples_cd = session.exec(
                select(InstanceTriple)
                .where(
                    InstanceTriple.status == ConceptStatusType.GENERATED,
                    )
                .limit(max_number)
            ).all()
            if len(instance_triples_cd) == 0:
                return []

            unique_labels = {t.concept_label for t in instance_triples_cd}

            concept_stats = session.exec(
                select(InstanceTriple.concept_label,
                       InstanceTriple.concept_id)
                .where(
                    InstanceTriple.concept_label.in_(unique_labels),
                    InstanceTriple.concept_id.is_not(None)
                    )
                ).all()

            auto_map = {row.concept_label: row.concept_id for row in concept_stats}

            if auto_map:

                triples_to_map = [
                    t for t in instance_triples_cd
                    if t.concept_label in auto_map
                ]

                relevant_entities = {t.entity_id for t in triples_to_map}
                existing_pairs_query = select(InstanceTriple.entity_id, InstanceTriple.concept_id).where(
                    InstanceTriple.entity_id.in_(relevant_entities),
                    InstanceTriple.concept_id.is_not(None)
                )
                collision_set = set(session.exec(existing_pairs_query).all())

                update_data = []
                to_delete_ids = []

                for t in triples_to_map:
                    new_cid = auto_map.get(t.concept_label)
                    future_pair = (t.entity_id, new_cid)
                    
                    if future_pair in collision_set:
                        to_delete_ids.append(t.original_triple_id)
                    else:
                        update_data.append({
                            "original_triple_id": t.original_triple_id,
                            "concept_id": new_cid,
                            "status": ConceptStatusType.FINISHED
                        })
                        collision_set.add(future_pair)

                if to_delete_ids:
                    session.execute(
                        delete(InstanceTriple).where(InstanceTriple.original_triple_id.in_(to_delete_ids))
                    )
                    session.execute(
                        delete(Triple).where(Triple.triple_id.in_(to_delete_ids))
                    )
                if update_data:
                    session.bulk_update_mappings(InstanceTriple, update_data)

                session.commit()
                logger.info(f"Cache CD for {len(update_data)} triples. {len(to_delete_ids)} triple(s) deleted due to collision.")
            else:
                logger.info(f"No CD cache.")

            triples_cd = [
                t for t in instance_triples_cd
                if t.concept_label not in auto_map
            ]
            if not triples_cd:
                logger.info("All found triples' concepts were cached. No CD.")
                return []
            triples_list = []
            triples_set = set()
            concept_count = session.execute(select(func.count(Concept.id))).scalar_one()
            for triple in triples_cd:
                if concept_count > 100:
                    label_to_check = str(triple.concept_label).lower().strip()
                else:
                    label_to_check = str(triple.concept_label)[:3].lower().strip()
                if label_to_check not in triples_set:
                    triples_list.append(triple)
                    triples_set.add(label_to_check)
            batches = [triples_list[i:i + max_batch_size] for i in
                       range(0, len(triples_list), max_batch_size)]
            logger.info(f"Prepared {len(batches)} batches of {len(triples_list)} instance triples for CD.")
        
        return batches

    def _get_nedg_triples_batch(self, max_number: int, max_batch_size:int, number_of_batches: int = 3):
        with Session(self.db_engine) as session:
            triples_nedg = session.exec(
                select(Triple)
                .where(
                    Triple.object_type == ObjectType.NE,
                    Triple.object_status == ObjectStatusType.NEDFINISHED1,
                )
                .limit(max_number)
            ).all()

            if len(triples_nedg) == 0:
                return []

            batches = [triples_nedg[i:i + max_batch_size] for i in
                       range(0, len(triples_nedg), max_batch_size)]
            logger.info(f"Prepared {len(batches)} batches of {len(triples_nedg)} triples for NEDG.")
        return batches
    
    def _get_pdg_triples_batch(self, max_number: int, max_batch_size:int, number_of_batches: int = 3):
        with Session(self.db_engine) as session:
            triples_pdg = session.exec(
                select(Triple)
                .where(
                    Triple.predicate_status == PredicateStatusType.PDFINISHED,
                    )
                .limit(max_number)
            ).all()

            if len(triples_pdg) == 0:
                return []
            batches = [triples_pdg[i:i + max_batch_size] for i in
                       range(0, len(triples_pdg), max_batch_size)]
            logger.info(f"Prepared {len(batches)} batches of {len(triples_pdg)} triples for PDG.")
        
        return batches
    
    def _get_cdg_instance_triples_batch(self, max_number: int, max_batch_size:int, number_of_batches: int = 3):
        with Session(self.db_engine) as session:
            instance_triples_cdg = session.exec(
                select(InstanceTriple)
                .where(
                    InstanceTriple.status == ConceptStatusType.CDFINISHED,
                    )
                .limit(max_number)
            ).all()

            if len(instance_triples_cdg) == 0:
                return []

            batches = [instance_triples_cdg[i:i + max_batch_size] for i in
                       range(0, len(instance_triples_cdg), max_batch_size)]
            logger.info(f"Prepared {len(batches)} batches of {len(instance_triples_cdg)} instance triples for CDG.")
        
        return batches
    

    def _reset_items_of_cancelled_batches(self):
        batch_id_column = {
            JobType.ELICITATION.value: InstanceNode.explored_batch_id,
            JobType.NAMED_ENTITY_RECOGNITION.value: Triple.ner_batch_id,
            JobType.NAMED_ENTITY_DISAMBIGUATION_SOURCE_TRIPLE.value: Triple.ned1_batch_id,
            JobType.NAMED_ENTITY_DESCRIPTION_GEN.value: Triple.nedg_batch_id,
            JobType.NAMED_ENTITY_DISAMBIGUATION_DESCRIPTION.value: Triple.ned2_batch_id,
            JobType.PREDICATE_DISAMBIGUATION.value: Triple.pd_batch_id,
            JobType.PREDICATE_DESCRIPTION_GEN.value: Triple.pdg_batch_id,
            JobType.CONCEPT_DISAMBIGUATION.value: InstanceTriple.cd_batch_id,
            JobType.CONCEPT_DESCRIPTION_GEN.value: InstanceTriple.cdg_batch_id,
        }
        handlers = {
            JobType.ELICITATION.value: self._process_one_completed_elicitation_batch,
            JobType.NAMED_ENTITY_RECOGNITION.value: self._process_one_completed_ner_batch,
            JobType.NAMED_ENTITY_DISAMBIGUATION_SOURCE_TRIPLE.value: self._process_one_completed_ned1_batch,
            JobType.NAMED_ENTITY_DESCRIPTION_GEN.value: self._process_one_completed_nedg_batch,
            JobType.NAMED_ENTITY_DISAMBIGUATION_DESCRIPTION.value: self._process_one_completed_ned2_batch,
            JobType.PREDICATE_DISAMBIGUATION.value: self._process_one_completed_pd_batch,
            JobType.PREDICATE_DESCRIPTION_GEN.value: self._process_one_completed_pdg_batch,
            JobType.CONCEPT_DISAMBIGUATION.value: self._process_one_completed_cd_batch,
            JobType.CONCEPT_DESCRIPTION_GEN.value: self._process_one_completed_cdg_batch,
        }
        with Session(self.db_engine) as session:
            cancelled = session.exec(select(Batch.id, Batch.job_type).where(Batch.status == "cancelled")).all()
            stuck = [
                (batch_id, job_type) for batch_id, job_type in cancelled
                if job_type in batch_id_column
                and session.exec(select(batch_id_column[job_type]).where(batch_id_column[job_type] == batch_id).limit(1)).first()
            ]
        if not stuck:
            return
        logger.info(f"{len(stuck)} cancelled batches still hold items, moving them back to their previous status.")
        for batch_id, job_type in stuck:
            with self._stage(f"Resetting items of cancelled {job_type} batch {batch_id}"):
                handlers[job_type](SimpleNamespace(id=batch_id, status="cancelled"))

    @contextmanager
    def _stage(self, name: str):
        try:
            yield
        except Exception:
            self._round_had_error = True
            logger.exception(f"{name} stage failed in this round, continuing with the next stage.")

    def loop(
            self,
            max_nes_explored: int,
            batchwise: bool = True,
            max_batch_size: int = None,
            max_queue_size: int = None,
            max_ned_triples_iter: int = None,
            max_pd_triples_iter: int = None,
            max_cd_triples_iter: int = None,
            max_nedg_triples_iter: int = None,
            max_pdg_triples_iter: int = None,
            max_cdg_triples_iter: int = None,
            single_chunk_size: int = 20,
            single_fetch_size: int = 500,
            max_idle_rounds: int = 3,
    ):

        if batchwise:
            batch_params = {
                "max_batch_size": max_batch_size,
                "max_queue_size": max_queue_size,
                "max_ned_triples_iter": max_ned_triples_iter,
                "max_pd_triples_iter": max_pd_triples_iter,
                "max_cd_triples_iter": max_cd_triples_iter,
                "max_nedg_triples_iter": max_nedg_triples_iter,
                "max_pdg_triples_iter": max_pdg_triples_iter,
                "max_cdg_triples_iter": max_cdg_triples_iter,
            }
            missing = [name for name, value in batch_params.items() if value is None]
            if missing:
                raise ValueError(f"Batch mode requires: {', '.join(missing)}")
            number_of_batches = 5
            max_ned2_triples_iter = max_ned_triples_iter
            elicitation_batch_size, elicitation_number_of_batches = 300, 30
            ner_batch_size, ner_number_of_batches = 1000, 10
        else:
            max_batch_size = single_chunk_size
            max_queue_size = math.inf
            max_ned_triples_iter = max_pd_triples_iter = max_cd_triples_iter = single_fetch_size
            max_ned2_triples_iter = single_chunk_size
            max_nedg_triples_iter = max_pdg_triples_iter = max_cdg_triples_iter = single_chunk_size
            number_of_batches = 1
            elicitation_batch_size, elicitation_number_of_batches = single_chunk_size, 1
            ner_batch_size, ner_number_of_batches = single_chunk_size, 1

        backends = self.batch_backends if batchwise else self.single_backends
        openai_roles = [role for role, backend in backends.items() if backend.client is None]
        if openai_roles:
            config_name = "batch_backends" if batchwise else "single_backends"
            raise ValueError(f"Roles {openai_roles} have no backend config and would use OpenAI: set OPENAI_API_KEY "
                             f"or configure them in {config_name}.")

        mode = "batch" if batchwise else "single-request"
        logger.info(f"Start the GPT-KBC runner loop in {mode} mode, goal: {max_nes_explored} inodes to explore.")

        num_nes_explored = 0

        completed_batch_handlers = {
            JobType.NAMED_ENTITY_RECOGNITION.value:                  self._process_one_completed_ner_batch,
            JobType.NAMED_ENTITY_DISAMBIGUATION_SOURCE_TRIPLE.value: self._process_one_completed_ned1_batch,
            JobType.NAMED_ENTITY_DESCRIPTION_GEN.value:              self._process_one_completed_nedg_batch,
            JobType.NAMED_ENTITY_DISAMBIGUATION_DESCRIPTION.value:   self._process_one_completed_ned2_batch,
            JobType.PREDICATE_DISAMBIGUATION.value:                  self._process_one_completed_pd_batch,
            JobType.PREDICATE_DESCRIPTION_GEN.value:                 self._process_one_completed_pdg_batch,
            JobType.CONCEPT_DISAMBIGUATION.value:                    self._process_one_completed_cd_batch,
            JobType.CONCEPT_DESCRIPTION_GEN.value:                   self._process_one_completed_cdg_batch,
        }

        with self._stage("Resetting items of cancelled batches"):
            self._reset_items_of_cancelled_batches()

        idle_rounds = 0
        while True:

            self._round_had_error = False
            outstanding_batch_ids, finished_batches, if_NED, if_PD, if_CD = self._check_batch_queue()

            did_work = bool(outstanding_batch_ids) or any(finished_batches.values())

            finished_elicitation_batches = finished_batches[JobType.ELICITATION.value]
            if len(finished_elicitation_batches) > 0:
                for batch in finished_elicitation_batches:
                    with self._stage(f"Processing elicitation batch {batch.id}"):
                        num_nes_explored += self._process_one_completed_elicitation_batch(batch)
                logger.info(f"{num_nes_explored} inodes explored so far.")

            for job_type, handler in completed_batch_handlers.items():
                for batch in finished_batches[job_type]:
                    with self._stage(f"Processing {job_type} batch {batch.id}"):
                        handler(batch)

            outstanding_sum = len(outstanding_batch_ids)
            batch_quota = max_queue_size - outstanding_sum
            if batchwise:
                logger.info(f"{batch_quota} batches quota left.")

            if if_NED:
                with self._stage("NED status reset"), Session(self.db_engine) as session:
                    statement1 = (
                        update(Triple)
                        .where(Triple.object_status == ObjectStatusType.ONNED1,
                            Triple.object_type==ObjectType.NE)
                        .values(
                            object_status=ObjectStatusType.NERFINISHED
                        )
                    )
                    session.execute(statement1)
                    statement2 = (
                        update(Triple)
                        .where(Triple.object_status == ObjectStatusType.ONDG,
                            Triple.object_type==ObjectType.NE)
                        .values(
                            object_status=ObjectStatusType.NEDFINISHED1
                        )
                    )
                    session.execute(statement2)
                    statement3 = (
                        update(Triple)
                        .where(Triple.object_status == ObjectStatusType.ONNED2,
                            Triple.object_type==ObjectType.NE)
                        .values(
                            object_status=ObjectStatusType.DGFINISHED
                        )
                    )
                    session.execute(statement3)
                    session.commit()

            if batch_quota > 0:
                with self._stage("NEDG"):
                    if nedg_batches := self._get_nedg_triples_batch(max_number=max_nedg_triples_iter, max_batch_size=max_batch_size, number_of_batches=number_of_batches):
                        did_work = True
                        if_NED = False
                        for nedg_batch in nedg_batches:
                            self._create_nedg_batches(nedg_batch, poll_interval=2, batchwise=batchwise)
                            batch_quota -= 1
            if batch_quota > 0:
                with self._stage("NED2"):
                    if ned2_batches := self._get_ned2_triples_batch(max_number=max_ned2_triples_iter, max_batch_size=max_batch_size, number_of_batches=number_of_batches):
                        did_work = True
                        if_NED = False
                        for ned_batch in ned2_batches:
                            self._create_ned2_batches(ned_batch, poll_interval=2, batchwise=batchwise)
            if batch_quota > 0:
                if if_NED:
                    with self._stage("NED1"):
                        if ned1_batches := self._get_ned1_triples_batch(max_number=max_ned_triples_iter, max_batch_size=max_batch_size, number_of_batches=number_of_batches, batchwise=batchwise):
                            did_work = True
                            for ned_batch in ned1_batches:
                                self._create_ned1_batches(ned_batch, poll_interval=2, batchwise=batchwise)

            if batch_quota > 0 and num_nes_explored < max_nes_explored:
                with self._stage("Elicitation"):
                    if batches_subjects := self._get_unexplored_inodes_batch(max_batch_size=elicitation_batch_size, number_of_batches=elicitation_number_of_batches):
                        did_work = True
                        for subject_batch in batches_subjects:
                            num_nes_explored += self._create_elicitation_batch(subject_batch, poll_interval=2, batchwise=batchwise)
                            batch_quota -= 1
                        if not batchwise:
                            logger.info(f"{num_nes_explored} inodes explored so far.")

            if batch_quota > 0:
                with self._stage("PDG"):
                    if pdg_batches := self._get_pdg_triples_batch(max_number=max_pdg_triples_iter, max_batch_size=max_batch_size, number_of_batches=number_of_batches):
                        did_work = True
                        if_PD = False
                        for pdg_batch in pdg_batches:
                            self._create_pdg_batches(pdg_batch, poll_interval=2, batchwise=batchwise)
                            batch_quota -= 1
            if batch_quota > 0:
                with self._stage("CDG"):
                    if cdg_batches := self._get_cdg_instance_triples_batch(max_number=max_cdg_triples_iter, max_batch_size=max_batch_size, number_of_batches=number_of_batches):
                        did_work = True
                        if_CD = False
                        for cdg_batch in cdg_batches:
                            self._create_cdg_batches(cdg_batch, poll_interval=2, batchwise=batchwise)

            if batch_quota > 0:
                if if_PD:
                    with self._stage("PD"):
                        if pd_batches := self._get_pd_triples_batch(max_number=max_pd_triples_iter, max_batch_size=max_batch_size, number_of_batches=number_of_batches):
                            did_work = True
                            for pd_batch in pd_batches:
                                self._create_pd_batches(pd_batch, poll_interval=2, batchwise=batchwise)
                                batch_quota -= 1
            if batch_quota > 0:
                if if_CD:
                    with self._stage("CD"):
                        if cd_batches := self._get_cd_instance_triples_batch(max_number=max_cd_triples_iter, max_batch_size=max_batch_size, number_of_batches=number_of_batches):
                            did_work = True
                            for cd_batch in cd_batches:
                                self._create_cd_batches(cd_batch, poll_interval=2, batchwise=batchwise)
                                batch_quota -= 1

            if batch_quota > 0:
                with self._stage("NER"):
                    if ner_batches := self._get_ner_triples_batch(max_batch_size=ner_batch_size, number_of_batches=ner_number_of_batches):
                        did_work = True
                        for ner_batch in ner_batches:
                            self._create_ner_batch(ner_batch, poll_interval=2, batchwise=batchwise)
                            batch_quota -= 1

            if did_work or self._round_had_error:
                idle_rounds = 0
            else:
                idle_rounds += 1
                if idle_rounds >= max_idle_rounds:
                    logger.info(f"Nothing left to do for {idle_rounds} rounds, stopping. {num_nes_explored} inodes explored in this run.")
                    return

            if batchwise or outstanding_batch_ids or self._round_had_error:
                time.sleep(5)

