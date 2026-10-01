from datetime import datetime
from enum import Enum
import sqlalchemy as sa
from sqlalchemy import Index, text
from sqlmodel import Field, SQLModel, Column, UniqueConstraint, Relationship

from typing import Optional, List, Any


class Batch(SQLModel, table=True):
    id: str = Field(primary_key=True, nullable=False)
    input_file_id: str
    status: str = Field(index=True)
    output_file_id: str = Field(default=None)

    job_type: str = Field(index=True, nullable=False)

    created_at: datetime = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),
        sa_column_kwargs={"server_default": sa.func.now()},
        nullable=False,
    )

    def __repr__(self):
        return f"< {self.__class__.__name__}: {self.id}, {self.status} >"

class StatusType(Enum):
    UNEXPLORED = "unexplored"
    EXPLORING = "exploring"
    EXPLORED = "explored"

class ObjectStatusType(Enum):
    GENERATED = "generated"
    ONNER = "onner"
    NERFINISHED = "ner_finished"
    ONNED1 = "onned1"
    NEDFINISHED1 = "ned_finished1"
    ONDG = "ondg"
    DGFINISHED = "dg_finished"
    ONNED2 = "onned2"
    FINISHED = "finished"

class ConceptStatusType(Enum):
    GENERATED = "generated"
    ONCD = "oncd"
    CDFINISHED = "cd_finished"
    ONDG = "ondg"
    FINISHED = "finished"

class PredicateStatusType(Enum):
    GENERATED = "generated"
    ONPD = "onpd"
    PDFINISHED = "pd_finished"
    ONDG = "ondg"
    FINISHED = "finished"

class ObjectType(Enum):
    LITERAL = "literal"
    NE = "ne"
    CONCEPT = "class"
    UNRECOGNIZED = "unrecognized"

class JobType(Enum):
    ELICITATION = "elicitation"
    NAMED_ENTITY_RECOGNITION = "ner"
    NAMED_ENTITY_DISAMBIGUATION_SOURCE_TRIPLE = "ned_source_triple"
    NAMED_ENTITY_DISAMBIGUATION_DESCRIPTION = "ned_description"
    PREDICATE_DISAMBIGUATION = "pd"
    CONCEPT_DISAMBIGUATION = "cd"
    NAMED_ENTITY_DESCRIPTION_GEN = "nedg"
    PREDICATE_DESCRIPTION_GEN = "pdg"
    CONCEPT_DESCRIPTION_GEN = "cdg"

class InstanceNode(SQLModel, table=True):
    id: str = Field(primary_key=True, nullable=False)
    status: StatusType = Field(
            default=StatusType.UNEXPLORED,
            index= True,
        )
    description: str = Field(nullable=False)
    label: str = Field(index=True, nullable=False)
    embedding_index: Optional[int] = Field(index=True, unique=True)
    first_appeared: Optional[str] = Field(default=None)
    explored_batch_id: Optional[str] = Field(default=None, foreign_key="batch.id", index=True)
    created_at: datetime = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),
        sa_column_kwargs={"server_default": sa.func.now()},
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("label", "description", name="uq_label_description"),
        Index(
            "id_label_description_instancenode",
            "id", "label", "description"
        )
    )

    def __repr__(self):
        return f"< {self.__class__.__name__}: {self.id}, {self.label} >"


class Predicate(SQLModel, table=True):
    id: str = Field(primary_key=True, nullable=False)
    label: str = Field(index=True, nullable=False)
    embedding_index: Optional[int] = Field(index=True, unique=True)
    description: str = Field(nullable=False)
    first_appeared: Optional[str] = Field(default=None)

    __table_args__ = (
        UniqueConstraint("label", "description", name="uq_label_description"),
        Index(
            "id_label_description_predicate",
            "id", "label", "description"
        )
    )

    def __repr__(self):
        return f"< {self.__class__.__name__}: {self.id}, {self.label} >"

class Concept(SQLModel, table=True):
    id: str = Field(primary_key=True, nullable=False)
    label: str = Field(index=True, nullable=False)
    embedding_index: Optional[int] = Field(index=True, unique=True)
    description: str = Field(nullable=False)
    first_appeared: Optional[str] = Field(default=None)

    __table_args__ = (
        UniqueConstraint("label", "description", name="uq_label_description"),
        Index(
            "id_label_description_concept",
            "id", "label", "description"
        )
    )
    
    def __repr__(self):
        return f"< {self.__class__.__name__}: {self.id}, {self.label} >"


class Triple(SQLModel, table=True):
    triple_id: str = Field(primary_key=True, nullable=False)
    subject_id: str = Field(index=True, nullable=False)

    subject_label: str = Field(index=True, nullable=False)
    predicate_label: str = Field(index=True, nullable=False)
    predicate_id: Optional[str] = Field(default=None, index=True)
    predicate_status: PredicateStatusType = Field(default=PredicateStatusType.GENERATED, index=True)
    object_label: str = Field(index=True, nullable=False)
    object_id: Optional[str] = Field(default=None, index=True, nullable=True)
    object_type : ObjectType = Field(default=ObjectType.UNRECOGNIZED, index=True)
    object_status: ObjectStatusType = Field(
        default=ObjectStatusType.GENERATED,
        index= True,
        )
    object_description : Optional[str] = Field(default=None, index=True, nullable=True)
    ned_candidates: List["InstanceNode"] = Relationship(
        sa_relationship_kwargs={
            "secondary": "tripleinstancenodelink",
            "order_by": "TripleInstanceNodeLink.position"},
        )
    ned2_candidates: List["InstanceNode"] = Relationship(
        sa_relationship_kwargs={
            "secondary": "tripleinstancenodedescriptionlink",
            "order_by": "TripleInstanceNodeDescriptionLink.position"},
        )
    pd_candidates: List["Predicate"] = Relationship(
        sa_relationship_kwargs={
            "secondary": "triplepredicatelink",
            "order_by": "TriplePredicateLink.position"},
        )
    creating_batch_id: Optional[str] = Field(default=None, 
                                             foreign_key="batch.id")

    ner_batch_id: Optional[str] = Field(default=None,
                              foreign_key="batch.id",
                              index=True)

    ned1_batch_id: Optional[str] = Field(default=None,
                              foreign_key="batch.id",
                              index=True)
    
    ned2_batch_id: Optional[str] = Field(default=None,
                              foreign_key="batch.id",
                              index=True)
    
    nedg_batch_id: Optional[str] = Field(default=None,
                              foreign_key="batch.id",
                              index=True)
    
    pd_batch_id: Optional[str] = Field(default=None,
                              foreign_key="batch.id",
                              index=True)
    
    pdg_batch_id: Optional[str] = Field(default=None,
                              foreign_key="batch.id",
                              index=True)

    created_at: datetime = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),
        sa_column_kwargs={"server_default": sa.func.now()},
        nullable=False,
    )

    __table_args__ = (
        Index(
            "triple_unique",
            "subject_id", "subject_label", "predicate_id", "predicate_label", "object_id", "object_label",
            unique=True
            ),
        Index(
            "predicate_object_triple_caching",
            "object_type", "object_label", "predicate_label", "object_id"
            ),
        Index(
            "ned_merged",
            "object_id", "object_label", "object_type"
            ),
        Index(
            "pd_caching",
            "predicate_label", "predicate_id"
        )
    )


class InstanceTriple(SQLModel, table=True):
    original_triple_id: str = Field(primary_key=True, index=True, nullable=False, foreign_key="triple.triple_id")
    entity_id: str = Field(index=True, nullable=False, foreign_key="instancenode.id")
    concept_label: str = Field(index=True, nullable=False)
    concept_id: Optional[str] = Field(index=True, default=None, foreign_key="concept.id")
    cd_candidates: List["Concept"] = Relationship(
        sa_relationship_kwargs={
            "secondary": "instancetripleconceptlink",
            "order_by": "InstanceTripleConceptLink.position"},
        )
    cd_batch_id: Optional[str] = Field(default=None,
                              foreign_key="batch.id",
                              index=True)
    
    cdg_batch_id: Optional[str] = Field(default=None,
                              foreign_key="batch.id",
                              index=True)

    created_at: datetime = Field(
        default=None,
        sa_type=sa.DateTime(timezone=True),
        sa_column_kwargs={"server_default": sa.func.now()},
        nullable=False,
    )

    status: ConceptStatusType = Field(
        default=ConceptStatusType.GENERATED,
        index= True)
    
    __table_args__ = (
        Index(
            "cd_caching",
            "concept_label", "concept_id"
        ),
        Index(
            "entity_concept_unique_check", 
            "entity_id", "concept_id"),
    )
    

class TripleInstanceNodeLink(SQLModel, table=True):
    triple_id: str = Field(
        foreign_key="triple.triple_id",
        primary_key=True
    )
    instance_node_id: str = Field(
        foreign_key="instancenode.id",
        primary_key=True
    )
    position: int = Field(nullable=False)

class TripleInstanceNodeDescriptionLink(SQLModel, table=True):
    triple_id: str = Field(
        foreign_key="triple.triple_id",
        primary_key=True
    )
    instance_node_id: str = Field(
        foreign_key="instancenode.id",
        primary_key=True
    )
    position: int = Field(nullable=False)

class TriplePredicateLink(SQLModel, table=True):
    triple_id: str = Field(
        foreign_key="triple.triple_id",
        primary_key=True
    )
    predicate_id: str = Field(
        foreign_key="predicate.id",
        primary_key=True
    )
    position: int = Field(nullable=False)

class InstanceTripleConceptLink(SQLModel, table=True):
    original_triple_id: str = Field(
        foreign_key="instancetriple.original_triple_id",
        primary_key=True
    )
    concept_id: str = Field(
        foreign_key="concept.id",
        primary_key=True
    )
    position: int = Field(nullable=False)