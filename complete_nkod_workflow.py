#!/usr/bin/env python3
"""
Kompletní NKOD LangGraph workflow
Integruje všechny komponenty do jednoho automatizovaného systému
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Literal, TypedDict
from pathlib import Path
import schedule
import time
import logging
from dataclasses import dataclass

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# Import všech uzlů
from nkod_langchain_nodes import (
    NKODConfig, 
    DataFetchNode, 
    MetadataParseNode, 
    OutputNode,
    NKODState
)
from json_diff_node import JSONDiffNode
from embedding_node import EmbeddingNode, EmbeddingConfig

# Nastavení loggingu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_nkod_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompleteWorkflowState(TypedDict):
    """Kompletní stav pro celý workflow"""
    # Základní NKOD stav
    sparql_endpoint: str
    session_headers: Dict[str, str]
    output_dir: str
    datasets: List[str]
    processed_datasets: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    limit: Optional[int]
    total_count: int
    hvd_count: int
    
    # Workflow management
    workflow_start_time: str
    workflow_status: Literal["starting", "checking", "backing_up", "scraping", "saving", "diffing", "embedding", "cleanup", "completed", "error"]
    workflow_id: str
    trigger_type: Literal["manual", "scheduled"]
    error_message: Optional[str]
    
    # File management
    previous_dataset_exists: bool
    current_dataset_path: str
    previous_dataset_path: str
    backup_created: bool
    files_managed: List[str]
    
    # Diff processing
    dataset_changes: List[Dict[str, Any]]
    changes_summary: Dict[str, Any]
    new_or_modified_datasets: List[Dict[str, Any]]
    diff_completed: bool
    diff_error: Optional[str]
    
    # Embedding processing
    embeddings_processed: int
    embeddings_dimension: int
    vector_db_collection: str
    vector_db_type: str
    embeddings_error: Optional[str]
    vector_db_initialized: bool
    
    # Cleanup tracking
    old_backups_removed: int
    cleanup_completed: bool


class CompleteWorkflowConfig(BaseModel):
    """Kompletní konfigurace workflow"""
    # NKOD scraping
    nkod_config: NKODConfig = Field(default_factory=NKODConfig)
    
    # File management
    output_base_dir: str = "complete_nkod_data"
    current_dataset_name: str = "dataset_current.json"
    previous_dataset_name: str = "dataset_previous.json"
    backup_retention_days: int = 30
    
    # Scheduling
    enable_scheduler: bool = True
    schedule_time: str = "09:00"  # Pondělí v 9:00
    schedule_day: str = "monday"
    
    # Embedding
    embedding_config: EmbeddingConfig = Field(default_factory=lambda: EmbeddingConfig(
        embedding_provider="huggingface",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_db="chroma",
        collection_name="complete_nkod_datasets",
        vector_db_config={"persist_directory": "./complete_nkod_vector_db"}
    ))
    
    # Workflow settings
    max_workflow_duration_minutes: int = 120
    retry_attempts: int = 3
    
    class Config:
        arbitrary_types_allowed = True


@dataclass
class WorkflowResult:
    """Výsledek workflow pro reporting"""
    workflow_id: str
    trigger_type: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    status: str
    datasets_processed: int
    changes_detected: int
    embeddings_created: int
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "trigger_type": self.trigger_type,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "datasets_processed": self.datasets_processed,
            "changes_detected": self.changes_detected,
            "embeddings_created": self.embeddings_created,
            "errors": self.errors
        }


class WorkflowTriggerNode:
    """Vstupní uzel - inicializace workflow"""
    
    def __init__(self, config: CompleteWorkflowConfig):
        self.config = config
        
    def __call__(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        workflow_id = f"nkod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        trigger_type = state.get("trigger_type", "manual")
        
        logger.info(f"🚀 Spouštím NKOD workflow: {workflow_id}")
        logger.info(f"📋 Typ spuštění: {trigger_type}")
        
        # Vytvoření výstupních adresářů
        base_dir = Path(self.config.output_base_dir)
        base_dir.mkdir(exist_ok=True)
        
        # Aktualizace stavu
        state.update({
            "workflow_id": workflow_id,
            "workflow_start_time": datetime.now().isoformat(),
            "workflow_status": "starting",
            "trigger_type": trigger_type,
            "current_dataset_path": str(base_dir / self.config.current_dataset_name),
            "previous_dataset_path": str(base_dir / self.config.previous_dataset_name),
            "files_managed": [],
            "error_message": None
        })
        
        return state


class FileManagementNode:
    """Uzel pro správu souborů - kontrola a zálohování"""
    
    def __init__(self, config: CompleteWorkflowConfig):
        self.config = config
        
    def __call__(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        logger.info("📁 Správa souborů...")
        
        try:
            current_path = Path(state["current_dataset_path"])
            previous_path = Path(state["previous_dataset_path"])
            
            # Kontrola existence současného datasetu
            previous_exists = current_path.exists()
            files_managed = []
            
            if previous_exists:
                logger.info(f"📄 Nalezen současný dataset: {current_path}")
                
                # Pokud už existuje previous, vytvoří timestampovanou zálohu
                if previous_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = previous_path.parent / f"dataset_backup_{timestamp}.json"
                    shutil.move(str(previous_path), str(backup_path))
                    files_managed.append(str(backup_path))
                    logger.info(f"🗂️ Stará záloha přesunuta: {backup_path}")
                
                # Přejmenování current na previous
                shutil.move(str(current_path), str(previous_path))
                files_managed.append(str(previous_path))
                logger.info(f"🔄 Dataset přejmenován: current → previous")
                
                state["backup_created"] = True
            else:
                logger.info("📄 Žádný předchozí dataset nenalezen")
                state["backup_created"] = False
            
            state.update({
                "previous_dataset_exists": previous_exists,
                "files_managed": files_managed,
                "workflow_status": "checking"
            })
            
        except Exception as e:
            error_msg = f"Chyba při správě souborů: {e}"
            logger.error(error_msg)
            state.update({
                "workflow_status": "error",
                "error_message": error_msg
            })
            
        return state


class DataScrapingNode:
    """Uzel pro stahování a parsování NKOD dat"""
    
    def __init__(self, config: CompleteWorkflowConfig):
        self.config = config
        self.data_fetch_node = DataFetchNode(config.nkod_config)
        self.metadata_parse_node = MetadataParseNode(config.nkod_config)
        
    def __call__(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        logger.info("🌐 Stahování NKOD dat...")
        
        try:
            state["workflow_status"] = "scraping"
            
            # Konverze na základní NKOD stav
            nkod_state = NKODState(
                sparql_endpoint=state.get("sparql_endpoint", ""),
                session_headers=state.get("session_headers", {}),
                output_dir=state.get("output_dir", ""),
                datasets=state.get("datasets", []),
                processed_datasets=state.get("processed_datasets", []),
                statistics=state.get("statistics", {}),
                limit=state.get("limit"),
                total_count=state.get("total_count", 0),
                hvd_count=state.get("hvd_count", 0)
            )
            
            # 1. Stahování seznamu datasetů
            logger.info("📊 Získávání seznamu datasetů...")
            nkod_state = self.data_fetch_node(nkod_state)
            logger.info(f"✅ Nalezeno {len(nkod_state['datasets'])} datasetů")
            
            # 2. Parsování metadat
            logger.info(f"🔍 Parsování {len(nkod_state['datasets'])} datasetů...")
            nkod_state = self.metadata_parse_node(nkod_state)
            logger.info(f"✅ Zpracováno {len(nkod_state['processed_datasets'])} datasetů")
            
            # Aktualizace hlavního stavu
            state.update(nkod_state)
            
        except Exception as e:
            error_msg = f"Chyba při stahování dat: {e}"
            logger.error(error_msg)
            state.update({
                "workflow_status": "error",
                "error_message": error_msg
            })
            
        return state


class DataSavingNode:
    """Uzel pro uložení dat do JSON souboru"""
    
    def __init__(self, config: CompleteWorkflowConfig):
        self.config = config
        
    def __call__(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        logger.info("💾 Ukládání dat...")
        
        try:
            # Příprava dat pro uložení
            output_data = {
                "metadata": {
                    "workflow_id": state["workflow_id"],
                    "created_at": state["workflow_start_time"],
                    "trigger_type": state["trigger_type"],
                    "total_datasets": len(state["processed_datasets"]),
                    "hvd_datasets": len([d for d in state["processed_datasets"] if d.get("isHVD")]),
                    "scraper_version": "2.1-HVD-LangChain-Complete",
                    "statistics": state.get("statistics", {})
                },
                "datasets": state["processed_datasets"]
            }
            
            # Uložení do current datasetu
            current_path = Path(state["current_dataset_path"])
            with open(current_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Data uložena: {current_path}")
            logger.info(f"📊 Celkem datasetů: {output_data['metadata']['total_datasets']}")
            logger.info(f"🏷️ HVD datasetů: {output_data['metadata']['hvd_datasets']}")
            
            state["workflow_status"] = "saving"
            
        except Exception as e:
            error_msg = f"Chyba při ukládání dat: {e}"
            logger.error(error_msg)
            state.update({
                "workflow_status": "error",
                "error_message": error_msg
            })
            
        return state


class DiffProcessingNode:
    """Uzel pro porovnání dat - wrapper nad JSONDiffNode"""
    
    def __init__(self, config: CompleteWorkflowConfig):
        self.config = config
        self.diff_node = JSONDiffNode(config)
        
    def __call__(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        logger.info("🔍 Porovnávání dat...")
        
        try:
            # Spuštění diff uzlu
            diff_result = self.diff_node(state)
            
            # Aktualizace stavu
            state.update(diff_result)
            
            if state.get("diff_completed", False):
                summary = state.get("changes_summary", {})
                logger.info("✅ Porovnání dokončeno:")
                logger.info(f"  📈 Nové datasety: {summary.get('new_datasets', 0)}")
                logger.info(f"  📝 Změněné datasety: {summary.get('modified_datasets', 0)}")
                logger.info(f"  📉 Smazané datasety: {summary.get('deleted_datasets', 0)}")
                logger.info(f"  🎯 K embedování: {len(state.get('new_or_modified_datasets', []))}")
            
        except Exception as e:
            error_msg = f"Chyba při porovnávání dat: {e}"
            logger.error(error_msg)
            state.update({
                "diff_error": error_msg,
                "diff_completed": False
            })
            
        return state


class EmbeddingProcessingNode:
    """Uzel pro vytváření embeddingů - wrapper nad EmbeddingNode"""
    
    def __init__(self, config: CompleteWorkflowConfig):
        self.config = config
        self.embedding_node = EmbeddingNode(config.embedding_config)
        
    def __call__(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        logger.info("🔮 Vytváření embeddingů...")
        
        try:
            # Spuštění embedding uzlu
            embedding_result = self.embedding_node(state)
            
            # Aktualizace stavu
            state.update(embedding_result)
            
            if state.get("vector_db_initialized", False):
                logger.info("✅ Embeddingy vytvořeny:")
                logger.info(f"  🧮 Zpracováno: {state.get('embeddings_processed', 0)}")
                logger.info(f"  📐 Dimenze: {state.get('embeddings_dimension', 0)}")
                logger.info(f"  🗄️ Databáze: {state.get('vector_db_type', 'N/A')}")
                logger.info(f"  📚 Kolekce: {state.get('vector_db_collection', 'N/A')}")
                
                state["workflow_status"] = "embedding"
            
        except Exception as e:
            error_msg = f"Chyba při vytváření embeddingů: {e}"
            logger.error(error_msg)
            state.update({
                "embeddings_error": error_msg,
                "vector_db_initialized": False
            })
            
        return state


class CleanupNode:
    """Uzel pro úklid starých souborů"""
    
    def __init__(self, config: CompleteWorkflowConfig):
        self.config = config
        
    def __call__(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        logger.info("🧹 Úklid starých souborů...")
        
        try:
            base_dir = Path(self.config.output_base_dir)
            retention_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
            
            # Najde a vymaže staré zálohy
            deleted_files = 0
            for backup_file in base_dir.glob("dataset_backup_*.json"):
                if backup_file.stat().st_mtime < retention_date.timestamp():
                    backup_file.unlink()
                    deleted_files += 1
                    logger.info(f"🗑️ Vymazána stará záloha: {backup_file.name}")
            
            if deleted_files > 0:
                logger.info(f"✅ Vymazáno {deleted_files} starých záloh")
            else:
                logger.info("ℹ️ Žádné staré zálohy k vymazání")
            
            state.update({
                "old_backups_removed": deleted_files,
                "cleanup_completed": True,
                "workflow_status": "cleanup"
            })
            
        except Exception as e:
            error_msg = f"Chyba při úklidu: {e}"
            logger.error(error_msg)
            state.update({
                "cleanup_completed": False,
                "error_message": error_msg
            })
            
        return state


class WorkflowFinalizationNode:
    """Finalizační uzel - dokončení workflow"""
    
    def __init__(self, config: CompleteWorkflowConfig):
        self.config = config
        
    def __call__(self, state: CompleteWorkflowState) -> CompleteWorkflowState:
        logger.info("🏁 Dokončování workflow...")
        
        # Výpočet statistik
        start_time = datetime.fromisoformat(state["workflow_start_time"])
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Vytvoření výsledku
        result = WorkflowResult(
            workflow_id=state["workflow_id"],
            trigger_type=state["trigger_type"],
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            status=state.get("workflow_status", "unknown"),
            datasets_processed=len(state.get("processed_datasets", [])),
            changes_detected=len(state.get("new_or_modified_datasets", [])),
            embeddings_created=state.get("embeddings_processed", 0),
            errors=[e for e in [
                state.get("error_message"),
                state.get("diff_error"),
                state.get("embeddings_error")
            ] if e]
        )
        
        # Uložení výsledku
        results_dir = Path(self.config.output_base_dir) / "workflow_results"
        results_dir.mkdir(exist_ok=True)
        
        result_file = results_dir / f"result_{state['workflow_id']}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Finální report
        if result.errors:
            logger.error(f"❌ Workflow dokončen s chybami: {state['workflow_id']}")
            for error in result.errors:
                logger.error(f"  🚨 {error}")
            state["workflow_status"] = "error"
        else:
            logger.info(f"✅ Workflow úspěšně dokončen: {state['workflow_id']}")
            logger.info(f"⏱️ Trvání: {duration:.1f}s")
            logger.info(f"📊 Zpracováno datasetů: {result.datasets_processed}")
            logger.info(f"🔄 Změn detekováno: {result.changes_detected}")
            logger.info(f"🔮 Embeddingů vytvořeno: {result.embeddings_created}")
            state["workflow_status"] = "completed"
        
        return state


# Kondicionální funkce pro routing
def should_backup(state: CompleteWorkflowState) -> Literal["scraping", "error"]:
    """Rozhoduje o pokračování workflow"""
    return "scraping" if state["workflow_status"] == "checking" else "error"


def should_diff(state: CompleteWorkflowState) -> Literal["diffing", "error"]:
    """Rozhoduje o spuštění diff"""
    return "diffing" if state["workflow_status"] == "saving" else "error"


def should_embed(state: CompleteWorkflowState) -> Literal["embedding", "cleanup"]:
    """Rozhoduje o spuštění embedding"""
    if state.get("diff_completed", False) and len(state.get("new_or_modified_datasets", [])) > 0:
        return "embedding"
    else:
        return "cleanup"


def should_finalize(state: CompleteWorkflowState) -> Literal["finalization", "error"]:
    """Rozhoduje o finalizaci"""
    return "finalization" if not state.get("error_message") else "error"


class CompleteNKODWorkflow:
    """Hlavní třída pro kompletní NKOD workflow"""
    
    def __init__(self, config: CompleteWorkflowConfig = None):
        self.config = config or CompleteWorkflowConfig()
        self.workflow_graph = None
        self.app = None
        self._build_workflow()
        
    def _build_workflow(self):
        """Sestaví LangGraph workflow"""
        logger.info("🏗️ Sestavuji workflow graf...")
        
        # Vytvoření uzlů
        trigger_node = WorkflowTriggerNode(self.config)
        file_mgmt_node = FileManagementNode(self.config)
        scraping_node = DataScrapingNode(self.config)
        saving_node = DataSavingNode(self.config)
        diff_node = DiffProcessingNode(self.config)
        embedding_node = EmbeddingProcessingNode(self.config)
        cleanup_node = CleanupNode(self.config)
        finalization_node = WorkflowFinalizationNode(self.config)
        
        # Vytvoření grafu
        self.workflow_graph = StateGraph(CompleteWorkflowState)
        
        # Přidání uzlů
        self.workflow_graph.add_node("trigger", trigger_node)
        self.workflow_graph.add_node("file_management", file_mgmt_node)
        self.workflow_graph.add_node("scraping", scraping_node)
        self.workflow_graph.add_node("saving", saving_node)
        self.workflow_graph.add_node("diffing", diff_node)
        self.workflow_graph.add_node("embedding", embedding_node)
        self.workflow_graph.add_node("cleanup", cleanup_node)
        self.workflow_graph.add_node("finalization", finalization_node)
        
        # Definice přechodů
        self.workflow_graph.add_edge(START, "trigger")
        self.workflow_graph.add_edge("trigger", "file_management")
        
        self.workflow_graph.add_conditional_edges(
            "file_management",
            should_backup,
            {
                "scraping": "scraping",
                "error": "finalization"
            }
        )
        
        self.workflow_graph.add_edge("scraping", "saving")
        
        self.workflow_graph.add_conditional_edges(
            "saving",
            should_diff,
            {
                "diffing": "diffing",
                "error": "finalization"
            }
        )
        
        self.workflow_graph.add_conditional_edges(
            "diffing",
            should_embed,
            {
                "embedding": "embedding",
                "cleanup": "cleanup"
            }
        )
        
        self.workflow_graph.add_edge("embedding", "cleanup")
        
        self.workflow_graph.add_conditional_edges(
            "cleanup",
            should_finalize,
            {
                "finalization": "finalization",
                "error": "finalization"
            }
        )
        
        self.workflow_graph.add_edge("finalization", END)
        
        # Kompilace grafu
        self.app = self.workflow_graph.compile(checkpointer=MemorySaver())
        logger.info("✅ Workflow graf sestaven")
        
    def run_manual(self, limit: int = None) -> WorkflowResult:
        """Spustí workflow ručně"""
        logger.info("🖱️ Manuální spuštění workflow...")
        
        # Aktualizace konfigurace
        if limit:
            self.config.nkod_config.limit = limit
            
        # Inicializační stav
        initial_state = CompleteWorkflowState(
            # NKOD základní stav
            sparql_endpoint="",
            session_headers={},
            output_dir="",
            datasets=[],
            processed_datasets=[],
            statistics={},
            limit=limit,
            total_count=0,
            hvd_count=0,
            
            # Workflow management
            workflow_start_time="",
            workflow_status="starting",
            workflow_id="",
            trigger_type="manual",
            error_message=None,
            
            # File management
            previous_dataset_exists=False,
            current_dataset_path="",
            previous_dataset_path="",
            backup_created=False,
            files_managed=[],
            
            # Diff processing
            dataset_changes=[],
            changes_summary={},
            new_or_modified_datasets=[],
            diff_completed=False,
            diff_error=None,
            
            # Embedding processing
            embeddings_processed=0,
            embeddings_dimension=0,
            vector_db_collection="",
            vector_db_type="",
            embeddings_error=None,
            vector_db_initialized=False,
            
            # Cleanup
            old_backups_removed=0,
            cleanup_completed=False
        )
        
        # Spuštění workflow
        thread_config = {"configurable": {"thread_id": f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
        
        try:
            final_state = self.app.invoke(initial_state, thread_config)
            return self._extract_result(final_state)
        except Exception as e:
            logger.error(f"❌ Kritická chyba workflow: {e}")
            raise
            
    def run_scheduled(self) -> WorkflowResult:
        """Spustí workflow jako naplánovaný úkol"""
        logger.info("⏰ Naplánované spuštění workflow...")
        
        # Podobné jako manual, ale s trigger_type="scheduled"
        initial_state = CompleteWorkflowState(
            # ... stejné jako manual ...
            trigger_type="scheduled",
            # ... zbytek ...
            sparql_endpoint="",
            session_headers={},
            output_dir="",
            datasets=[],
            processed_datasets=[],
            statistics={},
            limit=self.config.nkod_config.limit,
            total_count=0,
            hvd_count=0,
            workflow_start_time="",
            workflow_status="starting",
            workflow_id="",
            error_message=None,
            previous_dataset_exists=False,
            current_dataset_path="",
            previous_dataset_path="",
            backup_created=False,
            files_managed=[],
            dataset_changes=[],
            changes_summary={},
            new_or_modified_datasets=[],
            diff_completed=False,
            diff_error=None,
            embeddings_processed=0,
            embeddings_dimension=0,
            vector_db_collection="",
            vector_db_type="",
            embeddings_error=None,
            vector_db_initialized=False,
            old_backups_removed=0,
            cleanup_completed=False
        )
        
        thread_config = {"configurable": {"thread_id": f"scheduled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
        
        try:
            final_state = self.app.invoke(initial_state, thread_config)
            return self._extract_result(final_state)
        except Exception as e:
            logger.error(f"❌ Kritická chyba naplánovaného workflow: {e}")
            raise
            
    def _extract_result(self, state: CompleteWorkflowState) -> WorkflowResult:
        """Extrahuje výsledek z finálního stavu"""
        start_time = datetime.fromisoformat(state["workflow_start_time"])
        end_time = datetime.now()
        
        return WorkflowResult(
            workflow_id=state["workflow_id"],
            trigger_type=state["trigger_type"],
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
            status=state["workflow_status"],
            datasets_processed=len(state.get("processed_datasets", [])),
            changes_detected=len(state.get("new_or_modified_datasets", [])),
            embeddings_created=state.get("embeddings_processed", 0),
            errors=[e for e in [
                state.get("error_message"),
                state.get("diff_error"), 
                state.get("embeddings_error")
            ] if e]
        )
        
    def start_scheduler(self):
        """Spustí automatický scheduler"""
        if not self.config.enable_scheduler:
            logger.info("📅 Scheduler je zakázán v konfiguraci")
            return
            
        # Naplánování na určený den a čas
        if self.config.schedule_day.lower() == "monday":
            schedule.every().monday.at(self.config.schedule_time).do(self._scheduled_job)
        elif self.config.schedule_day.lower() == "tuesday":
            schedule.every().tuesday.at(self.config.schedule_time).do(self._scheduled_job)
        # ... další dny podle potřeby
        else:
            logger.error(f"❌ Nepodporovaný den: {self.config.schedule_day}")
            return
            
        logger.info(f"📅 Scheduler nastaven na každé {self.config.schedule_day} v {self.config.schedule_time}")
        logger.info("🔄 Scheduler běží... (Ctrl+C pro ukončení)")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Kontrola každou minutu
        except KeyboardInterrupt:
            logger.info("⏹️ Scheduler ukončen uživatelem")
        finally:
            schedule.clear()
            
    def _scheduled_job(self):
        """Naplánovaná úloha"""
        try:
            logger.info("⏰ Spouštím naplánovaný NKOD workflow...")
            result = self.run_scheduled()
            if result.status == "completed":
                logger.info(f"✅ Naplánovaný workflow úspěšně dokončen: {result.workflow_id}")
            else:
                logger.error(f"❌ Naplánovaný workflow selhal: {result.workflow_id}")
        except Exception as e:
            logger.error(f"💥 Kritická chyba v naplánované úloze: {e}")


def main():
    """Hlavní funkce s CLI rozhraním"""
    import sys
    
    config = CompleteWorkflowConfig()
    workflow = CompleteNKODWorkflow(config)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "run":
            # Manuální spuštění
            limit = None
            if len(sys.argv) > 2:
                try:
                    limit = int(sys.argv[2])
                except ValueError:
                    print("❌ Neplatný limit, použije se výchozí")
            
            print("🚀 Spouštím kompletní NKOD workflow ručně...")
            result = workflow.run_manual(limit)
            print(f"\n📋 Výsledek workflow:")
            print(f"  ID: {result.workflow_id}")
            print(f"  Status: {result.status}")
            print(f"  Trvání: {result.duration_seconds:.1f}s")
            print(f"  Zpracováno: {result.datasets_processed} datasetů")
            print(f"  Změny: {result.changes_detected}")
            print(f"  Embeddingy: {result.embeddings_created}")
            
        elif command == "schedule":
            # Spuštění scheduleru
            print("📅 Spouštím automatický scheduler...")
            workflow.start_scheduler()
            
        elif command == "test":
            # Test s omezeným počtem
            print("🧪 Testovací spuštění s 3 datasety...")
            result = workflow.run_manual(3)
            print(f"✅ Test dokončen: {result.status}")
            
        elif command == "help":
            print("🔧 Kompletní NKOD LangGraph Workflow")
            print("Použití:")
            print(f"  {sys.argv[0]} run [limit]    # Manuální spuštění")
            print(f"  {sys.argv[0]} schedule       # Automatický scheduler")
            print(f"  {sys.argv[0]} test           # Test s 3 datasety")
            print(f"  {sys.argv[0]} help           # Tato nápověda")
            print()
            print("Funkce:")
            print("  🔄 Automatické stahování a parsování NKOD dat")
            print("  📊 Detekce změn mezi běhy")
            print("  🔮 Automatické vytváření embeddingů pro nová/změněná data")
            print("  📅 Schedulovaný běh každé pondělí v 9:00")  
            print("  🗂️ Správa záloh a úklid starých souborů")
            print("  📋 Kompletní logování a reporting")
            
        else:
            print(f"❌ Neznámý příkaz: {command}")
            print(f"Použijte '{sys.argv[0]} help' pro nápovědu")
            sys.exit(1)
    else:
        # Výchozí chování - manuální spuštění
        print("🚀 Spouštím kompletní NKOD workflow...")
        result = workflow.run_manual()
        print(f"📊 Workflow dokončen: {result.status}")


if __name__ == "__main__":
    main()