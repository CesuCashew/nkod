#!/usr/bin/env python3
"""
Týdenní NKOD scraping workflow s automatickým spuštěním každé pondělí
Využívá LangGraph pro orchestraci a scheduling
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, TypedDict, Optional, Literal
from pathlib import Path
import schedule
import time
import logging

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# Import uzlů z předchozího souboru
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
        logging.FileHandler('nkod_weekly.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WeeklyNKODState(TypedDict):
    """Rozšířený stav pro týdenní workflow"""
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
    
    # Týdenní specifické položky
    previous_dataset_exists: bool
    current_dataset_path: str
    previous_dataset_path: str
    backup_created: bool
    workflow_start_time: str
    workflow_status: Literal["starting", "checking", "backing_up", "scraping", "saving", "diffing", "embedding", "completed", "error"]
    error_message: Optional[str]
    
    # Diff specifické položky
    dataset_changes: List[Dict[str, Any]]
    changes_summary: Dict[str, Any]
    new_or_modified_datasets: List[Dict[str, Any]]
    diff_completed: bool
    diff_error: Optional[str]
    
    # Embedding specifické položky
    embeddings_processed: int
    embeddings_dimension: int
    vector_db_collection: str
    vector_db_type: str
    embeddings_error: Optional[str]
    vector_db_initialized: bool


class WeeklyConfig(BaseModel):
    """Konfigurace pro týdenní workflow"""
    nkod_config: NKODConfig = Field(default_factory=NKODConfig)
    output_base_dir: str = "weekly_nkod_data"
    current_dataset_name: str = "dataset_current.json"
    previous_dataset_name: str = "dataset_previous.json"
    backup_retention_days: int = 30
    enable_scheduler: bool = True
    schedule_time: str = "09:00"  # Pondělí v 9:00
    
    # Embedding konfigurace
    embedding_config: EmbeddingConfig = Field(default_factory=lambda: EmbeddingConfig(
        embedding_provider="huggingface",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_db="chroma",
        collection_name="nkod_weekly_datasets",
        vector_db_config={"persist_directory": "./nkod_vector_db"}
    ))


class FileCheckNode:
    """Uzel pro kontrolu existujících souborů"""
    
    def __init__(self, config: WeeklyConfig):
        self.config = config
        
    def __call__(self, state: WeeklyNKODState) -> WeeklyNKODState:
        """Zkontroluje existenci předchozích datasetů"""
        logger.info("Kontroluji existující datasety...")
        
        # Nastavení cest
        base_dir = Path(self.config.output_base_dir)
        current_path = base_dir / self.config.current_dataset_name
        previous_path = base_dir / self.config.previous_dataset_name
        
        # Vytvoření adresáře pokud neexistuje
        base_dir.mkdir(exist_ok=True)
        
        # Kontrola existence současného datasetu
        previous_exists = current_path.exists()
        
        logger.info(f"Současný dataset existuje: {previous_exists}")
        if previous_exists:
            logger.info(f"Nalezen dataset: {current_path}")
        
        # Aktualizace stavu
        state.update({
            "previous_dataset_exists": previous_exists,
            "current_dataset_path": str(current_path),
            "previous_dataset_path": str(previous_path),
            "backup_created": False,
            "workflow_status": "checking"
        })
        
        return state


class BackupNode:
    """Uzel pro zálohování předchozích dat"""
    
    def __init__(self, config: WeeklyConfig):
        self.config = config
        
    def __call__(self, state: WeeklyNKODState) -> WeeklyNKODState:
        """Přejmenuje současný dataset na previous"""
        if not state["previous_dataset_exists"]:
            logger.info("Žádný předchozí dataset k zálohování")
            state["backup_created"] = True
            state["workflow_status"] = "backing_up"
            return state
            
        try:
            current_path = Path(state["current_dataset_path"])
            previous_path = Path(state["previous_dataset_path"])
            
            # Pokud už existuje previous, vytvoří timestampovanou zálohu
            if previous_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = previous_path.parent / f"dataset_backup_{timestamp}.json"
                shutil.move(str(previous_path), str(backup_path))
                logger.info(f"Stará previous záloha přesunuta do: {backup_path}")
            
            # Přejmenování current na previous
            shutil.move(str(current_path), str(previous_path))
            logger.info(f"Dataset přejmenován: {current_path} -> {previous_path}")
            
            state.update({
                "backup_created": True,
                "workflow_status": "backing_up"
            })
            
        except Exception as e:
            error_msg = f"Chyba při zálohování: {e}"
            logger.error(error_msg)
            state.update({
                "backup_created": False,
                "workflow_status": "error",
                "error_message": error_msg
            })
            
        return state


class WeeklyScrapingNode:
    """Wrapper uzel pro spuštění NKOD scrapingu"""
    
    def __init__(self, config: WeeklyConfig):
        self.config = config
        self.data_fetch_node = DataFetchNode(config.nkod_config)
        self.metadata_parse_node = MetadataParseNode(config.nkod_config)
        
    def __call__(self, state: WeeklyNKODState) -> WeeklyNKODState:
        """Spustí kompletní NKOD scraping"""
        try:
            logger.info("Spouštím NKOD scraping...")
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
            
            # Spuštění jednotlivých uzlů
            logger.info("1. Stahování seznamu datasetů...")
            nkod_state = self.data_fetch_node(nkod_state)
            
            logger.info(f"2. Parsování {len(nkod_state['datasets'])} datasetů...")
            nkod_state = self.metadata_parse_node(nkod_state)
            
            # Aktualizace hlavního stavu
            state.update(nkod_state)
            state["workflow_status"] = "scraping"
            
            logger.info(f"Scraping dokončen. Zpracováno {len(nkod_state['processed_datasets'])} datasetů")
            
        except Exception as e:
            error_msg = f"Chyba při scrapingu: {e}"
            logger.error(error_msg)
            state.update({
                "workflow_status": "error",
                "error_message": error_msg
            })
            
        return state


class WeeklySaveNode:
    """Uzel pro uložení výsledků do weekly formátu"""
    
    def __init__(self, config: WeeklyConfig):
        self.config = config
        
    def __call__(self, state: WeeklyNKODState) -> WeeklyNKODState:
        """Uloží výsledky jako dataset_current.json"""
        try:
            logger.info("Ukládám výsledky...")
            state["workflow_status"] = "saving"
            
            # Příprava dat pro uložení
            output_data = {
                "metadata": {
                    "created_at": state["workflow_start_time"],
                    "total_datasets": len(state["processed_datasets"]),
                    "hvd_datasets": len([d for d in state["processed_datasets"] if d.get("isHVD")]),
                    "scraper_version": "2.1-HVD-LangChain-Weekly",
                    "statistics": state.get("statistics", {})
                },
                "datasets": state["processed_datasets"]
            }
            
            # Uložení do current datasetu
            current_path = Path(state["current_dataset_path"])
            with open(current_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Výsledky uloženy do: {current_path}")
            logger.info(f"Celkem datasetů: {output_data['metadata']['total_datasets']}")
            logger.info(f"HVD datasetů: {output_data['metadata']['hvd_datasets']}")
            
            state["workflow_status"] = "saving"
            
        except Exception as e:
            error_msg = f"Chyba při ukládání: {e}"
            logger.error(error_msg)
            state.update({
                "workflow_status": "error",
                "error_message": error_msg
            })
            
        return state


class CleanupNode:
    """Uzel pro úklid starých záloh"""
    
    def __init__(self, config: WeeklyConfig):
        self.config = config
        
    def __call__(self, state: WeeklyNKODState) -> WeeklyNKODState:
        """Vymaže staré zálohy podle retention policy"""
        try:
            base_dir = Path(self.config.output_base_dir)
            retention_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
            
            # Najde a vymaže staré zálohy
            deleted_files = 0
            for backup_file in base_dir.glob("dataset_backup_*.json"):
                if backup_file.stat().st_mtime < retention_date.timestamp():
                    backup_file.unlink()
                    deleted_files += 1
                    logger.info(f"Vymazána stará záloha: {backup_file}")
            
            if deleted_files > 0:
                logger.info(f"Vymazáno {deleted_files} starých záloh")
            else:
                logger.info("Žádné staré zálohy k vymazání")
                
        except Exception as e:
            logger.warning(f"Chyba při úklidu záloh: {e}")
            
        return state


def should_backup(state: WeeklyNKODState) -> Literal["backup", "scrape"]:
    """Kondicionální funkce - určuje zda zálohovat"""
    return "backup" if state["previous_dataset_exists"] else "scrape"


def workflow_successful(state: WeeklyNKODState) -> Literal["diff", "end"]:
    """Kondicionální funkce - pokračovat na diff nebo skončit"""
    return "diff" if state["workflow_status"] == "saving" else "end"


def diff_successful(state: WeeklyNKODState) -> Literal["embedding", "end"]:
    """Kondicionální funkce - pokračovat na embedding nebo skončit"""
    return "embedding" if state.get("diff_completed", False) else "end"


def embedding_successful(state: WeeklyNKODState) -> Literal["cleanup", "end"]:
    """Kondicionální funkce - embedding úspěšný"""
    return "cleanup" if state.get("vector_db_initialized", False) else "end"


def create_weekly_workflow(config: WeeklyConfig = None) -> StateGraph:
    """Vytvoří LangGraph workflow pro týdenní NKOD scraping"""
    if config is None:
        config = WeeklyConfig()
    
    # Vytvoření uzlů
    file_check_node = FileCheckNode(config)
    backup_node = BackupNode(config)
    scraping_node = WeeklyScrapingNode(config)
    save_node = WeeklySaveNode(config)
    diff_node = JSONDiffNode(config)
    embedding_node = EmbeddingNode(config.embedding_config)
    cleanup_node = CleanupNode(config)
    
    # Vytvoření grafu
    workflow = StateGraph(WeeklyNKODState)
    
    # Přidání uzlů
    workflow.add_node("file_check", file_check_node)
    workflow.add_node("backup", backup_node)
    workflow.add_node("scrape", scraping_node)
    workflow.add_node("save", save_node)
    workflow.add_node("diff", diff_node)
    workflow.add_node("embedding", embedding_node)
    workflow.add_node("cleanup", cleanup_node)
    
    # Definice přechodů
    workflow.add_edge(START, "file_check")
    workflow.add_conditional_edges(
        "file_check",
        should_backup,
        {
            "backup": "backup",
            "scrape": "scrape"
        }
    )
    workflow.add_edge("backup", "scrape")
    workflow.add_edge("scrape", "save")
    workflow.add_conditional_edges(
        "save",
        workflow_successful,
        {
            "diff": "diff",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "diff",
        diff_successful,
        {
            "embedding": "embedding",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "embedding",
        embedding_successful,
        {
            "cleanup": "cleanup",
            "end": END
        }
    )
    workflow.add_edge("cleanup", END)
    
    return workflow


def run_weekly_workflow(config: WeeklyConfig = None) -> WeeklyNKODState:
    """Spustí týdenní workflow jednou"""
    if config is None:
        config = WeeklyConfig()
    
    # Vytvoření workflow
    workflow = create_weekly_workflow(config)
    app = workflow.compile(checkpointer=MemorySaver())
    
    # Konfigurace pro thread
    thread_config = {"configurable": {"thread_id": f"weekly_nkod_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}}
    
    # Inicializační stav
    initial_state = WeeklyNKODState(
        sparql_endpoint="",
        session_headers={},
        output_dir="",
        datasets=[],
        processed_datasets=[],
        statistics={},
        limit=config.nkod_config.limit,
        total_count=0,
        hvd_count=0,
        previous_dataset_exists=False,
        current_dataset_path="",
        previous_dataset_path="",
        backup_created=False,
        workflow_start_time=datetime.now().isoformat(),
        workflow_status="starting",
        error_message=None,
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
        vector_db_initialized=False
    )
    
    logger.info("=== Spouštím týdenní NKOD workflow ===")
    
    try:
        # Spuštění workflow
        final_state = app.invoke(initial_state, thread_config)
        
        if final_state.get("vector_db_initialized", False):
            logger.info("✅ Týdenní workflow úspěšně dokončen!")
            
            # Vypíše statistiky diff
            summary = final_state.get("changes_summary", {})
            if summary:
                logger.info(f"📊 Změny v datech:")
                logger.info(f"  - Nové datasety: {summary.get('new_datasets', 0)}")
                logger.info(f"  - Změněné datasety: {summary.get('modified_datasets', 0)}")
                logger.info(f"  - Smazané datasety: {summary.get('deleted_datasets', 0)}")
                logger.info(f"  - Vráceno nových/změněných: {len(final_state.get('new_or_modified_datasets', []))}")
            
            # Vypíše statistiky embeddingu
            logger.info(f"🔮 Embedding statistiky:")
            logger.info(f"  - Zpracováno embeddingů: {final_state.get('embeddings_processed', 0)}")
            logger.info(f"  - Dimenze vektoru: {final_state.get('embeddings_dimension', 0)}")
            logger.info(f"  - Vektorová DB: {final_state.get('vector_db_type', 'N/A')}")
            logger.info(f"  - Kolekce: {final_state.get('vector_db_collection', 'N/A')}")
            
        elif final_state["workflow_status"] == "error":
            logger.error(f"❌ Workflow selhal: {final_state.get('error_message', 'Neznámá chyba')}")
        elif final_state.get("diff_error"):
            logger.error(f"❌ Diff selhal: {final_state.get('diff_error', 'Neznámá chyba')}")
        elif final_state.get("embeddings_error"):
            logger.error(f"❌ Embedding selhal: {final_state.get('embeddings_error', 'Neznámá chyba')}")
        
        return final_state
        
    except Exception as e:
        logger.error(f"❌ Kritická chyba ve workflow: {e}")
        raise


class WeeklyScheduler:
    """Scheduler pro automatické spouštění každé pondělí"""
    
    def __init__(self, config: WeeklyConfig = None):
        self.config = config or WeeklyConfig()
        self.running = False
        
    def scheduled_job(self):
        """Úloha spouštěná scheduleru"""
        logger.info("🕒 Spouští se naplánovaný týdenní scraping...")
        try:
            run_weekly_workflow(self.config)
        except Exception as e:
            logger.error(f"Chyba v naplánované úloze: {e}")
    
    def start_scheduler(self):
        """Spustí scheduler"""
        if not self.config.enable_scheduler:
            logger.info("Scheduler je zakázán v konfiguraci")
            return
            
        # Naplánování na každé pondělí
        schedule.every().monday.at(self.config.schedule_time).do(self.scheduled_job)
        
        logger.info(f"📅 Scheduler nastaven na každé pondělí v {self.config.schedule_time}")
        logger.info("Scheduler běží... (Ctrl+C pro ukončení)")
        
        self.running = True
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Kontrola každou minutu
        except KeyboardInterrupt:
            logger.info("Scheduler ukončen uživatelem")
        finally:
            self.running = False
    
    def stop_scheduler(self):
        """Zastaví scheduler"""
        self.running = False
        schedule.clear()


def main():
    """Hlavní funkce s CLI rozhraním"""
    import sys
    
    config = WeeklyConfig()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "run":
            # Jednorázové spuštění
            print("Spouštím týdenní workflow jednorázově...")
            run_weekly_workflow(config)
            
        elif command == "schedule":
            # Spuštění scheduleru
            scheduler = WeeklyScheduler(config)
            scheduler.start_scheduler()
            
        elif command == "test":
            # Test s omezeným počtem datasetů
            config.nkod_config.limit = 5
            print("Testovací spuštění s 5 datasety...")
            run_weekly_workflow(config)
            
        elif command == "help":
            print("Týdenní NKOD Scraper - LangGraph workflow")
            print("Použití:")
            print("  python weekly_nkod_workflow.py run       # Jednorázové spuštění")
            print("  python weekly_nkod_workflow.py schedule  # Spustí scheduler (každé pondělí)")
            print("  python weekly_nkod_workflow.py test      # Test s 5 datasety")
            print("  python weekly_nkod_workflow.py help      # Tato nápověda")
            
        else:
            print(f"Neznámý příkaz: {command}")
            print("Použijte 'python weekly_nkod_workflow.py help' pro nápovědu")
            sys.exit(1)
    else:
        # Výchozí chování - jednorázové spuštění
        print("Spouštím týdenní workflow (použijte 'help' pro další možnosti)...")
        run_weekly_workflow(config)


if __name__ == "__main__":
    main()