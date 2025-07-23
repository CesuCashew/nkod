#!/usr/bin/env python3
"""
JSON Diff Node pro LangGraph workflow
Porovnává dataset_previous.json a dataset_current.json a vrací rozdíly
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Literal, Set, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetChange:
    """Reprezentuje změnu v datasetu"""
    change_type: Literal["new", "modified", "deleted"]
    dataset_id: str
    dataset_uri: str
    previous_data: Optional[Dict[str, Any]] = None
    current_data: Optional[Dict[str, Any]] = None
    changed_fields: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Konverze na slovník"""
        return {
            "change_type": self.change_type,
            "dataset_id": self.dataset_id,
            "dataset_uri": self.dataset_uri,
            "previous_data": self.previous_data,
            "current_data": self.current_data,
            "changed_fields": self.changed_fields or []
        }


class JSONDiffAnalyzer:
    """Třída pro efektivní porovnání JSON datasetů"""
    
    def __init__(self):
        self.ignored_fields = {
            "harvested_at",
            "scraper_version"
        }
    
    def _get_dataset_id(self, dataset: Dict[str, Any]) -> str:
        """Získá unikátní ID datasetu pro porovnání"""
        # Priorita: identifier > URI > title hash
        if dataset.get("identifier"):
            return f"id_{dataset['identifier']}"
        elif dataset.get("uri"):
            return f"uri_{dataset['uri']}"
        elif dataset.get("title"):
            title_hash = hashlib.md5(dataset["title"].encode()).hexdigest()[:8]
            return f"title_{title_hash}"
        else:
            # Fallback - hash celého objektu
            obj_str = json.dumps(dataset, sort_keys=True)
            obj_hash = hashlib.md5(obj_str.encode()).hexdigest()[:8]
            return f"hash_{obj_hash}"
    
    def _normalize_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizuje dataset pro porovnání (odstraní ignorované pole)"""
        normalized = {k: v for k, v in dataset.items() 
                     if k not in self.ignored_fields}
        
        # Seřadí seznamy pro konzistentní porovnání
        for key in ["keywords", "themes", "applicableLegislation"]:
            if key in normalized and isinstance(normalized[key], list):
                normalized[key] = sorted(normalized[key])
        
        return normalized
    
    def _calculate_content_hash(self, dataset: Dict[str, Any]) -> str:
        """Vypočítá hash obsahu datasetu pro detekci změn"""
        normalized = self._normalize_dataset(dataset)
        content_str = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _find_changed_fields(self, prev_data: Dict[str, Any], curr_data: Dict[str, Any]) -> List[str]:
        """Najde konkrétní pole, která se změnila"""
        changed_fields = []
        prev_normalized = self._normalize_dataset(prev_data)
        curr_normalized = self._normalize_dataset(curr_data)
        
        # Najde všechny klíče z obou objektů
        all_keys = set(prev_normalized.keys()) | set(curr_normalized.keys())
        
        for key in all_keys:
            prev_val = prev_normalized.get(key)
            curr_val = curr_normalized.get(key)
            
            # Porovnání hodnot
            if prev_val != curr_val:
                changed_fields.append(key)
        
        return sorted(changed_fields)
    
    def compare_datasets(self, previous_datasets: List[Dict[str, Any]], 
                        current_datasets: List[Dict[str, Any]]) -> List[DatasetChange]:
        """Porovná dva seznamy datasetů a vrátí změny"""
        changes = []
        
        # Vytvoří indexy pro rychlé vyhledávání
        prev_index = {}
        curr_index = {}
        
        # Indexování předchozích datasetů
        for dataset in previous_datasets:
            dataset_id = self._get_dataset_id(dataset)
            content_hash = self._calculate_content_hash(dataset)
            prev_index[dataset_id] = {
                "data": dataset,
                "hash": content_hash
            }
        
        # Indexování současných datasetů
        for dataset in current_datasets:
            dataset_id = self._get_dataset_id(dataset)
            content_hash = self._calculate_content_hash(dataset)
            curr_index[dataset_id] = {
                "data": dataset,
                "hash": content_hash
            }
        
        # Najde nové a změněné datasety
        for dataset_id, curr_info in curr_index.items():
            curr_data = curr_info["data"]
            curr_hash = curr_info["hash"]
            
            if dataset_id not in prev_index:
                # Nový dataset
                changes.append(DatasetChange(
                    change_type="new",
                    dataset_id=dataset_id,
                    dataset_uri=curr_data.get("uri", ""),
                    current_data=curr_data
                ))
            else:
                # Existující dataset - kontrola změn
                prev_info = prev_index[dataset_id]
                prev_data = prev_info["data"]
                prev_hash = prev_info["hash"]
                
                if curr_hash != prev_hash:
                    # Dataset se změnil
                    changed_fields = self._find_changed_fields(prev_data, curr_data)
                    changes.append(DatasetChange(
                        change_type="modified",
                        dataset_id=dataset_id,
                        dataset_uri=curr_data.get("uri", ""),
                        previous_data=prev_data,
                        current_data=curr_data,
                        changed_fields=changed_fields
                    ))
        
        # Najde smazané datasety
        for dataset_id, prev_info in prev_index.items():
            if dataset_id not in curr_index:
                prev_data = prev_info["data"]
                changes.append(DatasetChange(
                    change_type="deleted",
                    dataset_id=dataset_id,
                    dataset_uri=prev_data.get("uri", ""),
                    previous_data=prev_data
                ))
        
        return changes


class JSONDiffNode:
    """LangGraph uzel pro porovnání JSON souborů"""
    
    def __init__(self, config=None):
        self.config = config
        self.analyzer = JSONDiffAnalyzer()
        
    def load_json_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Načte JSON soubor"""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Soubor neexistuje: {file_path}")
                return None
                
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"Načten soubor: {file_path}")
                return data
        except Exception as e:
            logger.error(f"Chyba při načítání {file_path}: {e}")
            return None
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Hlavní funkce uzlu pro porovnání JSON souborů"""
        logger.info("🔍 Spouštím porovnání JSON souborů...")
        
        try:
            # Načte soubory
            previous_path = state.get("previous_dataset_path")
            current_path = state.get("current_dataset_path")
            
            if not previous_path or not current_path:
                logger.error("Chybí cesty k souborům pro porovnání")
                state.update({
                    "diff_error": "Missing file paths",
                    "dataset_changes": [],
                    "changes_summary": {}
                })
                return state
            
            # Načte data
            previous_data = self.load_json_file(previous_path)
            current_data = self.load_json_file(current_path)
            
            if previous_data is None:
                logger.info("Předchozí soubor neexistuje - všechny současné datasety jsou nové")
                if current_data and "datasets" in current_data:
                    # Všechny současné datasety jsou nové
                    changes = []
                    for dataset in current_data["datasets"]:
                        dataset_id = self.analyzer._get_dataset_id(dataset)
                        changes.append(DatasetChange(
                            change_type="new",
                            dataset_id=dataset_id,
                            dataset_uri=dataset.get("uri", ""),
                            current_data=dataset
                        ))
                else:
                    changes = []
            else:
                if current_data is None:
                    logger.error("Současný soubor neexistuje")
                    state.update({
                        "diff_error": "Current file not found",
                        "dataset_changes": [],
                        "changes_summary": {}
                    })
                    return state
                
                # Provede porovnání
                prev_datasets = previous_data.get("datasets", [])
                curr_datasets = current_data.get("datasets", [])
                
                logger.info(f"Porovnávám {len(prev_datasets)} předchozích vs {len(curr_datasets)} současných datasetů")
                changes = self.analyzer.compare_datasets(prev_datasets, curr_datasets)
            
            # Vytvoří statistiky
            changes_summary = {
                "total_changes": len(changes),
                "new_datasets": len([c for c in changes if c.change_type == "new"]),
                "modified_datasets": len([c for c in changes if c.change_type == "modified"]),
                "deleted_datasets": len([c for c in changes if c.change_type == "deleted"])
            }
            
            # Konverze na slovníky pro JSON serialization
            changes_as_dicts = [change.to_dict() for change in changes]
            
            # Filtruje jen nové a změněné záznamy (podle požadavku)
            new_or_modified = [
                change.current_data for change in changes 
                if change.change_type in ["new", "modified"] and change.current_data
            ]
            
            logger.info(f"✅ Porovnání dokončeno:")
            logger.info(f"  - Nové datasety: {changes_summary['new_datasets']}")
            logger.info(f"  - Změněné datasety: {changes_summary['modified_datasets']}")  
            logger.info(f"  - Smazané datasety: {changes_summary['deleted_datasets']}")
            
            # Aktualizace stavu
            state.update({
                "dataset_changes": changes_as_dicts,
                "changes_summary": changes_summary,
                "new_or_modified_datasets": new_or_modified,
                "diff_completed": True,
                "workflow_status": "diffing"
            })
            
        except Exception as e:
            error_msg = f"Chyba při porovnání JSON: {e}"
            logger.error(error_msg)
            state.update({
                "diff_error": error_msg,
                "dataset_changes": [],
                "changes_summary": {},
                "diff_completed": False
            })
        
        return state


# Rozšíření stavu pro diff funkcionalitu
class ExtendedWeeklyNKODState(dict):
    """Rozšířený stav obsahující diff informace"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Přidá nové klíče pro diff
        self.setdefault("dataset_changes", [])
        self.setdefault("changes_summary", {})
        self.setdefault("new_or_modified_datasets", [])
        self.setdefault("diff_completed", False)
        self.setdefault("diff_error", None)


def create_test_data():
    """Vytvoří testovací data pro demo"""
    previous_data = {
        "metadata": {
            "created_at": "2025-01-15T09:00:00",
            "total_datasets": 2,
            "scraper_version": "2.1-test"
        },
        "datasets": [
            {
                "uri": "https://data.gov.cz/dataset/1",
                "identifier": "dataset-1",
                "title": "Test Dataset 1",
                "description": "První testovací dataset",
                "keywords": ["test", "data"],
                "themes": ["GOVE"],
                "isHVD": False
            },
            {
                "uri": "https://data.gov.cz/dataset/2", 
                "identifier": "dataset-2",
                "title": "Test Dataset 2",
                "description": "Druhý testovací dataset",
                "keywords": ["example"],
                "themes": ["TECH"],
                "isHVD": True
            }
        ]
    }
    
    current_data = {
        "metadata": {
            "created_at": "2025-01-22T09:00:00",
            "total_datasets": 3,
            "scraper_version": "2.1-test"
        },
        "datasets": [
            {
                "uri": "https://data.gov.cz/dataset/1",
                "identifier": "dataset-1", 
                "title": "Test Dataset 1 - Updated",  # Změna!
                "description": "První testovací dataset s aktualizovaným popisem",  # Změna!
                "keywords": ["test", "data", "updated"],  # Změna!
                "themes": ["GOVE"],
                "isHVD": False
            },
            {
                "uri": "https://data.gov.cz/dataset/2",
                "identifier": "dataset-2",
                "title": "Test Dataset 2", 
                "description": "Druhý testovací dataset",
                "keywords": ["example"],
                "themes": ["TECH"],
                "isHVD": True
            },
            {
                # Nový dataset!
                "uri": "https://data.gov.cz/dataset/3",
                "identifier": "dataset-3",
                "title": "Test Dataset 3 - New",
                "description": "Třetí testovací dataset - zcela nový",
                "keywords": ["new", "fresh"],
                "themes": ["ENVI"],
                "isHVD": False
            }
        ]
    }
    
    return previous_data, current_data


def test_diff_node():
    """Testovací funkce pro diff node"""
    print("🧪 Testování JSON Diff Node...")
    
    # Vytvoří testovací data
    previous_data, current_data = create_test_data()
    
    # Uloží testovací soubory
    test_dir = Path("test_diff_data")
    test_dir.mkdir(exist_ok=True)
    
    prev_path = test_dir / "dataset_previous.json"
    curr_path = test_dir / "dataset_current.json"
    
    with open(prev_path, "w", encoding="utf-8") as f:
        json.dump(previous_data, f, indent=2, ensure_ascii=False)
    
    with open(curr_path, "w", encoding="utf-8") as f:
        json.dump(current_data, f, indent=2, ensure_ascii=False)
    
    # Vytvoří mock stav
    test_state = ExtendedWeeklyNKODState({
        "previous_dataset_path": str(prev_path),
        "current_dataset_path": str(curr_path)
    })
    
    # Spustí diff node
    diff_node = JSONDiffNode()
    result_state = diff_node(test_state)
    
    # Vypíše výsledky
    print("\n📊 Výsledky porovnání:")
    print(f"Celkem změn: {result_state['changes_summary']['total_changes']}")
    print(f"Nové datasety: {result_state['changes_summary']['new_datasets']}")
    print(f"Změněné datasety: {result_state['changes_summary']['modified_datasets']}")
    print(f"Smazané datasety: {result_state['changes_summary']['deleted_datasets']}")
    
    print("\n📝 Detaily změn:")
    for change in result_state["dataset_changes"]:
        print(f"- {change['change_type'].upper()}: {change['dataset_id']}")
        if change["changed_fields"]:
            print(f"  Změněná pole: {', '.join(change['changed_fields'])}")
    
    print(f"\n🆕 Počet nových/změněných datasetů k vrácení: {len(result_state['new_or_modified_datasets'])}")
    
    # Vyčistí testovací soubory
    import shutil
    shutil.rmtree(test_dir)
    
    return result_state


if __name__ == "__main__":
    # Nastavení loggingu
    logging.basicConfig(level=logging.INFO)
    
    # Spustí test
    test_diff_node()