#!/usr/bin/env python3
"""
NKOD Scraper rozdělený do LangChain uzlů pro LangGraph
"""

import requests
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class NKODState(TypedDict):
    """Stav pro LangGraph workflow"""
    sparql_endpoint: str
    session_headers: Dict[str, str]
    output_dir: str
    datasets: List[str]
    processed_datasets: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    limit: int
    total_count: int
    hvd_count: int


class NKODConfig(BaseModel):
    """Konfigurace pro NKOD scraper"""
    sparql_endpoint: str = "https://data.gov.cz/sparql"
    output_dir: str = "nkod_data"
    limit: Optional[int] = None
    hvd_legislation_uri: str = "http://data.europa.eu/eli/reg_impl/2023/138/oj"
    
    class Config:
        arbitrary_types_allowed = True


class DataFetchNode:
    """Uzel pro stahování dat z NKOD pomocí SPARQL dotazů"""
    
    def __init__(self, config: NKODConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NKOD-Scraper/2.1-HVD-LangChain",
            "Accept": "application/sparql-results+json"
        })
    
    def sparql_query(self, query: str) -> Dict[str, Any]:
        """Vykoná SPARQL dotaz"""
        try:
            r = self.session.post(
                self.config.sparql_endpoint, 
                data={"query": query}, 
                timeout=30
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Chyba při dotazu SPARQL: {e}")
            return {}
    
    def count_datasets(self) -> int:
        """Spočítá celkový počet datasetů"""
        query = """
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        SELECT (COUNT(DISTINCT ?dataset) AS ?count) WHERE {
          ?dataset a dcat:Dataset .
        }
        """
        result = self.sparql_query(query)
        try:
            return int(result["results"]["bindings"][0]["count"]["value"])
        except:
            return 0
    
    def count_hvd_datasets(self) -> int:
        """Spočítá počet HVD datových sad"""
        query = """
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX dcatap: <http://data.europa.eu/r5r/>
        SELECT (COUNT(DISTINCT ?dataset) AS ?count) WHERE {
          ?dataset a dcat:Dataset .
          ?dataset dcatap:hvdCategory ?hvdCategory .
        }
        """
        result = self.sparql_query(query)
        try:
            return int(result["results"]["bindings"][0]["count"]["value"])
        except:
            return 0
    
    def list_datasets(self, limit: int = None) -> List[str]:
        """Získá seznam URI datasetů"""
        if limit is None:
            query = """
            PREFIX dcat: <http://www.w3.org/ns/dcat#>
            SELECT DISTINCT ?dataset WHERE {
              ?dataset a dcat:Dataset .
            }
            ORDER BY ?dataset
            """
        else:
            query = f"""
            PREFIX dcat: <http://www.w3.org/ns/dcat#>
            SELECT DISTINCT ?dataset WHERE {{
              ?dataset a dcat:Dataset .
            }}
            ORDER BY ?dataset
            LIMIT {limit}
            """
        
        result = self.sparql_query(query)
        datasets = []
        try:
            for item in result["results"]["bindings"]:
                datasets.append(item["dataset"]["value"])
        except:
            pass
        return datasets
    
    def __call__(self, state: NKODState) -> NKODState:
        """Spustí stahování základních informací"""
        # Inicializace výstupních adresářů
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(f"{self.config.output_dir}/datasets", exist_ok=True)
        os.makedirs(f"{self.config.output_dir}/hvd", exist_ok=True)
        
        # Spočítá datasety
        total_count = self.count_datasets()
        hvd_count = self.count_hvd_datasets()
        
        # Získá seznam datasetů
        datasets = self.list_datasets(self.config.limit)
        
        # Aktualizuje stav
        state.update({
            "sparql_endpoint": self.config.sparql_endpoint,
            "session_headers": dict(self.session.headers),
            "output_dir": self.config.output_dir,
            "datasets": datasets,
            "total_count": total_count,
            "hvd_count": hvd_count,
            "limit": self.config.limit or total_count
        })
        
        return state


class MetadataParseNode:
    """Uzel pro parsování metadat jednotlivých datasetů"""
    
    def __init__(self, config: NKODConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NKOD-Scraper/2.1-HVD-LangChain",
            "Accept": "application/sparql-results+json"
        })
    
    def sparql_query(self, query: str) -> Dict[str, Any]:
        """Vykoná SPARQL dotaz"""
        try:
            r = self.session.post(
                self.config.sparql_endpoint, 
                data={"query": query}, 
                timeout=30
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Chyba při dotazu SPARQL: {e}")
            return {}
    
    def is_hvd_dataset(self, metadata: Dict[str, Any]) -> bool:
        """Kontroluje zda je datová sada HVD na základě metadat"""
        if metadata.get("hvdCategory"):
            return True
        
        applicable_legislation = metadata.get("applicableLegislation", [])
        if isinstance(applicable_legislation, str):
            applicable_legislation = [applicable_legislation]
        
        return self.config.hvd_legislation_uri in applicable_legislation
    
    def get_comprehensive_metadata(self, dataset_uri: str) -> Dict[str, Any]:
        """Stáhne kompletní metadata podle DCAT-AP-CZ specifikace včetně HVD položek"""
        query = f"""
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX vcard: <http://www.w3.org/2006/vcard/ns#>
        PREFIX dcatap: <http://data.europa.eu/r5r/>
        
        SELECT ?p ?o ?lang WHERE {{
          <{dataset_uri}> ?p ?o .
          OPTIONAL {{ BIND(LANG(?o) AS ?lang) }}
          FILTER (?p IN (
            dct:title, dct:description, dct:publisher, dcat:keyword, 
            dcat:theme, dct:accrualPeriodicity, dct:spatial, dct:temporal,
            dcat:contactPoint, foaf:page, dct:conformsTo, dct:identifier,
            dcatap:hvdCategory, dcatap:applicableLegislation, dcat:inSeries,
            dcat:spatialResolutionInMeters, dcat:temporalResolution
          ))
        }}
        """
        result = self.sparql_query(query)
        metadata = {
            "uri": dataset_uri, 
            "keywords": [], 
            "themes": [],
            "applicableLegislation": []
        }
        
        for b in result.get("results", {}).get("bindings", []):
            p = b["p"]["value"]
            o = b["o"]["value"]
            lang = b.get("lang", {}).get("value", "")
            
            if p.endswith("title"):
                if not metadata.get("title") or lang == "cs":
                    metadata["title"] = o
            elif p.endswith("description"):
                if not metadata.get("description") or lang == "cs":
                    metadata["description"] = o
            elif p.endswith("publisher"):
                metadata["publisher"] = o
            elif p.endswith("keyword"):
                if o not in metadata["keywords"]:
                    metadata["keywords"].append(o)
            elif p.endswith("theme"):
                if o not in metadata["themes"]:
                    metadata["themes"].append(o)
            elif p.endswith("accrualPeriodicity"):
                metadata["accrualPeriodicity"] = o
            elif p.endswith("spatial"):
                metadata["spatial"] = o
            elif p.endswith("temporal"):
                metadata["temporal"] = o
            elif p.endswith("contactPoint"):
                metadata["contactPoint"] = o
            elif p.endswith("page"):
                metadata["documentation"] = o
            elif p.endswith("conformsTo"):
                metadata["specification"] = o
            elif p.endswith("identifier"):
                metadata["identifier"] = o
            elif p.endswith("hvdCategory"):
                metadata["hvdCategory"] = o
            elif p.endswith("applicableLegislation"):
                if o not in metadata["applicableLegislation"]:
                    metadata["applicableLegislation"].append(o)
            elif p.endswith("inSeries"):
                metadata["inSeries"] = o
            elif p.endswith("spatialResolutionInMeters"):
                metadata["spatialResolutionInMeters"] = o
            elif p.endswith("temporalResolution"):
                metadata["temporalResolution"] = o
        
        # Získáme také distribuce
        metadata["distributions"] = self.get_distributions(dataset_uri)
        
        # Označíme zda je to HVD dataset
        metadata["isHVD"] = self.is_hvd_dataset(metadata)
        
        # Přidáme timestamp a verzi
        metadata["harvested_at"] = datetime.now().isoformat()
        metadata["scraper_version"] = "2.1-HVD-LangChain"
        
        return metadata
    
    def get_distributions(self, dataset_uri: str) -> List[Dict[str, Any]]:
        """Získá seznam distribucí pro datovou sadu včetně HVD informací"""
        query = f"""
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX dcatap: <http://data.europa.eu/r5r/>
        
        SELECT ?distribution ?title ?downloadURL ?accessURL ?format ?mediaType ?applicableLegislation WHERE {{
          <{dataset_uri}> dcat:distribution ?distribution .
          OPTIONAL {{ ?distribution dct:title ?title }}
          OPTIONAL {{ ?distribution dcat:downloadURL ?downloadURL }}
          OPTIONAL {{ ?distribution dcat:accessURL ?accessURL }}
          OPTIONAL {{ ?distribution dct:format ?format }}
          OPTIONAL {{ ?distribution dcat:mediaType ?mediaType }}
          OPTIONAL {{ ?distribution dcatap:applicableLegislation ?applicableLegislation }}
        }}
        """
        result = self.sparql_query(query)
        distributions = []
        
        for b in result.get("results", {}).get("bindings", []):
            dist = {"uri": b["distribution"]["value"]}
            for key in ["title", "downloadURL", "accessURL", "format", "mediaType", "applicableLegislation"]:
                if key in b:
                    dist[key] = b[key]["value"]
            distributions.append(dist)
            
        return distributions
    
    def __call__(self, state: NKODState) -> NKODState:
        """Zpracuje metadata pro všechny datasety"""
        processed_datasets = []
        
        for i, dataset_uri in enumerate(state["datasets"], 1):
            try:
                print(f"Parsování {i}/{len(state['datasets'])}: ", end="", flush=True)
                metadata = self.get_comprehensive_metadata(dataset_uri)
                processed_datasets.append(metadata)
                
                hvd_marker = " [HVD]" if metadata.get("isHVD") else ""
                title = metadata.get("title", "Bez názvu")[:50]
                print(f"{title}{hvd_marker}")
                
            except Exception as e:
                print(f"CHYBA při {dataset_uri}: {e}")
                continue
        
        state["processed_datasets"] = processed_datasets
        return state


class OutputNode:
    """Uzel pro výstup dat do souborů a generování statistik"""
    
    def __init__(self, config: NKODConfig):
        self.config = config
    
    def create_filename_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Vytvoří název souboru na základě metadat s prefixem pro HVD"""
        def clean_filename(text):
            text = re.sub(r'[<>:"/\\|?*]', '', text)
            text = re.sub(r'\s+', '_', text.strip())
            return text[:100]
        
        prefix = "HVD_" if metadata.get("isHVD") else ""
        
        if metadata.get("identifier"):
            return prefix + clean_filename(metadata["identifier"]) + ".json"
        
        if metadata.get("title"):
            return prefix + clean_filename(metadata["title"]) + ".json"
        
        if metadata.get("keywords") and metadata.get("themes"):
            keyword = metadata["keywords"][0] if metadata["keywords"] else ""
            theme = self.extract_theme_name(metadata["themes"][0]) if metadata["themes"] else ""
            if keyword and theme:
                return prefix + clean_filename(f"{keyword}_{theme}") + ".json"
        
        if metadata.get("keywords"):
            return prefix + clean_filename(metadata["keywords"][0]) + ".json"
        
        if metadata.get("themes"):
            theme_name = self.extract_theme_name(metadata["themes"][0])
            if theme_name:
                return prefix + clean_filename(theme_name) + ".json"
        
        uri_parts = metadata["uri"].rstrip("/").split("/")
        if len(uri_parts) > 1:
            return prefix + clean_filename(uri_parts[-1]) + ".json"
        
        return f"{prefix}dataset_{hash(metadata['uri']) % 10000}.json"
    
    def extract_theme_name(self, theme_uri: str) -> str:
        """Extrahuje název tématu z URI"""
        if "data-theme" in theme_uri:
            return theme_uri.split("/")[-1].upper()
        elif "eurovoc.europa.eu" in theme_uri:
            return f"eurovoc_{theme_uri.split('/')[-1]}"
        elif "data.europa.eu/bna" in theme_uri:
            return f"hvd_{theme_uri.split('/')[-1]}"
        else:
            return theme_uri.split("/")[-1]
    
    def save_metadata(self, metadata: Dict[str, Any]) -> str:
        """Uloží metadata s inteligentním pojmenováním souboru do správné složky"""
        filename = self.create_filename_from_metadata(metadata)
        
        if metadata.get("isHVD"):
            path = os.path.join(self.config.output_dir, "hvd", filename)
        else:
            path = os.path.join(self.config.output_dir, "datasets", filename)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return filename
    
    def generate_statistics(self, processed_datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generuje statistiky o zpracovaných datech"""
        hvd_datasets = [d for d in processed_datasets if d.get("isHVD")]
        regular_datasets = [d for d in processed_datasets if not d.get("isHVD")]
        
        themes = {}
        publishers = {}
        hvd_categories = {}
        
        for metadata in processed_datasets:
            # Témata
            for theme in metadata.get("themes", []):
                theme_name = self.extract_theme_name(theme)
                themes[theme_name] = themes.get(theme_name, 0) + 1
            
            # Poskytovatelé
            publisher = metadata.get("publisher", "Neznámý")
            publishers[publisher] = publishers.get(publisher, 0) + 1
            
            # HVD kategorie
            if metadata.get("hvdCategory"):
                hvd_cat = self.extract_theme_name(metadata["hvdCategory"])
                hvd_categories[hvd_cat] = hvd_categories.get(hvd_cat, 0) + 1
        
        return {
            "total_processed": len(processed_datasets),
            "hvd_count": len(hvd_datasets),
            "regular_count": len(regular_datasets),
            "themes": themes,
            "publishers": publishers,
            "hvd_categories": hvd_categories
        }
    
    def print_statistics(self, statistics: Dict[str, Any]):
        """Vypíše statistiky"""
        print(f"\n=== STATISTIKY STAHOVÁNÍ ===")
        print(f"Celkem zpracováno datových sad: {statistics['total_processed']}")
        print(f"HVD datových sad: {statistics['hvd_count']}")
        print(f"Běžných datových sad: {statistics['regular_count']}")
        
        if statistics["themes"]:
            print(f"\nNejčastější témata:")
            for theme, count in sorted(statistics["themes"].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {theme}: {count}")
        
        if statistics["publishers"]:
            print(f"\nNejčastější poskytovatelé:")
            for publisher, count in sorted(statistics["publishers"].items(), key=lambda x: x[1], reverse=True)[:5]:
                publisher_name = publisher.split("/")[-1] if "/" in publisher else publisher
                print(f"  {publisher_name}: {count}")
        
        if statistics["hvd_categories"]:
            print(f"\nHVD kategorie:")
            for category, count in sorted(statistics["hvd_categories"].items(), key=lambda x: x[1], reverse=True):
                print(f"  {category}: {count}")
    
    def __call__(self, state: NKODState) -> NKODState:
        """Uloží data a vygeneruje statistiky"""
        saved_files = []
        
        for metadata in state["processed_datasets"]:
            try:
                filename = self.save_metadata(metadata)
                saved_files.append(filename)
            except Exception as e:
                print(f"Chyba při ukládání {metadata.get('uri', 'unknown')}: {e}")
        
        # Vygeneruje statistiky
        statistics = self.generate_statistics(state["processed_datasets"])
        self.print_statistics(statistics)
        
        print(f"\nStahování dokončeno. Soubory uloženy v:")
        print(f"  - Běžné datasety: {self.config.output_dir}/datasets/")
        print(f"  - HVD datasety: {self.config.output_dir}/hvd/")
        
        state["statistics"] = statistics
        return state


# Pomocné funkce pro vytvoření workflow
def create_nkod_workflow(config: NKODConfig = None):
    """Vytvoří LangGraph workflow pro NKOD scraping"""
    if config is None:
        config = NKODConfig()
    
    # Vytvoří uzly
    fetch_node = DataFetchNode(config)
    parse_node = MetadataParseNode(config)
    output_node = OutputNode(config)
    
    return {
        "data_fetch": fetch_node,
        "metadata_parse": parse_node,
        "output": output_node
    }


def run_nkod_workflow(limit: int = None):
    """Spustí kompletní NKOD workflow"""
    config = NKODConfig(limit=limit)
    nodes = create_nkod_workflow(config)
    
    # Inicializační stav
    state = NKODState(
        sparql_endpoint="",
        session_headers={},
        output_dir="",
        datasets=[],
        processed_datasets=[],
        statistics={},
        limit=limit or 0,
        total_count=0,
        hvd_count=0
    )
    
    print(f"Spouštím NKOD scraping workflow...")
    
    # Spustí uzly postupně
    state = nodes["data_fetch"](state)
    print(f"Nalezeno {len(state['datasets'])} datasetů k zpracování")
    
    state = nodes["metadata_parse"](state)
    print(f"Zpracováno {len(state['processed_datasets'])} datasetů")
    
    state = nodes["output"](state)
    
    return state


if __name__ == "__main__":
    # Příklad použití
    import sys
    
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print("Použití: python nkod_langchain_nodes.py [počet_datasetů]")
            sys.exit(1)
    
    run_nkod_workflow(limit)