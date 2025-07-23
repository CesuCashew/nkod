#!/usr/bin/env python3
import requests
import os
import sys
import json
import re
from datetime import datetime
from urllib.parse import urlparse

class NKODScraper:
    SPARQL_ENDPOINT = "https://data.gov.cz/sparql"
    OUTPUT_DIR = "nkod_data"
    DEFAULT_LIMIT = 100  # Neomezený počet - stáhne všechna data
    HVD_LEGISLATION_URI = "http://data.europa.eu/eli/reg_impl/2023/138/oj"

    def __init__(self, limit=None):
        self.limit = limit if limit is not None else self.DEFAULT_LIMIT
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(f"{self.OUTPUT_DIR}/datasets", exist_ok=True)
        os.makedirs(f"{self.OUTPUT_DIR}/hvd", exist_ok=True)  # Složka pro HVD datasety
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NKOD-Scraper/2.1-HVD",
            "Accept": "application/sparql-results+json"
        })

    def sparql_query(self, query):
        try:
            r = self.session.post(self.SPARQL_ENDPOINT, data={"query": query}, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"Chyba při dotazu SPARQL: {e}")
            return {}

    def count_datasets(self):
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

    def count_hvd_datasets(self):
        """Spočítá počet HVD datových sad"""
        query = f"""
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX dcatap: <http://data.europa.eu/r5r/>
        SELECT (COUNT(DISTINCT ?dataset) AS ?count) WHERE {{
          ?dataset a dcat:Dataset .
          ?dataset dcatap:hvdCategory ?hvdCategory .
        }}
        """
        result = self.sparql_query(query)
        try:
            return int(result["results"]["bindings"][0]["count"]["value"])
        except:
            return 0

    def list_datasets(self, limit=None):
        actual_limit = limit if limit is not None else self.limit
        
        if actual_limit is None:
            # Neomezený počet - stáhne všechna data
            query = f"""
            PREFIX dcat: <http://www.w3.org/ns/dcat#>
            SELECT DISTINCT ?dataset WHERE {{
              ?dataset a dcat:Dataset .
            }}
            ORDER BY ?dataset
            """
        else:
            query = f"""
            PREFIX dcat: <http://www.w3.org/ns/dcat#>
            SELECT DISTINCT ?dataset WHERE {{
              ?dataset a dcat:Dataset .
            }}
            ORDER BY ?dataset
            LIMIT {actual_limit}
            """
        result = self.sparql_query(query)
        datasets = []
        try:
            for item in result["results"]["bindings"]:
                datasets.append(item["dataset"]["value"])
        except:
            pass
        return datasets

    def is_hvd_dataset(self, metadata):
        """Kontroluje zda je datová sada HVD na základě metadat"""
        # Kontrola přítomnosti HVD kategorie
        if metadata.get("hvdCategory"):
            return True
        
        # Kontrola přítomnosti HVD legislativy
        applicable_legislation = metadata.get("applicableLegislation", [])
        if isinstance(applicable_legislation, str):
            applicable_legislation = [applicable_legislation]
        
        return self.HVD_LEGISLATION_URI in applicable_legislation

    def get_comprehensive_metadata(self, dataset_uri):
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
        
        return metadata

    def get_distributions(self, dataset_uri):
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

    def create_filename_from_metadata(self, metadata):
        """Vytvoří název souboru na základě metadat s prefixem pro HVD"""
        def clean_filename(text):
            # Odstraní diakritiku a nepovolené znaky
            text = re.sub(r'[<>:"/\\|?*]', '', text)
            text = re.sub(r'\s+', '_', text.strip())
            return text[:100]  # Omezí délku na 100 znaků
        
        # Prefix pro HVD datasety
        prefix = "HVD_" if metadata.get("isHVD") else ""
        
        # Pokusíme se najít vhodný název v tomto pořadí:
        # 1. Oficiální identifikátor
        if metadata.get("identifier"):
            return prefix + clean_filename(metadata["identifier"]) + ".json"
        
        # 2. Název datové sady
        if metadata.get("title"):
            return prefix + clean_filename(metadata["title"]) + ".json"
        
        # 3. První klíčové slovo + téma
        if metadata.get("keywords") and metadata.get("themes"):
            keyword = metadata["keywords"][0] if metadata["keywords"] else ""
            theme = self.extract_theme_name(metadata["themes"][0]) if metadata["themes"] else ""
            if keyword and theme:
                return prefix + clean_filename(f"{keyword}_{theme}") + ".json"
        
        # 4. Jen klíčové slovo
        if metadata.get("keywords"):
            return prefix + clean_filename(metadata["keywords"][0]) + ".json"
        
        # 5. Název tématu
        if metadata.get("themes"):
            theme_name = self.extract_theme_name(metadata["themes"][0])
            if theme_name:
                return prefix + clean_filename(theme_name) + ".json"
        
        # 6. Záložní řešení - použije část URI
        uri_parts = metadata["uri"].rstrip("/").split("/")
        if len(uri_parts) > 1:
            return prefix + clean_filename(uri_parts[-1]) + ".json"
        
        # 7. Poslední záložní řešení
        return f"{prefix}dataset_{hash(metadata['uri']) % 10000}.json"

    def extract_theme_name(self, theme_uri):
        """Extrahuje název tématu z URI"""
        if "data-theme" in theme_uri:
            return theme_uri.split("/")[-1].upper()
        elif "eurovoc.europa.eu" in theme_uri:
            return f"eurovoc_{theme_uri.split('/')[-1]}"
        elif "data.europa.eu/bna" in theme_uri:
            return f"hvd_{theme_uri.split('/')[-1]}"
        else:
            return theme_uri.split("/")[-1]

    def save_metadata(self, metadata):
        """Uloží metadata s inteligentním pojmenováním souboru do správné složky"""
        filename = self.create_filename_from_metadata(metadata)
        
        # Určí správnou složku podle typu datasetu
        if metadata.get("isHVD"):
            path = os.path.join(self.OUTPUT_DIR, "hvd", filename)
        else:
            path = os.path.join(self.OUTPUT_DIR, "datasets", filename)
        
        # Přidáme timestamp pro snadné sledování
        metadata["harvested_at"] = datetime.now().isoformat()
        metadata["scraper_version"] = "2.1-HVD"
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return filename

    def print_statistics(self, processed_datasets):
        """Vypíše statistiky o stažených datech včetně HVD informací"""
        print(f"\n=== STATISTIKY STAHOVÁNÍ ===")
        print(f"Celkem zpracováno datových sad: {len(processed_datasets)}")
        
        # Rozdělí datasety podle typu
        hvd_datasets = [d for d in processed_datasets if d.get("isHVD")]
        regular_datasets = [d for d in processed_datasets if not d.get("isHVD")]
        
        print(f"HVD datových sad: {len(hvd_datasets)}")
        print(f"Běžných datových sad: {len(regular_datasets)}")
        
        # Statistiky podle témat
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
        
        if themes:
            print(f"\nNejčastější témata:")
            for theme, count in sorted(themes.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {theme}: {count}")
        
        if publishers:
            print(f"\nNejčastější poskytovatelé:")
            for publisher, count in sorted(publishers.items(), key=lambda x: x[1], reverse=True)[:5]:
                publisher_name = publisher.split("/")[-1] if "/" in publisher else publisher
                print(f"  {publisher_name}: {count}")
        
        if hvd_categories:
            print(f"\nHVD kategorie:")
            for category, count in sorted(hvd_categories.items(), key=lambda x: x[1], reverse=True):
                print(f"  {category}: {count}")

    def run(self):
        total_count = self.count_datasets()
        hvd_count = self.count_hvd_datasets()
        
        if self.limit is None:
            actual_limit = total_count
            limit_text = "všechna data"
        else:
            actual_limit = min(self.limit, total_count)
            limit_text = f"{actual_limit} datových sad"
        
        print(f"Celkový počet datových sad v NKOD: {total_count}")
        print(f"Celkový počet HVD datových sad: {hvd_count}")
        print(f"Stáhnu: {limit_text}")
        print(f"Výstupní adresář: {self.OUTPUT_DIR}")
        print(f"  - Běžné datasety: {self.OUTPUT_DIR}/datasets/")
        print(f"  - HVD datasety: {self.OUTPUT_DIR}/hvd/")
        print()
        
        datasets = self.list_datasets(self.limit)
        processed_datasets = []
        hvd_counter = 0
        
        for i, ds in enumerate(datasets, 1):
            try:
                print(f"Stažení {i}/{len(datasets)}: ", end="", flush=True)
                metadata = self.get_comprehensive_metadata(ds)
                filename = self.save_metadata(metadata)
                processed_datasets.append(metadata)
                
                if metadata.get("isHVD"):
                    hvd_counter += 1
                    hvd_marker = " [HVD]"
                else:
                    hvd_marker = ""
                
                # Zobrazí název datové sady pokud je k dispozici
                title = metadata.get("title", "Bez názvu")[:50]
                print(f"{title}{hvd_marker} -> {filename}")
                
            except Exception as e:
                print(f"CHYBA při {ds}: {e}")
                continue
        
        print(f"\nPřehled zpracování:")
        print(f"  Staženo celkem: {len(processed_datasets)} datových sad")
        print(f"  HVD datových sad: {hvd_counter}")
        print(f"  Běžných datových sad: {len(processed_datasets) - hvd_counter}")
        
        self.print_statistics(processed_datasets)
        print(f"\nStahování dokončeno. Soubory uloženy v:")
        print(f"  - Běžné datasety: {self.OUTPUT_DIR}/datasets/")
        print(f"  - HVD datasety: {self.OUTPUT_DIR}/hvd/")

def main():
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "data":
            # Zobrazí počet datových sad včetně HVD
            scraper = NKODScraper()
            total_count = scraper.count_datasets()
            hvd_count = scraper.count_hvd_datasets()
            print(f"Počet datových sad v NKOD: {total_count}")
            print(f"Počet HVD datových sad: {hvd_count}")
            print(f"Počet běžných datových sad: {total_count - hvd_count}")
            sys.exit(0)
        
        elif command == "help" or command == "--help":
            print("NKOD Scraper - Enhanced version 2.1 s podporou HVD")
            print("Použití:")
            print("  python nkod_scraper.py                 # Stáhne VŠECHNA dostupná data")
            print("  python nkod_scraper.py <počet>         # Stáhne zadaný počet datových sad")
            print("  python nkod_scraper.py data            # Jen zobrazí celkový počet včetně HVD")
            print("  python nkod_scraper.py help            # Zobrazí tuto nápovědu")
            print()
            print("Funkce:")
            print("  - Automatické rozpoznání HVD datových sad")
            print("  - Ukládání HVD do složky nkod_data/hvd/")
            print("  - Ukládání běžných datových sad do nkod_data/datasets/")
            print("  - Inteligentní pojmenovávání souborů s HVD_ prefixem")
            print("  - Rozšířené statistiky o HVD kategoriích")
            print()
            print("HVD datasety jsou rozpoznány podle:")
            print("  - Přítomnosti dcatap:hvdCategory")
            print("  - Přítomnosti dcatap:applicableLegislation s HVD nařízením")
            sys.exit(0)
        
        else:
            # Pokusíme se interpretovat jako číslo
            try:
                limit = int(command)
                if limit <= 0:
                    print("Počet datových sad musí být kladné číslo")
                    sys.exit(1)
                scraper = NKODScraper(limit=limit)
            except ValueError:
                print(f"Neplatný argument: {command}")
                print("Použijte 'python nkod_scraper.py help' pro nápovědu")
                sys.exit(1)
    else:
        # Výchozí chování - stáhne VŠECHNA dostupná data
        scraper = NKODScraper()
    
    # Spuštění stahování
    scraper.run()

if __name__ == "__main__":
    main()