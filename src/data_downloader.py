#!/usr/bin/env python3
"""
Data Downloader pro NKOD
========================

Skript pro stahování skutečných datových souborů na základě metadat 
z JSON souborů vytvořených scraperem.

Autor: Claude AI
Datum: 2025-07-23
"""

import os
import json
import requests
import urllib.parse
from pathlib import Path
import time
from urllib.parse import urlparse
import logging

# Konfigurace
INPUT_DIR = "nkod_data/datasets"  # Složka s JSON soubory
OUTPUT_DIR = "downloaded_data"    # Složka pro stažená data
LOG_FILE = "data_downloader.log"

# Nastavení logování
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_filename_from_url(url):
    """
    Extrahuje název souboru z URL.
    
    Args:
        url (str): URL adresa souboru
        
    Returns:
        str: Název souboru nebo fallback název
    """
    try:
        # Parsování URL
        parsed_url = urlparse(url)
        
        # Získání názvu souboru z cesty
        filename = os.path.basename(parsed_url.path)
        
        # Pokud není název souboru, vytvoříme ho z domény a hash
        if not filename or filename == '/':
            domain = parsed_url.netloc.replace('.', '_')
            # Použijeme hash URL pro jedinečnost
            url_hash = str(hash(url))[-8:]
            filename = f"{domain}_{url_hash}.data"
        
        # Ošetření speciálních znaků v názvu souboru
        filename = "".join(c for c in filename if c.isalnum() or c in '.-_')
        
        return filename
    except Exception as e:
        logger.warning(f"Chyba při extrakci názvu souboru z URL {url}: {e}")
        return f"unknown_{str(hash(url))[-8:]}.data"

def download_file(url, output_path, timeout=30):
    """
    Stáhne soubor z URL a uloží ho do zadané cesty.
    
    Args:
        url (str): URL souboru ke stažení
        output_path (str): Cesta pro uložení souboru
        timeout (int): Timeout pro požadavek v sekundách
        
    Returns:
        bool: True pokud bylo stažení úspěšné, False jinak
    """
    try:
        logger.info(f"Stahuji: {url}")
        
        # HTTP GET požadavek s timeoutem
        headers = {
            'User-Agent': 'NKOD Data Downloader 1.0'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Uložení souboru
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        file_size = os.path.getsize(output_path)
        logger.info(f"Úspěšně staženo: {output_path} ({file_size} bytů)")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Chyba při stahování {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Neočekávaná chyba při stahování {url}: {e}")
        return False

def process_json_file(json_path):
    """
    Zpracuje jeden JSON soubor a stáhne všechny dostupné distribuce.
    
    Args:
        json_path (str): Cesta k JSON souboru
        
    Returns:
        dict: Statistiky stahování pro tento soubor
    """
    stats = {
        'processed': 0,
        'downloaded': 0,
        'failed': 0,
        'no_download_url': 0
    }
    
    try:
        # Načtení JSON souboru
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Získání distribucí
        distributions = data.get('distributions', [])
        
        if not distributions:
            logger.warning(f"Žádné distribuce v {json_path}")
            return stats
        
        # Zpracování každé distribuce
        for dist in distributions:
            stats['processed'] += 1
            
            download_url = dist.get('downloadURL')
            if not download_url:
                stats['no_download_url'] += 1
                logger.warning(f"Žádná downloadURL v distribuci: {dist.get('uri', 'N/A')}")
                continue
            
            # Vytvoření názvu souboru
            filename = extract_filename_from_url(download_url)
            output_path = os.path.join(OUTPUT_DIR, filename)
            
            # Kontrola, zda soubor už existuje
            if os.path.exists(output_path):
                logger.info(f"Soubor již existuje, přeskakuji: {output_path}")
                stats['downloaded'] += 1
                continue
            
            # Stažení souboru
            if download_file(download_url, output_path):
                stats['downloaded'] += 1
            else:
                stats['failed'] += 1
            
            # Malá pauza mezi požadavky
            time.sleep(0.5)
    
    except json.JSONDecodeError as e:
        logger.error(f"Chyba při parsování JSON {json_path}: {e}")
    except Exception as e:
        logger.error(f"Chyba při zpracování {json_path}: {e}")
    
    return stats

def main():
    """
    Hlavní funkce skriptu.
    """
    logger.info("Spouštím NKOD Data Downloader")
    
    # Kontrola vstupní složky
    if not os.path.exists(INPUT_DIR):
        logger.error(f"Vstupní složka neexistuje: {INPUT_DIR}")
        return
    
    # Vytvoření výstupní složky
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Výstupní složka: {OUTPUT_DIR}")
    
    # Získání seznamu JSON souborů
    json_files = []
    for file_name in os.listdir(INPUT_DIR):
        if file_name.endswith('.json'):
            json_files.append(os.path.join(INPUT_DIR, file_name))
    
    if not json_files:
        logger.warning(f"Žádné JSON soubory ve složce: {INPUT_DIR}")
        return
    
    logger.info(f"Nalezeno {len(json_files)} JSON souborů")
    
    # Celkové statistiky
    total_stats = {
        'files_processed': 0,
        'distributions_processed': 0,
        'files_downloaded': 0,
        'files_failed': 0,
        'no_download_url': 0
    }
    
    # Zpracování každého JSON souboru
    for json_file in json_files:
        logger.info(f"Zpracovávám: {os.path.basename(json_file)}")
        
        file_stats = process_json_file(json_file)
        
        # Aktualizace celkových statistik
        total_stats['files_processed'] += 1
        total_stats['distributions_processed'] += file_stats['processed']
        total_stats['files_downloaded'] += file_stats['downloaded']
        total_stats['files_failed'] += file_stats['failed']
        total_stats['no_download_url'] += file_stats['no_download_url']
    
    # Výpis finálních statistik
    logger.info("=" * 50)
    logger.info("FINÁLNÍ STATISTIKY")
    logger.info("=" * 50)
    logger.info(f"Zpracovaných JSON souborů: {total_stats['files_processed']}")
    logger.info(f"Zpracovaných distribucí: {total_stats['distributions_processed']}")
    logger.info(f"Úspěšně stažených souborů: {total_stats['files_downloaded']}")
    logger.info(f"Neúspěšných stažení: {total_stats['files_failed']}")
    logger.info(f"Distribucí bez downloadURL: {total_stats['no_download_url']}")
    
    if total_stats['distributions_processed'] > 0:
        success_rate = (total_stats['files_downloaded'] / total_stats['distributions_processed']) * 100
        logger.info(f"Úspěšnost stahování: {success_rate:.1f}%")
    
    logger.info("NKOD Data Downloader dokončen")

if __name__ == "__main__":
    main()