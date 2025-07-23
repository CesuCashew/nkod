#!/usr/bin/env python3
"""
Czech Dataset Financial Potential Evaluator
Evaluates the financial potential of Czech datasets using local LLM via Ollama
"""

import json
import time
import csv
import sys
import subprocess
import requests
from typing import List, Dict, Tuple
import argparse
from pathlib import Path


class OllamaDatasetEvaluator:
    def __init__(self, model_name: str = "llama3", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.api_endpoint = f"{ollama_url}/api/generate"
        
        # Czech system prompt for financial evaluation
        self.system_prompt = """Jsi ekonomický poradce pro startupy v České republice. Pro daný název datasetu ohodnoť jeho potenciál přinášet zisk ve firemním nebo komerčním využití v Česku. Hodnoť jako ve škole:
1 – vynikající ziskový potenciál,
2 – silný potenciál,
3 – mírný,
4 – slabý,
5 – téměř žádný.
Vrať výstup ve formátu: "Název datasetu | Hodnocení (1–5) | Stručné zdůvodnění"."""

    def check_ollama_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def start_ollama_server(self) -> bool:
        """Attempt to start Ollama server"""
        print("🚀 Spouštím Ollama server...")
        try:
            # Start Ollama in background
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for server to start
            for i in range(30):  # 30 second timeout
                if self.check_ollama_running():
                    print("✅ Ollama server běží!")
                    return True
                time.sleep(1)
                if i % 5 == 0:
                    print(f"⏳ Čekám na spuštění serveru... ({i}s)")
            
            print("❌ Nepodařilo se spustit Ollama server")
            return False
        except Exception as e:
            print(f"❌ Chyba při spouštění Ollama: {e}")
            return False

    def ensure_model_available(self) -> bool:
        """Check if model is available, pull if necessary"""
        try:
            # Check if model exists
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                
                if self.model_name not in model_names:
                    print(f"📥 Stahuji model {self.model_name}...")
                    pull_response = requests.post(
                        f"{self.ollama_url}/api/pull",
                        json={"name": self.model_name},
                        stream=True
                    )
                    
                    for line in pull_response.iter_lines():
                        if line:
                            data = json.loads(line)
                            status = data.get("status", "")
                            if "pulling" in status or "downloading" in status:
                                print(f"  {status}")
                    
                    print(f"✅ Model {self.model_name} je připraven!")
                return True
        except Exception as e:
            print(f"❌ Chyba při kontrole/stahování modelu: {e}")
            return False

    def query_llm(self, dataset_name: str, max_retries: int = 3) -> Tuple[int, str]:
        """Query the LLM for dataset evaluation"""
        prompt = f"{self.system_prompt}\n\nNázev datasetu: {dataset_name}"
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_endpoint,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.3  # Lower temperature for more consistent ratings
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    full_response = result.get("response", "")
                    
                    # Parse the response
                    rating, justification = self.parse_llm_response(full_response, dataset_name)
                    return rating, justification
                else:
                    print(f"⚠️  API chyba: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                print(f"⏰ Timeout při dotazu na {dataset_name} (pokus {attempt + 1}/{max_retries})")
            except Exception as e:
                print(f"❌ Chyba při dotazu: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2)
        
        return 5, "Nepodařilo se získat hodnocení"

    def parse_llm_response(self, response: str, dataset_name: str) -> Tuple[int, str]:
        """Parse LLM response to extract rating and justification"""
        try:
            # Try to parse the expected format: "Dataset name | Rating (1-5) | Justification"
            parts = response.strip().split("|")
            if len(parts) >= 3:
                # Extract rating from second part
                rating_part = parts[1].strip()
                # Look for number 1-5
                for i in range(1, 6):
                    if str(i) in rating_part:
                        rating = i
                        justification = parts[2].strip()
                        return rating, justification
            
            # Fallback: look for rating number in the entire response
            for i in range(1, 6):
                if f"hodnocení: {i}" in response.lower() or f"rating: {i}" in response.lower():
                    # Extract justification as everything after the rating
                    justification = response.split(str(i), 1)[1].strip()
                    return i, justification[:200]  # Limit length
            
            # If no clear rating found, return default
            return 3, response[:200] if response else "Nelze určit hodnocení"
            
        except Exception as e:
            print(f"⚠️  Chyba při parsování odpovědi: {e}")
            return 3, "Chyba při parsování odpovědi"

    def load_datasets(self, dataset_path: str) -> List[str]:
        """Load dataset names from JSON files or text file"""
        datasets = []
        try:
            path = Path(dataset_path)
            
            if path.is_dir():
                # Load from directory of JSON files
                json_files = list(path.glob("*.json"))
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            title = data.get("title", "")
                            if title:
                                datasets.append(title)
                    except Exception as e:
                        print(f"⚠️  Chyba při načítání {json_file}: {e}")
                        continue
                        
            elif path.is_file() and path.suffix == '.txt':
                # Load from text file (original format)
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    datasets = [line.strip() for line in f if line.strip()]
            else:
                print(f"❌ Neplatná cesta: {dataset_path}")
                return []
                
            print(f"📋 Načteno {len(datasets)} datasetů")
            return datasets
            
        except FileNotFoundError:
            print(f"❌ Cesta {dataset_path} nenalezena!")
            return []
        except Exception as e:
            print(f"❌ Chyba při načítání datasetů: {e}")
            return []

    def save_results(self, results: List[Dict], filename: str = "results.csv"):
        """Save results to CSV file"""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['dataset_name', 'rating', 'justification']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(results)
            
            print(f"💾 Výsledky uloženy do {filename}")
        except Exception as e:
            print(f"❌ Chyba při ukládání výsledků: {e}")

    def evaluate_datasets(self, dataset_path: str = "datasets.txt", output_file: str = "results.csv"):
        """Main evaluation process"""
        print("🔍 Czech Dataset Financial Evaluator")
        print("=" * 50)
        
        # Check and start Ollama if needed
        if not self.check_ollama_running():
            if not self.start_ollama_server():
                print("❌ Nelze spustit Ollama server. Ujistěte se, že je Ollama nainstalována.")
                print("   Instalace: brew install ollama")
                return
        
        # Ensure model is available
        if not self.ensure_model_available():
            print("❌ Nelze připravit model")
            return
        
        # Load datasets
        datasets = self.load_datasets(dataset_path)
        if not datasets:
            print("❌ Žádné datasety k vyhodnocení")
            return
        
        # Evaluate each dataset
        results = []
        print(f"\n🤖 Vyhodnocuji datasety pomocí modelu {self.model_name}...")
        print("-" * 50)
        
        for i, dataset_name in enumerate(datasets, 1):
            print(f"\n[{i}/{len(datasets)}] Analyzuji: {dataset_name}")
            
            rating, justification = self.query_llm(dataset_name)
            
            results.append({
                'dataset_name': dataset_name,
                'rating': rating,
                'justification': justification
            })
            
            # Print progress
            stars = "⭐" * (6 - rating)  # More stars = better rating
            print(f"   Hodnocení: {rating}/5 {stars}")
            print(f"   Zdůvodnění: {justification[:100]}...")
            
            # Small delay to avoid overwhelming the server
            if i < len(datasets):
                time.sleep(0.5)
        
        # Save results
        print("\n" + "=" * 50)
        self.save_results(results, output_file)
        
        # Print summary
        print("\n📊 Souhrn hodnocení:")
        ratings = [r['rating'] for r in results]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        print(f"   Průměrné hodnocení: {avg_rating:.2f}")
        print(f"   Nejlepší (1-2): {sum(1 for r in ratings if r <= 2)} datasetů")
        print(f"   Střední (3): {sum(1 for r in ratings if r == 3)} datasetů")
        print(f"   Slabé (4-5): {sum(1 for r in ratings if r >= 4)} datasetů")


def main():
    parser = argparse.ArgumentParser(
        description="Vyhodnotí finanční potenciál českých datasetů pomocí lokálního LLM"
    )
    parser.add_argument(
        "--model", 
        default="llama3", 
        help="Název Ollama modelu (výchozí: llama3)"
    )
    parser.add_argument(
        "--datasets", 
        default="data/nkod_data/datasets", 
        help="Cesta k souboru s názvy datasetů nebo složce s JSON soubory (výchozí: data/nkod_data/datasets)"
    )
    parser.add_argument(
        "--output", 
        default="results.csv", 
        help="Výstupní CSV soubor (výchozí: results.csv)"
    )
    parser.add_argument(
        "--ollama-url", 
        default="http://localhost:11434", 
        help="URL Ollama serveru (výchozí: http://localhost:11434)"
    )
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = OllamaDatasetEvaluator(
        model_name=args.model,
        ollama_url=args.ollama_url
    )
    
    evaluator.evaluate_datasets(
        dataset_path=args.datasets,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
