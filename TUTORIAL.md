# 📚 NKOD LangGraph Workflow - Kompletní Tutorial

## 🎯 Co je to NKOD Workflow?

NKOD (Národní katalog otevřených dat) Workflow je automatizovaný systém postavený na **LangGraph**, který:

- 🌐 **Stahuje** metadata z českého NKOD (data.gov.cz)
- 🔍 **Porovnává** změny mezi běhy
- 🔮 **Vytváří embeddingy** jen pro nová/změněná data  
- 💾 **Ukládá** do vektorové databáze pro podobnostní vyhledávání
- 📅 **Běží automaticky** každé pondělí nebo manuálně

---

## 🚀 Rychlý start

### 1. Instalace závislostí

```bash
# Přejděte do adresáře
cd /Users/filiphirt/Desktop/nkod

# Nainstalujte požadované balíčky
pip3 install -r requirements.txt

# Nainstalujte dodatečné balíčky pro embeddingy
pip3 install sentence-transformers chromadb
```

### 2. První spuštění

```bash
# Test s 3 datasety (rychlé)
python3 complete_nkod_workflow.py test

# Manuální spuštění s 10 datasety
python3 complete_nkod_workflow.py run 10

# Zobrazení nápovědy
python3 complete_nkod_workflow.py help
```

---

## 📖 Podrobný návod

### 🔧 Konfigurace

Workflow používá třídu `CompleteWorkflowConfig` s těmito hlavními nastaveními:

```python
# Základní nastavení
output_base_dir = "complete_nkod_data"          # Výstupní složka
backup_retention_days = 30                      # Jak dlouho uchovávat zálohy

# Scheduling
enable_scheduler = True                          # Povolit automatické spouštění
schedule_time = "09:00"                         # Čas spuštění (pondělí)
schedule_day = "monday"                         # Den týdne

# Embedding nastavení
embedding_provider = "huggingface"              # Ollama nebo HuggingFace
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
vector_db = "chroma"                            # ChromaDB, Qdrant, nebo Weaviate
```

### 📁 Struktura souborů

Po spuštění workflow vytvoří následující strukturu:

```
complete_nkod_data/
├── dataset_current.json         # Nejnovější data
├── dataset_previous.json        # Data z předchozího běhu
├── dataset_backup_*.json        # Starší zálohy (timestamp)
├── workflow_results/            # Výsledky jednotlivých běhů
│   ├── result_nkod_20250723_103125.json
│   └── result_nkod_20250723_103153.json
└── complete_nkod_vector_db/     # ChromaDB vektorová databáze
    ├── chroma.sqlite3
    └── [embeddings data]
```

---

## 🎮 Způsoby spuštění

### 1. 🧪 Testovací režim
```bash
python3 complete_nkod_workflow.py test
```
- Stáhne **3 datasety** (rychlé pro testování)
- Ideální pro ověření, že vše funguje

### 2. 🖱️ Manuální spuštění
```bash
# Bez limitu (všechna data - může trvat dlouho!)
python3 complete_nkod_workflow.py run

# S limitem (doporučeno)
python3 complete_nkod_workflow.py run 50
python3 complete_nkod_workflow.py run 100
```

### 3. ⏰ Automatický scheduler
```bash
python3 complete_nkod_workflow.py schedule
```
- Běží kontinuálně
- Spustí workflow každé **pondělí v 9:00**
- Ukončíte pomocí `Ctrl+C`

---

## 🔍 Jak workflow funguje

### 📊 Fáze workflow

1. **🚀 Trigger** - Inicializace, vytvoření ID workflow
2. **📁 File Management** - Zálohování předchozích dat
3. **🌐 Data Scraping** - Stahování z NKOD pomocí SPARQL
4. **💾 Data Saving** - Uložení do JSON souboru
5. **🔍 Diffing** - Porovnání s předchozími daty
6. **🔮 Embedding** - Vytvoření vektorů jen pro změny
7. **🧹 Cleanup** - Úklid starých záloh
8. **🏁 Finalization** - Vytvoření reportu

### 🤖 LangGraph konstrukce

```python
# StateGraph s kondicionálními přechody
workflow = StateGraph(CompleteWorkflowState)

# Uzly
workflow.add_node("trigger", trigger_node)
workflow.add_node("file_management", file_mgmt_node)
# ... další uzly

# Kondicionální routing
workflow.add_conditional_edges(
    "diffing",
    should_embed,  # Funkce rozhodování
    {
        "embedding": "embedding",    # Jsou-li změny → embedding
        "cleanup": "cleanup"         # Nejsou změny → přeskočit
    }
)
```

---

## 📊 Výsledky a monitoring

### 📋 Workflow Results

Každý běh vytvoří soubor s výsledky:

```json
{
  "workflow_id": "nkod_20250723_103153",
  "trigger_type": "manual",
  "start_time": "2025-07-23T10:31:25.358244",
  "end_time": "2025-07-23T10:31:26.550968", 
  "duration_seconds": 1.192724,
  "status": "completed",
  "datasets_processed": 5,
  "changes_detected": 2,
  "embeddings_created": 2,
  "errors": []
}
```

### 📊 Dataset metadata

Každý dataset obsahuje:

```json
{
  "metadata": {
    "workflow_id": "nkod_20250723_103153",
    "created_at": "2025-07-23T10:31:53.193849",
    "trigger_type": "manual",
    "total_datasets": 5,
    "hvd_datasets": 5,
    "scraper_version": "2.1-HVD-LangChain-Complete"
  },
  "datasets": [
    {
      "uri": "https://data.gov.cz/zdroj/datové-sady/...",
      "title": "Název datové sady",
      "description": "Popis datasetu...",
      "keywords": ["klíčová", "slova"],
      "isHVD": true,
      "harvested_at": "2025-07-23T10:31:53.785318"
    }
  ]
}
```

### 📈 Logování

Workflow loguje do několika míst:

1. **Konzole** - Průběžné informace během běhu
2. **complete_nkod_workflow.log** - Kompletní log soubor
3. **Workflow results** - JSON souhrny jednotlivých běhů

---

## 🔮 Práce s embeddingy

### 🗄️ Vektorová databáze

Workflow automaticky:
- Vytváří **384-dimenzionální** vektory (sentence-transformers)
- Ukládá do **ChromaDB** kolekce `complete_nkod_datasets`
- Indexuje podle **ID**, **URL**, nebo **title hash**

### 🔍 Vyhledávání (příklad použití)

```python
from embedding_node import EmbeddingNode, EmbeddingConfig

# Inicializace
config = EmbeddingConfig()
embedding_node = EmbeddingNode(config)

# Vyhledání podobných datasetů
results = embedding_node.search_similar(
    "dopravní statistiky praha", 
    limit=5
)

for result in results:
    print(f"Podobnost: {result['score']}")
    print(f"Název: {result['metadata']['title']}")
    print(f"URL: {result['metadata']['url']}")
```

---

## ⚙️ Pokročilá konfigurace

### 🔧 Vlastní konfigurace

```python
from complete_nkod_workflow import CompleteWorkflowConfig, CompleteNKODWorkflow
from embedding_node import EmbeddingConfig

# Vlastní konfigurace
config = CompleteWorkflowConfig(
    # Základní
    output_base_dir="my_nkod_data",
    backup_retention_days=60,
    
    # Scheduling  
    schedule_time="08:00",
    schedule_day="tuesday",
    
    # Embedding
    embedding_config=EmbeddingConfig(
        embedding_provider="ollama",
        embedding_model="nomic-embed-text", 
        vector_db="qdrant",  # Pokud máte Qdrant server
        vector_db_config={"url": "localhost", "port": 6333}
    )
)

# Spuštění s vlastní konfigurací
workflow = CompleteNKODWorkflow(config)
result = workflow.run_manual(limit=20)
```

### 🗄️ Jiné vektorové databáze

#### Qdrant
```bash
# Spuštění Qdrant serveru
docker run -p 6333:6333 qdrant/qdrant

# Konfigurace
embedding_config = EmbeddingConfig(
    vector_db="qdrant",
    vector_db_config={"url": "localhost", "port": 6333}
)
```

#### Weaviate
```bash
# Spuštění Weaviate
docker run -p 8080:8080 weaviate/weaviate:latest

# Konfigurace  
embedding_config = EmbeddingConfig(
    vector_db="weaviate",
    vector_db_config={"url": "http://localhost:8080"}
)
```

---

## 🐛 Řešení problémů

### ❌ Časté chyby a řešení

#### 1. **Import chyby**
```bash
# Chyba: ModuleNotFoundError: No module named 'sentence_transformers'
pip3 install sentence-transformers

# Chyba: ModuleNotFoundError: No module named 'chromadb' 
pip3 install chromadb
```

#### 2. **SPARQL endpoint nedostupný**
```
Chyba při dotazu SPARQL: 502 Server Error: Bad Gateway
```
**Řešení**: NKOD server je dočasně nedostupný, zkuste později.

#### 3. **ChromaDB metadata chyby**
```
Expected metadata value to be a str, int, float, bool, or None, got [...] which is a list
```
**Řešení**: Automaticky vyřešeno - workflow převádí seznamy na stringy.

#### 4. **Nedostatek místa na disku**
```
OSError: [Errno 28] No space left on device
```
**Řešení**: 
- Vyčistěte staré zálohy: `rm complete_nkod_data/dataset_backup_*.json`
- Snižte `backup_retention_days`

### 🔧 Debug režim

```python
import logging

# Zapnutí debug loggingu
logging.getLogger().setLevel(logging.DEBUG)

# Nebo spusťte s verbose logováním
python3 complete_nkod_workflow.py run 5 --verbose
```

---

## 📈 Optimalizace a tipy

### ⚡ Výkonové tipy

1. **Začněte s malými limity** (10-50 datasetů) pro testování
2. **Používejte scheduler** pro pravilné aktualizace
3. **Monitorujte disk space** - embeddingy zabírají místo
4. **Pravidelně čistěte zálohy** starší než 30 dní

### 🎯 Doporučené workflow

```bash
# 1. Prvotní test
python3 complete_nkod_workflow.py test

# 2. Malý produkční běh  
python3 complete_nkod_workflow.py run 100

# 3. Spuštění scheduleru pro automatizaci
python3 complete_nkod_workflow.py schedule
```

### 📊 Monitoring produkce

```bash
# Sledování logů
tail -f complete_nkod_workflow.log

# Kontrola výsledků
ls -la complete_nkod_data/workflow_results/

# Velikost vektorové databáze
du -sh complete_nkod_data/complete_nkod_vector_db/
```

---

## 🔗 Integrace s dalšími systémy

### 📡 API endpoint (příklad)

```python
from flask import Flask, jsonify
from embedding_node import EmbeddingNode, EmbeddingConfig

app = Flask(__name__)
embedding_node = EmbeddingNode(EmbeddingConfig())

@app.route('/search/<query>')
def search_datasets(query):
    results = embedding_node.search_similar(query, limit=10)
    return jsonify({
        'query': query,
        'results': results,
        'count': len(results)
    })

if __name__ == '__main__':
    app.run(port=5000)
```

### 📊 Jupyter Notebook analýza

```python
import json
import pandas as pd
from pathlib import Path

# Načtení dat
with open('complete_nkod_data/dataset_current.json', 'r') as f:
    data = json.load(f)

# Konverze na DataFrame
df = pd.DataFrame(data['datasets'])

# Analýza
print(f"Celkem datasetů: {len(df)}")
print(f"HVD datasetů: {df['isHVD'].sum()}")
print(f"Nejčastější klíčová slova: {df['keywords'].explode().value_counts().head()}")
```

---

## 🎓 Pokročilé použití

### 🔀 Vlastní uzly

```python
class CustomProcessingNode:
    """Vlastní uzel pro specifické zpracování"""
    
    def __init__(self, config):
        self.config = config
        
    def __call__(self, state):
        # Vlastní logika
        logger.info("🔥 Spouštím vlastní zpracování...")
        
        # Zpracování dat
        processed_data = self.process_datasets(state['processed_datasets'])
        
        # Aktualizace stavu
        state['custom_processing_completed'] = True
        state['custom_results'] = processed_data
        
        return state
        
    def process_datasets(self, datasets):
        # Implementace vlastní logiky
        return datasets

# Integrace do workflow
workflow.add_node("custom", CustomProcessingNode(config))
workflow.add_edge("embedding", "custom")
workflow.add_edge("custom", "cleanup")
```

### 📊 Vlastní metriky

```python
class MetricsCollector:
    """Sběr vlastních metrik"""
    
    def collect_metrics(self, state):
        metrics = {
            'processing_time': time.time() - start_time,
            'memory_usage': psutil.Process().memory_info().rss,
            'datasets_per_second': len(datasets) / duration,
            'hvd_ratio': hvd_count / total_count
        }
        
        # Uložení metrik
        with open('metrics.json', 'w') as f:
            json.dump(metrics, f)
            
        return metrics
```

---

## 📞 Podpora a FAQ

### ❓ Často kladené otázky

**Q: Jak dlouho trvá zpracování všech dat?**
A: Závisí na počtu datasetů. ~1000 datasetů = 5-10 minut.

**Q: Mohu změnit embedding model?**
A: Ano, v konfiguraci nastavte `embedding_model` na jakýkoli HuggingFace model.

**Q: Jak mohu vyhledávat v datech?**
A: Použijte `embedding_node.search_similar("váš dotaz")` nebo vytvořte vlastní API.

**Q: Běží workflow i bez internetu?**
A: Ne, potřebuje přístup k data.gov.cz a HuggingFace modelům.

**Q: Mohu upravit čas schedulingu?**
A: Ano, změňte `schedule_time` a `schedule_day` v konfiguraci.

### 🆘 Podpora

- **GitHub Issues**: Reportujte chyby a požadavky
- **Dokumentace**: Tento tutorial a komentáře v kódu
- **Logy**: Vždy zkontrolujte `complete_nkod_workflow.log`

---

## 🎉 Závěr

Gratulujeme! Nyní máte kompletní automatizovaný systém pro:

- ✅ **Stahování** NKOD dat
- ✅ **Detekci změn** mezi běhy  
- ✅ **Vytváření embeddingů** pro podobnostní vyhledávání
- ✅ **Automatické spouštění** každé pondělí
- ✅ **Kompletní monitoring** a logování

**Začněte s:** `python3 complete_nkod_workflow.py test`

**Produkční nasazení:** `python3 complete_nkod_workflow.py schedule`

---

*📝 Naposledy aktualizováno: 23. července 2025*  
*🚀 NKOD LangGraph Workflow v2.1*