#!/usr/bin/env python3
"""
Embedding Node pro LangChain workflow
Převádí záznamy na text, generuje embeddingy a ukládá do vektorové databáze
"""

import json
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Literal, Union
from pathlib import Path
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

# LangChain embeddings
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

# Vektorové databáze
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingRecord:
    """Záznam s embeddingem"""
    id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


class EmbeddingConfig(BaseModel):
    """Konfigurace pro embedding node"""
    embedding_provider: Literal["ollama", "huggingface"] = "ollama"
    embedding_model: str = "nomic-embed-text"  # Pro Ollama
    vector_db: Literal["chroma", "qdrant", "weaviate"] = "chroma"
    collection_name: str = "nkod_datasets"
    vector_db_config: Dict[str, Any] = Field(default_factory=dict)
    text_template: str = "Název: {title}\nPopis: {description}\nURL: {url}"
    chunk_size: int = 100  # Velikost dávky pro embedding
    
    class Config:
        arbitrary_types_allowed = True


class VectorDBInterface(ABC):
    """Abstraktní rozhraní pro vektorové databáze"""
    
    @abstractmethod
    def initialize(self, collection_name: str, dimension: int) -> bool:
        """Inicializuje databázi a kolekci"""
        pass
    
    @abstractmethod
    def add_embeddings(self, records: List[EmbeddingRecord]) -> bool:
        """Přidá embeddingy do databáze"""
        pass
    
    @abstractmethod
    def search(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Vyhledá podobné záznamy"""
        pass
    
    @abstractmethod
    def delete_collection(self) -> bool:
        """Smaže kolekci"""
        pass


class ChromaVectorDB(VectorDBInterface):
    """Implementace pro ChromaDB"""
    
    def __init__(self, config: Dict[str, Any]):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB není nainstalováno. Spusťte: pip install chromadb")
        
        self.persist_directory = config.get("persist_directory", "./chroma_db")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = None
        self.collection_name = None
    
    def initialize(self, collection_name: str, dimension: int) -> bool:
        try:
            self.collection_name = collection_name
            # Pokusí se získat existující kolekci nebo vytvoří novou
            try:
                self.collection = self.client.get_collection(collection_name)
                logger.info(f"Použita existující ChromaDB kolekce: {collection_name}")
            except:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Vytvořena nová ChromaDB kolekce: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Chyba při inicializaci ChromaDB: {e}")
            return False
    
    def add_embeddings(self, records: List[EmbeddingRecord]) -> bool:
        try:
            ids = [record.id for record in records]
            embeddings = [record.embedding for record in records]
            documents = [record.text for record in records]
            metadatas = [record.metadata for record in records]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Přidáno {len(records)} záznamů do ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Chyba při přidávání do ChromaDB: {e}")
            return False
    
    def search(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit
            )
            return results
        except Exception as e:
            logger.error(f"Chyba při vyhledávání v ChromaDB: {e}")
            return []
    
    def delete_collection(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Smazána ChromaDB kolekce: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Chyba při mazání ChromaDB kolekce: {e}")
            return False


class QdrantVectorDB(VectorDBInterface):
    """Implementace pro Qdrant"""
    
    def __init__(self, config: Dict[str, Any]):
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant client není nainstalován. Spusťte: pip install qdrant-client")
        
        self.url = config.get("url", "localhost")
        self.port = config.get("port", 6333)
        self.client = QdrantClient(host=self.url, port=self.port)
        self.collection_name = None
    
    def initialize(self, collection_name: str, dimension: int) -> bool:
        try:
            self.collection_name = collection_name
            
            # Zkontroluje zda kolekce existuje
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == collection_name for col in collections)
            
            if not collection_exists:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                )
                logger.info(f"Vytvořena nová Qdrant kolekce: {collection_name}")
            else:
                logger.info(f"Použita existující Qdrant kolekce: {collection_name}")
            
            return True
        except Exception as e:
            logger.error(f"Chyba při inicializaci Qdrant: {e}")
            return False
    
    def add_embeddings(self, records: List[EmbeddingRecord]) -> bool:
        try:
            points = []
            for record in records:
                point = PointStruct(
                    id=record.id,
                    vector=record.embedding,
                    payload={
                        "text": record.text,
                        **record.metadata
                    }
                )
                points.append(point)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Přidáno {len(records)} záznamů do Qdrant")
            return True
        except Exception as e:
            logger.error(f"Chyba při přidávání do Qdrant: {e}")
            return False
    
    def search(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit
            )
            return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results]
        except Exception as e:
            logger.error(f"Chyba při vyhledávání v Qdrant: {e}")
            return []
    
    def delete_collection(self) -> bool:
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Smazána Qdrant kolekce: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Chyba při mazání Qdrant kolekce: {e}")
            return False


class WeaviateVectorDB(VectorDBInterface):
    """Implementace pro Weaviate"""
    
    def __init__(self, config: Dict[str, Any]):
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate client není nainstalován. Spusťte: pip install weaviate-client")
        
        self.url = config.get("url", "http://localhost:8080")
        self.client = weaviate.Client(self.url)
        self.collection_name = None
    
    def initialize(self, collection_name: str, dimension: int) -> bool:
        try:
            self.collection_name = collection_name.capitalize()  # Weaviate vyžaduje velké první písmeno
            
            # Zkontroluje zda třída existuje
            if not self.client.schema.contains({"class": self.collection_name}):
                class_schema = {
                    "class": self.collection_name,
                    "description": f"NKOD datasets collection: {collection_name}",
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"],
                            "description": "Full text content"
                        },
                        {
                            "name": "title", 
                            "dataType": ["text"],
                            "description": "Dataset title"
                        },
                        {
                            "name": "description",
                            "dataType": ["text"], 
                            "description": "Dataset description"
                        },
                        {
                            "name": "url",
                            "dataType": ["text"],
                            "description": "Dataset URL"
                        }
                    ]
                }
                self.client.schema.create_class(class_schema)
                logger.info(f"Vytvořena nová Weaviate třída: {self.collection_name}")
            else:
                logger.info(f"Použita existující Weaviate třída: {self.collection_name}")
            
            return True
        except Exception as e:
            logger.error(f"Chyba při inicializaci Weaviate: {e}")
            return False
    
    def add_embeddings(self, records: List[EmbeddingRecord]) -> bool:
        try:
            with self.client.batch as batch:
                for record in records:
                    properties = {
                        "text": record.text,
                        **record.metadata
                    }
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.collection_name,
                        uuid=record.id,
                        vector=record.embedding
                    )
            
            logger.info(f"Přidáno {len(records)} záznamů do Weaviate")
            return True
        except Exception as e:
            logger.error(f"Chyba při přidávání do Weaviate: {e}")
            return False
    
    def search(self, embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        try:
            result = (
                self.client.query
                .get(self.collection_name, ["text", "title", "description", "url"])
                .with_near_vector({"vector": embedding})
                .with_limit(limit)
                .with_additional(["certainty", "id"])
                .do()
            )
            return result.get("data", {}).get("Get", {}).get(self.collection_name, [])
        except Exception as e:
            logger.error(f"Chyba při vyhledávání ve Weaviate: {e}")
            return []
    
    def delete_collection(self) -> bool:
        try:
            self.client.schema.delete_class(self.collection_name)
            logger.info(f"Smazána Weaviate třída: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Chyba při mazání Weaviate třídy: {e}")
            return False


class EmbeddingNode:
    """LangChain uzel pro generování embeddingů a ukládání do vektorové DB"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embeddings: Optional[Embeddings] = None
        self.vector_db: Optional[VectorDBInterface] = None
        self._initialize_embeddings()
        self._initialize_vector_db()
    
    def _initialize_embeddings(self):
        """Inicializuje embedding model"""
        try:
            if self.config.embedding_provider == "ollama":
                self.embeddings = OllamaEmbeddings(
                    model=self.config.embedding_model,
                    base_url="http://localhost:11434"
                )
                logger.info(f"Inicializován Ollama embedding model: {self.config.embedding_model}")
            
            elif self.config.embedding_provider == "huggingface":
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.embedding_model
                )
                logger.info(f"Inicializován HuggingFace embedding model: {self.config.embedding_model}")
            
            else:
                raise ValueError(f"Nepodporovaný embedding provider: {self.config.embedding_provider}")
                
        except Exception as e:
            logger.error(f"Chyba při inicializaci embeddingu: {e}")
            raise
    
    def _initialize_vector_db(self):
        """Inicializuje vektorovou databázi"""
        try:
            if self.config.vector_db == "chroma":
                self.vector_db = ChromaVectorDB(self.config.vector_db_config)
            elif self.config.vector_db == "qdrant":
                self.vector_db = QdrantVectorDB(self.config.vector_db_config)
            elif self.config.vector_db == "weaviate":
                self.vector_db = WeaviateVectorDB(self.config.vector_db_config)
            else:
                raise ValueError(f"Nepodporovaná vektorová databáze: {self.config.vector_db}")
                
            logger.info(f"Inicializována vektorová databáze: {self.config.vector_db}")
            
        except Exception as e:
            logger.error(f"Chyba při inicializaci vektorové DB: {e}")
            raise
    
    def _record_to_text(self, record: Dict[str, Any]) -> str:
        """Převádí záznam na text podle template"""
        try:
            # Defaultní hodnoty pro chybějící pole
            safe_record = {
                "title": record.get("title", "Bez názvu"),
                "description": record.get("description", "Bez popisu"),
                "url": record.get("uri", record.get("url", "Bez URL")),
                **record  # Přidá všechna ostatní pole
            }
            
            # Formátování podle template
            text = self.config.text_template.format(**safe_record)
            return text.strip()
            
        except KeyError as e:
            logger.warning(f"Chybí pole v záznamu: {e}")
            # Fallback na jednoduchý formát
            return f"Název: {record.get('title', 'N/A')}\nPopis: {record.get('description', 'N/A')}"
    
    def _generate_id(self, record: Dict[str, Any]) -> str:
        """Generuje unikátní ID pro záznam"""
        # Priorita: identifier > uri > uuid > hash
        if record.get("identifier"):
            return f"id_{record['identifier']}"
        elif record.get("uri"):
            return f"uri_{hashlib.md5(record['uri'].encode()).hexdigest()[:8]}"
        elif record.get("id"):
            return str(record["id"])
        else:
            # Hash z textu
            text = self._record_to_text(record)
            return f"hash_{hashlib.md5(text.encode()).hexdigest()[:8]}"
    
    def _process_batch(self, records: List[Dict[str, Any]]) -> List[EmbeddingRecord]:
        """Zpracuje dávku záznamů"""
        embedding_records = []
        
        # Převede záznamy na texty
        texts = []
        for record in records:
            text = self._record_to_text(record)
            texts.append(text)
        
        # Generuje embeddingy pro celou dávku
        try:
            embeddings = self.embeddings.embed_documents(texts)
            
            # Vytvoří EmbeddingRecord objekty
            for i, record in enumerate(records):
                embedding_record = EmbeddingRecord(
                    id=self._generate_id(record),
                    text=texts[i],
                    embedding=embeddings[i],
                    metadata={
                        "title": record.get("title", ""),
                        "description": record.get("description", "")[:500] if record.get("description") else "",  # Omezení délky
                        "url": record.get("uri", record.get("url", "")),
                        "isHVD": record.get("isHVD", False),
                        "keywords": ", ".join(record.get("keywords", [])),  # Převod na string
                        "themes": ", ".join(record.get("themes", []))  # Převod na string
                    }
                )
                embedding_records.append(embedding_record)
                
            return embedding_records
            
        except Exception as e:
            logger.error(f"Chyba při generování embeddingů: {e}")
            return []
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Hlavní funkce uzlu"""
        logger.info("🔮 Spouštím embedding node...")
        
        try:
            # Získá záznamy ze stavu
            records = state.get("new_or_modified_datasets", [])
            if not records:
                logger.info("Žádné záznamy k zpracování")
                state.update({
                    "embeddings_processed": 0,
                    "embeddings_error": None,
                    "vector_db_initialized": False
                })
                return state
            
            logger.info(f"Zpracovávám {len(records)} záznamů...")
            
            # Test embeddingu pro zjištění dimenze
            test_embedding = self.embeddings.embed_query("test")
            dimension = len(test_embedding)
            logger.info(f"Dimenze embeddingu: {dimension}")
            
            # Inicializuje vektorovou databázi
            if not self.vector_db.initialize(self.config.collection_name, dimension):
                raise Exception("Nepodařilo se inicializovat vektorovou databázi")
            
            # Zpracovává záznamy po dávkách
            total_processed = 0
            chunk_size = self.config.chunk_size
            
            for i in range(0, len(records), chunk_size):
                batch = records[i:i + chunk_size]
                logger.info(f"Zpracovávám dávku {i//chunk_size + 1}/{(len(records)-1)//chunk_size + 1} ({len(batch)} záznamů)")
                
                # Generuje embeddingy
                embedding_records = self._process_batch(batch)
                
                if embedding_records:
                    # Ukládá do vektorové databáze
                    if self.vector_db.add_embeddings(embedding_records):
                        total_processed += len(embedding_records)
                        logger.info(f"Uloženo {len(embedding_records)} embeddingů")
                    else:
                        logger.error(f"Chyba při ukládání dávky {i//chunk_size + 1}")
            
            logger.info(f"✅ Embedding dokončen. Zpracováno {total_processed}/{len(records)} záznamů")
            
            # Aktualizuje stav
            state.update({
                "embeddings_processed": total_processed,
                "embeddings_dimension": dimension,
                "vector_db_collection": self.config.collection_name,
                "vector_db_type": self.config.vector_db,
                "embeddings_error": None,
                "vector_db_initialized": True
            })
            
        except Exception as e:
            error_msg = f"Chyba v embedding node: {e}"
            logger.error(error_msg)
            state.update({
                "embeddings_processed": 0,
                "embeddings_error": error_msg,
                "vector_db_initialized": False
            })
        
        return state
    
    def search_similar(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Vyhledá podobné záznamy podle textového dotazu"""
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.vector_db.search(query_embedding, limit)
            return results
        except Exception as e:
            logger.error(f"Chyba při vyhledávání: {e}")
            return []


def create_test_records() -> List[Dict[str, Any]]:
    """Vytvoří testovací záznamy ve formátu NKOD"""
    return [
        {
            "identifier": "test-dataset-1",
            "uri": "https://data.gov.cz/dataset/1",
            "title": "Demografická data České republiky",
            "description": "Kompletní statistiky o obyvatelstvu ČR včetně věkové struktury, regionálního rozložení a demografických trendů za posledních 10 let.",
            "keywords": ["demografie", "statistiky", "obyvatelstvo", "česká republika"],
            "themes": ["SOCI", "GOVE"],
            "isHVD": False
        },
        {
            "identifier": "test-dataset-2", 
            "uri": "https://data.gov.cz/dataset/2",
            "title": "Dopravní nehody v Praze",
            "description": "Databáze všech dopravních nehod v Praze s podrobnými údaji o místě, čase, příčinách a následcích nehod.",
            "keywords": ["doprava", "nehody", "praha", "bezpečnost"],
            "themes": ["TRAN"],
            "isHVD": True
        },
        {
            "identifier": "test-dataset-3",
            "uri": "https://data.gov.cz/dataset/3", 
            "title": "Kvalita ovzduší - měřicí stanice",
            "description": "Hodinové měření kvality ovzduší ze všech měřicích stanic v ČR včetně koncentrací PM2.5, PM10, NO2, SO2 a ozónu.",
            "keywords": ["ovzduší", "kvalita", "měření", "environement", "znečištění"],
            "themes": ["ENVI"],
            "isHVD": False
        }
    ]


def test_embedding_node():
    """Testovací funkce"""
    print("🧪 Testování Embedding Node...")
    
    # Konfigurace (používá ChromaDB jako nejjednodušší možnost)
    config = EmbeddingConfig(
        embedding_provider="huggingface",  # HuggingFace je spolehlivější než Ollama pro test
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        vector_db="chroma",
        collection_name="test_nkod_datasets",
        vector_db_config={"persist_directory": "./test_chroma_db"}
    )
    
    try:
        # Vytvoří embedding node
        embedding_node = EmbeddingNode(config)
        
        # Testovací data
        test_records = create_test_records()
        
        # Mock stav
        test_state = {
            "new_or_modified_datasets": test_records
        }
        
        # Spustí embedding
        result_state = embedding_node(test_state)
        
        # Vypíše výsledky
        print(f"\n📊 Výsledky:")
        print(f"Zpracováno záznamů: {result_state.get('embeddings_processed', 0)}")
        print(f"Dimenze embeddingu: {result_state.get('embeddings_dimension', 'N/A')}")
        print(f"Vektorová DB: {result_state.get('vector_db_type', 'N/A')}")
        print(f"Kolekce: {result_state.get('vector_db_collection', 'N/A')}")
        
        if result_state.get('embeddings_error'):
            print(f"❌ Chyba: {result_state['embeddings_error']}")
        else:
            print("✅ Embedding úspěšný!")
            
            # Test vyhledávání
            print("\n🔍 Test vyhledávání...")
            query = "dopravní statistiky praha"
            results = embedding_node.search_similar(query, limit=2)
            print(f"Dotaz: '{query}'")
            print(f"Nalezeno výsledků: {len(results)}")
            for i, result in enumerate(results[:2]):
                print(f"  {i+1}. {result}")
        
        return result_state
        
    except ImportError as e:
        print(f"❌ Chybí závislosti: {e}")
        print("Nainstalujte potřebné balíčky:")
        print("  pip install chromadb sentence-transformers")
        return {}
    except Exception as e:
        print(f"❌ Chyba při testování: {e}")
        return {}


if __name__ == "__main__":
    # Nastavení loggingu
    logging.basicConfig(level=logging.INFO)
    
    # Spustí test
    test_embedding_node()