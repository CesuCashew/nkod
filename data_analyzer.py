#!/usr/bin/env python3
"""
NKOD Data Analyzer - Nástroj pro analýzu a čitelné zobrazení dat
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re

class NKODDataAnalyzer:
    """Analyzátor NKOD dat pro čitelné zobrazení a analýzu"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data = None
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Načte JSON data"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # Konverze na DataFrame pro snadnější analýzu
            if 'datasets' in self.data:
                self.df = pd.DataFrame(self.data['datasets'])
                print(f"✅ Načteno {len(self.df)} datasetů z {self.data_path}")
            else:
                print("❌ Soubor neobsahuje 'datasets' klíč")
                
        except Exception as e:
            print(f"❌ Chyba při načítání {self.data_path}: {e}")
    
    def show_basic_info(self):
        """Zobrazí základní informace o datech"""
        print("\n" + "="*60)
        print("📊 ZÁKLADNÍ INFORMACE O DATECH")
        print("="*60)
        
        if not self.data:
            print("❌ Žádná data k zobrazení")
            return
            
        # Metadata
        metadata = self.data.get('metadata', {})
        print(f"🆔 Workflow ID: {metadata.get('workflow_id', 'N/A')}")
        print(f"📅 Vytvořeno: {metadata.get('created_at', 'N/A')}")
        print(f"🎯 Typ spuštění: {metadata.get('trigger_type', 'N/A')}")
        print(f"📈 Celkem datasetů: {metadata.get('total_datasets', 0)}")
        print(f"🏷️ HVD datasetů: {metadata.get('hvd_datasets', 0)}")
        print(f"🔧 Verze scraperu: {metadata.get('scraper_version', 'N/A')}")
        
        if self.df is not None:
            print(f"\n📋 STRUKTURA DAT:")
            print(f"   Řádky (datasety): {len(self.df)}")
            print(f"   Sloupce: {len(self.df.columns)}")
            
            # HVD statistiky
            hvd_count = self.df['isHVD'].sum() if 'isHVD' in self.df.columns else 0
            print(f"   HVD datasety: {hvd_count} ({hvd_count/len(self.df)*100:.1f}%)")
    
    def show_datasets_summary(self, limit: int = 10):
        """Zobrazí přehled datasetů"""
        print("\n" + "="*60)
        print(f"📋 PŘEHLED DATASETŮ (prvních {limit})")
        print("="*60)
        
        if self.df is None:
            print("❌ Žádná data k zobrazení")
            return
            
        for i, row in self.df.head(limit).iterrows():
            print(f"\n🔹 Dataset #{i+1}")
            print(f"   📄 Název: {row.get('title', 'Bez názvu')[:80]}...")
            print(f"   🔗 URI: {row.get('uri', 'N/A')}")
            print(f"   🏷️ HVD: {'✅' if row.get('isHVD', False) else '❌'}")
            
            # Klíčová slova (první 3)
            keywords = row.get('keywords', [])
            if keywords and isinstance(keywords, list):
                kw_display = ', '.join(keywords[:3])
                if len(keywords) > 3:
                    kw_display += f" (+{len(keywords)-3} dalších)"
                print(f"   🔍 Klíčová slova: {kw_display}")
            
            # Popis (zkrácený)
            desc = row.get('description', '')
            if desc:
                desc_short = desc[:120] + "..." if len(desc) > 120 else desc
                print(f"   📝 Popis: {desc_short}")
    
    def analyze_keywords(self, top_n: int = 15):
        """Analyzuje nejčastější klíčová slova"""
        print("\n" + "="*60)
        print(f"🔍 NEJČASTĚJŠÍ KLÍČOVÁ SLOVA (top {top_n})")
        print("="*60)
        
        if self.df is None or 'keywords' not in self.df.columns:
            print("❌ Žádná klíčová slova k analýze")
            return
            
        # Flatten všechna klíčová slova
        all_keywords = []
        for keywords in self.df['keywords'].dropna():
            if isinstance(keywords, list):
                all_keywords.extend([kw.lower().strip() for kw in keywords])
            elif isinstance(keywords, str):
                # Pokud jsou klíčová slova jako string oddělený čárkami
                all_keywords.extend([kw.lower().strip() for kw in keywords.split(',')])
        
        # Počítání
        keyword_counts = Counter(all_keywords)
        
        print(f"📊 Celkem unikátních klíčových slov: {len(keyword_counts)}")
        print(f"📈 Celkem použití: {sum(keyword_counts.values())}")
        print(f"\n🏆 Nejčastější klíčová slova:")
        
        for i, (keyword, count) in enumerate(keyword_counts.most_common(top_n), 1):
            percentage = (count / len(self.df)) * 100
            print(f"   {i:2d}. {keyword:<30} | {count:3d}x ({percentage:5.1f}%)")
        
        return keyword_counts
    
    def analyze_themes(self, top_n: int = 10):
        """Analyzuje témata datasetů"""
        print("\n" + "="*60)
        print(f"🎨 ANALÝZA TÉMAT (top {top_n})")
        print("="*60)
        
        if self.df is None or 'themes' not in self.df.columns:
            print("❌ Žádná témata k analýze")
            return
            
        # Flatten všechna témata
        all_themes = []
        for themes in self.df['themes'].dropna():
            if isinstance(themes, list):
                all_themes.extend(themes)
            elif isinstance(themes, str):
                all_themes.extend([t.strip() for t in themes.split(',')])
        
        # Extrakce čitelných názvů témat
        theme_names = []
        for theme in all_themes:
            if 'data-theme' in theme:
                theme_names.append(theme.split('/')[-1].upper())
            elif 'eurovoc' in theme:
                theme_names.append(f"EUROVOC-{theme.split('/')[-1]}")
            else:
                theme_names.append(theme.split('/')[-1])
        
        theme_counts = Counter(theme_names)
        
        print(f"📊 Celkem unikátních témat: {len(theme_counts)}")
        print(f"\n🏆 Nejčastější témata:")
        
        for i, (theme, count) in enumerate(theme_counts.most_common(top_n), 1):
            percentage = (count / len(self.df)) * 100
            print(f"   {i:2d}. {theme:<25} | {count:3d}x ({percentage:5.1f}%)")
        
        return theme_counts
    
    def analyze_publishers(self, top_n: int = 10):
        """Analyzuje poskytovatele dat"""
        print("\n" + "="*60)
        print(f"🏢 POSKYTOVATELÉ DAT (top {top_n})")
        print("="*60)
        
        if self.df is None or 'publisher' not in self.df.columns:
            print("❌ Žádní poskytovatelé k analýze")
            return
            
        # Extrakce názvů poskytovatelů
        publishers = self.df['publisher'].dropna()
        publisher_names = []
        
        for pub in publishers:
            if isinstance(pub, str):
                # Extrakce posledního segmentu z URI
                if '/' in pub:
                    publisher_names.append(pub.split('/')[-1])
                else:
                    publisher_names.append(pub)
        
        publisher_counts = Counter(publisher_names)
        
        print(f"📊 Celkem poskytovatelů: {len(publisher_counts)}")
        print(f"\n🏆 Nejaktivnější poskytovatelé:")
        
        for i, (publisher, count) in enumerate(publisher_counts.most_common(top_n), 1):
            percentage = (count / len(self.df)) * 100
            print(f"   {i:2d}. {publisher:<20} | {count:3d}x ({percentage:5.1f}%)")
        
        return publisher_counts
    
    def analyze_hvd_datasets(self):
        """Analyzuje HVD (High-Value Datasets) datasety"""
        print("\n" + "="*60)
        print("🏷️ ANALÝZA HVD DATASETŮ")
        print("="*60)
        
        if self.df is None:
            print("❌ Žádná data k analýze")
            return
            
        hvd_datasets = self.df[self.df.get('isHVD', False) == True]
        regular_datasets = self.df[self.df.get('isHVD', False) == False]
        
        print(f"📊 HVD datasety: {len(hvd_datasets)} ({len(hvd_datasets)/len(self.df)*100:.1f}%)")
        print(f"📊 Běžné datasety: {len(regular_datasets)} ({len(regular_datasets)/len(self.df)*100:.1f}%)")
        
        if len(hvd_datasets) > 0:
            print(f"\n🏷️ HVD Datasety:")
            for i, row in hvd_datasets.iterrows():
                title = row.get('title', 'Bez názvu')[:60]
                hvd_cat = row.get('hvdCategory', 'N/A')
                print(f"   • {title}...")
                if hvd_cat != 'N/A':
                    cat_name = hvd_cat.split('/')[-1] if '/' in hvd_cat else hvd_cat
                    print(f"     📂 Kategorie: {cat_name}")
        
        return hvd_datasets, regular_datasets
    
    def search_datasets(self, query: str, limit: int = 5):
        """Vyhledá datasety podle klíčových slov"""
        print("\n" + "="*60)
        print(f"🔍 VYHLEDÁVÁNÍ: '{query}' (max {limit} výsledků)")
        print("="*60)
        
        if self.df is None:
            print("❌ Žádná data k vyhledávání")
            return
            
        query_lower = query.lower()
        matches = []
        
        for i, row in self.df.iterrows():
            score = 0
            match_fields = []
            
            # Hledání v názvu
            title = str(row.get('title', '')).lower()
            if query_lower in title:
                score += 10
                match_fields.append('název')
            
            # Hledání v klíčových slovech
            keywords = row.get('keywords', [])
            if isinstance(keywords, list):
                keywords_text = ' '.join(keywords).lower()
            else:
                keywords_text = str(keywords).lower()
                
            if query_lower in keywords_text:
                score += 5
                match_fields.append('klíčová slova')
            
            # Hledání v popisu
            description = str(row.get('description', '')).lower()
            if query_lower in description:
                score += 3
                match_fields.append('popis')
            
            if score > 0:
                matches.append({
                    'index': i,
                    'score': score,
                    'row': row,
                    'match_fields': match_fields
                })
        
        # Seřazení podle skóre
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        if not matches:
            print(f"❌ Nenalezeny žádné výsledky pro '{query}'")
            return
            
        print(f"✅ Nalezeno {len(matches)} výsledků:")
        
        for i, match in enumerate(matches[:limit], 1):
            row = match['row']
            print(f"\n🔹 Výsledek #{i} (skóre: {match['score']})")
            print(f"   📄 Název: {row.get('title', 'Bez názvu')}")
            print(f"   🔗 URI: {row.get('uri', 'N/A')}")
            print(f"   🎯 Shoda v: {', '.join(match['match_fields'])}")
            print(f"   🏷️ HVD: {'✅' if row.get('isHVD', False) else '❌'}")
            
            # Klíčová slova
            keywords = row.get('keywords', [])
            if keywords:
                if isinstance(keywords, list):
                    kw_display = ', '.join(keywords[:3])
                    if len(keywords) > 3:
                        kw_display += f" (+{len(keywords)-3})"
                else:
                    kw_display = str(keywords)[:50] + "..."
                print(f"   🔍 Klíčová slova: {kw_display}")
        
        return matches
    
    def create_summary_report(self, output_file: str = None):
        """Vytvoří souhrnný report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"nkod_analysis_report_{timestamp}.txt"
        
        print(f"\n📝 Vytvářím souhrnný report: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("NKOD DATA ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Základní info
            metadata = self.data.get('metadata', {})
            f.write(f"Workflow ID: {metadata.get('workflow_id', 'N/A')}\n")
            f.write(f"Vytvořeno: {metadata.get('created_at', 'N/A')}\n")
            f.write(f"Celkem datasetů: {metadata.get('total_datasets', 0)}\n")
            f.write(f"HVD datasetů: {metadata.get('hvd_datasets', 0)}\n\n")
            
            # Statistiky
            if self.df is not None:
                hvd_count = self.df['isHVD'].sum() if 'isHVD' in self.df.columns else 0
                f.write(f"HVD poměr: {hvd_count}/{len(self.df)} ({hvd_count/len(self.df)*100:.1f}%)\n\n")
                
                # Top klíčová slova
                keyword_counts = self.analyze_keywords(10)
                f.write("TOP 10 KLÍČOVÝCH SLOV:\n")
                for keyword, count in keyword_counts.most_common(10):
                    f.write(f"  {keyword}: {count}x\n")
                f.write("\n")
                
                # Top témata
                theme_counts = self.analyze_themes(10)
                f.write("TOP 10 TÉMAT:\n")
                for theme, count in theme_counts.most_common(10):
                    f.write(f"  {theme}: {count}x\n")
        
        print(f"✅ Report uložen: {output_file}")
        return output_file
    
    def interactive_menu(self):
        """Interaktivní menu pro analýzu"""
        while True:
            print("\n" + "="*60)
            print("🔍 NKOD DATA ANALYZER - MENU")
            print("="*60)
            print("1. 📊 Základní informace")
            print("2. 📋 Přehled datasetů")
            print("3. 🔍 Analýza klíčových slov")
            print("4. 🎨 Analýza témat")
            print("5. 🏢 Analýza poskytovatelů")
            print("6. 🏷️ Analýza HVD datasetů")
            print("7. 🔍 Vyhledávání v datech")
            print("8. 📝 Vytvořit souhrnný report")
            print("9. ❌ Ukončit")
            
            try:
                choice = input("\n🎯 Vyberte možnost (1-9): ").strip()
                
                if choice == '1':
                    self.show_basic_info()
                elif choice == '2':
                    limit = input("📋 Počet datasetů k zobrazení (výchozí 10): ").strip()
                    limit = int(limit) if limit.isdigit() else 10
                    self.show_datasets_summary(limit)
                elif choice == '3':
                    top_n = input("🔍 Počet top klíčových slov (výchozí 15): ").strip()
                    top_n = int(top_n) if top_n.isdigit() else 15
                    self.analyze_keywords(top_n)
                elif choice == '4':
                    top_n = input("🎨 Počet top témat (výchozí 10): ").strip()
                    top_n = int(top_n) if top_n.isdigit() else 10
                    self.analyze_themes(top_n)
                elif choice == '5':
                    top_n = input("🏢 Počet top poskytovatelů (výchozí 10): ").strip()
                    top_n = int(top_n) if top_n.isdigit() else 10
                    self.analyze_publishers(top_n)
                elif choice == '6':
                    self.analyze_hvd_datasets()
                elif choice == '7':
                    query = input("🔍 Zadejte hledaný výraz: ").strip()
                    if query:
                        limit = input("📊 Max počet výsledků (výchozí 5): ").strip()
                        limit = int(limit) if limit.isdigit() else 5
                        self.search_datasets(query, limit)
                elif choice == '8':
                    filename = input("📝 Název souboru (nebo Enter pro auto): ").strip()
                    filename = filename if filename else None
                    self.create_summary_report(filename)
                elif choice == '9':
                    print("👋 Děkujeme za použití NKOD Data Analyzer!")
                    break
                else:
                    print("❌ Neplatná volba, zkuste znovu")
                    
            except KeyboardInterrupt:
                print("\n👋 Ukončuji...")
                break
            except Exception as e:
                print(f"❌ Chyba: {e}")


def main():
    """Hlavní funkce"""
    import sys
    
    print("🔍 NKOD Data Analyzer")
    print("=" * 30)
    
    # Pokud je zadán soubor jako argument
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # Pokusí se najít nejnovější dataset
        possible_files = [
            "complete_nkod_data/dataset_current.json",
            "complete_nkod_data/dataset_previous.json",
            "weekly_nkod_data/dataset_current.json",
            "weekly_nkod_data/dataset_previous.json"
        ]
        
        data_file = None
        for file_path in possible_files:
            if Path(file_path).exists():
                data_file = file_path
                break
        
        if not data_file:
            print("❌ Nenalezen žádný dataset soubor!")
            print("💡 Použití: python3 data_analyzer.py [cesta_k_souboru.json]")
            print("💡 Nebo spusťte nejdříve workflow pro vytvoření dat")
            sys.exit(1)
    
    print(f"📁 Analyzuji soubor: {data_file}")
    
    try:
        analyzer = NKODDataAnalyzer(data_file)
        
        if len(sys.argv) > 2:
            # Rychlá analýza pro konkrétní příkaz
            command = sys.argv[2].lower()
            if command == 'info':
                analyzer.show_basic_info()
            elif command == 'datasets':
                analyzer.show_datasets_summary()
            elif command == 'keywords':
                analyzer.analyze_keywords()
            elif command == 'themes':
                analyzer.analyze_themes()
            elif command == 'hvd':
                analyzer.analyze_hvd_datasets()
            elif command == 'report':
                analyzer.create_summary_report()
            else:
                print(f"❌ Neznámý příkaz: {command}")
        else:
            # Interaktivní menu
            analyzer.interactive_menu()
            
    except Exception as e:
        print(f"❌ Chyba při analýze: {e}")


if __name__ == "__main__":
    main()