import os
import json
from datetime import datetime
from research_processor import ResearchProcessor

def load_research_data(file_path: str) -> dict:
    """Load research data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    # Path to your research data file
    data_file = "nkod_data/datasets/Výsledky_výzkumu,_experimentálního_vývoje_a_inovací_uplatněné_v_roce_2023.json"
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/research_results_{timestamp}.json"
        
        # Load the research data
        print(f"Loading research data from {data_file}...")
        research_data = load_research_data(data_file)
        
        # Check if the data is a list or needs to be wrapped in a list
        if not isinstance(research_data, list):
            research_data = [research_data]
        
        # Limit the number of items to process (set to None to process all)
        max_items_to_process = None  # Change this to process more or fewer items
        
        # Initialize the processor
        print("\nInitializing research processor...")
        processor = ResearchProcessor(max_results=max_items_to_process)
        
        # Process the research data
        print(f"\nProcessing up to {max_items_to_process if max_items_to_process else 'all'} research results...")
        result = processor.process(research_data)
        
        # Print the results
        print("\n=== Processing Complete ===")
        print(f"Status: {result['analysis']}")
        
        if result["processed_results"]:
            print("\n=== Analysis Results ===")
            for i, processed in enumerate(result["processed_results"], 1):
                print(f"\nVýsledek {i} - {processed.title} ({processed.year}):")
                print("-" * 80)
                print(processed.analysis)
                print("\n" + "=" * 80)
            
            # Save results to file
            processor.save_results(result["processed_results"], output_file)
            print(f"\nDetailed results saved to: {output_file}")
                
    except FileNotFoundError:
        print(f"Chyba: Soubor {data_file} nebyl nalezen.")
        print("Zkontrolujte prosím cestu k souboru.")
    except json.JSONDecodeError:
        print(f"Chyba: Soubor {data_file} není platný JSON.")
    except Exception as e:
        print(f"Nastala neočekávaná chyba: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
