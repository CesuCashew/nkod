# NKOD Data Pipeline

A Python toolkit for working with the National Catalog of Open Data (NKOD - Národní katalog otevřených dat). This project provides tools for scraping metadata and downloading actual data files from the NKOD portal.

## Features

- **Metadata Scraping**: Extract metadata about available datasets from NKOD
- **Data Downloading**: Download actual data files based on scraped metadata
- **Structured Storage**: Organized storage of downloaded data and metadata
- **Configurable**: Customize scraping and downloading parameters

## Project Structure

```
nkod/
├── data/                    # Directory for downloaded data files
│   └── downloaded_data/     # Actual data files are stored here
├── src/                     # Source code
│   ├── data_downloader.py   # Script for downloading data files
│   └── nkod_scraper.py      # Script for scraping metadata
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Requirements

- Python 3.9+
- Required Python packages (see `requirements.txt`):
  - requests
  - langchain-core
  - langchain-community
  - langgraph
  - pydantic
  - sentence-transformers
  - transformers
  - chromadb (optional, for vector storage)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/CesuCashew/nkod.git
   cd nkod
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Scrape Metadata
Run the scraper to collect metadata about available datasets:
```bash
python src/nkod_scraper.py
```

### 2. Download Data Files
After scraping metadata, download the actual data files:
```bash
python src/data_downloader.py
```

## Configuration

The scripts are pre-configured with reasonable defaults, but you can modify them as needed:
- `nkod_scraper.py`: Adjust scraping parameters like timeouts and retry logic
- `data_downloader.py`: Configure download locations and file handling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Claude AI
