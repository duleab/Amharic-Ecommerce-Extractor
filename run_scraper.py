"""
Script to run the Telegram scraper for Amharic e-commerce data.
"""

import os
import asyncio
import dotenv
from pathlib import Path
from src.data.telegram_scraper import TelegramScraper

# Load environment variables from .env file
dotenv.load_dotenv()

# Get API credentials from environment variables
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
PHONE = os.getenv("TELEGRAM_PHONE")

# List of Ethiopian e-commerce Telegram channels to scrape
CHANNELS = [
    "@tikvahethmart",
    "@abaymart", 
    "@nejashionlinemarketing",
    "@Leyueqa",
    "@helloomarketethiopia",
    "@shilngie",
    "@balemoyamarket",
    "@efuyegellaMarket"
]

async def main():
    # Define output directories
    data_dir = Path(__file__).parent / "data"
    raw_dir = data_dir / "raw"
    checkpoint_dir = data_dir / "checkpoints"
    
    # Create directories
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("Starting Telegram scraper with enhanced Amharic text processing...")
    
    # Initialize scraper with enhanced filtering
    scraper = TelegramScraper(
        API_ID, 
        API_HASH, 
        PHONE,
        text_filter_mode='amharic_only',  # Options: 'amharic_only', 'remove_emojis', 'remove_english_numbers', 'remove_emojis_english_numbers', 'analysis_only'
        min_amharic_percentage=30.0,      # Minimum percentage of Amharic characters to include message
        max_retries=3,                    # Maximum number of retries for failed operations
        retry_delay=5,                    # Base delay between retries (will use exponential backoff)
        checkpoint_interval=50            # Create checkpoint after every 50 messages
    )
    
    print(f"Will scrape {len(CHANNELS)} channels with the following settings:")
    print(f"- Text filter mode: {scraper.text_filter_mode}")
    print(f"- Minimum Amharic percentage: {scraper.min_amharic_percentage}%")
    print(f"- Clear existing data: Yes")
    print(f"- Output directory: {raw_dir}")
    print(f"- Checkpoint directory: {checkpoint_dir}")
    
    try:
        # Connect to Telegram
        await scraper.connect()
        
        # Scrape channels
        await scraper.scrape_multiple_channels(
            CHANNELS,
            limit_per_channel=200,
            output_dir=str(raw_dir),
            output_format="json",        # or "csv"
            filter_non_amharic=True,     # Set to False to include all messages regardless of Amharic content
            generate_analytics=True,     # Generate detailed analytics
            clear_existing_data=True,    # Remove previous data files
            checkpoint_dir=str(checkpoint_dir)  # Directory for saving checkpoints
        )
        
        print("Scraping completed successfully!")
    except Exception as e:
        print(f"Error during scraping: {e}")
    finally:
        # Disconnect from Telegram
        await scraper.disconnect()
        print("Disconnected from Telegram.")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main()) 