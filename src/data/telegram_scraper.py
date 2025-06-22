"""
Enhanced Telegram Scraper Module for Amharic E-commerce Data Extraction.

This module provides functionality to scrape messages from Telegram channels
that contain Amharic e-commerce data with advanced text filtering capabilities.
"""

import os
import json
import logging
import re
import unicodedata
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from telethon import TelegramClient
from telethon.errors import FloodWaitError, SessionPasswordNeededError, ChatAdminRequiredError, ChannelPrivateError
from telethon.tl.functions.channels import JoinChannelRequest
from telethon.tl.types import Message, MessageMediaPhoto, MessageMediaDocument
import pandas as pd

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class AmharicTextProcessor:
    """
    A class to process and filter Amharic text content.
    """
    
    def __init__(self, custom_informal_patterns: Optional[List[str]] = None):
        # Amharic Unicode ranges
        self.amharic_ranges = [
            (0x1200, 0x137F),  # Ethiopic
            (0x1380, 0x139F),  # Ethiopic Supplement
            (0x2D80, 0x2DDF),  # Ethiopic Extended
            (0xAB00, 0xAB2F),  # Ethiopic Extended-A
        ]
        
        # Common Amharic punctuation and symbols
        self.amharic_punctuation = {
            '።', '፣', '፤', '፥', '፦', '፧', '፨', '፠', '፡'
        }
        
        # Emoji pattern (more comprehensive)
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
            "\U00002600-\U000026FF"  # miscellaneous symbols
            "\U00002700-\U000027BF"  # dingbats
            "]+", 
            flags=re.UNICODE
        )
        
        # English letters and numbers pattern
        self.english_numbers_pattern = re.compile(r'[a-zA-Z0-9]')
        
        # Common social media abbreviations and informal text patterns
        self.default_informal_patterns = [
            r'\b(lol|omg|wtf|brb|ttyl|imo|imho|fyi|btw|asap)\b',
            r'\b(u|ur|r|n|2|4|b4|2day|2morrow)\b',
            r'[xX]{2,}',  # Multiple x's
            r'[hH]{3,}',  # Multiple h's (hahaha)
            r'[!]{2,}',   # Multiple exclamation marks
            r'[?]{2,}',   # Multiple question marks
        ]
        
        # Use custom patterns if provided, otherwise use defaults
        self.informal_patterns = custom_informal_patterns or self.default_informal_patterns
        
    def is_amharic_char(self, char: str) -> bool:
        """Check if a character is Amharic."""
        char_code = ord(char)
        return any(start <= char_code <= end for start, end in self.amharic_ranges)
    
    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        return self.emoji_pattern.sub('', text)
    
    def remove_english_and_numbers(self, text: str) -> str:
        """Remove English letters and numbers from text."""
        return self.english_numbers_pattern.sub('', text)
    
    def extract_amharic_only(self, text: str) -> str:
        """Extract only Amharic characters and punctuation."""
        filtered_chars = []
        for char in text:
            if (self.is_amharic_char(char) or 
                char in self.amharic_punctuation or 
                char.isspace()):
                filtered_chars.append(char)
        
        # Clean up multiple spaces
        result = ''.join(filtered_chars)
        result = re.sub(r'\s+', ' ', result).strip()
        return result
    
    def detect_mixed_content(self, text: str) -> Dict[str, Any]:
        """Analyze text content composition."""
        stats = {
            'total_chars': len(text),
            'amharic_chars': 0,
            'english_chars': 0,
            'numbers': 0,
            'emojis': 0,
            'spaces': 0,
            'punctuation': 0,
            'has_mixed_content': False,
            'is_informal': False,
            'amharic_percentage': 0.0
        }
        
        # Count emojis
        emoji_matches = self.emoji_pattern.findall(text)
        stats['emojis'] = len(''.join(emoji_matches))
        
        # Count character types
        for char in text:
            if self.is_amharic_char(char):
                stats['amharic_chars'] += 1
            elif char.isalpha() and ord(char) < 128:  # English letters
                stats['english_chars'] += 1
            elif char.isdigit():
                stats['numbers'] += 1
            elif char.isspace():
                stats['spaces'] += 1
            elif not char.isalnum():
                stats['punctuation'] += 1
        
        # Calculate percentages
        non_space_chars = stats['total_chars'] - stats['spaces']
        if non_space_chars > 0:
            stats['amharic_percentage'] = (stats['amharic_chars'] / non_space_chars) * 100
        
        # Detect mixed content
        stats['has_mixed_content'] = (
            stats['amharic_chars'] > 0 and 
            (stats['english_chars'] > 0 or stats['numbers'] > 0)
        )
        
        # Detect informal language patterns
        text_lower = text.lower()
        stats['is_informal'] = any(
            re.search(pattern, text_lower, re.IGNORECASE) 
            for pattern in self.informal_patterns
        )
        
        return stats
    
    def process_text(self, text: str, filter_mode: str = 'amharic_only') -> Dict[str, Any]:
        """
        Process text based on specified filter mode.
        
        Args:
            text: Input text to process
            filter_mode: 'amharic_only', 'remove_emojis', 'remove_english_numbers', 
                        'remove_emojis_english_numbers', 'analysis_only'
        
        Returns:
            Dictionary containing processed text and analysis
        """
        if not text or not isinstance(text, str):
            return {
                'original_text': text,
                'processed_text': '',
                'analysis': {},
                'filter_mode': filter_mode
            }
        
        # Analyze original text
        analysis = self.detect_mixed_content(text)
        
        # Apply filtering based on mode
        if filter_mode == 'amharic_only':
            processed_text = self.extract_amharic_only(text)
        elif filter_mode == 'remove_emojis':
            processed_text = self.remove_emojis(text)
        elif filter_mode == 'remove_english_numbers':
            temp_text = self.remove_english_and_numbers(text)
            processed_text = re.sub(r'\s+', ' ', temp_text).strip()
        elif filter_mode == 'remove_emojis_english_numbers':
            temp_text = self.remove_emojis(text)
            temp_text = self.remove_english_and_numbers(temp_text)
            processed_text = re.sub(r'\s+', ' ', temp_text).strip()
        else:  # analysis_only
            processed_text = text
        
        return {
            'original_text': text,
            'processed_text': processed_text,
            'analysis': analysis,
            'filter_mode': filter_mode,
            'length_reduction': len(text) - len(processed_text),
            'is_amharic_dominant': analysis['amharic_percentage'] > 50
        }

class TelegramScraper:
    """
    Enhanced Telegram scraper with Amharic text processing capabilities.
    """
    
    def __init__(
        self, 
        api_id: str, 
        api_hash: str, 
        phone: str, 
        session_name: str = "amharic_scraper",
        text_filter_mode: str = 'amharic_only',
        min_amharic_percentage: float = 30.0,
        custom_informal_patterns: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        checkpoint_interval: int = 50
    ):
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.session_name = session_name
        self.client = None
        self.text_processor = AmharicTextProcessor(custom_informal_patterns)
        self.text_filter_mode = text_filter_mode
        self.min_amharic_percentage = min_amharic_percentage
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.checkpoint_interval = checkpoint_interval
        
    async def connect(self) -> None:
        """Connect to Telegram API."""
        self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
        await self.client.start(phone=self.phone)
        
        if not await self.client.is_user_authorized():
            await self.client.send_code_request(self.phone)
            try:
                await self.client.sign_in(self.phone, input('Enter the code: '))
            except SessionPasswordNeededError:
                await self.client.sign_in(password=input('Password: '))
                
        logger.info("Successfully connected to Telegram API")
        
    async def join_channel(self, channel_username: str) -> bool:
        """
        Join a Telegram channel with retry logic.
        
        Returns:
            bool: True if joined successfully, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                channel = await self.client.get_entity(channel_username)
                await self.client(JoinChannelRequest(channel))
                logger.info(f"Successfully joined channel: {channel_username}")
                return True
            except FloodWaitError as e:
                wait_time = e.seconds
                logger.warning(f"Rate limited when joining {channel_username}. Waiting {wait_time} seconds. Attempt {attempt+1}/{self.max_retries}")
                await asyncio.sleep(wait_time)
            except (ChatAdminRequiredError, ChannelPrivateError) as e:
                logger.error(f"Cannot join {channel_username}: {e}. Channel may be private or require admin approval.")
                return False
            except Exception as e:
                logger.error(f"Failed to join channel {channel_username} (Attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    return False
        
        return False
            
    async def scrape_channel(
        self, 
        channel_username: str, 
        limit: int = 100, 
        offset_date: Optional[datetime] = None,
        filter_non_amharic: bool = True,
        checkpoint_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Scrape messages from a Telegram channel with enhanced filtering and checkpointing."""
        try:
            channel = await self.client.get_entity(channel_username)
            logger.info(f"Scraping messages from channel: {channel_username}")
            messages = []
            filtered_count = 0
            checkpoint_count = 0
            
            # Create checkpoint directory if specified
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)

            message_iter = self.client.iter_messages(
                channel,
                limit=limit,
                offset_date=offset_date
            )

            count = 0
            async for message in message_iter:
                if not message.text:
                    continue

                # Process text with filtering
                text_result = self.text_processor.process_text(
                    message.text, 
                    self.text_filter_mode
                )
                
                # Skip messages with insufficient Amharic content if filtering is enabled
                if (filter_non_amharic and 
                    text_result['analysis']['amharic_percentage'] < self.min_amharic_percentage):
                    filtered_count += 1
                    continue
                
                # Skip if processed text is empty after filtering
                if not text_result['processed_text'].strip():
                    filtered_count += 1
                    continue

                message_data = {
                    "id": message.id,
                    "date": message.date.isoformat(),
                    "original_text": message.text,
                    "processed_text": text_result['processed_text'],
                    "text_analysis": text_result['analysis'],
                    "views": getattr(message, "views", 0),
                    "channel": channel_username,
                    "has_media": message.media is not None,
                    "media_type": self._get_media_type(message),
                    "filter_mode": self.text_filter_mode,
                    "is_amharic_dominant": text_result['is_amharic_dominant'],
                    "length_reduction": text_result['length_reduction']
                }

                messages.append(message_data)
                count += 1
                checkpoint_count += 1
                
                # Create checkpoint if needed
                if checkpoint_dir and checkpoint_count >= self.checkpoint_interval:
                    checkpoint_file = os.path.join(
                        checkpoint_dir, 
                        f"{channel_username.replace('@', '')}_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    with open(checkpoint_file, "w", encoding="utf-8") as f:
                        json.dump(messages, f, ensure_ascii=False, indent=2)
                    logger.info(f"Created checkpoint with {len(messages)} messages at {checkpoint_file}")
                    checkpoint_count = 0
                
                if count % 10 == 0 or count == limit:
                    logger.info(f"{count}/{limit} messages scraped from {channel_username} ({filtered_count} filtered out)")

            logger.info(f"Successfully scraped {len(messages)} messages from {channel_username} ({filtered_count} filtered out)")
            return messages

        except FloodWaitError as e:
            wait_time = e.seconds
            logger.warning(f"Rate limited when scraping {channel_username}. Need to wait {wait_time} seconds.")
            await asyncio.sleep(wait_time)
            return await self.scrape_channel(channel_username, limit, offset_date, filter_non_amharic, checkpoint_dir)
        except Exception as e:
            logger.error(f"Failed to scrape channel {channel_username}: {e}")
            return []
            
    def _get_media_type(self, message: Message) -> str:
        """Get the media type of a message with enhanced detection."""
        if message.media:
            if isinstance(message.media, MessageMediaPhoto):
                return "photo"
            elif isinstance(message.media, MessageMediaDocument):
                # Try to determine document type
                if message.media.document.mime_type:
                    mime_type = message.media.document.mime_type
                    if mime_type.startswith('image/'):
                        return "image"
                    elif mime_type.startswith('video/'):
                        return "video"
                    elif mime_type.startswith('audio/'):
                        return "audio"
                    else:
                        return mime_type
                return "document"
            else:
                return str(type(message.media).__name__)
        return "none"
        
    async def scrape_multiple_channels(
        self, 
        channel_usernames: List[str], 
        limit_per_channel: int = 100,
        output_dir: str = "../../data/raw",
        output_format: str = "json",
        filter_non_amharic: bool = True,
        generate_analytics: bool = True,
        clear_existing_data: bool = False,
        checkpoint_dir: Optional[str] = None
    ) -> None:
        """Scrape messages from multiple Telegram channels with analytics and checkpointing."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear existing data if requested
        if clear_existing_data:
            self._clear_directory(output_dir)
            logger.info(f"Cleared existing data in {output_dir}")
        
        # Create checkpoint directory if needed
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        all_messages = []
        channel_stats = {}
        channel_success = {}

        for channel in channel_usernames:
            try:
                # Try to join the channel
                joined = await self.join_channel(channel)
                if not joined:
                    logger.warning(f"Skipping channel {channel} as we couldn't join it")
                    channel_success[channel] = False
                    continue
                
                # Scrape messages with retry logic
                for attempt in range(self.max_retries):
                    try:
                        messages = await self.scrape_channel(
                            channel, 
                            limit=limit_per_channel,
                            filter_non_amharic=filter_non_amharic,
                            checkpoint_dir=checkpoint_dir
                        )
                        
                        if messages:
                            all_messages.extend(messages)
                            channel_stats[channel] = self._generate_channel_stats(messages)
                            channel_success[channel] = True
                            
                            # Save channel-specific data
                            channel_filename = f"{channel.replace('@', '')}_messages"
                            self._save_data(messages, output_dir, channel_filename, output_format)
                        else:
                            logger.warning(f"No messages retrieved from {channel}")
                            channel_success[channel] = False
                            
                        break  # Exit retry loop if successful
                        
                    except FloodWaitError as e:
                        wait_time = e.seconds
                        logger.warning(f"Hit rate limit for {channel}. Need to wait {wait_time} seconds")
                        await asyncio.sleep(wait_time)
                    except Exception as e:
                        logger.error(f"Error scraping channel {channel} (Attempt {attempt+1}/{self.max_retries}): {e}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                        else:
                            channel_success[channel] = False

            except Exception as e:
                logger.error(f"Unexpected error processing channel {channel}: {e}")
                channel_success[channel] = False

        # Save all messages and analytics
        if all_messages:
            self._save_data(all_messages, output_dir, "all_messages", output_format)
            
            if generate_analytics:
                analytics = self._generate_overall_analytics(all_messages, channel_stats, channel_success)
                self._save_data([analytics], output_dir, "scraping_analytics", "json")
                
            logger.info(f"Successfully scraped {len(all_messages)} messages from {sum(channel_success.values())} channels")
        else:
            logger.warning("No messages were scraped from any channel")
    
    def _clear_directory(self, directory: str) -> None:
        """Clear all files in a directory but keep the directory itself."""
        for file_path in Path(directory).glob("*"):
            if file_path.is_file():
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.error(f"Error deleting file {file_path}: {e}")
        
    def _generate_channel_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics for a channel."""
        if not messages:
            return {}
            
        total_messages = len(messages)
        amharic_dominant = sum(1 for msg in messages if msg.get('is_amharic_dominant', False))
        mixed_content = sum(1 for msg in messages if msg.get('text_analysis', {}).get('has_mixed_content', False))
        informal_content = sum(1 for msg in messages if msg.get('text_analysis', {}).get('is_informal', False))
        
        # Calculate average message length
        avg_original_length = sum(len(msg.get('original_text', '')) for msg in messages) / total_messages
        avg_processed_length = sum(len(msg.get('processed_text', '')) for msg in messages) / total_messages
        
        # Calculate average reduction percentage
        avg_reduction_percentage = (1 - (avg_processed_length / avg_original_length)) * 100 if avg_original_length > 0 else 0
        
        avg_amharic_percentage = sum(
            msg.get('text_analysis', {}).get('amharic_percentage', 0) 
            for msg in messages
        ) / total_messages
        
        # Count messages with media
        media_count = sum(1 for msg in messages if msg.get('has_media', False))
        
        # Group by media type
        media_types = {}
        for msg in messages:
            if msg.get('has_media', False):
                media_type = msg.get('media_type', 'unknown')
                media_types[media_type] = media_types.get(media_type, 0) + 1
        
        return {
            'total_messages': total_messages,
            'amharic_dominant_count': amharic_dominant,
            'mixed_content_count': mixed_content,
            'informal_content_count': informal_content,
            'avg_amharic_percentage': round(avg_amharic_percentage, 2),
            'amharic_dominant_percentage': round((amharic_dominant / total_messages) * 100, 2),
            'mixed_content_percentage': round((mixed_content / total_messages) * 100, 2),
            'informal_content_percentage': round((informal_content / total_messages) * 100, 2),
            'avg_original_length': round(avg_original_length, 2),
            'avg_processed_length': round(avg_processed_length, 2),
            'avg_reduction_percentage': round(avg_reduction_percentage, 2),
            'media_count': media_count,
            'media_percentage': round((media_count / total_messages) * 100, 2),
            'media_types': media_types
        }
    
    def _generate_overall_analytics(
        self, 
        all_messages: List[Dict[str, Any]], 
        channel_stats: Dict[str, Dict[str, Any]],
        channel_success: Dict[str, bool]
    ) -> Dict[str, Any]:
        """Generate overall analytics for all scraped data."""
        # Calculate success rate
        total_channels = len(channel_success)
        successful_channels = sum(1 for success in channel_success.values() if success)
        
        # Get date range
        dates = [datetime.fromisoformat(msg['date']) for msg in all_messages if 'date' in msg]
        date_range = {
            'oldest': min(dates).isoformat() if dates else None,
            'newest': max(dates).isoformat() if dates else None,
            'span_days': (max(dates) - min(dates)).days + 1 if dates else 0
        }
        
        return {
            'scraping_timestamp': datetime.now().isoformat(),
            'total_messages_scraped': len(all_messages),
            'total_channels_attempted': total_channels,
            'successful_channels': successful_channels,
            'success_rate': round((successful_channels / total_channels) * 100, 2) if total_channels > 0 else 0,
            'date_range': date_range,
            'filter_mode_used': self.text_filter_mode,
            'min_amharic_percentage_threshold': self.min_amharic_percentage,
            'channel_statistics': channel_stats,
            'overall_statistics': self._generate_channel_stats(all_messages) if all_messages else {}
        }
        
    def _save_data(
        self, 
        data: List[Dict[str, Any]], 
        output_dir: str, 
        filename: str, 
        format: str
    ) -> None:
        """Save scraped data to file with error handling."""
        if not data:
            logger.warning(f"No data to save for {filename}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{filename}_{timestamp}")

        try:
            if format.lower() == "json":
                with open(f"{output_path}.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            elif format.lower() == "csv":
                # Flatten nested dictionaries for CSV
                flattened_data = []
                for item in data:
                    flat_item = item.copy()
                    if 'text_analysis' in flat_item:
                        analysis = flat_item.pop('text_analysis')
                        for key, value in analysis.items():
                            flat_item[f'analysis_{key}'] = value
                    flattened_data.append(flat_item)
                
                df = pd.DataFrame(flattened_data)
                df.to_csv(f"{output_path}.csv", index=False, encoding="utf-8")
            else:
                logger.error(f"Unsupported output format: {format}")
                return

            logger.info(f"Saved {len(data)} records to {output_path}.{format}")
        except Exception as e:
            logger.error(f"Error saving data to {output_path}.{format}: {e}")
        
    async def disconnect(self) -> None:
        """Disconnect from Telegram API."""
        if self.client:
            await self.client.disconnect()
            logger.info("Disconnected from Telegram API")

if __name__ == "__main__":
    import asyncio
    import dotenv
    from pathlib import Path

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
        # Initialize scraper with enhanced filtering
        scraper = TelegramScraper(
            API_ID, 
            API_HASH, 
            PHONE,
            text_filter_mode='amharic_only',  # Options: 'amharic_only', 'remove_emojis', 'remove_english_numbers', 'remove_emojis_english_numbers', 'analysis_only'
            min_amharic_percentage=30.0,  # Minimum percentage of Amharic characters to include message
            max_retries=3,               # Maximum number of retries for failed operations
            retry_delay=5,               # Base delay between retries (will use exponential backoff)
            checkpoint_interval=50       # Create checkpoint after every 50 messages
        )
        
        # Define output directories
        data_dir = Path(__file__).parent.parent.parent / "data"
        raw_dir = data_dir / "raw"
        checkpoint_dir = data_dir / "checkpoints"
        
        # Create directories
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        await scraper.connect()
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
        await scraper.disconnect()

    # Run the main function
    asyncio.run(main())
