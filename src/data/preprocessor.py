"""
Preprocessor Module for Amharic E-commerce Data.

This module provides functionality to preprocess raw Amharic text data
from Telegram channels for NER tasks.
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class AmharicPreprocessor:
    """
    A class to preprocess Amharic text data for NER tasks.
    """
    
    def __init__(self):
        """
        Initialize the AmharicPreprocessor.
        """
        # Common Amharic punctuation and special characters
        self.punctuation = [
            "።", "፡", "፣", "፤", "፥", "፦", "፧", "፨", "?", "!", ".", ",", ";", ":", "-", "(", ")", "[", "]", "{", "}", 
            "\"", "'", "/", "\\", "@", "#", "$", "%", "&", "*", "+", "=", "<", ">", "|", "~", "^", "_"
        ]
        
    def load_data(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load data from a JSON or CSV file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            List of dictionaries containing message data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []
            
        try:
            if file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data
            elif file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path, encoding="utf-8")
                return df.to_dict("records")
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return []
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            return []
            
    def normalize_text(self, text: str) -> str:
        """
        Normalize Amharic text by removing extra whitespace and normalizing characters.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Replace common non-standard characters with their standard equivalents
        # This is a simplified example - may need to be expanded for Amharic
        replacements = {
            '፡፡': '።',  # Replace double colon with Amharic full stop
            '...': '…',  # Replace triple dots with ellipsis
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text
        
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Amharic text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # Simple whitespace tokenization
        tokens = text.split()
        
        # Handle punctuation - this is a simplified approach
        # For Amharic, more sophisticated tokenization might be needed
        processed_tokens = []
        for token in tokens:
            # Check if token ends with punctuation
            if token and token[-1] in self.punctuation:
                processed_tokens.append(token[:-1])
                processed_tokens.append(token[-1])
            else:
                processed_tokens.append(token)
                
        return [t for t in processed_tokens if t]  # Remove empty tokens
        
    def clean_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a message by normalizing and tokenizing the text.
        
        Args:
            message: Dictionary containing message data
            
        Returns:
            Cleaned message dictionary
        """
        if not message or "text" not in message:
            return message
            
        cleaned_message = message.copy()
        
        # Normalize text
        cleaned_message["normalized_text"] = self.normalize_text(message["text"])
        
        # Tokenize text
        cleaned_message["tokens"] = self.tokenize(cleaned_message["normalized_text"])
        
        return cleaned_message
        
    def extract_potential_entities(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract potential entities from a message based on patterns.
        
        Args:
            message: Dictionary containing message data
            
        Returns:
            Message dictionary with potential entities
        """
        if not message or "normalized_text" not in message:
            return message
            
        message_with_entities = message.copy()
        text = message["normalized_text"]
        
        # Extract potential product names (this is a simplified approach)
        # In a real implementation, more sophisticated NLP techniques would be used
        
        # Extract potential prices (looking for numbers followed by "ብር" or "ETB")
        price_pattern = r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:ብር|ETB|Birr|birr)'
        prices = re.findall(price_pattern, text, re.IGNORECASE)
        
        # Extract potential locations (this is a simplified approach)
        # Common Ethiopian locations
        locations = ["አዲስ አበባ", "ባህር ዳር", "ጎንደር", "መቀሌ", "ሐዋሳ", "ድሬዳዋ", "ጅማ", 
                    "ደሴ", "አዳማ", "ኮምቦልቻ", "ደብረ ብርሃን", "ደብረ ማርቆስ", "ደብረ ዘይት", 
                    "ሰሜን ሸዋ", "ደቡብ ወሎ", "ምስራቅ ጎጃም", "ምዕራብ ጎጃም", "አርሲ", "ባሌ", "ወላይታ",
                    "ቦሌ", "አያት", "ሲኤምሲ", "ቂርቆስ", "ካሳንቺስ", "4ኪሎ", "6ኪሎ", "መገናኛ"]
                    
        found_locations = [loc for loc in locations if loc.lower() in text.lower()]
        
        # Add extracted entities to message
        message_with_entities["potential_entities"] = {
            "prices": prices,
            "locations": found_locations
        }
        
        return message_with_entities
        
    def process_data(
        self, 
        data: List[Dict[str, Any]], 
        output_dir: str = "../../data/processed",
        output_filename: str = "processed_data"
    ) -> List[Dict[str, Any]]:
        """
        Process a list of messages by cleaning and extracting potential entities.
        
        Args:
            data: List of message dictionaries
            output_dir: Directory to save processed data
            output_filename: Name of file to save processed data
            
        Returns:
            List of processed message dictionaries
        """
        if not data:
            logger.warning("No data to process")
            return []
            
        processed_data = []
        
        for message in tqdm(data, desc="Processing messages"):
            # Clean message
            cleaned_message = self.clean_message(message)
            
            # Extract potential entities
            processed_message = self.extract_potential_entities(cleaned_message)
            
            processed_data.append(processed_message)
            
        # Save processed data
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename}.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(processed_data)} processed messages to {output_path}")
        
        return processed_data
        
    def prepare_for_ner(
        self, 
        data: List[Dict[str, Any]], 
        output_dir: str = "../../data/processed",
        output_filename: str = "ner_ready_data"
    ) -> pd.DataFrame:
        """
        Prepare data for NER labeling by extracting relevant fields.
        
        Args:
            data: List of processed message dictionaries
            output_dir: Directory to save NER-ready data
            output_filename: Name of file to save NER-ready data
            
        Returns:
            DataFrame with data ready for NER labeling
        """
        if not data:
            logger.warning("No data to prepare for NER")
            return pd.DataFrame()
            
        ner_data = []
        
        for message in data:
            if "tokens" not in message:
                continue
                
            # Create a record for each token
            for token in message["tokens"]:
                ner_record = {
                    "message_id": message.get("id", ""),
                    "channel": message.get("channel", ""),
                    "token": token,
                    "entity": "O"  # Default entity is Outside (O)
                }
                
                ner_data.append(ner_record)
                
            # Add a blank line between messages (for CoNLL format)
            ner_data.append({
                "message_id": "",
                "channel": "",
                "token": "",
                "entity": ""
            })
            
        # Convert to DataFrame
        df = pd.DataFrame(ner_data)
        
        # Save NER-ready data
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_filename}.csv")
        
        df.to_csv(output_path, index=False, encoding="utf-8")
        
        logger.info(f"Saved NER-ready data to {output_path}")
        
        return df


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Initialize preprocessor
    preprocessor = AmharicPreprocessor()
    
    # Load data
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    data_file = list(data_dir.glob("all_messages_*.json"))
    
    if data_file:
        # Use the most recent file
        data_file = sorted(data_file)[-1]
        
        # Load data
        data = preprocessor.load_data(data_file)
        
        # Process data
        processed_data = preprocessor.process_data(
            data,
            output_dir=str(Path(__file__).parent.parent.parent / "data" / "processed"),
            output_filename="processed_messages"
        )
        
        # Prepare data for NER
        ner_data = preprocessor.prepare_for_ner(
            processed_data,
            output_dir=str(Path(__file__).parent.parent.parent / "data" / "processed"),
            output_filename="ner_ready_data"
        )
    else:
        logger.error("No data files found in the raw data directory") 