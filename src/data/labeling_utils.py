"""
Labeling Utilities Module for Amharic E-commerce Data.
Provides functionality for converting between CSV and CoNLL formats,
validating data, and generating NER statistics.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class NERLabeler:
    def __init__(self):
        self.entity_types = [
            "B-Product", "I-Product",
            "B-PRICE", "I-PRICE",
            "B-LOC", "I-LOC",
            "B-DELIVERY_FEE", "I-DELIVERY_FEE",
            "B-CONTACT_INFO", "I-CONTACT_INFO",
            "O"  # Outside (not an entity)
        ]
    
    def csv_to_conll(self, csv_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Convert a CSV file with token and entity columns to CoNLL format.
        
        Args:
            csv_path: Path to the CSV file
            output_path: Path to save the CoNLL file (optional)
            
        Returns:
            CoNLL formatted string
        """
        try:
            df = pd.read_csv(csv_path)
            
            if "token" not in df.columns or "entity" not in df.columns:
                logger.error("CSV file must have token and entity columns")
                return ""
                
            conll_lines = []
            
            for _, row in df.iterrows():
                if pd.isna(row["token"]) or row["token"] == "":
                    conll_lines.append("")  # Empty line between sentences
                else:
                    conll_lines.append(f"{row['token']} {row['entity']}")
                    
            conll_text = "\n".join(conll_lines)
            
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(conll_text)
                logger.info(f"Saved CoNLL file to {output_path}")
                
            return conll_text
            
        except Exception as e:
            logger.error(f"Error converting CSV to CoNLL: {e}")
            return ""
    
    def conll_to_csv(self, conll_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Convert a CoNLL file to CSV format.
        
        Args:
            conll_path: Path to the CoNLL file
            output_path: Path to save the CSV file (optional)
            
        Returns:
            DataFrame with token and entity columns
        """
        try:
            tokens = []
            entities = []
            
            with open(conll_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            tokens.append(parts[0])
                            entities.append(parts[1])
                    else:
                        # Empty line indicates sentence boundary
                        tokens.append("")
                        entities.append("")
                        
            df = pd.DataFrame({
                "token": tokens,
                "entity": entities
            })
            
            if output_path:
                df.to_csv(output_path, index=False, encoding="utf-8")
                logger.info(f"Saved CSV file to {output_path}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error converting CoNLL to CSV: {e}")
            return pd.DataFrame()
    
    def validate_labels(self, labels: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate NER labels for consistency.
        
        Args:
            labels: List of NER labels
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        for i, label in enumerate(labels):
            if label not in self.entity_types:
                errors.append(f"Invalid label '{label}' at position {i}")
                
            # Check for invalid I- tags without preceding B- tags
            if label.startswith("I-"):
                entity_type = label[2:]
                if i == 0 or not labels[i-1].endswith(entity_type):
                    errors.append(f"I-{entity_type} at position {i} without preceding B-{entity_type}")
                    
        return len(errors) == 0, errors
    
    def create_labeling_template(self, messages: List[str], output_path: Union[str, Path]):
        """
        Create a CSV template for manual labeling from message data.
        
        Args:
            messages: List of message texts
            output_path: Path to save the CSV template
        """
        try:
            tokens = []
            entities = []
            
            for message in tqdm(messages, desc="Processing messages"):
                if message and message.strip():
                    # Simple tokenization - split by whitespace
                    message_tokens = message.strip().split()
                    tokens.extend(message_tokens)
                    entities.extend(['O'] * len(message_tokens))
                    
                    # Add empty row for sentence boundary
                    tokens.append('')
                    entities.append('')
            
            df = pd.DataFrame({
                'token': tokens,
                'entity': entities
            })
            
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Created labeling template with {len(df)} rows: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating labeling template: {e}")
    
    def get_entity_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate statistics about labeled entities.
        
        Args:
            df: DataFrame with token and entity columns
            
        Returns:
            Dictionary with entity statistics
        """
        try:
            # Filter out empty tokens and O labels
            labeled_df = df[(df['token'] != '') & (df['entity'] != 'O')]
            
            stats = {
                'total_tokens': len(df[df['token'] != '']),
                'total_entities': len(labeled_df),
                'entity_coverage': len(labeled_df) / len(df[df['token'] != '']) if len(df[df['token'] != '']) > 0 else 0,
                'entity_counts': labeled_df['entity'].value_counts().to_dict(),
                'unique_entities': labeled_df['entity'].nunique()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating entity statistics: {e}")
            return {}