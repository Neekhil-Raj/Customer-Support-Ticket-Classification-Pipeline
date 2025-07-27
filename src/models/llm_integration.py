import openai
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                         pipeline, AutoModel, AutoModelForCausalLM)
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from config.config import Config

class LLMIntegration:
    def __init__(self):
        self.config = Config()
        self.openai_client = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI
        if self.config.api.openai_api_key:
            openai.api_key = self.config.api.openai_api_key
    
    def load_huggingface_model(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Load Hugging Face model for text generation/classification"""
        try:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.hf_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.logger.info(f"Loaded Hugging Face model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error loading Hugging Face model: {e}")
    
    def classify_with_llm(self, ticket_text: str, categories: List[str]) -> Dict:
        """Classify ticket using LLM with few-shot prompting"""
        
        # Few-shot examples
        examples = [
            {
                "text": "My password is not working and I cannot login to my account",
                "category": "Authentication"
            },
            {
                "text": "The software crashes every time I try to save a document",
                "category": "Technical"
            },
            {
                "text": "I want to cancel my subscription immediately",
                "category": "Billing"
            },
            {
                "text": "How do I export my data from the platform?",
                "category": "General Inquiry"
            }
        ]
        
        # Construct prompt
        prompt = "Classify the following customer support tickets into categories.\n\n"
        
        # Add examples
        for example in examples:
            prompt += f"Text: {example['text']}\nCategory: {example['category']}\n\n"
        
        # Add the ticket to classify
        prompt += f"Text: {ticket_text}\nCategory:"
        
        try:
            if self.config.api.openai_api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a customer support ticket classifier. Classify tickets into one of these categories: {', '.join(categories)}"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=50,
                    temperature=0.1
                )
                
                predicted_category = response.choices[0].message.content.strip()
                confidence = 0.85  # Placeholder confidence score
                
                return {
                    "predicted_category": predicted_category,
                    "confidence": confidence,
                    "method": "OpenAI GPT"
                }
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
        
        # Fallback to Hugging Face if available
        if self.hf_model and self.hf_tokenizer:
            return self._classify_with_hf(ticket_text, categories)
        
        return {"predicted_category": "Unknown", "confidence": 0.0, "method": "None"}
    
    def _classify_with_hf(self, ticket_text: str, categories: List[str]) -> Dict:
        """Classify using Hugging Face model"""
        try:
            # Use a classification pipeline
            classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            result = classifier(ticket_text, categories)
            
            return {
                "predicted_category": result['labels'][0],
                "confidence": result['scores'][0],
                "method": "Hugging Face BART"
            }
        except Exception as e:
            self.logger.error(f"Hugging Face classification error: {e}")
            return {"predicted_category": "Unknown", "confidence": 0.0, "method": "Error"}
    
    def summarize_ticket(self, ticket_text: str, max_length: int = 100) -> str:
        """Summarize ticket content using LLM"""
        try:
            if self.config.api.openai_api_key:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "Summarize the following customer support ticket in a concise manner."
                        },
                        {
                            "role": "user",
                            "content": ticket_text
                        }
                    ],
                    max_tokens=max_length,
                    temperature=0.3
                )
                
                return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"Error summarizing ticket: {e}")
        
        # Fallback to simple truncation
        return ticket_text[:max_length] + "..." if len(ticket_text) > max_length else ticket_text
    
    def suggest_routing(self, ticket_text: str, ticket_category: str) -> Dict:
        """Suggest ticket routing using RAG approach"""
        
        # Knowledge base for routing (in practice, this would be a vector database)
        routing_knowledge = {
            "Authentication": {
                "department": "IT Security",
                "priority": "High",
                "estimated_resolution": "2-4 hours"
            },
            "Technical": {
                "department": "Technical Support",
                "priority": "Medium",
                "estimated_resolution": "4-24 hours"
            },
            "Billing": {
                "department": "Finance",
                "priority": "Medium",
                "estimated_resolution": "1-3 business days"
            },
            "General Inquiry": {
                "department": "Customer Service",
                "priority": "Low",
                "estimated_resolution": "1-2 business days"
            }
        }
        
        # Get base routing info
        base_routing = routing_knowledge.get(ticket_category, {
            "department": "General Support",
            "priority": "Medium",
            "estimated_resolution": "2-5 business days"
        })
        
        # Use LLM to enhance routing decision
        try:
            if self.config.api.openai_api_key:
                prompt = f"""
                Based on this customer support ticket, provide routing recommendations:
                
                Ticket: {ticket_text}
                Category: {ticket_category}
                
                Consider:
                1. Urgency indicators
                2. Complexity level
                3. Required expertise
                
                Provide recommendations for:
                - Priority level (Low/Medium/High/Critical)
                - Specialized skills needed
                - Estimated resolution time
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at routing customer support tickets to the appropriate teams."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=200,
                    temperature=0.2
                )
                
                llm_suggestions = response.choices[0].message.content.strip()
                base_routing["llm_suggestions"] = llm_suggestions
        
        except Exception as e:
            self.logger.error(f"Error getting routing suggestions: {e}")
        
        return base_routing
    
    def generate_response_template(self, ticket_category: str, ticket_text: str) -> str:
        """Generate response template for agents"""
        try:
            if self.config.api.openai_api_key:
                prompt = f"""
                Create a professional response template for a customer support agent to address this ticket:
                
                Category: {ticket_category}
                Customer Issue: {ticket_text}
                
                The template should:
                1. Acknowledge the customer's concern
                2. Provide initial guidance or next steps
                3. Set appropriate expectations
                4. Maintain a professional and empathetic tone
                
                Include placeholders for personalization (e.g., [Customer Name], [Specific Details])
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at creating customer support response templates."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=300,
                    temperature=0.4
                )
                
                return response.choices[0].message.content.strip()
        
        except Exception as e:
            self.logger.error(f"Error generating response template: {e}")
        
        # Fallback template
        return f"""
        Dear [Customer Name],

        Thank you for contacting our support team regarding your {ticket_category.lower()} issue.

        We have received your request and are reviewing the details you provided. Our team will investigate this matter and get back to you with a resolution as soon as possible.

        In the meantime, if you have any additional information that might help us resolve this issue faster, please don't hesitate to reply to this message.

        We appreciate your patience and will keep you updated on our progress.

        Best regards,
        [Agent Name]
        Customer Support Team
        """
