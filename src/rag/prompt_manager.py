"""
System Prompt Management for RAG System
Handles dynamic prompt selection and customization for different use cases.
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from loguru import logger

class PromptManager:
    """Manages system prompts for different use cases and contexts."""
    
    def __init__(self, prompts_file: str = "config/prompts.json"):
        self.prompts_file = Path(prompts_file)
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from JSON file."""
        if self.prompts_file.exists():
            try:
                with open(self.prompts_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading prompts: {e}")
                return self._get_default_prompts()
        else:
            # Create default prompts file
            default_prompts = self._get_default_prompts()
            self._save_prompts(default_prompts)
            return default_prompts
    
    def _save_prompts(self, prompts: Dict[str, Any]) -> bool:
        """Save prompts to JSON file."""
        try:
            # Ensure directory exists
            self.prompts_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.prompts_file, 'w', encoding='utf-8') as f:
                json.dump(prompts, f, indent=2, ensure_ascii=False)
            
            self.prompts = prompts
            logger.info(f"Prompts saved to {self.prompts_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving prompts: {e}")
            return False
    
    def _get_default_prompts(self) -> Dict[str, Any]:
        """Get default prompt templates for different use cases."""
        return {
            "default": {
                "name": "Default Assistant",
                "description": "General purpose helpful assistant",
                "system_prompt": "You are a helpful assistant. Answer based on the provided context. Be concise and professional.",
                "use_cases": ["general"],
                "active": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "customer_service": {
                "name": "Customer Service",
                "description": "Friendly customer service representative",
                "system_prompt": """You are a friendly and helpful customer service representative. 

Your role:
- Answer customer questions based on the provided company information
- Be empathetic and understanding
- Provide clear, actionable solutions
- If you don't know something, direct them to contact support
- Always maintain a positive, professional tone

Guidelines:
- Use the customer's name if provided
- Acknowledge their concerns
- Provide step-by-step solutions when needed
- End with asking if there's anything else you can help with""",
                "use_cases": ["support", "customer_service"],
                "active": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "sales_assistant": {
                "name": "Sales Assistant",
                "description": "Knowledgeable sales representative",
                "system_prompt": """You are an expert sales assistant with deep knowledge of our products.

Your role:
- Help customers understand our products and services
- Highlight benefits that match their needs
- Provide accurate pricing and availability information
- Guide them through the decision-making process
- Create urgency when appropriate

Guidelines:
- Ask qualifying questions to understand their needs
- Present solutions, not just features
- Handle objections professionally
- Always be honest about product capabilities
- Focus on value, not just price""",
                "use_cases": ["sales", "product_info", "e-commerce"],
                "active": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "technical_support": {
                "name": "Technical Support",
                "description": "Technical expert for troubleshooting",
                "system_prompt": """You are a technical support specialist with expertise in troubleshooting.

Your role:
- Diagnose technical issues based on user descriptions
- Provide step-by-step troubleshooting instructions
- Explain technical concepts in simple terms
- Escalate complex issues when necessary

Guidelines:
- Ask diagnostic questions to narrow down the issue
- Provide clear, numbered steps
- Explain why each step is important
- Offer alternative solutions when possible
- Verify the solution worked before closing""",
                "use_cases": ["technical", "troubleshooting", "support"],
                "active": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "health_nutrition": {
                "name": "Health & Nutrition Expert",
                "description": "Knowledgeable health and nutrition advisor",
                "system_prompt": """You are a knowledgeable health and nutrition expert specializing in weight management and healthy living.

Your expertise:
- Nutrition science and dietary recommendations
- Weight loss and weight management strategies
- Exercise and lifestyle advice
- Product recommendations based on health goals

Guidelines:
- Always base advice on the provided company information
- Emphasize the importance of consulting healthcare professionals
- Provide evidence-based information
- Be supportive and motivational
- Address safety concerns appropriately
- Never diagnose medical conditions

Important: Always recommend consulting with a healthcare professional for personalized medical advice.""",
                "use_cases": ["health", "nutrition", "weight_loss", "fitness"],
                "active": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            },
            "brand_voice_nupo": {
                "name": "Nupo Brand Voice",
                "description": "Official Nupo brand personality",
                "system_prompt": """You are the official voice of Nupo, a leading weight management and nutrition company.

Brand personality:
- Supportive and encouraging
- Science-backed and trustworthy
- Approachable and friendly
- Focused on sustainable health solutions
- Empowering customers on their health journey

Your expertise:
- Nupo products and their benefits
- Weight management strategies
- Nutrition science
- Healthy lifestyle advice

Communication style:
- Use "we" when referring to Nupo
- Be encouraging about health journeys
- Emphasize sustainable, healthy approaches
- Mention scientific backing when relevant
- Always prioritize customer safety and well-being

Key messages:
- Health is a journey, not a destination
- Sustainable changes lead to lasting results
- Our products support your health goals
- Professional guidance is always recommended""",
                "use_cases": ["nupo", "brand", "marketing", "health"],
                "active": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        }
    
    def get_prompt(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific prompt by ID."""
        return self.prompts.get(prompt_id)
    
    def get_all_prompts(self) -> Dict[str, Any]:
        """Get all prompts."""
        return self.prompts
    
    def get_active_prompts(self) -> Dict[str, Any]:
        """Get only active prompts."""
        return {k: v for k, v in self.prompts.items() if v.get('active', True)}
    
    def get_prompts_by_use_case(self, use_case: str) -> Dict[str, Any]:
        """Get prompts that match a specific use case."""
        matching_prompts = {}
        for prompt_id, prompt_data in self.prompts.items():
            if use_case in prompt_data.get('use_cases', []):
                matching_prompts[prompt_id] = prompt_data
        return matching_prompts
    
    def add_prompt(self, prompt_id: str, prompt_data: Dict[str, Any]) -> bool:
        """Add a new prompt."""
        try:
            # Add metadata
            prompt_data['created_at'] = datetime.now().isoformat()
            prompt_data['updated_at'] = datetime.now().isoformat()
            
            # Set defaults
            if 'active' not in prompt_data:
                prompt_data['active'] = True
            if 'use_cases' not in prompt_data:
                prompt_data['use_cases'] = ["general"]
            
            self.prompts[prompt_id] = prompt_data
            return self._save_prompts(self.prompts)
        except Exception as e:
            logger.error(f"Error adding prompt {prompt_id}: {e}")
            return False
    
    def update_prompt(self, prompt_id: str, prompt_data: Dict[str, Any]) -> bool:
        """Update an existing prompt."""
        try:
            if prompt_id not in self.prompts:
                logger.error(f"Prompt {prompt_id} not found")
                return False
            
            # Preserve created_at, update updated_at
            if 'created_at' not in prompt_data and 'created_at' in self.prompts[prompt_id]:
                prompt_data['created_at'] = self.prompts[prompt_id]['created_at']
            
            prompt_data['updated_at'] = datetime.now().isoformat()
            
            self.prompts[prompt_id] = prompt_data
            return self._save_prompts(self.prompts)
        except Exception as e:
            logger.error(f"Error updating prompt {prompt_id}: {e}")
            return False
    
    def delete_prompt(self, prompt_id: str) -> bool:
        """Delete a prompt."""
        try:
            if prompt_id not in self.prompts:
                logger.error(f"Prompt {prompt_id} not found")
                return False
            
            del self.prompts[prompt_id]
            return self._save_prompts(self.prompts)
        except Exception as e:
            logger.error(f"Error deleting prompt {prompt_id}: {e}")
            return False
    
    def get_system_prompt(self, prompt_id: str = "default", context: Optional[Dict[str, Any]] = None) -> str:
        """Get the system prompt text for a given prompt ID."""
        prompt_data = self.get_prompt(prompt_id)
        if not prompt_data:
            logger.warning(f"Prompt {prompt_id} not found, using default")
            prompt_data = self.get_prompt("default")
        
        if not prompt_data:
            # Fallback if even default doesn't exist
            return "You are a helpful assistant. Answer based on the provided context."
        
        system_prompt = prompt_data.get('system_prompt', '')
        
        # TODO: Add context-based prompt customization here
        # For example, inject user name, company info, etc.
        if context:
            # Future enhancement: template variable substitution
            pass
        
        return system_prompt
    
    def suggest_prompt(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Suggest the best prompt based on query content and context."""
        query_lower = query.lower()
        
        # Simple keyword-based prompt selection
        if any(word in query_lower for word in ['buy', 'purchase', 'price', 'cost', 'order']):
            return "sales_assistant"
        elif any(word in query_lower for word in ['problem', 'issue', 'error', 'broken', 'not working']):
            return "technical_support"
        elif any(word in query_lower for word in ['weight', 'diet', 'nutrition', 'health', 'calories', 'exercise']):
            return "health_nutrition"
        elif any(word in query_lower for word in ['nupo', 'product', 'shake', 'meal replacement']):
            return "brand_voice_nupo"
        elif any(word in query_lower for word in ['help', 'support', 'question', 'how']):
            return "customer_service"
        else:
            return "default"
    
    def get_prompt_stats(self) -> Dict[str, Any]:
        """Get statistics about prompts."""
        total_prompts = len(self.prompts)
        active_prompts = len(self.get_active_prompts())
        
        use_cases = set()
        for prompt_data in self.prompts.values():
            use_cases.update(prompt_data.get('use_cases', []))
        
        return {
            'total_prompts': total_prompts,
            'active_prompts': active_prompts,
            'inactive_prompts': total_prompts - active_prompts,
            'unique_use_cases': len(use_cases),
            'use_cases': sorted(list(use_cases))
        }
