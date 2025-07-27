from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TicketPriority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class TicketStatus(str, Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    CLOSED = "Closed"

class TicketChannel(str, Enum):
    EMAIL = "Email"
    CHAT = "Chat"
    PHONE = "Phone"
    WEB = "Web"

class TicketRequest(BaseModel):
    customer_name: str
    customer_email: EmailStr
    customer_age: Optional[int] = None
    customer_gender: Optional[str] = None
    product_purchased: Optional[str] = None
    ticket_subject: str
    ticket_description: str
    ticket_priority: Optional[TicketPriority] = TicketPriority.MEDIUM
    ticket_channel: Optional[TicketChannel] = TicketChannel.WEB
    image_data: Optional[str] = None  # Base64 encoded image

class TicketResponse(BaseModel):
    ticket_id: str
    predicted_category: str
    confidence_score: float
    suggested_department: str
    estimated_resolution_time: str
    priority_recommendation: TicketPriority
    response_template: str
    created_at: datetime

class ClassificationRequest(BaseModel):
    text: str
    use_llm: bool = False

class ClassificationResponse(BaseModel):
    predicted_category: str
    confidence_score: float
    all_predictions: Dict[str, float]
    method_used: str

class BatchClassificationRequest(BaseModel):
    texts: List[str]
    use_llm: bool = False

class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]
    total_processed: int

class ModelMetrics(BaseModel):
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    roc_auc: Optional[float] = None

class ModelStatus(BaseModel):
    model_name: str
    version: str
    last_trained: datetime
    metrics: ModelMetrics
    status: str

class HealthResponse(BaseModel):
    status: str
    message: str
