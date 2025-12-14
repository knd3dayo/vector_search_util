from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Any


import vector_search_util.log.log_settings as log_settings
logger = log_settings.getLogger(__name__)

class EmbeddingData(BaseModel):
    page_content: str
    source_id: str
    category: str = ""
    metadata : dict[str, Any] = Field(default_factory=dict)
    
