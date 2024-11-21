import torch
from typing import Dict, Type
from .base_model import BaseModel

class ModelFactory:
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type[BaseModel]) -> Type[BaseModel]:
            if name in cls._models:
                raise ValueError(f'Model {name} already exists')
            cls._models[name] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
    @classmethod
    def create_model(cls, name: str, **kwargs) -> BaseModel:
        if name not in cls._models:
            raise ValueError(f'Model {name} not found. Available models: {list(cls._models.keys())}')
            
        model_class = cls._models[name]
        model = model_class(**kwargs)
        return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))