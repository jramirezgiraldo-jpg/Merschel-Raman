from pydantic import BaseModel  
class SpectrumInput(BaseModel):  
    name: str  
    x: list[float]  
    y: list[float]  
try:  
    s = SpectrumInput(**{'name':'test', 'x':[1.0], 'y':[1.0], 'originalIndex': 5})  
    print('SUCCESS')  
except Exception as e:  
    print('ERROR', e)  
