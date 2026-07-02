import requests  
import json  
url = 'http://localhost:8000/api/predict'  
payload = {  
    'train_spectra': [{'name':'A', 'x':[1000.0, 1100.0, 1200.0], 'y':[0.1, 0.2, 0.3], 'label':'Clase1'}, {'name':'B', 'x':[1000.0, 1100.0, 1200.0], 'y':[0.2, 0.3, 0.4], 'label':'Clase1'}, {'name':'C', 'x':[1000.0, 1100.0, 1200.0], 'y':[0.5, 0.6, 0.7], 'label':'Clase2'}],  
    'test_spectra': [{'name':'test1', 'x':[1000.0, 1100.0, 1200.0], 'y':[0.3, 0.4, 0.5], 'originalIndex': 0}],  
    'n_components': 2,  
    'algorithm': 'PLS-DA'  
}  
try:  
    from backend.main import predict_plsda, PredictPayload  
    import asyncio  
    p = PredictPayload(**payload)  
    res = asyncio.run(predict_plsda(p))  
    print('RESULT:', res)  
except Exception as e:  
    print('ERROR:', e)  
