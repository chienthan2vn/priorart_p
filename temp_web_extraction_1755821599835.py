
import sys
import json
import os
sys.path.append('/home/merlin/project/priorart_p')
from src.core.extractor import CoreConceptExtractor

def main():
    input_text = """Problem: hhh
Technical: hhhhhh"""
    
    try:
        # Initialize extractor with web_mode=True
        extractor = CoreConceptExtractor(web_mode=True)
        
        # Send initial status
        print(json.dumps({"type": "status", "message": "Starting extraction..."}))
        sys.stdout.flush()
        
        # Run extraction
        result = extractor.extract_keywords(input_text)
        
        # Send final results
        serialized_result = {}
        for key, value in result.items():
            if hasattr(value, 'dict'):
                serialized_result[key] = value.dict()
            elif isinstance(value, list):
                serialized_result[key] = [
                    item.dict() if hasattr(item, 'dict') else item 
                    for item in value
                ]
            else:
                serialized_result[key] = value
        
        output = {
            "type": "results",
            "data": serialized_result
        }
        print(json.dumps(output))
        sys.stdout.flush()
        
    except Exception as e:
        import traceback
        error_output = {
            "type": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_output))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
