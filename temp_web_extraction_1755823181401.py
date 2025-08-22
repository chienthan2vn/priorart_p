
import sys
import json
import os
sys.path.append('/home/merlin/project/priorart_p')
from src.core.extractor import CoreConceptExtractor

def main():
    input_text = """Problem: **User problem**: Traditional irrigation systems either over-water or under-water crops because 
    they operate on fixed schedules without considering actual soil moisture, weather conditions, 
    or crop-specific needs. This leads to water waste, increased costs, and potentially reduced 
    crop yields.
Technical: **Idea title**: Smart Irrigation System with IoT Sensors

    **User scenario**: A farmer managing a large agricultural field needs to optimize water usage 
    while ensuring crops receive adequate moisture. The farmer wants to monitor soil conditions 
    remotely and automatically adjust irrigation based on real-time data from multiple field locations."""
    
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
            elif hasattr(value, 'summary'):  # Handle SummaryResponse object
                serialized_result[key] = value.summary
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
