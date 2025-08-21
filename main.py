"""
Main entry point for the Patent AI Agent
Provides a simple interface to run the patent keyword extraction system
"""

from src.core.extractor import CoreConceptExtractor
import gradio as gr

extractor = CoreConceptExtractor()

def run_extraction(problem, technical):
    input_text = f"Problem: {problem}\nTechnical: {technical}"
    results = extractor.extract_keywords(input_text)
    # Format results for display
    output = ""
    for key, value in results.items():
        if value is None:
            continue
        output += f"### {key}\n"
        if hasattr(value, "dict"):
            for subkey, subval in value.dict().items():
                output += f"- **{subkey.replace('_', ' ').title()}**: {subval}\n"
        elif isinstance(value, dict):
            for subkey, subval in value.items():
                output += f"- **{subkey}**: {subval}\n"
        elif isinstance(value, list):
            for i, item in enumerate(value, 1):
                if isinstance(item, dict):
                    output += f"{i}. " + ", ".join([f"{k}: {v}" for k, v in item.items()]) + "\n"
                else:
                    output += f"{i}. {item}\n"
        else:
            output += f"{value}\n"
    return output

demo = gr.Interface(
    fn=run_extraction,
    inputs=[
        gr.Textbox(label="Problem", lines=2, placeholder="Nhập vấn đề kỹ thuật..."),
        gr.Textbox(label="Technical", lines=2, placeholder="Nhập nội dung kỹ thuật...")
    ],
    outputs=gr.Markdown(),
    title="Patent Keyword Extraction",
    description="Nhập thông tin vào 2 ô bên dưới. Kết quả sẽ hiển thị đầy đủ, đơn giản và dễ nhìn.",
    theme="default"
)

if __name__ == "__main__":
    demo.launch()
