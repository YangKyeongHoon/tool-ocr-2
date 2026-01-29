import requests
import base64
import os
from pathlib import Path
import sys

MODELS = [
    "yasserrmd/Nanonets-OCR-s:latest",
    "MedAIBase/PaddleOCR-VL:0.9b",
    "deepseek-ocr:latest"
]

IMAGE_DIRECTORY = "resources/receipts"
BASE_OUTPUT_DIRECTORY = "result/ocr_outputs"
COMPARISON_REPORT_PATH = "result/ollama_ocr_comparison_results.md"
NUM_SAMPLE_IMAGES = 3 # Process only a subset of images to avoid timeout

def run_ollama_ocr_integrated(model_name, image_dir, base_output_dir, num_sample_images):
    url = "http://localhost:11434/api/generate"
    
    # Create a model-specific output directory
    model_output_dir_name = model_name.replace('/', '_').replace(':', '_')
    model_output_path = Path(base_output_dir) / model_output_dir_name
    model_output_path.mkdir(parents=True, exist_ok=True)
    
    all_image_files = [f for f in Path(image_dir).iterdir() if f.is_file() and f.suffix.lower() in ['.jpeg', '.jpg', '.png']]
    
    image_files = all_image_files[:int(num_sample_images)]

    if not image_files:
        print(f"No image files found in {image_dir} or selected samples.", file=sys.stderr)
        return False

    success_all = True
    print(f"\n--- Running OCR for model: {model_name} ---")
    for image_path in image_files:
        try:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}", file=sys.stderr)
            success_all = False
            continue
        except Exception as e:
            print(f"Error reading image file {image_path}: {e}", file=sys.stderr)
            success_all = False
            continue

        prompt = "Extract all text from this image. Provide only the extracted text."
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "images": [image_data]
        }

        print(f"Running OCR with model: {model_name} on image: {image_path.name}")
        try:
            response = requests.post(url, json=payload, timeout=600) # Increased timeout
            response.raise_for_status()
            
            result = response.json()
            ocr_text = result.get("response", "").strip()

            output_filename = model_output_path / f"{image_path.stem}.txt"
            with open(output_filename, "w", encoding="utf-8") as outfile:
                outfile.write(ocr_text)
            print(f"OCR result for {image_path.name} saved to {output_filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API for model {model_name} on image {image_path.name}: {e}", file=sys.stderr)
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}", file=sys.stderr)
            success_all = False
        except Exception as e:
            print(f"An unexpected error occurred for image {image_path.name}: {e}", file=sys.stderr)
            success_all = False
    return success_all

def generate_comparison_report(models, base_output_dir, report_path, image_dir, num_sample_images):
    print("\n--- Generating comparison report ---")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("### 📝 **Ollama OCR 모델 비교 결과**\n\n")
        f.write("다양한 OCR 모델들의 성능을 비교했습니다.\n\n")

        # Get a list of image names for which OCR was performed
        sample_images = []
        image_path = Path(image_dir)
        for img_file in image_path.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpeg', '.jpg', '.png']:
                sample_images.append(img_file.stem)
        
        # Limit to NUM_SAMPLE_IMAGES for report
        sample_images = sample_images[:int(num_sample_images)]
        if not sample_images:
            f.write("No sample images processed to include in report.\n")
            return

        for model_name in models:
            f.write(f"#### **{model_name}** ✨\n")
            model_output_dir_name = model_name.replace('/', '_').replace(':', '_')
            model_output_path = Path(base_output_dir) / model_output_dir_name

            if not model_output_path.exists():
                f.write("**Status:** OCR run failed or output directory not found. 😞\n\n")
                continue

            for sample_image_stem in sample_images:
                ocr_output_file = model_output_path / f"{sample_image_stem}.txt"
                f.write(f"##### **이미지: {sample_image_stem}.jpeg/png**\n")
                
                if ocr_output_file.exists():
                    try:
                        with open(ocr_output_file, "r", encoding="utf-8") as ocr_f:
                            extracted_text = ocr_f.read().strip()
                            if extracted_text:
                                f.write("**추출된 텍스트:**\n")
                                f.write("```\n")
                                f.write(extracted_text)
                                f.write("\n```\n")
                            else:
                                f.write("**추출된 텍스트:** (없음)\n")
                        f.write("**평가:** 수동 검토 필요. 이 모델이 이 이미지에서 텍스트를 얼마나 잘 추출했는지 확인해주세요. 🤔\n\n")
                    except Exception as e:
                        f.write(f"**Error reading OCR output for {sample_image_stem}:** {e}\n\n")
                else:
                    f.write("**추출된 텍스트:** (파일 없음)\n")
                    f.write("**평가:** OCR 결과 파일을 찾을 수 없습니다. 모델이 이 이미지를 처리하지 못했거나 오류가 발생했을 수 있습니다. ❌\n\n")
            f.write("---\n\n")
        
        f.write("**종합 요약:**\n")
        f.write("각 모델의 상세 평가는 위에 제시된 개별 이미지 결과와 함께 수동으로 진행되어야 합니다. "
                "전반적인 성능은 추출된 텍스트의 양과 정확성을 바탕으로 판단할 수 있습니다. 🌟\n")
    print(f"Comparison report generated at {report_path}")

def main():
    # Ensure output directory exists
    Path(BASE_OUTPUT_DIRECTORY).mkdir(parents=True, exist_ok=True)
    
    for model in MODELS:
        run_ollama_ocr_integrated(model, IMAGE_DIRECTORY, BASE_OUTPUT_DIRECTORY, NUM_SAMPLE_IMAGES)
    
    generate_comparison_report(MODELS, BASE_OUTPUT_DIRECTORY, COMPARISON_REPORT_PATH, IMAGE_DIRECTORY, NUM_SAMPLE_IMAGES)

if __name__ == "__main__":
    main()