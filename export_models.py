from src.models import build_model
from src.export_onnx import export_to_onnx 

def main():
    print("Starting ONNX model export ")
    print("=" * 80)

    models_to_export = ["SmallCNN", "ResNet18", "MobileNetV2"]

    for name in models_to_export:
        print("-"*80)
        # build raw pytorch model
        model = build_model(name)

        # run export function
        export_to_onnx(model, model_name=name)

    print("="*80)
    print("All models exported successfully.")

if __name__ == "__main__":
    main()