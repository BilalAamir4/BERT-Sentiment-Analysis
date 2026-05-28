import sys

def check_setup():
    missing_packages = []
    print("--- Environment Setup Check ---")
    
    # 1 & 2. Torch and CUDA
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"CUDA available — GPU training enabled. GPU: {gpu_name}")
        else:
            print("CUDA not available — training will run on CPU (slow)")
    except ImportError:
        print("PyTorch missing")
        missing_packages.append("torch")

    # 3. Transformers
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers missing")
        missing_packages.append("transformers")

    # 4. Datasets
    try:
        import datasets
        print(f"Datasets version: {datasets.__version__}")
    except ImportError:
        print("Datasets missing")
        missing_packages.append("datasets")

    # 5. Streamlit
    try:
        import streamlit
        print("Streamlit OK")
    except ImportError:
        print("Streamlit missing")
        missing_packages.append("streamlit")
        
    print("-" * 31)
    
    # 6. Summary
    if missing_packages:
        print(f"Setup check failed. Missing packages: {', '.join(missing_packages)}")
    else:
        print("Setup check complete. Ready to train.")

if __name__ == "__main__":
    check_setup()
