import sys
import os
import importlib

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

def test_imports():
    modules = [
        'data_loader',
        'preprocessing',
        'whisper_finetune',
        'evaluation',
        'error_analysis',
        'number_normalizer',
        'english_detector',
        'spelling_checker',
        'lattice_builder'
    ]
    
    success = []
    failed = []
    
    for mod in modules:
        try:
            importlib.import_module(mod)
            success.append(mod)
            print(f"✓ Successfully imported {mod}")
        except ImportError as e:
            failed.append((mod, str(e)))
            print(f"✗ Failed to import {mod}: {e}")
        except Exception as e:
            failed.append((mod, f"Init error: {e}"))
            print(f"⚠ Error initializing {mod}: {e}")

    if not failed:
        print("\nAll core modules imported successfully!")
    else:
        print(f"\nIssues found in {len(failed)} modules.")
        sys.exit(1)

if __name__ == "__main__":
    test_imports()
