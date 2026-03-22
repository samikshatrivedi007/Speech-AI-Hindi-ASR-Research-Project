import os
import ast

def scan_syntax(directory):
    errors = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                except SyntaxError as e:
                    errors.append((path, str(e)))
                except Exception as e:
                    errors.append((path, f"Unexpected error: {e}"))
    return errors

if __name__ == "__main__":
    src_dir = 'src'
    syntax_errors = scan_syntax(src_dir)
    if syntax_errors:
        print(f"Found {len(syntax_errors)} syntax errors:")
        for path, err in syntax_errors:
            print(f"- {path}: {err}")
    else:
        print("No syntax errors found in src directory.")
