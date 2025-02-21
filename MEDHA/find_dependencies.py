import os
import ast
import pkg_resources

def find_imports(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            tree = ast.parse(file.read())
            
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
                    
        # Filter out standard library modules
        std_libs = {pkg.key for pkg in pkg_resources.working_set}
        dependencies = {imp for imp in imports if imp.lower() in std_libs}
        return dependencies
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return set()

def analyze_project(directory):
    dependencies = set()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                dependencies.update(find_imports(file_path))
    return dependencies

# Usage
project_dir = '.'  # Current directory, change this to your project path
dependencies = analyze_project(project_dir)
print("Found dependencies:", dependencies)