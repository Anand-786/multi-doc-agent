# src/ast_parser_test.py
import clang.cindex

# A simple C++ code snippet to parse
CPP_CODE = """
int add(int a, int b) {
    if (a > 0) {
        return a + b;
    }
    return 0;
}
"""

def traverse_ast(node, level=0):
    """
    A recursive function to traverse the AST and print node details.
    """
    indent = "  " * level
    # Print the kind of node (e.g., FUNCTION_DECL) and its name (spelling)
    print(f"{indent}{node.kind.name}: {node.spelling}")

    # Recurse for all children of this node
    for child in node.get_children():
        traverse_ast(child, level + 1)

def main():
    # Set the path to the libclang library if needed (usually not required on Linux)
    # clang.cindex.Config.set_library_file('/usr/lib/x86_64-linux-gnu/libclang-14.so.1')

    # Create an index and parse the code
    index = clang.cindex.Index.create()
    # We use 'x.cpp' as a dummy filename
    tu = index.parse('x.cpp', args=['-std=c++11'], unsaved_files=[('x.cpp', CPP_CODE)])

    print("--- Abstract Syntax Tree ---")
    traverse_ast(tu.cursor)
    print("--------------------------")

if __name__ == "__main__":
    main()