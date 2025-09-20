import os
import shutil

# --- Configuration ---
# Directories to clean
DIRECTORIES_TO_CLEAN = ['intent_classifiers', 'multi_agent_chroma_db']
# The keyword for files/folders that should NOT be deleted
KEEP_KEYWORD = 'gem5'

def clean_directory(dir_path):
    """
    Iterates through a directory and deletes all files and sub-directories
    that do not contain the KEEP_KEYWORD in their name.
    """
    if not os.path.exists(dir_path):
        print(f"Directory not found: '{dir_path}'. Skipping.")
        return

    print(f"\n--- Scanning directory: '{dir_path}' ---")
    
    items_to_delete = []
    
    # First, identify all items to be deleted
    for item_name in os.listdir(dir_path):
        if KEEP_KEYWORD not in item_name:
            items_to_delete.append(item_name)
        else:
            print(f"  ✅ Keeping: {item_name}")

    # Now, delete the identified items
    if not items_to_delete:
        print("  No items to clean in this directory.")
        return

    print("\n  The following items will be deleted:")
    for item_name in items_to_delete:
        print(f"    - {item_name}")

    # Perform the actual deletion
    for item_name in items_to_delete:
        full_path = os.path.join(dir_path, item_name)
        try:
            if os.path.isfile(full_path):
                os.remove(full_path)
                print(f"  🗑️ Deleted file: {item_name}")
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
                print(f"  🗑️ Deleted folder: {item_name}")
        except Exception as e:
            print(f"  ❌ Error deleting {item_name}: {e}")

def main():
    """
    Main function to run the cleanup process.
    """
    print("*" * 50)
    print("DEPLOYMENT CLEANUP SCRIPT")
    print("*" * 50)
    print("\nThis script will permanently delete all non-'gem5' agents")
    print("from the following directories:")
    for d in DIRECTORIES_TO_CLEAN:
        print(f"  - {d}")
    print("\nThis action cannot be undone.")
    
    confirmation = input("Type 'YES' to confirm and proceed: ")

    if confirmation.strip().upper() == 'YES':
        print("\nStarting cleanup...")
        for directory in DIRECTORIES_TO_CLEAN:
            clean_directory(directory)
        print("\n✨ Cleanup complete! Your project is ready for deployment.")
    else:
        print("\nOperation cancelled.")

if __name__ == "__main__":
    main()