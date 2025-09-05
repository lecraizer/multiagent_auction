import os
import shutil
import argparse

RESULTS_DIR = 'results'
MODELS_DIR = 'models'

def clear_folder(path: str, label: str = None) -> None:
    """
    Delete all contents of the specified folder.

    Args:
        path (str): Path to the folder to clear.
        label (str, optional): Optional label to display after clearing.
    """
    if os.path.exists(path):
        os.system(f'rm -rf {path}/*')
        if label:
            print(f"✔ Cleared: {label}")

def clear_results() -> None:
    """
    Checks the 'results/' directory and deletes contents of all subdirectories
    except those used for GIFs or temporary files.
    """
    if os.path.exists(RESULTS_DIR):
        for subdir in os.listdir(RESULTS_DIR):
            full_path = os.path.join(RESULTS_DIR, subdir)
            if os.path.isdir(full_path) and subdir not in ['gifs', '.tmp']:
                clear_folder(full_path, f'results/{subdir}')
    else:
        print("⚠ No 'results/' directory found.")

def clear_models() -> None:
    """
    Deletes contents of the 'models/actor' and 'models/critic' directories.
    """
    clear_folder(os.path.join(MODELS_DIR, 'actor'), 'models/actor')
    clear_folder(os.path.join(MODELS_DIR, 'critic'), 'models/critic')

def clear_gifs() -> None:
    """
    Clear all saved GIFs used for training visualization.
    """
    clear_folder(os.path.join(RESULTS_DIR, 'gifs'), 'results/gifs')

def clear_tmp() -> None:
    """
    Clear all temporary PNG files used for GIF creation.
    """
    clear_folder(os.path.join(RESULTS_DIR, '.tmp'), 'results/.tmp')

def clear_pycache() -> None:
    """
    Recursively delete Python cache files: __pycache__, .pyc, and .pyo files.
    """
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                shutil.rmtree(os.path.join(root, dir_name))
        for file in files:
            if file.endswith('.pyc') or file.endswith('.pyo'):
                try:
                    os.remove(os.path.join(root, file))
                except:
                    pass
    print("✔ Cleared: Python cache files (__pycache__, .pyc, .pyo)")

def main() -> None:
    """
    Command-line interface for clearing saved data and generated files.
    Supports clearing specific folders or all related data, including:
    - model checkpoints
    - auction results
    - GIFs
    - temporary images
    - Python cache files

    Command-line Arguments:
        --force         (bool): Clear all data without confirmation.
        --models-only   (bool): Clear only models (actor/critic).
        --results-only  (bool): Clear only auction result folders.
        --gifs-only     (bool): Clear only GIFs folder.
        --tmp-only      (bool): Clear only temporary PNGs.
        --cache-only    (bool): Clear only Python cache files.
    """
    parser = argparse.ArgumentParser(description=
                                     'Clear saved data and generated files for auction experiments.')
    parser.add_argument('--force', action='store_true', 
                        help='Clear everything without confirmation prompt')
    parser.add_argument('--models-only', action='store_true', 
                        help='Clear only models (actor/critic)')
    parser.add_argument('--results-only', action='store_true', 
                        help='Clear only auction result folders')
    parser.add_argument('--gifs-only', action='store_true', 
                        help='Clear only GIFs folder')
    parser.add_argument('--tmp-only', action='store_true', 
                        help='Clear only temporary PNGs for gif creation')
    parser.add_argument('--cache-only', action='store_true', 
                        help='Clear only Python __pycache__ and .pyc/.pyo files')

    args = parser.parse_args()

    clear_all = args.force or not any([
        args.models_only, args.results_only, args.gifs_only, args.tmp_only, args.cache_only
    ])

    if not args.force and clear_all:
        confirm = input("Are you sure you want to clear ALL data? (y/n): ").strip().lower()
        if confirm and confirm not in ['y', 'yes', '']:
            print("❌ Operation cancelled.")
            return

    if args.models_only or clear_all: clear_models()
    if args.results_only or clear_all: clear_results()
    if args.gifs_only or clear_all: clear_gifs()
    if args.tmp_only or clear_all: clear_tmp()
    if args.cache_only or clear_all: clear_pycache()

    print("\n✅ Done.")

if __name__ == "__main__":
    main()