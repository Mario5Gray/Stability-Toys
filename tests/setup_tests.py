#!/usr/bin/env python3
"""
Automated setup script for Yume test suite.
Run this from your project root to set up the proper directory structure.
"""

import os
import shutil
import sys
from pathlib import Path


def create_directory_structure():
    """Create the proper directory structure for Yume."""
    
    print("🚀 Setting up Yume test directory structure...\n")
    
    # Get project root (where this script is)
    project_root = Path.cwd()
    
    # Create main directories
    yume_dir = project_root / "yume"
    tests_dir = project_root / "tests"
    
    print(f"Project root: {project_root}")
    print(f"Creating directories...\n")
    
    # Create yume/ directory if it doesn't exist
    if not yume_dir.exists():
        yume_dir.mkdir()
        print(f"✓ Created {yume_dir}")
    else:
        print(f"✓ {yume_dir} already exists")
    
    # Create tests/ directory
    if not tests_dir.exists():
        tests_dir.mkdir()
        print(f"✓ Created {tests_dir}")
    else:
        print(f"✓ {tests_dir} already exists")
    
    # Create __init__.py files
    yume_init = yume_dir / "__init__.py"
    tests_init = tests_dir / "__init__.py"
    
    if not yume_init.exists():
        yume_init.write_text('"""\nYume - Latent Space Exploration Library\n"""\n\n__version__ = "0.1.0"\n')
        print(f"✓ Created {yume_init}")
    else:
        print(f"✓ {yume_init} already exists")
    
    if not tests_init.exists():
        tests_init.write_text('"""\nTest suite for Yume library.\n"""\n')
        print(f"✓ Created {tests_init}")
    else:
        print(f"✓ {tests_init} already exists")
    
    return project_root, yume_dir, tests_dir


def move_test_files(project_root, tests_dir):
    """Move test files to tests/ directory."""
    
    print("\n📦 Moving test files...\n")
    
    test_files = [
        "test_dream_worker.py",
        "test_scoring.py",
        "test_integration.py",
        "conftest.py",
        "pytest.ini",
    ]
    
    for filename in test_files:
        src = project_root / filename
        dst = tests_dir / filename
        
        if src.exists():
            if dst.exists():
                print(f"⚠ {filename} already exists in tests/, skipping")
            else:
                shutil.copy2(src, dst)
                print(f"✓ Moved {filename} to tests/")
        else:
            print(f"⚠ {filename} not found in project root")


def create_setup_files(project_root):
    """Create setup.py and requirements files."""
    
    print("\n📝 Creating setup files...\n")
    
    setup_py = project_root / "setup.py"
    if not setup_py.exists():
        print(f"⚠ setup.py not found - you may need to create it manually")
        print(f"  See SETUP_GUIDE.md for an example")
    else:
        print(f"✓ setup.py already exists")
    
    req_test = project_root / "requirements-test.txt"
    if req_test.exists():
        print(f"✓ requirements-test.txt found")
    else:
        print(f"⚠ requirements-test.txt not found")


def verify_structure(project_root, yume_dir, tests_dir):
    """Verify the directory structure is correct."""
    
    print("\n🔍 Verifying structure...\n")
    
    checks = [
        (yume_dir.exists(), f"yume/ directory exists"),
        (tests_dir.exists(), f"tests/ directory exists"),
        ((yume_dir / "__init__.py").exists(), f"yume/__init__.py exists"),
        ((tests_dir / "__init__.py").exists(), f"tests/__init__.py exists"),
        ((tests_dir / "conftest.py").exists(), f"tests/conftest.py exists"),
        ((tests_dir / "pytest.ini").exists(), f"tests/pytest.ini exists"),
    ]
    
    all_good = True
    for check, desc in checks:
        if check:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc}")
            all_good = False
    
    return all_good


def print_next_steps(all_good):
    """Print next steps for the user."""
    
    print("\n" + "="*60)
    
    if all_good:
        print("✅ Setup complete!\n")
        print("Next steps:")
        print("1. Move your source files to yume/ directory")
        print("   - dream_worker.py")
        print("   - scoring.py")
        print("   - backends/")
        print("")
        print("2. Install the package in development mode:")
        print("   pip install -e .")
        print("")
        print("3. Install test dependencies:")
        print("   pip install -r requirements-test.txt")
        print("")
        print("4. Run tests:")
        print("   pytest tests/")
        print("   # or")
        print("   python run_tests.py all")
        print("")
        print("See SETUP_GUIDE.md for detailed instructions!")
    else:
        print("⚠ Setup incomplete\n")
        print("Some files are missing. Please check the output above.")
        print("See SETUP_GUIDE.md for manual setup instructions.")
    
    print("="*60)


def main():
    """Main setup function."""
    
    print("\n" + "="*60)
    print("  Yume Test Suite Setup")
    print("="*60 + "\n")
    
    # Check if we're in the right place
    if not Path("run_tests.py").exists() and not Path("test_dream_worker.py").exists():
        print("⚠ Warning: Test files not found in current directory")
        print("Make sure you're running this from your project root")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 1
    
    try:
        # Create directory structure
        project_root, yume_dir, tests_dir = create_directory_structure()
        
        # Move test files
        move_test_files(project_root, tests_dir)
        
        # Check setup files
        create_setup_files(project_root)
        
        # Verify
        all_good = verify_structure(project_root, yume_dir, tests_dir)
        
        # Print next steps
        print_next_steps(all_good)
        
        return 0 if all_good else 1
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("See SETUP_GUIDE.md for manual setup instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
