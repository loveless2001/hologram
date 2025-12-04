#!/usr/bin/env python3
"""
Hologram Setup Script

Interactive CLI tool to initialize and synchronize the global configuration.
Can be run standalone or integrated into the launch script.

Usage:
    python scripts/setup_hologram.py          # Interactive setup
    python scripts/setup_hologram.py --init   # Force initialize
    python scripts/setup_hologram.py --sync   # Auto-sync (non-interactive)
    python scripts/setup_hologram.py --show   # Show current config
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hologram.config import Config
from hologram.global_config import (
    global_project_exists,
    init_global_project,
    load_global_config,
    save_global_config,
    sync_config,
    apply_global_config,
    SyncStatus,
)


def print_header():
    print("=" * 60)
    print("  Hologram Configuration Setup")
    print("=" * 60)
    print()


def print_config(config_dict: dict, title: str = "Current Configuration"):
    print(f"\n{title}:")
    print("-" * 40)
    
    # Group by section
    sections = {}
    for key, value in sorted(config_dict.items()):
        section, field = key.split(".", 1)
        if section not in sections:
            sections[section] = []
        sections[section].append((field, value))
    
    for section, fields in sections.items():
        print(f"\n  [{section}]")
        for field, value in fields:
            print(f"    {field}: {value}")
    print()


def print_diff(differences: dict):
    print("\nConfiguration Differences:")
    print("-" * 40)
    for key, (local, global_val) in sorted(differences.items()):
        print(f"  {key}:")
        print(f"    Local:  {local}")
        print(f"    Global: {global_val}")
    print()


def prompt_choice(question: str, options: list) -> str:
    """Prompt user to choose from options."""
    print(f"\n{question}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    
    while True:
        try:
            choice = input("\nEnter choice (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except (ValueError, IndexError):
            pass
        print("Invalid choice. Try again.")


def interactive_setup():
    """Run interactive setup wizard."""
    print_header()
    
    # Check current status
    status, differences = sync_config()
    
    if status == SyncStatus.GLOBAL_MISSING:
        print("📦 No global configuration found.")
        print(f"   Location: {Config.get_global_db_path()}")
        
        choice = prompt_choice(
            "Would you like to initialize the global configuration?",
            ["Yes, initialize with current settings", "No, skip for now"]
        )
        
        if "Yes" in choice:
            success, msg = init_global_project()
            if success:
                print(f"\n✅ {msg}")
            else:
                print(f"\n❌ Failed: {msg}")
        else:
            print("\n⏭️  Skipped global config initialization.")
    
    elif status == SyncStatus.SYNCED:
        print("✅ Configuration is synchronized!")
        print(f"   Global config: {Config.get_global_db_path()}")
        print_config(Config.to_dict())
    
    elif status == SyncStatus.CONFLICT:
        print("⚠️  Configuration conflict detected!")
        print_diff(differences)
        
        choice = prompt_choice(
            "How would you like to resolve?",
            [
                "Use Local → Overwrite Global config",
                "Use Global → Update Local from Global config",
                "Skip → Do nothing (keep both as-is)"
            ]
        )
        
        if "Local" in choice:
            success, msg = save_global_config()
            print(f"\n{'✅' if success else '❌'} {msg}")
        elif "Global" in choice:
            success, msg = apply_global_config()
            print(f"\n{'✅' if success else '❌'} {msg}")
        else:
            print("\n⏭️  Skipped conflict resolution.")
    
    print("\n" + "=" * 60)
    print("  Setup complete!")
    print("=" * 60)


def auto_sync():
    """Non-interactive sync - applies global config if exists."""
    status, differences = sync_config()
    
    if status == SyncStatus.GLOBAL_MISSING:
        print("[Setup] No global config found, using defaults.")
        success, msg = init_global_project()
        print(f"[Setup] {msg}")
    elif status == SyncStatus.SYNCED:
        print("[Setup] Config synchronized.")
    elif status == SyncStatus.CONFLICT:
        print("[Setup] Config conflict detected. Applying global config...")
        success, msg = apply_global_config()
        print(f"[Setup] {msg}")


def main():
    parser = argparse.ArgumentParser(description="Hologram Configuration Setup")
    parser.add_argument("--init", action="store_true", help="Force initialize global config")
    parser.add_argument("--sync", action="store_true", help="Auto-sync (non-interactive)")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--global", dest="show_global", action="store_true", 
                       help="Show global configuration")
    
    args = parser.parse_args()
    
    if args.init:
        success, msg = init_global_project(force=True)
        print(f"{'✅' if success else '❌'} {msg}")
    elif args.sync:
        auto_sync()
    elif args.show:
        print_config(Config.to_dict(), "Local Configuration")
    elif args.show_global:
        global_cfg = load_global_config()
        if global_cfg:
            print_config(global_cfg, "Global Configuration")
        else:
            print("No global configuration found.")
    else:
        interactive_setup()


if __name__ == "__main__":
    main()
