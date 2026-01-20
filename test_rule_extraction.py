#!/usr/bin/env python3
"""
Test script for the learned rules extraction system.
Run this to verify rule extraction is working correctly.

Usage:
    python test_rule_extraction.py              # Run all tests
    python test_rule_extraction.py --extract    # Test extraction only
    python test_rule_extraction.py --list       # List existing rules
    python test_rule_extraction.py --inject     # Test prompt injection
"""
import asyncio
import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Test cases: (question, response, feedback, expected_actionable)
TEST_CASES = [
    (
        "Wie viele Bomag Maschinen haben wir?",
        "Sie haben 45 Bomag-Maschinen im Bestand. Davon sind 30 Mietmaschinen.",
        "Zeig mir bitte die Seriennummern statt der IDs",
        True,  # Should extract a rule about showing serial numbers
    ),
    (
        "Welche Fertiger sind verfuegbar?",
        "Hier sind 5 verfuegbare Fertiger: ...",
        "Danke!",
        False,  # Just praise, not actionable
    ),
    (
        "Liste alle Mietmaschinen",
        "Hier sind alle 120 Mietmaschinen...",
        "Die Antwort war zu lang, bitte kuerzere Listen",
        True,  # Should extract rule about shorter lists
    ),
    (
        "Zeig mir Details zum Super 1800",
        "Der Super 1800 hat folgende Eigenschaften...",
        "Gut",
        False,  # Just approval, not actionable
    ),
]


async def test_extraction():
    """Test the LLM-based rule extraction."""
    print("\n=== Testing Rule Extraction ===\n")

    try:
        from rag.learned_rules import learned_rules_service
    except ImportError as e:
        print(f"[ERROR] Cannot import learned_rules_service: {e}")
        return False

    all_passed = True
    for i, (question, response, feedback, expected_actionable) in enumerate(TEST_CASES, 1):
        print(f"Test {i}: Feedback = '{feedback[:40]}...'")
        print(f"  Expected actionable: {expected_actionable}")

        try:
            rule = await learned_rules_service.extract_rule_from_feedback(
                question=question,
                response=response,
                feedback=feedback
            )

            if rule is None:
                actual_actionable = False
                print(f"  Result: No rule extracted (None)")
            else:
                actual_actionable = rule.get('is_actionable', False)
                print(f"  Result: is_actionable={actual_actionable}")
                if actual_actionable:
                    print(f"  Rule text: {rule.get('rule_text', 'N/A')}")
                    print(f"  Category: {rule.get('category', 'N/A')}")
                    print(f"  Confidence: {rule.get('confidence_score', 'N/A')}")

            if actual_actionable == expected_actionable:
                print(f"  [PASS]\n")
            else:
                print(f"  [FAIL] Expected {expected_actionable}, got {actual_actionable}\n")
                all_passed = False

        except Exception as e:
            print(f"  [ERROR] Exception during extraction: {e}\n")
            all_passed = False

    return all_passed


def list_rules():
    """List all rules in the database."""
    print("\n=== Existing Learned Rules ===\n")

    try:
        from rag.admin_logger import admin_logger

        if not admin_logger.available:
            print("[ERROR] Admin logger not available - check database configuration")
            return

        rules = admin_logger.get_all_rules(include_inactive=True)

        if not rules:
            print("No rules found in database.")
            print("\nPossible reasons:")
            print("  1. No feedback has been submitted yet")
            print("  2. All feedback was non-actionable")
            print("  3. Database table doesn't exist (run migration)")
            return

        print(f"Found {len(rules)} rules:\n")
        for rule in rules:
            status = "ACTIVE" if rule.get('is_active') else "inactive"
            print(f"#{rule.get('id')} [{status}]")
            print(f"  Rule: {rule.get('rule_text', 'N/A')}")
            print(f"  Category: {rule.get('category', 'N/A')}")
            print(f"  Confidence: {rule.get('confidence_score', 'N/A')}")
            print(f"  Usage count: {rule.get('usage_count', 0)}")
            print(f"  Created: {rule.get('created_at', 'N/A')}")
            if rule.get('source_feedback'):
                print(f"  Source feedback: {rule.get('source_feedback', '')[:60]}...")
            print()

    except Exception as e:
        print(f"[ERROR] Failed to list rules: {e}")


def test_injection():
    """Test the prompt injection system."""
    print("\n=== Testing Prompt Injection ===\n")

    try:
        from rag.learned_rules import learned_rules_service

        rules_section = learned_rules_service.build_rules_prompt_section()

        if not rules_section:
            print("No rules to inject (empty section returned)")
            print("\nThis means either:")
            print("  1. No active rules exist")
            print("  2. Database is not connected")
            return

        print("Rules section that would be injected into system prompt:\n")
        print("-" * 60)
        print(rules_section)
        print("-" * 60)

        # Count rules
        active_rules = learned_rules_service.get_all_active_rules()
        print(f"\nTotal active rules: {len(active_rules)}")

    except Exception as e:
        print(f"[ERROR] Failed to test injection: {e}")


def check_database():
    """Check database connectivity and learned_rules table."""
    print("\n=== Checking Database ===\n")

    try:
        from rag.admin_logger import admin_logger

        print(f"Admin logger available: {admin_logger.available}")

        if not admin_logger.available:
            print("\n[ERROR] Admin logger is not available!")
            print("Check the following environment variables:")
            print("  - ADMIN_POSTGRES_HOST (or POSTGRES_HOST)")
            print("  - ADMIN_POSTGRES_PORT (or POSTGRES_PORT)")
            print("  - ADMIN_POSTGRES_DB")
            print("  - ADMIN_POSTGRES_USER (or POSTGRES_USER)")
            print("  - ADMIN_POSTGRES_PASSWORD")
            return False

        # Try to query the learned_rules table
        try:
            rules = admin_logger.get_active_rules()
            print(f"[OK] learned_rules table accessible, {len(rules)} active rules")
            return True
        except Exception as e:
            print(f"[ERROR] Cannot query learned_rules table: {e}")
            print("\nYou may need to run the migration:")
            print("  cd admin_dashboard && python run_migration.py")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to check database: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Test learned rules extraction system")
    parser.add_argument('--extract', action='store_true', help="Test rule extraction")
    parser.add_argument('--list', action='store_true', help="List existing rules")
    parser.add_argument('--inject', action='store_true', help="Test prompt injection")
    parser.add_argument('--check', action='store_true', help="Check database only")
    args = parser.parse_args()

    # If no specific test requested, run all
    run_all = not (args.extract or args.list or args.inject or args.check)

    print("=" * 60)
    print("Learned Rules Extraction Test")
    print("=" * 60)

    # Always check database first
    db_ok = check_database()

    if args.check:
        return

    if run_all or args.extract:
        if await test_extraction():
            print("\n[OK] All extraction tests passed")
        else:
            print("\n[WARN] Some extraction tests failed")

    if run_all or args.list:
        list_rules()

    if run_all or args.inject:
        test_injection()

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
