#!/usr/bin/env python3
"""
Backfill script to extract rules from existing feedback.
Run this to retroactively process feedback that was submitted before
the rule extraction feature was enabled.

Usage:
    python backfill_rules.py           # Dry run - show what would be extracted
    python backfill_rules.py --save    # Actually save the rules
"""
import asyncio
import argparse
import os
from dotenv import load_dotenv

load_dotenv()


async def backfill_rules(save: bool = False):
    """Extract rules from existing feedback entries."""
    print("=" * 60)
    print("Backfill Rules from Existing Feedback")
    print("=" * 60)
    print(f"Mode: {'SAVE' if save else 'DRY RUN (use --save to persist)'}")
    print()

    try:
        from rag.admin_logger import admin_logger
        from rag.learned_rules import learned_rules_service
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return

    if not admin_logger.available:
        print("[ERROR] Admin logger not available - check database config")
        return

    # Get all feedback entries
    print("Fetching feedback entries...")
    feedback_entries = admin_logger.get_all_feedback(limit=100)

    if not feedback_entries:
        print("No feedback entries found.")
        return

    print(f"Found {len(feedback_entries)} feedback entries\n")

    extracted = 0
    skipped = 0
    errors = 0

    for entry in feedback_entries:
        feedback_text = entry.get('feedback', '')
        question = entry.get('user_question', '')
        response = entry.get('assistant_response', '')
        feedback_at = entry.get('feedback_at', 'Unknown')

        print(f"Processing feedback from {feedback_at}:")
        print(f"  Feedback: {feedback_text[:60]}...")
        print(f"  Question: {(question or 'N/A')[:50]}...")

        if not question or not response:
            print(f"  [SKIP] Missing question or response context\n")
            skipped += 1
            continue

        try:
            rule = await learned_rules_service.extract_rule_from_feedback(
                question=question,
                response=response,
                feedback=feedback_text
            )

            if not rule:
                print(f"  [SKIP] No rule extracted (None returned)\n")
                skipped += 1
                continue

            if not rule.get('is_actionable'):
                print(f"  [SKIP] Not actionable: {rule.get('rule_text', 'N/A')[:40]}...\n")
                skipped += 1
                continue

            print(f"  [EXTRACTED] {rule.get('rule_text', 'N/A')}")
            print(f"  Category: {rule.get('category', 'N/A')}, Confidence: {rule.get('confidence_score', 'N/A')}")

            if save:
                # Check for duplicates
                if learned_rules_service.check_duplicate(rule.get('rule_text', '')):
                    print(f"  [SKIP] Duplicate rule exists\n")
                    skipped += 1
                    continue

                saved = learned_rules_service.save_rule(rule)
                if saved:
                    print(f"  [SAVED]\n")
                    extracted += 1
                else:
                    print(f"  [ERROR] Failed to save\n")
                    errors += 1
            else:
                print(f"  [DRY RUN] Would save this rule\n")
                extracted += 1

        except Exception as e:
            print(f"  [ERROR] {e}\n")
            errors += 1

    print("=" * 60)
    print(f"Summary:")
    print(f"  Total feedback entries: {len(feedback_entries)}")
    print(f"  Rules extracted: {extracted}")
    print(f"  Skipped (not actionable/duplicate/missing context): {skipped}")
    print(f"  Errors: {errors}")

    if not save and extracted > 0:
        print(f"\nRun with --save to persist the {extracted} extracted rules.")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Backfill rules from existing feedback")
    parser.add_argument('--save', action='store_true', help="Actually save the rules (default is dry run)")
    args = parser.parse_args()

    asyncio.run(backfill_rules(save=args.save))


if __name__ == "__main__":
    main()
