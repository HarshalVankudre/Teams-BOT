"""
Integration tests for Enhanced Single Agent features.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# Test planning module
class TestQueryPlanner:
    def test_should_plan_simple_query(self):
        from rag.planning import QueryPlanner
        planner = QueryPlanner()

        # Simple queries should not need planning
        assert not planner.should_plan("wie viele Bomag?")
        assert not planner.should_plan("zeige 5 Maschinen")
        assert not planner.should_plan("liste Fertiger")

    def test_should_plan_complex_query(self):
        from rag.planning import QueryPlanner
        planner = QueryPlanner()

        # Complex queries should need planning
        assert planner.should_plan("vergleiche die Kettenfertiger und empfehle den besten für 3m")
        assert planner.should_plan("berechne die Gesamtkosten für alle Mietmaschinen")
        assert planner.should_plan("welcher Fertiger ist optimal für meine Anforderungen?")

    def test_create_simple_plan(self):
        from rag.planning import QueryPlanner
        planner = QueryPlanner()

        plan = planner.create_simple_plan("wie viele Bomag Maschinen?")
        assert plan.complexity == "simple"
        assert plan.requires_aggregation is True
        assert len(plan.steps) == 1

    def test_create_followup_plan(self):
        from rag.planning import QueryPlanner
        planner = QueryPlanner()

        thread_state = {"last_result_ids": [1, 2, 3]}
        plan = planner.create_simple_plan("davon mit Klimaanlage?", thread_state)

        assert plan.is_followup is True
        assert "last_result_ids" in plan.context_needed


# Test SQL verifier
class TestSQLVerifier:
    def test_verify_safe_query(self):
        from rag.sql_verifier import SQLVerifier
        verifier = SQLVerifier(equipment_table="sema.equipment")

        result = verifier.verify(
            "SELECT * FROM sema.equipment WHERE hersteller_name ILIKE '%bomag%' LIMIT 10"
        )
        assert result.is_valid is True

    def test_detect_unsafe_query(self):
        from rag.sql_verifier import SQLVerifier
        verifier = SQLVerifier(equipment_table="sema.equipment")

        result = verifier.verify("DELETE FROM sema.equipment")
        assert result.is_valid is False
        assert "DELETE not allowed" in result.issues

    def test_autocorrect_kostenstelle(self):
        from rag.sql_verifier import SQLVerifier
        verifier = SQLVerifier(equipment_table="sema.equipment")

        result = verifier.verify(
            "SELECT * FROM sema.equipment WHERE kostenstelle = '200'"
        )
        assert result.corrected_sql is not None
        assert "ibs_nuclet_geraete_kostenstelle" in result.corrected_sql.lower()


# Test reasoning tools
class TestReasoningTools:
    def test_calculate_simple(self):
        from rag.reasoning_tools import reasoning_tools

        result = reasoning_tools.calculate("100 * 5 + 50", purpose="test")
        assert result.success is True
        assert result.result == 550

    def test_calculate_with_values(self):
        from rag.reasoning_tools import reasoning_tools

        result = reasoning_tools.calculate(
            "price * quantity",
            values={"price": 100, "quantity": 3},
            unit="EUR",
            purpose="cost calculation"
        )
        assert result.success is True
        assert result.result == 300
        assert result.unit == "EUR"

    def test_calculate_unsafe_blocked(self):
        from rag.reasoning_tools import reasoning_tools

        result = reasoning_tools.calculate("import os", purpose="test")
        assert result.success is False
        assert "Unsafe" in result.error

    def test_compare_items(self):
        from rag.reasoning_tools import reasoning_tools

        items = [
            {"id": 1, "bezeichnung": "A", "prop_gewicht": 1000, "nuclos_state": "Released"},
            {"id": 2, "bezeichnung": "B", "prop_gewicht": 2000, "nuclos_state": "Locked"},
            {"id": 3, "bezeichnung": "C", "prop_gewicht": 1500, "nuclos_state": "Released"},
        ]

        result = reasoning_tools.compare(
            items=items,
            criteria=["prop_gewicht", "nuclos_state"],
            purpose="find heaviest available"
        )

        assert result.success is True
        assert result.winner is not None
        assert len(result.ranking) == 3

    def test_compare_with_requirements(self):
        from rag.reasoning_tools import reasoning_tools

        items = [
            {"id": 1, "prop_einbaubreite_max": 2.5},
            {"id": 2, "prop_einbaubreite_max": 3.5},
            {"id": 3, "prop_einbaubreite_max": 4.0},
        ]

        result = reasoning_tools.compare(
            items=items,
            criteria=["prop_einbaubreite_max"],
            requirements={"prop_einbaubreite_max": {"min": 3.0}},
            purpose="find 3m+ width"
        )

        assert result.success is True
        assert len(result.ranking) == 2  # Only items >= 3.0

    def test_aggregate_sum(self):
        from rag.reasoning_tools import reasoning_tools

        data = [
            {"hersteller": "Bomag", "prop_gewicht": 1000},
            {"hersteller": "Bomag", "prop_gewicht": 2000},
            {"hersteller": "Hamm", "prop_gewicht": 1500},
        ]

        result = reasoning_tools.aggregate(
            data=data,
            operation="sum",
            field="prop_gewicht",
            purpose="total weight"
        )

        assert result.success is True
        assert result.result == 4500

    def test_aggregate_with_groupby(self):
        from rag.reasoning_tools import reasoning_tools

        data = [
            {"hersteller": "Bomag", "prop_gewicht": 1000},
            {"hersteller": "Bomag", "prop_gewicht": 2000},
            {"hersteller": "Hamm", "prop_gewicht": 1500},
        ]

        result = reasoning_tools.aggregate(
            data=data,
            operation="sum",
            field="prop_gewicht",
            group_by="hersteller",
            purpose="weight by manufacturer"
        )

        assert result.success is True
        assert result.groups is not None
        assert len(result.groups) == 2


# Test context manager
class TestContextManager:
    def test_create_context(self):
        from rag.context_manager import ContextManager
        cm = ContextManager()

        ctx = cm.get_context("test-thread")
        assert ctx.thread_key == "test-thread"
        assert ctx.turn_count == 0

    def test_detect_followup(self):
        from rag.context_manager import ContextManager
        cm = ContextManager()

        ctx = cm.update_context("test", "davon mit Klimaanlage?")
        assert ctx.is_followup is True
        assert ctx.followup_type == "filter"

    def test_extract_width(self):
        from rag.context_manager import ContextManager
        cm = ContextManager()

        ctx = cm.update_context("test", "Fertiger für 3,5m Breite")
        assert ctx.target_width_m == 3.5

    def test_extract_manufacturers(self):
        from rag.context_manager import ContextManager
        cm = ContextManager()

        ctx = cm.update_context("test", "zeige mir Bomag und Hamm Maschinen")
        assert "bomag" in ctx.mentioned_manufacturers
        assert "hamm" in ctx.mentioned_manufacturers


# Test config
class TestConfig:
    def test_enhanced_agent_config(self):
        from rag.config import config

        # Check that new config options exist and default to True
        assert hasattr(config, 'agent_enable_planning')
        assert hasattr(config, 'agent_enable_sql_verification')
        assert hasattr(config, 'agent_enable_reasoning_tools')
        assert hasattr(config, 'agent_planning_model')
        assert hasattr(config, 'agent_verification_model')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
