"""Tests for ColumnCatalog semantic column resolution."""
import pytest
from rag.column_catalog import ColumnCatalog, column_catalog


class TestColumnCatalog:
    def test_initialization(self):
        """Test catalog initializes and loads columns."""
        catalog = ColumnCatalog()
        catalog.initialize()
        columns = catalog.get_all_columns()
        assert len(columns) > 100  # Should have 170+ columns

    def test_get_prompt_section(self):
        """Test prompt section is generated."""
        catalog = ColumnCatalog()
        catalog.initialize()
        prompt = catalog.get_prompt_section()
        assert "PROPERTY COLUMNS CATALOG" in prompt
        # Check for some known columns
        assert "prop_e" in prompt.lower()

    def test_search_columns_grabtiefe(self):
        """Test searching for Grabtiefe column."""
        catalog = ColumnCatalog()
        catalog.initialize()
        results = catalog.search_columns("grabtiefe")
        assert len(results) >= 1
        assert any("grabtiefe" in r.display_name.lower() for r in results)

    def test_search_columns_breite(self):
        """Test searching for Breite column."""
        catalog = ColumnCatalog()
        catalog.initialize()
        results = catalog.search_columns("breite")
        assert len(results) >= 1

    def test_search_columns_klimaanlage(self):
        """Test searching for Klimaanlage column."""
        catalog = ColumnCatalog()
        catalog.initialize()
        results = catalog.search_columns("klimaanlage")
        assert len(results) >= 1
        # Should be categorized as boolean
        assert any(r.category == "boolean" for r in results)

    def test_column_categorization_dimension(self):
        """Test dimension columns have units."""
        catalog = ColumnCatalog()
        catalog.initialize()

        # Find a column with mm unit
        breite = catalog.search_columns("breite")
        if breite:
            mm_cols = [c for c in breite if c.unit == "mm"]
            if mm_cols:
                assert mm_cols[0].category == "dimension"

    def test_column_categorization_boolean(self):
        """Test boolean columns are categorized correctly."""
        catalog = ColumnCatalog()
        catalog.initialize()

        # Find klimaanlage column
        klima = catalog.search_columns("klimaanlage")
        if klima:
            assert klima[0].category == "boolean"

    def test_global_singleton(self):
        """Test global singleton works."""
        column_catalog.initialize()
        assert column_catalog.get_prompt_section() != ""
        assert len(column_catalog.get_all_columns()) > 100

    def test_search_columns_gewicht(self):
        """Test searching for Gewicht (weight) column."""
        catalog = ColumnCatalog()
        catalog.initialize()
        results = catalog.search_columns("gewicht")
        assert len(results) >= 1
        # Should have kg unit
        assert any(r.unit == "kg" for r in results)

    def test_search_columns_leistung(self):
        """Test searching for Leistung (power) column."""
        catalog = ColumnCatalog()
        catalog.initialize()
        results = catalog.search_columns("leistung")
        assert len(results) >= 1
        # Should have kW unit
        assert any(r.unit == "kW" for r in results)

    def test_slugify(self):
        """Test slugify produces correct column name suffixes."""
        catalog = ColumnCatalog()

        # Test with unit
        slug = catalog._slugify("Breite [mm]")
        assert "mm" not in slug  # Unit should be removed
        assert "breite" in slug

        # Test with umlauts
        slug = catalog._slugify("Höhe [m]")
        assert "hohe" in slug or "hoehe" in slug  # Umlaut normalized

    def test_prompt_contains_sql_patterns(self):
        """Test prompt includes SQL usage patterns."""
        catalog = ColumnCatalog()
        catalog.initialize()
        prompt = catalog.get_prompt_section()

        # Should include SQL pattern hints
        assert "regexp_replace" in prompt
        assert "CAST" in prompt
        assert "NUMERIC" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
