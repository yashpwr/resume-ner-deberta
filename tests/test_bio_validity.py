"""
Tests for BIO validity checking and fixing.
"""

import pytest
from src.label_space import LabelSpaceManager


class TestBIOValidity:
    """Test cases for BIO validity checking and fixing."""
    
    @pytest.fixture
    def label_manager(self):
        """Create a label manager instance for testing."""
        return LabelSpaceManager()
    
    def test_fix_bio_validity_i_without_b(self, label_manager):
        """Test fixing I- tag without preceding B- tag."""
        labels = ['O', 'I-NAME', 'I-NAME', 'O', 'B-COMPANY']
        fixed = label_manager.fix_bio_validity(labels)
        
        # First I-NAME should become B-NAME
        assert fixed[1] == 'B-NAME'
        assert fixed[2] == 'I-NAME'  # Second I-NAME is fine now
        
        # COMPANY should remain unchanged
        assert fixed[4] == 'B-COMPANY'
    
    def test_fix_bio_validity_multiple_sequences(self, label_manager):
        """Test fixing multiple invalid I- sequences."""
        labels = ['I-NAME', 'I-NAME', 'O', 'I-COMPANY', 'O', 'B-TITLE', 'I-TITLE']
        fixed = label_manager.fix_bio_validity(labels)
        
        # First I-NAME should become B-NAME
        assert fixed[0] == 'B-NAME'
        assert fixed[1] == 'I-NAME'
        
        # I-COMPANY should become B-COMPANY
        assert fixed[3] == 'B-COMPANY'
        
        # TITLE sequence should remain unchanged
        assert fixed[5] == 'B-TITLE'
        assert fixed[6] == 'I-TITLE'
    
    def test_fix_bio_validity_valid_sequence(self, label_manager):
        """Test that valid BIO sequences remain unchanged."""
        labels = ['B-NAME', 'I-NAME', 'O', 'B-COMPANY', 'O', 'B-TITLE']
        fixed = label_manager.fix_bio_validity(labels)
        
        # Should be identical
        assert fixed == labels
    
    def test_fix_bio_validity_empty_sequence(self, label_manager):
        """Test fixing empty sequence."""
        labels = []
        fixed = label_manager.fix_bio_validity(labels)
        
        assert fixed == []
    
    def test_fix_bio_validity_single_token(self, label_manager):
        """Test fixing single token sequence."""
        labels = ['I-NAME']
        fixed = label_manager.fix_bio_validity(labels)
        
        assert fixed == ['B-NAME']
    
    def test_fix_bio_validity_complex_sequence(self, label_manager):
        """Test fixing complex sequence with multiple issues."""
        labels = [
            'O', 'I-NAME', 'I-NAME', 'O',  # Invalid I-NAME sequence
            'B-COMPANY', 'I-COMPANY', 'O',  # Valid sequence
            'I-TITLE', 'O',                 # Invalid I-TITLE
            'B-SKILL', 'I-SKILL'           # Valid sequence
        ]
        fixed = label_manager.fix_bio_validity(labels)
        
        # Check fixes
        assert fixed[1] == 'B-NAME'   # First I-NAME -> B-NAME
        assert fixed[2] == 'I-NAME'   # Second I-NAME stays
        assert fixed[4] == 'B-COMPANY'  # Unchanged
        assert fixed[5] == 'I-COMPANY'  # Unchanged
        assert fixed[7] == 'B-TITLE'   # I-TITLE -> B-TITLE
        assert fixed[9] == 'B-SKILL'   # Unchanged
        assert fixed[10] == 'I-SKILL'  # Unchanged
    
    def test_fix_bio_validity_mixed_case(self, label_manager):
        """Test fixing mixed case with O tags interrupting sequences."""
        labels = ['I-NAME', 'O', 'I-NAME', 'B-COMPANY', 'I-COMPANY']
        fixed = label_manager.fix_bio_validity(labels)
        
        # First I-NAME should become B-NAME
        assert fixed[0] == 'B-NAME'
        # Second I-NAME should become B-NAME (O tag interrupted)
        assert fixed[2] == 'B-NAME'
        # COMPANY sequence should remain unchanged
        assert fixed[3] == 'B-COMPANY'
        assert fixed[4] == 'I-COMPANY'
    
    def test_fix_bio_validity_all_o_tags(self, label_manager):
        """Test sequence with only O tags."""
        labels = ['O', 'O', 'O', 'O']
        fixed = label_manager.fix_bio_validity(labels)
        
        # Should remain unchanged
        assert fixed == labels
    
    def test_fix_bio_validity_all_b_tags(self, label_manager):
        """Test sequence with only B- tags."""
        labels = ['B-NAME', 'B-COMPANY', 'B-TITLE']
        fixed = label_manager.fix_bio_validity(labels)
        
        # Should remain unchanged
        assert fixed == labels
    
    def test_fix_bio_validity_alternating_entities(self, label_manager):
        """Test alternating entity types."""
        labels = ['B-NAME', 'I-NAME', 'B-COMPANY', 'I-COMPANY', 'B-NAME', 'I-NAME']
        fixed = label_manager.fix_bio_validity(labels)
        
        # Should remain unchanged
        assert fixed == labels


class TestLabelNormalization:
    """Test cases for label normalization."""
    
    @pytest.fixture
    def label_manager(self):
        """Create a label manager instance for testing."""
        return LabelSpaceManager()
    
    def test_normalize_label_no_bio_prefix(self, label_manager):
        """Test normalizing label without BIO prefix."""
        normalized = label_manager.normalize_label('PERSON')
        assert normalized == 'NAME'  # Assuming PERSON -> NAME in synonyms
    
    def test_normalize_label_with_bio_prefix(self, label_manager):
        """Test normalizing label with BIO prefix."""
        normalized = label_manager.normalize_label('B-PERSON')
        assert normalized == 'B-NAME'
        
        normalized = label_manager.normalize_label('I-PERSON')
        assert normalized == 'I-NAME'
    
    def test_normalize_label_unknown_label(self, label_manager):
        """Test normalizing unknown label."""
        normalized = label_manager.normalize_label('UNKNOWN_LABEL')
        assert normalized == 'UNKNOWN_LABEL'  # Should remain unchanged
    
    def test_normalize_label_empty_string(self, label_manager):
        """Test normalizing empty string."""
        normalized = label_manager.normalize_label('')
        assert normalized == ''
    
    def test_normalize_label_none(self, label_manager):
        """Test normalizing None value."""
        with pytest.raises(AttributeError):
            label_manager.normalize_label(None)


class TestBIOConversion:
    """Test cases for converting labels to BIO format."""
    
    @pytest.fixture
    def label_manager(self):
        """Create a label manager instance for testing."""
        return LabelSpaceManager()
    
    def test_convert_to_bio_simple(self, label_manager):
        """Test simple BIO conversion."""
        labels = ['NAME', 'O', 'COMPANY', 'O', 'TITLE']
        bio_labels = label_manager.convert_to_bio(labels)
        
        expected = ['B-NAME', 'O', 'B-COMPANY', 'O', 'B-TITLE']
        assert bio_labels == expected
    
    def test_convert_to_bio_consecutive_entities(self, label_manager):
        """Test BIO conversion with consecutive entities."""
        labels = ['NAME', 'NAME', 'O', 'COMPANY', 'COMPANY', 'COMPANY']
        bio_labels = label_manager.convert_to_bio(labels)
        
        expected = ['B-NAME', 'I-NAME', 'O', 'B-COMPANY', 'I-COMPANY', 'I-COMPANY']
        assert bio_labels == expected
    
    def test_convert_to_bio_all_o(self, label_manager):
        """Test BIO conversion with all O labels."""
        labels = ['O', 'O', 'O', 'O']
        bio_labels = label_manager.convert_to_bio(labels)
        
        assert bio_labels == labels
    
    def test_convert_to_bio_single_entity(self, label_manager):
        """Test BIO conversion with single entity."""
        labels = ['NAME']
        bio_labels = label_manager.convert_to_bio(labels)
        
        assert bio_labels == ['B-NAME']
    
    def test_convert_to_bio_empty_sequence(self, label_manager):
        """Test BIO conversion with empty sequence."""
        labels = []
        bio_labels = label_manager.convert_to_bio(labels)
        
        assert bio_labels == []


if __name__ == "__main__":
    pytest.main([__file__])
