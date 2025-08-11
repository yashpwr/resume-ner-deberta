"""
Tests for the postprocess module.
"""

import pytest
from src.postprocess import create_post_processor, EntityPostProcessor


class TestEntityPostProcessor:
    """Test cases for EntityPostProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance for testing."""
        return create_post_processor()
    
    def test_merge_spans_basic(self, processor):
        """Test basic span merging."""
        tokens = ['John', 'Smith', 'works', 'at', 'Google']
        labels = ['B-NAME', 'I-NAME', 'O', 'O', 'B-COMPANY']
        
        entities = processor.merge_spans(tokens, labels)
        
        assert len(entities) == 2
        assert entities[0]['label'] == 'NAME'
        assert entities[0]['text'] == 'John Smith'
        assert entities[0]['start'] == 0
        assert entities[0]['end'] == 1
        
        assert entities[1]['label'] == 'COMPANY'
        assert entities[1]['text'] == 'Google'
        assert entities[1]['start'] == 4
        assert entities[1]['end'] == 4
    
    def test_merge_spans_single_token(self, processor):
        """Test merging single token entities."""
        tokens = ['John', 'works', 'at', 'Google']
        labels = ['B-NAME', 'O', 'O', 'B-COMPANY']
        
        entities = processor.merge_spans(tokens, labels)
        
        assert len(entities) == 2
        assert entities[0]['text'] == 'John'
        assert entities[1]['text'] == 'Google'
    
    def test_merge_spans_invalid_i_tag(self, processor):
        """Test handling of invalid I- tags without B-."""
        tokens = ['John', 'Smith', 'works']
        labels = ['I-NAME', 'I-NAME', 'O']
        
        entities = processor.merge_spans(tokens, labels)
        
        # Should convert I- to B- when no preceding B-
        assert len(entities) == 1
        assert entities[0]['label'] == 'NAME'
        assert entities[0]['text'] == 'John Smith'
    
    def test_normalize_email_valid(self, processor):
        """Test email normalization with valid email."""
        entity = {'label': 'EMAIL', 'text': 'john@example.com'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['is_valid'] is True
        assert normalized['domain'] == 'example.com'
        assert normalized['username'] == 'john'
    
    def test_normalize_email_invalid(self, processor):
        """Test email normalization with invalid email."""
        entity = {'label': 'EMAIL', 'text': 'invalid-email'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['is_valid'] is False
        assert normalized['domain'] is None
        assert normalized['username'] is None
    
    def test_normalize_phone_valid(self, processor):
        """Test phone normalization with valid phone."""
        entity = {'label': 'PHONE', 'text': '+1-555-123-4567'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['is_valid'] is True
        assert normalized['clean_number'] == '15551234567'
        assert normalized['country_code'] == '+1'
    
    def test_normalize_phone_invalid(self, processor):
        """Test phone normalization with invalid phone."""
        entity = {'label': 'PHONE', 'text': 'not-a-phone'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['is_valid'] is False
        assert normalized['clean_number'] is None
    
    def test_normalize_website_valid(self, processor):
        """Test website normalization with valid URL."""
        entity = {'label': 'WEBSITE', 'text': 'https://github.com/username'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['is_valid'] is True
        assert normalized['domain'] == 'github.com'
        assert normalized['social_type'] == 'GITHUB'
    
    def test_normalize_website_no_protocol(self, processor):
        """Test website normalization without protocol."""
        entity = {'label': 'WEBSITE', 'text': 'linkedin.com/in/username'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['is_valid'] is True
        assert normalized['full_url'] == 'https://linkedin.com/in/username'
        assert normalized['social_type'] == 'LINKEDIN'
    
    def test_normalize_name_full(self, processor):
        """Test name normalization with full name."""
        entity = {'label': 'NAME', 'text': 'John Michael Smith'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['first_name'] == 'John'
        assert normalized['last_name'] == 'Smith'
        assert normalized['middle_names'] == ['Michael']
        assert normalized['formatted_name'] == 'John Michael Smith'
    
    def test_normalize_name_single(self, processor):
        """Test name normalization with single name."""
        entity = {'label': 'NAME', 'text': 'John'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['first_name'] == 'John'
        assert normalized['last_name'] is None
        assert normalized['middle_names'] == []
    
    def test_normalize_company_with_suffix(self, processor):
        """Test company normalization with suffix."""
        entity = {'label': 'COMPANY', 'text': 'Google Inc'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['clean_name'] == 'Google'
        assert normalized['suffix'] == 'Inc'
    
    def test_normalize_title_with_seniority(self, processor):
        """Test title normalization with seniority."""
        entity = {'label': 'TITLE', 'text': 'Senior Software Engineer'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['seniority'] == 'Senior'
        assert normalized['formatted_title'] == 'Senior Software Engineer'
    
    def test_normalize_degree_standard(self, processor):
        """Test degree normalization with standard abbreviation."""
        entity = {'label': 'DEGREE', 'text': 'MS'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['abbreviation'] == 'MS'
        assert normalized['is_standard'] is True
    
    def test_normalize_degree_custom(self, processor):
        """Test degree normalization with custom degree."""
        entity = {'label': 'DEGREE', 'text': 'Master of Computer Science'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['abbreviation'] is None
        assert normalized['is_standard'] is False
    
    def test_normalize_year_valid(self, processor):
        """Test year normalization with valid year."""
        entity = {'label': 'GRAD_YEAR', 'text': '2020'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['is_valid'] is True
        assert normalized['year'] == 2020
        assert normalized['decade'] == 2020
    
    def test_normalize_year_invalid(self, processor):
        """Test year normalization with invalid year."""
        entity = {'label': 'GRAD_YEAR', 'text': 'not-a-year'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['is_valid'] is False
        assert normalized['year'] is None
    
    def test_normalize_skill_programming(self, processor):
        """Test skill normalization with programming language."""
        entity = {'label': 'SKILL', 'text': 'Python'}
        normalized = processor.normalize_entity(entity)
        
        assert normalized['category'] == 'programming_language'
        assert normalized['formatted_name'] == 'Python'
    
    def test_calculate_confidence(self, processor):
        """Test confidence calculation."""
        # Valid entity
        entity = {'is_valid': True, 'text': 'john@example.com'}
        confidence = processor._calculate_confidence(entity)
        assert confidence > 0.8
        
        # Entity with normalization
        entity = {'is_valid': True, 'text': 'john@example.com', 'domain': 'example.com'}
        confidence = processor._calculate_confidence(entity)
        assert confidence > 0.85
        
        # Short entity
        entity = {'is_valid': True, 'text': 'a'}
        confidence = processor._calculate_confidence(entity)
        assert confidence < 0.8
    
    def test_postprocess_predictions_complete(self, processor):
        """Test complete post-processing pipeline."""
        tokens = ['John', 'Smith', 'works', 'at', 'Google', 'contact', 'john@email.com']
        labels = ['B-NAME', 'I-NAME', 'O', 'O', 'B-COMPANY', 'O', 'B-EMAIL']
        
        results = processor.postprocess_predictions(tokens, labels)
        
        assert 'entities' in results
        assert 'entity_counts' in results
        assert 'normalized' in results
        assert 'total_entities' in results
        
        assert results['total_entities'] == 3
        assert results['entity_counts']['NAME'] == 1
        assert results['entity_counts']['COMPANY'] == 1
        assert results['entity_counts']['EMAIL'] == 1
        
        # Check normalized values
        assert 'name' in results['normalized']
        assert 'company' in results['normalized']
        assert 'email' in results['normalized']


class TestFactoryFunction:
    """Test cases for factory function."""
    
    def test_create_post_processor(self):
        """Test processor creation."""
        processor = create_post_processor()
        assert isinstance(processor, EntityPostProcessor)
        assert processor.patterns is not None
        assert processor.domain_mappings is not None


if __name__ == "__main__":
    pytest.main([__file__])
