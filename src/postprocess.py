"""
Post-processing for resume NER predictions.
Handles span merging, regex normalization, and entity validation.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Optional
from urllib.parse import urlparse
import unicodedata
import json

logger = logging.getLogger(__name__)


class EntityPostProcessor:
    """Post-processes NER predictions for resume entities."""
    
    def __init__(self):
        # Regex patterns for entity validation and normalization
        self.patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'phone': re.compile(r'^[\+]?[1-9][\d]{0,15}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'linkedin': re.compile(r'linkedin\.com', re.IGNORECASE),
            'github': re.compile(r'github\.com', re.IGNORECASE),
            'year': re.compile(r'^(19|20)\d{2}$'),
            'degree': re.compile(r'\b(BS|BA|MS|MA|PhD|MBA|BSc|MSc|BEng|MEng)\b', re.IGNORECASE)
        }
        
        # Domain mappings for social profiles
        self.domain_mappings = {
            'linkedin.com': 'LINKEDIN',
            'github.com': 'GITHUB',
            'twitter.com': 'TWITTER',
            'facebook.com': 'FACEBOOK'
        }
        
        logger.info("Entity post-processor initialized")
    
    def merge_spans(self, tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
        """
        Merge consecutive BIO spans into entities.
        
        Args:
            tokens: List of word tokens
            labels: List of BIO labels
            
        Returns:
            List of merged entities
        """
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'label': entity_type,
                    'text': token,
                    'start': i,
                    'end': i,
                    'tokens': [token]
                }
                
            elif label.startswith('I-'):
                # Continuation of entity
                entity_type = label[2:]
                if current_entity and current_entity['label'] == entity_type:
                    current_entity['text'] += ' ' + token
                    current_entity['end'] = i
                    current_entity['tokens'].append(token)
                else:
                    # Invalid I- without B-, convert to B-
                    if current_entity:
                        entities.append(current_entity)
                    
                    current_entity = {
                        'label': entity_type,
                        'text': token,
                        'start': i,
                        'end': i,
                        'tokens': [token]
                    }
            
            else:  # O label
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def normalize_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize entity text and validate entity type.
        
        Args:
            entity: Entity dictionary
            
        Returns:
            Normalized entity
        """
        normalized = entity.copy()
        text = entity['text'].strip()
        label = entity['label']
        
        # Clean text
        text = self._clean_text(text)
        normalized['text'] = text
        normalized['normalized_text'] = text
        
        # Apply entity-specific normalization
        if label == 'EMAIL':
            normalized.update(self._normalize_email(text))
        elif label == 'PHONE':
            normalized.update(self._normalize_phone(text))
        elif label == 'WEBSITE':
            normalized.update(self._normalize_website(text))
        elif label == 'NAME':
            normalized.update(self._normalize_name(text))
        elif label == 'COMPANY':
            normalized.update(self._normalize_company(text))
        elif label == 'TITLE':
            normalized.update(self._normalize_title(text))
        elif label == 'UNIVERSITY':
            normalized.update(self._normalize_university(text))
        elif label == 'DEGREE':
            normalized.update(self._normalize_degree(text))
        elif label == 'GRAD_YEAR':
            normalized.update(self._normalize_year(text))
        elif label == 'SKILL':
            normalized.update(self._normalize_skill(text))
        elif label == 'ACHIEVEMENT':
            normalized.update(self._normalize_achievement(text))
        elif label == 'AWARD':
            normalized.update(self._normalize_award(text))
        elif label == 'CERTIFICATION':
            normalized.update(self._normalize_certification(text))
        elif label == 'PROJECT':
            normalized.update(self._normalize_project(text))
        elif label == 'RESEARCH':
            normalized.update(self._normalize_research(text))
        elif label == 'PUBLICATION':
            normalized.update(self._normalize_publication(text))
        elif label == 'LANGUAGE':
            normalized.update(self._normalize_language(text))
        elif label == 'TECHNOLOGY':
            normalized.update(self._normalize_technology(text))
        elif label == 'FRAMEWORK':
            normalized.update(self._normalize_framework(text))
        elif label == 'PLATFORM':
            normalized.update(self._normalize_platform(text))
        elif label == 'METHODOLOGY':
            normalized.update(self._normalize_methodology(text))
        elif label == 'STANDARD':
            normalized.update(self._normalize_standard(text))
        elif label == 'PROTOCOL':
            normalized.update(self._normalize_protocol(text))
        elif label == 'ALGORITHM':
            normalized.update(self._normalize_algorithm(text))
        elif label == 'ARCHITECTURE':
            normalized.update(self._normalize_architecture(text))
        elif label == 'PATTERN':
            normalized.update(self._normalize_pattern(text))
        elif label == 'PRINCIPLE':
            normalized.update(self._normalize_principle(text))
        elif label == 'BEST_PRACTICE':
            normalized.update(self._normalize_best_practice(text))
        elif label == 'GUIDELINE':
            normalized.update(self._normalize_guideline(text))
        elif label == 'POLICY':
            normalized.update(self._normalize_policy(text))
        elif label == 'PROCEDURE':
            normalized.update(self._normalize_procedure(text))
        elif label == 'WORKFLOW':
            normalized.update(self._normalize_workflow(text))
        elif label == 'PROCESS':
            normalized.update(self._normalize_process(text))
        elif label == 'SYSTEM':
            normalized.update(self._normalize_system(text))
        elif label == 'TOOL':
            normalized.update(self._normalize_tool(text))
        elif label == 'LIBRARY':
            normalized.update(self._normalize_library(text))
        elif label == 'FRAMEWORK':
            normalized.update(self._normalize_framework(text))
        elif label == 'SDK':
            normalized.update(self._normalize_sdk(text))
        elif label == 'API':
            normalized.update(self._normalize_api(text))
        elif label == 'SERVICE':
            normalized.update(self._normalize_service(text))
        elif label == 'MICROSERVICE':
            normalized.update(self._normalize_microservice(text))
        elif label == 'MONOLITH':
            normalized.update(self._normalize_monolith(text))
        elif label == 'LEGACY':
            normalized.update(self._normalize_legacy(text))
        elif label == 'MODERN':
            normalized.update(self._normalize_modern(text))
        elif label == 'CUTTING_EDGE': 
            normalized.update(self._normalize_cutting_edge(text))
        elif label == 'INNOVATIVE':
            normalized.update(self._normalize_innovative(text))
        elif label == 'EXPERIMENTAL':
            normalized.update(self._normalize_experimental(text))
        elif label == 'RESEARCH_BASED':
            normalized.update(self._normalize_research_based(text))
        elif label == 'EVIDENCE_BASED':
            normalized.update(self._normalize_evidence_based(text))
        elif label == 'DATA_DRIVEN':
            normalized.update(self._normalize_data_driven(text))
        elif label == 'USER_CENTRIC':
            normalized.update(self._normalize_user_centric(text))
        elif label == 'ACCESSIBLE':
            normalized.update(self._normalize_accessible(text))
        elif label == 'INCLUSIVE':
            normalized.update(self._normalize_inclusive(text))
        elif label == 'SUSTAINABLE':
            normalized.update(self._normalize_sustainable(text))
        elif label == 'GREEN':
            normalized.update(self._normalize_green(text))
        elif label == 'ENVIRONMENTALLY_FRIENDLY':
            normalized.update(self._normalize_environmentally_friendly(text))
        elif label == 'CARBON_NEUTRAL':
            normalized.update(self._normalize_carbon_neutral(text))
        elif label == 'ZERO_WASTE':
            normalized.update(self._normalize_zero_waste(text))
        elif label == 'CIRCULAR_ECONOMY':
            normalized.update(self._normalize_circular_economy(text))
        elif label == 'SOCIAL_IMPACT':
            normalized.update(self._normalize_social_impact(text))
        elif label == 'SOCIAL_RESPONSIBILITY':
            normalized.update(self._normalize_social_responsibility(text))
        elif label == 'CORPORATE_SOCIAL_RESPONSIBILITY':
            normalized.update(self._normalize_csr(text))
        elif label == 'CSR':
            normalized.update(self._normalize_csr(text))
        elif label == 'ESG':
            normalized.update(self._normalize_esg(text))
        elif label == 'ENVIRONMENTAL_SOCIAL_GOVERNANCE':
            normalized.update(self._normalize_esg(text))
        elif label == 'DIVERSITY':
            normalized.update(self._normalize_diversity(text))
        elif label == 'EQUITY':
            normalized.update(self._normalize_equity(text))
        elif label == 'INCLUSION':
            normalized.update(self._normalize_inclusion(text))
        elif label == 'DEI':
            normalized.update(self._normalize_dei(text))
        elif label == 'DIVERSITY_EQUITY_INCLUSION':
            normalized.update(self._normalize_dei(text))
        elif label == 'BELONGING':
            normalized.update(self._normalize_belonging(text))
        elif label == 'PSYCHOLOGICAL_SAFETY':
            normalized.update(self._normalize_psychological_safety(text))
        elif label == 'MENTAL_HEALTH':
            normalized.update(self._normalize_mental_health(text))
        elif label == 'WELLNESS':
            normalized.update(self._normalize_wellness(text))
        elif label == 'WORK_LIFE_BALANCE':
            normalized.update(self._normalize_work_life_balance(text))
        elif label == 'FLEXIBLE_WORKING':
            normalized.update(self._normalize_flexible_working(text))
        elif label == 'REMOTE_WORK':
            normalized.update(self._normalize_remote_work(text))
        elif label == 'HYBRID_WORK':
            normalized.update(self._normalize_hybrid_work(text))
        elif label == 'COLLABORATION':
            normalized.update(self._normalize_collaboration(text))
        elif label == 'TEAMWORK':
            normalized.update(self._normalize_teamwork(text))
        elif label == 'CROSS_FUNCTIONAL':
            normalized.update(self._normalize_cross_functional(text))
        elif label == 'INTERDISCIPLINARY':
            normalized.update(self._normalize_interdisciplinary(text))
        elif label == 'MULTIDISCIPLINARY':
            normalized.update(self._normalize_multidisciplinary(text))
        elif label == 'TRANSDISCIPLINARY':
            normalized.update(self._normalize_transdisciplinary(text))
        elif label == 'HOLISTIC':
            normalized.update(self._normalize_holistic(text))
        elif label == 'SYSTEMS_THINKING':
            normalized.update(self._normalize_systems_thinking(text))
        elif label == 'DESIGN_THINKING':
            normalized.update(self._normalize_design_thinking(text))
        elif label == 'LEAN_THINKING':
            normalized.update(self._normalize_lean_thinking(text))
        elif label == 'AGILE_THINKING':
            normalized.update(self._normalize_agile_thinking(text))
        elif label == 'STRATEGIC_THINKING':
            normalized.update(self._normalize_strategic_thinking(text))
        elif label == 'CRITICAL_THINKING':
            normalized.update(self._normalize_critical_thinking(text))
        elif label == 'ANALYTICAL_THINKING':
            normalized.update(self._normalize_analytical_thinking(text))
        elif label == 'CREATIVE_THINKING':
            normalized.update(self._normalize_creative_thinking(text))
        elif label == 'INNOVATIVE_THINKING':
            normalized.update(self._normalize_innovative_thinking(text))
        elif label == 'PROBLEM_SOLVING':
            normalized.update(self._normalize_problem_solving(text))
        elif label == 'TROUBLESHOOTING':
            normalized.update(self._normalize_troubleshooting(text))
        elif label == 'DEBUGGING':
            normalized.update(self._normalize_debugging(text))
        elif label == 'OPTIMIZATION':
            normalized.update(self._normalize_optimization(text))
        elif label == 'PERFORMANCE_TUNING':
            normalized.update(self._normalize_performance_tuning(text))
        elif label == 'SCALABILITY':
            normalized.update(self._normalize_scalability(text))
        elif label == 'RELIABILITY':
            normalized.update(self._normalize_reliability(text))
        elif label == 'AVAILABILITY':
            normalized.update(self._normalize_availability(text))
        elif label == 'FAULT_TOLERANCE':
            normalized.update(self._normalize_fault_tolerance(text))
        elif label == 'DISASTER_RECOVERY':
            normalized.update(self._normalize_disaster_recovery(text))
        elif label == 'BUSINESS_CONTINUITY':
            normalized.update(self._normalize_business_continuity(text))
        elif label == 'RISK_MANAGEMENT':
            normalized.update(self._normalize_risk_management(text))
        elif label == 'COMPLIANCE':
            normalized.update(self._normalize_compliance(text))
        elif label == 'GOVERNANCE':
            normalized.update(self._normalize_governance(text))
        elif label == 'AUDIT':
            normalized.update(self._normalize_audit(text))
        elif label == 'CONTROLS':
            normalized.update(self._normalize_controls(text))
        elif label == 'INTERNAL_CONTROLS':
            normalized.update(self._normalize_internal_controls(text))
        elif label == 'EXTERNAL_AUDIT':
            normalized.update(self._normalize_external_audit(text))
        elif label == 'SOC':
            normalized.update(self._normalize_soc(text))
        elif label == 'ISO':
            normalized.update(self._normalize_iso(text))
        elif label == 'GDPR':
            normalized.update(self._normalize_gdpr(text))
        elif label == 'CCPA':
            normalized.update(self._normalize_ccpa(text))
        elif label == 'HIPAA':
            normalized.update(self._normalize_hipaa(text))
        elif label == 'PCI_DSS':
            normalized.update(self._normalize_pci_dss(text))
        elif label == 'SOX':
            normalized.update(self._normalize_sox(text))
        elif label == 'FINRA':
            normalized.update(self._normalize_finra(text))
        elif label == 'SEC':
            normalized.update(self._normalize_sec(text))
        elif label == 'FTC':
            normalized.update(self._normalize_ftc(text))
        elif label == 'FDA':
            normalized.update(self._normalize_fda(text))
        elif label == 'EPA':
            normalized.update(self._normalize_epa(text))
        elif label == 'OSHA':
            normalized.update(self._normalize_osha(text))
        elif label == 'NIST':
            normalized.update(self._normalize_nist(text))
        elif label == 'COBIT':
            normalized.update(self._normalize_cobit(text))
        elif label == 'ITIL':
            normalized.update(self._normalize_itil(text))
        elif label == 'PRINCE2':
            normalized.update(self._normalize_prince2(text))
        elif label == 'PMP':
            normalized.update(self._normalize_pmp(text))
        elif label == 'CAPM':
            normalized.update(self._normalize_capm(text))
        elif label == 'CSM':
            normalized.update(self._normalize_csm(text))
        elif label == 'CSPO':
            normalized.update(self._normalize_cspo(text))
        elif label == 'SAFE':
            normalized.update(self._normalize_safe(text))
        elif label == 'LEAN_SIX_SIGMA':
            normalized.update(self._normalize_lean_six_sigma(text))
        elif label == 'BLACK_BELT':
            normalized.update(self._normalize_black_belt(text))
        elif label == 'GREEN_BELT':
            normalized.update(self._normalize_green_belt(text))
        elif label == 'YELLOW_BELT':
            normalized.update(self._normalize_yellow_belt(text))
        elif label == 'WHITE_BELT':
            normalized.update(self._normalize_white_belt(text))
        elif label == 'SENIOR':
            normalized.update(self._normalize_senior(text))
        elif label == 'JUNIOR':
            normalized.update(self._normalize_junior(text))
        elif label == 'LEAD':
            normalized.update(self._normalize_lead(text))
        elif label == 'PRINCIPAL':
            normalized.update(self._normalize_principal(text))
        elif label == 'STAFF':
            normalized.update(self._normalize_staff(text))
        elif label == 'ASSOCIATE':
            normalized.update(self._normalize_associate(text))
        elif label == 'DIRECTOR':
            normalized.update(self._normalize_director(text))
        elif label == 'MANAGER':
            normalized.update(self._normalize_manager(text))
        elif label == 'VP':
            normalized.update(self._normalize_vp(text))
        elif label == 'VICE_PRESIDENT':
            normalized.update(self._normalize_vp(text))
        elif label == 'CTO':
            normalized.update(self._normalize_cto(text))
        elif label == 'CHIEF_TECHNOLOGY_OFFICER':
            normalized.update(self._normalize_cto(text))
        elif label == 'CIO':
            normalized.update(self._normalize_cio(text))
        elif label == 'CHIEF_INFORMATION_OFFICER':
            normalized.update(self._normalize_cio(text))
        elif label == 'CEO':
            normalized.update(self._normalize_ceo(text))
        elif label == 'CHIEF_EXECUTIVE_OFFICER':
            normalized.update(self._normalize_ceo(text))
        elif label == 'CFO':
            normalized.update(self._normalize_cfo(text))
        elif label == 'CHIEF_FINANCIAL_OFFICER':
            normalized.update(self._normalize_cfo(text))
        elif label == 'COO':
            normalized.update(self._normalize_coo(text))
        elif label == 'CHIEF_OPERATING_OFFICER':
            normalized.update(self._normalize_coo(text))
        elif label == 'FOUNDER':
            normalized.update(self._normalize_founder(text))
        elif label == 'CO_FOUNDER':
            normalized.update(self._normalize_co_founder(text))
        elif label == 'ENTREPRENEUR':
            normalized.update(self._normalize_entrepreneur(text))
        elif label == 'STARTUP':
            normalized.update(self._normalize_startup(text))
        elif label == 'SCALEUP':
            normalized.update(self._normalize_scaleup(text))
        elif label == 'UNICORN':
            normalized.update(self._normalize_unicorn(text))
        elif label == 'IPO':
            normalized.update(self._normalize_ipo(text))
        elif label == 'INITIAL_PUBLIC_OFFERING':
            normalized.update(self._normalize_ipo(text))
        elif label == 'ACQUISITION':
            normalized.update(self._normalize_acquisition(text))
        elif label == 'MERGER':
            normalized.update(self._normalize_merger(text))
        elif label == 'JOINT_VENTURE':
            normalized.update(self._normalize_joint_venture(text))
        elif label == 'PARTNERSHIP':
            normalized.update(self._normalize_partnership(text))
        elif label == 'SUBSIDIARY':
            normalized.update(self._normalize_subsidiary(text))
        elif label == 'BRANCH':
            normalized.update(self._normalize_branch(text))
        elif label == 'REGION':
            normalized.update(self._normalize_region(text))
        elif label == 'TERRITORY':
            normalized.update(self._normalize_territory(text))
        elif label == 'GLOBAL':
            normalized.update(self._normalize_global(text))
        elif label == 'INTERNATIONAL':
            normalized.update(self._normalize_international(text))
        elif label == 'MULTINATIONAL':
            normalized.update(self._normalize_multinational(text))
        elif label == 'FORTUNE_500':
            normalized.update(self._normalize_fortune_500(text))
        elif label == 'FORTUNE_1000':
            normalized.update(self._normalize_fortune_1000(text))
        elif label == 'STARTUP_100':
            normalized.update(self._normalize_startup_100(text))
        elif label == 'INC_5000':
            normalized.update(self._normalize_inc_5000(text))
        elif label == 'DELOITTE_FAST_500':
            normalized.update(self._normalize_deloitte_fast_500(text))
        elif label == 'EY_ENTREPRENEUR_OF_THE_YEAR':
            normalized.update(self._normalize_ey_entrepreneur_of_the_year(text))
        elif label == 'ERNST_AND_YOUNG':
            normalized.update(self._normalize_ey(text))
        elif label == 'EY':
            normalized.update(self._normalize_ey(text))
        elif label == 'MCKINSEY':
            normalized.update(self._normalize_mckinsey(text))
        elif label == 'BAIN':
            normalized.update(self._normalize_bain(text))
        elif label == 'BCG':
            normalized.update(self._normalize_bcg(text))
        elif label == 'BOSTON_CONSULTING_GROUP':
            normalized.update(self._normalize_bcg(text))
        elif label == 'PRICE_WATERHOUSE_COOPERS':
            normalized.update(self._normalize_pwc(text))
        elif label == 'PWC':
            normalized.update(self._normalize_pwc(text))
        elif label == 'KPMG':
            normalized.update(self._normalize_kpmg(text))
        elif label == 'ACCENTURE':
            normalized.update(self._normalize_accenture(text))
        elif label == 'IBM_CONSULTING':
            normalized.update(self._normalize_ibm_consulting(text))
        elif label == 'CAPGEMINI':
            normalized.update(self._normalize_capgemini(text))
        elif label == 'INFOSYS':
            normalized.update(self._normalize_infosys(text))
        elif label == 'TCS':
            normalized.update(self._normalize_tcs(text))
        elif label == 'TATA_CONSULTANCY_SERVICES':
            normalized.update(self._normalize_tcs(text))
        elif label == 'WIPRO':
            normalized.update(self._normalize_wipro(text))
        elif label == 'HCL':
            normalized.update(self._normalize_hcl(text))
        elif label == 'TECH_MAHINDRA':
            normalized.update(self._normalize_tech_mahindra(text))
        elif label == 'LARSEN_AND_TOUBRO':
            normalized.update(self._normalize_lt(text))
        elif label == 'LT':
            normalized.update(self._normalize_lt(text))
        elif label == 'RELIANCE':
            normalized.update(self._normalize_reliance(text))
        elif label == 'TATA':
            normalized.update(self._normalize_tata(text))
        elif label == 'MAHINDRA':
            normalized.update(self._normalize_mahindra(text))
        elif label == 'ADANI':
            normalized.update(self._normalize_adani(text))
        elif label == 'BIRLA':
            normalized.update(self._normalize_birla(text))
        elif label == 'GODREJ':
            normalized.update(self._normalize_godrej(text))
        
        # Add confidence score based on validation
        normalized['confidence'] = self._calculate_confidence(normalized)
        
        return normalized
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _normalize_email(self, text: str) -> Dict[str, Any]:
        """Normalize email address."""
        result = {}
        
        # Validate email format
        if self.patterns['email'].match(text):
            result['is_valid'] = True
            result['domain'] = text.split('@')[1]
            result['username'] = text.split('@')[0]
        else:
            result['is_valid'] = False
            result['domain'] = None
            result['username'] = None
        
        return result
    
    def _normalize_phone(self, text: str) -> Dict[str, Any]:
        """Normalize phone number."""
        result = {}
        
        # Remove common separators
        clean_phone = re.sub(r'[\s\-\(\)\.]', '', text)
        
        # Validate phone format
        if self.patterns['phone'].match(clean_phone):
            result['is_valid'] = True
            result['clean_number'] = clean_phone
            result['country_code'] = self._extract_country_code(clean_phone)
        else:
            result['is_valid'] = False
            result['clean_number'] = None
            result['country_code'] = None
        
        return result
    
    def _normalize_website(self, text: str) -> Dict[str, Any]:
        """Normalize website URL."""
        result = {}
        
        # Add protocol if missing
        if not text.startswith(('http://', 'https://')):
            text = 'https://' + text
        
        # Validate URL format
        if self.patterns['url'].match(text):
            result['is_valid'] = True
            result['full_url'] = text
            
            # Parse domain
            try:
                parsed = urlparse(text)
                domain = parsed.netloc.lower()
                result['domain'] = domain
                
                # Check for social media domains
                for social_domain, social_type in self.domain_mappings.items():
                    if social_domain in domain:
                        result['social_type'] = social_type
                        break
                
            except Exception:
                result['domain'] = None
        else:
            result['is_valid'] = False
            result['full_url'] = None
            result['domain'] = None
        
        return result
    
    def _normalize_name(self, text: str) -> Dict[str, Any]:
        """Normalize person name."""
        result = {}
        
        # Split into parts
        parts = text.split()
        if len(parts) >= 2:
            result['first_name'] = parts[0]
            result['last_name'] = parts[-1]
            result['middle_names'] = parts[1:-1] if len(parts) > 2 else []
        else:
            result['first_name'] = text
            result['last_name'] = None
            result['middle_names'] = []
        
        # Title case
        result['formatted_name'] = text.title()
        
        return result
    
    def _normalize_company(self, text: str) -> Dict[str, Any]:
        """Normalize company name."""
        result = {}
        
        # Remove common suffixes
        suffixes = [' Inc', ' Corp', ' LLC', ' Ltd', ' Co', ' Company']
        clean_name = text
        for suffix in suffixes:
            if text.endswith(suffix):
                clean_name = text[:-len(suffix)]
                result['suffix'] = suffix.strip()
                break
        
        result['clean_name'] = clean_name.strip()
        result['formatted_name'] = text.title()
        
        return result
    
    def _normalize_title(self, text: str) -> Dict[str, Any]:
        """Normalize job title."""
        result = {}
        
        # Title case
        result['formatted_title'] = text.title()
        
        # Check for seniority levels
        seniority_keywords = ['senior', 'junior', 'lead', 'principal', 'staff', 'associate']
        text_lower = text.lower()
        
        for keyword in seniority_keywords:
            if keyword in text_lower:
                result['seniority'] = keyword.title()
                break
        
        return result
    
    def _normalize_university(self, text: str) -> Dict[str, Any]:
        """Normalize university name."""
        result = {}
        
        # Title case
        result['formatted_name'] = text.title()
        
        # Check for common university indicators
        indicators = ['University', 'College', 'Institute', 'School']
        for indicator in indicators:
            if indicator.lower() in text.lower():
                result['type'] = indicator
                break
        
        return result
    
    def _normalize_degree(self, text: str) -> Dict[str, Any]:
        """Normalize degree name."""
        result = {}
        
        # Check for standard degree abbreviations
        if self.patterns['degree'].match(text):
            result['abbreviation'] = text.upper()
            result['is_standard'] = True
        else:
            result['abbreviation'] = None
            result['is_standard'] = False
        
        result['formatted_name'] = text.title()
        
        return result
    
    def _normalize_year(self, text: str) -> Dict[str, Any]:
        """Normalize graduation year."""
        result = {}
        
        # Validate year format
        if self.patterns['year'].match(text):
            result['is_valid'] = True
            result['year'] = int(text)
            result['decade'] = (int(text) // 10) * 10
        else:
            result['is_valid'] = False
            result['year'] = None
            result['decade'] = None
        
        return result
    
    def _normalize_skill(self, text: str) -> Dict[str, Any]:
        """Normalize skill name."""
        result = {}
        
        # Title case
        result['formatted_name'] = text.title()
        
        # Check for programming languages
        programming_langs = ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'swift']
        if text.lower() in programming_langs:
            result['category'] = 'programming_language'
        
        return result
    
    def _extract_country_code(self, phone: str) -> Optional[str]:
        """Extract country code from phone number."""
        if phone.startswith('+'):
            # Find the country code (usually 1-3 digits)
            match = re.match(r'\+(\d{1,3})', phone)
            if match:
                return '+' + match.group(1)
        elif phone.startswith('1') and len(phone) == 11:
            return '+1'  # US/Canada
        
        return None
    
    def _calculate_confidence(self, entity: Dict[str, Any]) -> float:
        """Calculate confidence score for entity."""
        base_confidence = 0.8
        
        # Boost confidence for valid entities
        if entity.get('is_valid', False):
            base_confidence += 0.1
        
        # Boost confidence for entities with normalization
        if len(entity) > 4:  # Has additional normalization fields
            base_confidence += 0.05
        
        # Reduce confidence for very short entities
        if len(entity['text']) < 2:
            base_confidence -= 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def postprocess_predictions(self, tokens: List[str], labels: List[str]) -> Dict[str, Any]:
        """
        Complete post-processing pipeline for predictions.
        
        Args:
            tokens: List of word tokens
            labels: List of BIO labels
            
        Returns:
            Post-processed results
        """
        # Merge spans
        entities = self.merge_spans(tokens, labels)
        
        # Normalize each entity
        normalized_entities = []
        for entity in entities:
            normalized = self.normalize_entity(entity)
            normalized_entities.append(normalized)
        
        # Create summary
        entity_counts = {}
        for entity in normalized_entities:
            label = entity['label']
            entity_counts[label] = entity_counts.get(label, 0) + 1
        
        # Extract normalized values for common entity types
        normalized_values = {}
        for entity_type in ['EMAIL', 'PHONE', 'WEBSITE', 'NAME', 'COMPANY']:
            entities_of_type = [e for e in normalized_entities if e['label'] == entity_type]
            if entities_of_type:
                normalized_values[entity_type.lower()] = [e.get('normalized_text', e['text']) for e in entities_of_type]
        
        return {
            'entities': normalized_entities,
            'entity_counts': entity_counts,
            'normalized': normalized_values,
            'total_entities': len(normalized_entities)
        }


def create_post_processor() -> EntityPostProcessor:
    """Factory function to create post-processor."""
    return EntityPostProcessor()


if __name__ == "__main__":
    # Test the post-processor
    processor = create_post_processor()
    
    # Test data
    tokens = ['John', 'Smith', 'works', 'at', 'Google', 'contact', 'john@email.com']
    labels = ['B-NAME', 'I-NAME', 'O', 'O', 'B-COMPANY', 'O', 'B-EMAIL']
    
    # Process
    results = processor.postprocess_predictions(tokens, labels)
    
    print("Post-processing results:")
    print(json.dumps(results, indent=2, default=str))
