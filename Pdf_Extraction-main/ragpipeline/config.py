
import os

class Config:
    
    SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
    MODEL_CACHE_DIR = "/app/models"
    MAX_MODEL_SIZE_MB = 500
    
    MAX_DOCUMENTS = 10
    MAX_FILE_SIZE_MB = 50
    PROCESSING_TIMEOUT_SECONDS = 60
    MAX_SECTIONS_PER_DOC = 10
    MAX_TOTAL_SUBSECTIONS = 15
    
    MIN_SECTION_LENGTH = 50
    MIN_SUBSECTION_LENGTH = 100
    MAX_CONTENT_PREVIEW_LENGTH = 200
    MAX_REFINED_TEXT_WORDS = 300
    
    KEYWORD_MATCH_WEIGHT = 0.3
    SEMANTIC_SIMILARITY_WEIGHT = 0.4
    CONTENT_QUALITY_WEIGHT = 0.2
    STRUCTURE_WEIGHT = 0.1
    
    MIN_SECTION_RELEVANCE = 20
    MIN_SUBSECTION_RELEVANCE = 30
    
    MAX_KEYWORDS_EXTRACTED = 20
    TFIDF_MAX_FEATURES = 1000
    NGRAM_RANGE = (1, 2)
    
    OUTPUT_PRECISION = 2  
    INCLUDE_DEBUG_INFO = False
    
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_env_config(cls):
        return {
            'processing_timeout': int(os.getenv('PROCESSING_TIMEOUT', cls.PROCESSING_TIMEOUT_SECONDS)),
            'max_documents': int(os.getenv('MAX_DOCUMENTS', cls.MAX_DOCUMENTS)),
            'max_file_size_mb': int(os.getenv('MAX_FILE_SIZE_MB', cls.MAX_FILE_SIZE_MB)),
            'min_section_relevance': float(os.getenv('MIN_SECTION_RELEVANCE', cls.MIN_SECTION_RELEVANCE)),
            'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        }

DOMAIN_KEYWORDS = {
    'academic': [
        'research', 'study', 'analysis', 'methodology', 'results',
        'literature', 'hypothesis', 'experiment', 'data', 'findings',
        'publication', 'peer-review', 'statistical', 'significant',
        'correlation', 'validation', 'benchmark', 'baseline'
    ],
    'business': [
        'revenue', 'profit', 'market', 'strategy', 'competitive',
        'customer', 'growth', 'investment', 'roi', 'kpi',
        'business', 'commercial', 'financial', 'performance',
        'sales', 'marketing', 'operations', 'management'
    ],
    'technical': [
        'system', 'algorithm', 'implementation', 'architecture',
        'software', 'hardware', 'network', 'database',
        'performance', 'optimization', 'scalability', 'security',
        'api', 'framework', 'protocol', 'interface'
    ],
    'medical': [
        'patient', 'clinical', 'treatment', 'diagnosis', 'therapy',
        'medical', 'health', 'disease', 'symptom', 'drug',
        'trial', 'efficacy', 'safety', 'dosage', 'adverse'
    ],
    'financial': [
        'financial', 'accounting', 'budget', 'cost', 'expense',
        'income', 'cash', 'flow', 'balance', 'sheet',
        'assets', 'liabilities', 'equity', 'investment', 'return'
    ],
    'travel': [
        'travel', 'trip', 'destination', 'hotel', 'restaurant',
        'attraction', 'activity', 'transportation', 'accommodation',
        'booking', 'itinerary', 'sightseeing', 'tour', 'guide',
        'location', 'city', 'culture', 'history', 'cuisine',
        'budget', 'cost', 'price', 'recommendation', 'review',
        'experience', 'vacation', 'holiday', 'journey', 'visit'
    ],
    'planning': [
        'plan', 'schedule', 'organize', 'coordinate', 'arrange',
        'timeline', 'duration', 'day', 'time', 'group',
        'friends', 'college', 'student', 'young', 'budget',
        'affordable', 'cheap', 'fun', 'entertainment', 'social',
        'party', 'nightlife', 'adventure', 'explore', 'discover'
    ]
}

SECTION_PATTERNS = {
    'abstract': [
        r'\babstract\b', r'\bsummary\b', r'\boverview\b',
        r'\bexecutive summary\b'
    ],
    'introduction': [
        r'\bintroduction\b', r'\bbackground\b', r'\bmotivation\b',
        r'\bobjective\b', r'\bpurpose\b'
    ],
    'methodology': [
        r'\bmethodology\b', r'\bmethod\b', r'\bapproach\b',
        r'\btechnique\b', r'\bprocedure\b', r'\bexperimental\b'
    ],
    'results': [
        r'\bresults?\b', r'\bfindings?\b', r'\boutcome\b',
        r'\bdata\b', r'\bobservation\b'
    ],
    'discussion': [
        r'\bdiscussion\b', r'\banalysis\b', r'\binterpretation\b',
        r'\bimplication\b'
    ],
    'conclusion': [
        r'\bconclusion\b', r'\bsummary\b', r'\bfuture work\b',
        r'\brecommendation\b'
    ],
    'financial': [
        r'\bfinancial\b', r'\brevenue\b', r'\bprofit\b',
        r'\bearnings\b', r'\bincome statement\b', r'\bbalance sheet\b'
    ],
    'travel_destinations': [
        r'\bcities\b', r'\bdestinations\b', r'\bplaces\b',
        r'\blocations\b', r'\battraction\b', r'\bsightseeing\b'
    ],
    'accommodation': [
        r'\bhotel\b', r'\baccommodation\b', r'\blodging\b',
        r'\bstay\b', r'\bresort\b', r'\bhostel\b'
    ],
    'dining': [
        r'\brestaurant\b', r'\bcuisine\b', r'\bfood\b',
        r'\bdining\b', r'\bbar\b', r'\bcafe\b'
    ],
    'activities': [
        r'\bactivity\b', r'\bthings to do\b', r'\battraction\b',
        r'\bentertainment\b', r'\btour\b', r'\bexperience\b'
    ],
    'culture': [
        r'\bculture\b', r'\btradition\b', r'\bhistory\b',
        r'\bheritage\b', r'\bart\b', r'\bmuseum\b'
    ],
    'tips': [
        r'\btips\b', r'\btricks\b', r'\badvice\b',
        r'\brecommendation\b', r'\bguide\b', r'\bhint\b'
    ]
}

PERSONA_PREFERENCES = {
    'researcher': {
        'preferred_sections': ['methodology', 'results', 'discussion'],
        'keyword_boost': DOMAIN_KEYWORDS['academic'],
        'min_content_complexity': 0.6
    },
    'student': {
        'preferred_sections': ['introduction', 'abstract', 'conclusion'],
        'keyword_boost': ['concept', 'definition', 'example', 'explanation'],
        'min_content_complexity': 0.3
    },
    'analyst': {
        'preferred_sections': ['results', 'financial', 'discussion'],
        'keyword_boost': DOMAIN_KEYWORDS['business'] + DOMAIN_KEYWORDS['financial'],
        'min_content_complexity': 0.5
    },
    'manager': {
        'preferred_sections': ['abstract', 'conclusion', 'financial'],
        'keyword_boost': ['summary', 'recommendation', 'decision', 'strategy'],
        'min_content_complexity': 0.4
    },
    'travel planner': {
        'preferred_sections': ['activities', 'accommodation', 'dining', 'tips'],
        'keyword_boost': DOMAIN_KEYWORDS['travel'] + DOMAIN_KEYWORDS['planning'],
        'min_content_complexity': 0.3
    },
    'travel_planner': {
        'preferred_sections': ['activities', 'accommodation', 'dining', 'tips'],
        'keyword_boost': DOMAIN_KEYWORDS['travel'] + DOMAIN_KEYWORDS['planning'],
        'min_content_complexity': 0.3
    }
}