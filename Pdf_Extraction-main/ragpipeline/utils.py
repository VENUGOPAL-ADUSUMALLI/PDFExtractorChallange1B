import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

import fitz  
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

@dataclass
class DocumentSection:
    document_name: str
    page_number: int
    section_title: str
    content: str
    importance_score: float
    section_type: str = "content"

@dataclass
class SubSection:
    document_name: str
    page_number: int
    content: str
    refined_text: str
    relevance_score: float

class DocumentIntelligenceSystem:
    def __init__(self):
        self.setup_logging()
        self.sentence_model = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.stop_words = set(stopwords.words('english'))
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_sentence_model(self):
        if self.sentence_model is None:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        try:
            doc = fitz.open(pdf_path)
            pages_text = {}
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                pages_text[page_num + 1] = text
                
            doc.close()
            return pages_text
        except Exception as e:
            self.logger.error(f"Error extracting text from {pdf_path}: {e}")
            return {}

    def identify_sections(self, pages_text: Dict[int, str], doc_name: str) -> List[DocumentSection]:
        sections = []
        
        for page_num, text in pages_text.items():
            if not text.strip():
                continue
                
            section_patterns = [
                r'\n([A-Z][A-Z\s]{10,50})\n', 
                r'\n(\d+\.?\s+[A-Z][^.\n]{10,100})\n', 
                r'\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?\s*)\n',  
                r'\n(Abstract|Introduction|Methodology|Results|Discussion|Conclusion|References)\b', 
            ]
            
            sections_found = []
            for pattern in section_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    sections_found.append((match.start(), match.group(1).strip()))
            
            if not sections_found:
                paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
                for i, para in enumerate(paragraphs[:5]): 
                    section_title = f"Section {i+1}"
                    if len(para) > 100:
                        sections.append(DocumentSection(
                            document_name=doc_name,
                            page_number=page_num,
                            section_title=section_title,
                            content=para,
                            importance_score=0.0
                        ))
            else:
                sections_found.sort(key=lambda x: x[0])
                
                for i, (pos, title) in enumerate(sections_found):
                    start_pos = pos
                    end_pos = sections_found[i+1][0] if i+1 < len(sections_found) else len(text)
                    content = text[start_pos:end_pos].strip()
                    
                    if len(content) > 50:  
                        sections.append(DocumentSection(
                            document_name=doc_name,
                            page_number=page_num,
                            section_title=title,
                            content=content,
                            importance_score=0.0
                        ))
        
        return sections

    def extract_persona_keywords(self, persona_description: str, job_description: str) -> List[str]:
        combined_text = f"{persona_description} {job_description}"
        
        words = word_tokenize(combined_text.lower())
        keywords = [word for word in words if word.isalpha() and word not in self.stop_words and len(word) > 3]
        
        from config import DOMAIN_KEYWORDS, PERSONA_PREFERENCES
        
        text_lower = combined_text.lower()
        domain_keywords = []
        
        if any(term in text_lower for term in ['travel', 'trip', 'planner', 'vacation', 'tour']):
            domain_keywords.extend(DOMAIN_KEYWORDS.get('travel', []))
            domain_keywords.extend(DOMAIN_KEYWORDS.get('planning', []))
        
        if any(term in text_lower for term in ['research', 'phd', 'academic', 'study']):
            domain_keywords.extend(DOMAIN_KEYWORDS.get('academic', []))
        
        if any(term in text_lower for term in ['business', 'analyst', 'investment', 'financial']):
            domain_keywords.extend(DOMAIN_KEYWORDS.get('business', []))
            domain_keywords.extend(DOMAIN_KEYWORDS.get('financial', []))
        
        if any(term in text_lower for term in ['technical', 'software', 'system', 'engineering']):
            domain_keywords.extend(DOMAIN_KEYWORDS.get('technical', []))
        
        travel_patterns = [
            r'\b(plan|planning|itinerary|schedule|organize)\w*\b',
            r'\b(day|days|trip|vacation|holiday|travel)\w*\b',
            r'\b(group|friends|college|student|young)\w*\b',
            r'\b(budget|affordable|cost|price|cheap)\w*\b',
            r'\b(hotel|restaurant|activity|attraction|city)\w*\b',
            r'\b(food|cuisine|dining|culture|history)\w*\b'
        ]
        
        for pattern in travel_patterns:
            matches = re.findall(pattern, combined_text.lower())
            keywords.extend(matches)
        
        keywords.extend(domain_keywords)
        
        important_patterns = [
            r'\b(research|analysis|study|review|methodology|data|performance|trend|strategy)\w*\b',
            r'\b(financial|technical|academic|business|scientific|clinical)\w*\b',
            r'\b(PhD|analyst|student|researcher|manager|specialist|planner)\w*\b'
        ]
        
        for pattern in important_patterns:
            matches = re.findall(pattern, combined_text.lower())
            keywords.extend(matches)
        
        unique_keywords = list(set(keywords))
        
        keyword_freq = {}
        for keyword in unique_keywords:
            freq = combined_text.lower().count(keyword)
            if keyword in domain_keywords:
                freq *= 2
            keyword_freq[keyword] = freq
        
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, freq in sorted_keywords[:25]]

    def calculate_relevance_score(self, section: DocumentSection, persona_keywords: List[str], 
                                job_description: str) -> float:
        content_lower = section.content.lower()
        title_lower = section.section_title.lower()
        
        keyword_matches = sum(1 for keyword in persona_keywords if keyword in content_lower)
        keyword_score = min(keyword_matches * 2, 40)
        
        title_matches = sum(1 for keyword in persona_keywords if keyword in title_lower)
        title_score = min(title_matches * 5, 20)
        
        content_quality = min(len(section.content) / 500 * 10, 20)
        
        semantic_score = 0
        if self.sentence_model:
            try:
                job_embedding = self.sentence_model.encode([job_description])
                content_embedding = self.sentence_model.encode([section.content[:512]])  
                similarity = cosine_similarity(job_embedding, content_embedding)[0][0]
                semantic_score = similarity * 20
            except Exception as e:
                self.logger.warning(f"Semantic similarity calculation failed: {e}")
        
        total_score = keyword_score + title_score + content_quality + semantic_score
        return min(total_score, 100)  

    def rank_sections(self, sections: List[DocumentSection], persona_description: str, 
                     job_description: str) -> List[DocumentSection]:
        self.load_sentence_model()
        persona_keywords = self.extract_persona_keywords(persona_description, job_description)
        
        for section in sections:
            section.importance_score = self.calculate_relevance_score(
                section, persona_keywords, job_description
            )
        
        sections.sort(key=lambda x: x.importance_score, reverse=True)
        
        for i, section in enumerate(sections):
            section.importance_rank = i + 1
            
        return sections

    def extract_subsections(self, top_sections: List[DocumentSection], 
                          persona_description: str, job_description: str) -> List[SubSection]:
        subsections = []
        persona_keywords = self.extract_persona_keywords(persona_description, job_description)
        
        for section in top_sections[:10]:  
            sentences = sent_tokenize(section.content)
            
            current_subsection = []
            for sentence in sentences:
                current_subsection.append(sentence)
                
                if len(current_subsection) >= 2 and len(' '.join(current_subsection)) > 200:
                    subsection_text = ' '.join(current_subsection)
                    refined_text = self.refine_text(subsection_text, persona_keywords)
                    
                    relevance_score = self.calculate_subsection_relevance(
                        subsection_text, persona_keywords, job_description
                    )
                    
                    if relevance_score > 30:  
                        subsections.append(SubSection(
                            document_name=section.document_name,
                            page_number=section.page_number,
                            content=subsection_text,
                            refined_text=refined_text,
                            relevance_score=relevance_score
                        ))
                    
                    current_subsection = []
            
            if current_subsection and len(' '.join(current_subsection)) > 100:
                subsection_text = ' '.join(current_subsection)
                refined_text = self.refine_text(subsection_text, persona_keywords)
                relevance_score = self.calculate_subsection_relevance(
                    subsection_text, persona_keywords, job_description
                )
                
                if relevance_score > 30:
                    subsections.append(SubSection(
                        document_name=section.document_name,
                        page_number=section.page_number,
                        content=subsection_text,
                        refined_text=refined_text,
                        relevance_score=relevance_score
                    ))
        
        subsections.sort(key=lambda x: x.relevance_score, reverse=True)
        return subsections[:15] 

    def refine_text(self, text: str, keywords: List[str]) -> str:
        refined = re.sub(r'\s+', ' ', text.strip())
        
        sentences = sent_tokenize(refined)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence)
            elif len(sentence) > 100 and len(relevant_sentences) < 3:
                relevant_sentences.append(sentence)
        
        result = ' '.join(relevant_sentences)
        words = result.split()
        if len(words) > 300:
            result = ' '.join(words[:300]) + "..."
            
        return result

    def calculate_subsection_relevance(self, text: str, keywords: List[str], 
                                     job_description: str) -> float:
        text_lower = text.lower()
        
        keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
        keyword_score = min(keyword_matches * 15, 60)
        
        quality_score = min(len(text) / 200 * 20, 40)
        
        return keyword_score + quality_score

    def process_documents(self, pdf_paths: List[str], persona_description: str, 
                         job_description: str) -> Dict[str, Any]:
        start_time = time.time()
        
        self.logger.info(f"Processing {len(pdf_paths)} documents")
        
        all_sections = []
        document_names = []
        
        for pdf_path in pdf_paths:
            doc_name = os.path.basename(pdf_path)
            document_names.append(doc_name)
            
            self.logger.info(f"Processing {doc_name}")
            pages_text = self.extract_text_from_pdf(pdf_path)
            sections = self.identify_sections(pages_text, doc_name)
            all_sections.extend(sections)
        
        ranked_sections = self.rank_sections(all_sections, persona_description, job_description)
        
        top_sections = ranked_sections[:20]
        
        subsections = self.extract_subsections(top_sections, persona_description, job_description)
        
        processing_time = time.time() - start_time
        
        output = {
            "metadata": {
                "input_documents": document_names,
                "persona": persona_description,
                "job_to_be_done": job_description,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(processing_time, 2),
                "total_sections_analyzed": len(all_sections),
                "top_sections_selected": len(top_sections)
            },
            "extracted_sections": [
                {
                    "document": section.document_name,
                    "page_number": section.page_number,
                    "section_title": section.section_title,
                    "importance_rank": i + 1,
                    "relevance_score": round(section.importance_score, 2),
                    "content_preview": section.content[:200] + "..." if len(section.content) > 200 else section.content
                }
                for i, section in enumerate(top_sections)
            ],
            "subsection_analysis": [
                {
                    "document": subsection.document_name,
                    "page_number": subsection.page_number,
                    "refined_text": subsection.refined_text,
                    "relevance_score": round(subsection.relevance_score, 2),
                    "original_content_length": len(subsection.content)
                }
                for subsection in subsections
            ]
        }
        
        self.logger.info(f"Processing completed in {processing_time:.2f} seconds")
        return output

    def load_challenge_input(self, input_file: str) -> Dict[str, Any]:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                challenge_data = json.load(f)
            
            self.logger.info(f"Loaded challenge: {challenge_data.get('challenge_info', {}).get('challenge_id', 'unknown')}")
            return challenge_data
        except Exception as e:
            self.logger.error(f"Failed to load challenge input: {e}")
            raise

    def process_challenge(self, input_file: str, pdf_directory: str) -> Dict[str, Any]:
        challenge_data = self.load_challenge_input(input_file)
        
        challenge_info = challenge_data.get('challenge_info', {})
        documents_info = challenge_data.get('documents', [])
        persona_info = challenge_data.get('persona', {})
        job_info = challenge_data.get('job_to_be_done', {})
        
        persona_role = persona_info.get('role', 'Professional')
        persona_description = f"{persona_role} with expertise in planning and coordination."
        
        if 'expertise' in persona_info:
            persona_description += f" Specializes in {persona_info['expertise']}."
        if 'focus_areas' in persona_info:
            persona_description += f" Focuses on {', '.join(persona_info['focus_areas'])}."
        
        job_description = job_info.get('task', 'Complete the assigned task.')
        
        pdf_paths = []
        document_titles = {}
        
        for doc_info in documents_info:
            filename = doc_info.get('filename', '')
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                if os.path.exists(pdf_path):
                    pdf_paths.append(pdf_path)
                    document_titles[filename] = doc_info.get('title', filename.replace('.pdf', ''))
                else:
                    self.logger.warning(f"PDF file not found: {pdf_path}")
        
        if not pdf_paths:
            raise ValueError("No valid PDF files found for processing")
        
        result = self.process_documents(pdf_paths, persona_description, job_description)
        
        result['metadata'].update({
            'challenge_id': challenge_info.get('challenge_id', 'unknown'),
            'test_case_name': challenge_info.get('test_case_name', 'unknown'),
            'challenge_description': challenge_info.get('description', ''),
            'document_titles': document_titles,
            'persona_role': persona_role,
            'original_task': job_description
        })
        
        for section in result['extracted_sections']:
            filename = section['document']
            if filename in document_titles:
                section['document_title'] = document_titles[filename]
        
        for subsection in result['subsection_analysis']:
            filename = subsection['document']
            if filename in document_titles:
                subsection['document_title'] = document_titles[filename]
        
        return result

def main():
    import sys
    
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
        
        input_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else '.'
        pdf_directory = os.path.join(input_dir, 'pdfs') if os.path.exists(os.path.join(input_dir, 'pdfs')) else input_dir
        
        system = DocumentIntelligenceSystem()
        result = system.process_challenge(input_file, pdf_directory)
        
        challenge_id = result['metadata'].get('challenge_id', 'challenge')
        test_case = result['metadata'].get('test_case_name', 'output')
        output_file = f"{challenge_id}_{test_case}_output.json"
        
    elif len(sys.argv) == 3:
        input_file = sys.argv[1]
        pdf_directory = sys.argv[2]
        
        system = DocumentIntelligenceSystem()
        result = system.process_challenge(input_file, pdf_directory)
        
        challenge_id = result['metadata'].get('challenge_id', 'challenge')
        test_case = result['metadata'].get('test_case_name', 'output')
        output_file = f"{challenge_id}_{test_case}_output.json"
        
    elif len(sys.argv) == 4:
        pdf_directory = sys.argv[1]
        persona_file = sys.argv[2]
        job_file = sys.argv[3]
        
        with open(persona_file, 'r') as f:
            persona_description = f.read().strip()
        
        with open(job_file, 'r') as f:
            job_description = f.read().strip()
        
        pdf_paths = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_paths:
            print("No PDF files found in the specified directory")
            sys.exit(1)
        
        system = DocumentIntelligenceSystem()
        result = system.process_documents(pdf_paths, persona_description, job_description)
        output_file = 'challenge1b_output.json'
        
    else:
        print("Usage:")
        print("  New format: python main.py <challenge_input.json> [pdf_directory]")
        print("  Legacy format: python main.py <pdf_directory> <persona_file> <job_file>")
        sys.exit(1)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Output saved to {output_file}")
    print(f"Challenge: {result['metadata'].get('challenge_id', 'N/A')}")
    print(f"Test case: {result['metadata'].get('test_case_name', 'N/A')}")
    print(f"Processed {len(result['metadata']['input_documents'])} documents in {result['metadata']['processing_time_seconds']} seconds")

if __name__ == "__main__":
    main()