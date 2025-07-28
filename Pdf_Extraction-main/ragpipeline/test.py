#!/usr/bin/env python3
"""
Test runner for the document intelligence system.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, Any

# Add the main module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import DocumentIntelligenceSystem
from utils import validate_input_files, estimate_processing_time

class TestRunner:
    """Test runner for document intelligence system."""
    
    def __init__(self):
        self.system = DocumentIntelligenceSystem()
        self.test_cases = []
        
    def add_test_case(self, name: str, pdf_paths: list, persona: str, job: str):
        """Add a test case."""
        self.test_cases.append({
            'name': name,
            'pdf_paths': pdf_paths,
            'persona': persona,
            'job': job
        })
    
    def add_challenge_test(self, name: str, input_file: str, pdf_directory: str):
        """Add a test case with JSON challenge input format."""
        self.test_cases.append({
            'name': name,
            'input_file': input_file,
            'pdf_directory': pdf_directory,
            'format': 'json'
        })
    
    def run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case."""
        print(f"\n{'='*60}")
        print(f"Running test case: {test_case['name']}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            
            if test_case.get('format') == 'json':
                # New JSON format
                result = self.system.process_challenge(
                    test_case['input_file'],
                    test_case['pdf_directory']
                )
                
                # Validate input files from JSON
                with open(test_case['input_file'], 'r') as f:
                    challenge_data = json.load(f)
                
                pdf_paths = []
                for doc_info in challenge_data.get('documents', []):
                    filename = doc_info.get('filename', '')
                    if filename.endswith('.pdf'):
                        pdf_path = os.path.join(test_case['pdf_directory'], filename)
                        if os.path.exists(pdf_path):
                            pdf_paths.append(pdf_path)
                
                valid, message = validate_input_files(pdf_paths)
                if not valid:
                    return {
                        'name': test_case['name'],
                        'status': 'FAILED',
                        'error': message,
                        'processing_time': 0
                    }
                
            else:
                # Legacy format
                valid, message = validate_input_files(test_case['pdf_paths'])
                if not valid:
                    return {
                        'name': test_case['name'],
                        'status': 'FAILED',
                        'error': message,
                        'processing_time': 0
                    }
                
                result = self.system.process_documents(
                    test_case['pdf_paths'],
                    test_case['persona'],
                    test_case['job']
                )
            
            processing_time = time.time() - start_time
            
            # Validate output
            validation_result = self.validate_output(result)
            
            print(f"✅ Test completed successfully in {processing_time:.2f} seconds")
            print(f"📊 Sections extracted: {len(result['extracted_sections'])}")
            print(f"📋 Subsections analyzed: {len(result['subsection_analysis'])}")
            
            if test_case.get('format') == 'json':
                print(f"🎯 Challenge ID: {result['metadata'].get('challenge_id', 'N/A')}")
                print(f"📝 Test Case: {result['metadata'].get('test_case_name', 'N/A')}")
            
            return {
                'name': test_case['name'],
                'status': 'PASSED',
                'result': result,
                'processing_time': processing_time,
                'validation': validation_result
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Test failed: {str(e)}")
            return {
                'name': test_case['name'],
                'status': 'FAILED',
                'error': str(e),
                'processing_time': processing_time
            }
    
    def validate_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the output format and content."""
        validation = {
            'format_valid': True,
            'content_valid': True,
            'warnings': []
        }
        
        # Check required fields
        required_fields = ['metadata', 'extracted_sections', 'subsection_analysis']
        for field in required_fields:
            if field not in result:
                validation['format_valid'] = False
                validation['warnings'].append(f"Missing required field: {field}")
        
        # Check metadata
        if 'metadata' in result:
            metadata_fields = ['input_documents', 'persona', 'job_to_be_done', 'processing_timestamp']
            for field in metadata_fields:
                if field not in result['metadata']:
                    validation['warnings'].append(f"Missing metadata field: {field}")
        
        # Check sections
        if 'extracted_sections' in result:
            for i, section in enumerate(result['extracted_sections']):
                section_fields = ['document', 'page_number', 'section_title', 'importance_rank']
                for field in section_fields:
                    if field not in section:
                        validation['warnings'].append(f"Section {i} missing field: {field}")
        
        # Check processing time constraint
        if 'metadata' in result and 'processing_time_seconds' in result['metadata']:
            if result['metadata']['processing_time_seconds'] > 60:
                validation['content_valid'] = False
                validation['warnings'].append("Processing time exceeded 60 seconds")
        
        return validation
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases."""
        print("🚀 Starting Document Intelligence System Tests")
        print(f"Total test cases: {len(self.test_cases)}")
        
        results = []
        passed = 0
        failed = 0
        total_time = 0
        
        for test_case in self.test_cases:
            result = self.run_single_test(test_case)
            results.append(result)
            
            if result['status'] == 'PASSED':
                passed += 1
            else:
                failed += 1
            
            total_time += result['processing_time']
        
        # Generate summary
        summary = {
            'total_tests': len(self.test_cases),
            'passed': passed,
            'failed': failed,
            'total_processing_time': total_time,
            'average_processing_time': total_time / len(self.test_cases) if self.test_cases else 0,
            'results': results
        }
        
        self.print_summary(summary)
        return summary
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"⏱️  Total Processing Time: {summary['total_processing_time']:.2f} seconds")
        print(f"📈 Average Processing Time: {summary['average_processing_time']:.2f} seconds")
        
        if summary['failed'] > 0:
            print("\n❌ Failed Tests:")
            for result in summary['results']:
                if result['status'] == 'FAILED':
                    print(f"  - {result['name']}: {result.get('error', 'Unknown error')}")
        
        print(f"\n{'='*60}")

def create_sample_test_cases():
    """Create sample test cases for demonstration."""
    runner = TestRunner()
    
    # Test Case 1: Travel Planner (New JSON format)
    runner.add_challenge_test(
        name="Travel Planner - South of France",
        input_file="sample_travel_input.json",
        pdf_directory="input/pdfs"
    )
    
    # Test Case 2: Academic Research (Legacy format)
    runner.add_test_case(
        name="Academic Research - Graph Neural Networks",
        pdf_paths=[],  # Would be populated with actual PDF paths
        persona="PhD Researcher in Computational Biology with expertise in machine learning applications for drug discovery. Focuses on graph neural networks, molecular representation learning, and benchmarking methodologies.",
        job="Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks for graph neural networks in drug discovery applications."
    )
    
    # Test Case 3: Business Analysis (Legacy format)
    runner.add_test_case(
        name="Business Analysis - Tech Companies",
        pdf_paths=[],  # Would be populated with actual PDF paths
        persona="Investment Analyst specializing in technology sector companies. Focuses on financial performance, market positioning, and growth strategies.",
        job="Analyze revenue trends, R&D investments, and market positioning strategies from annual reports of competing tech companies."
    )
    
    return runner

def main():
    """Main test execution."""
    if len(sys.argv) < 2:
        print("Usage: python test_runner.py <test_data_directory> [--json-format]")
        print("The test data directory should contain:")
        print("  For JSON format: challenge_input.json files with corresponding PDF directories")
        print("  For legacy format: subdirectories with PDFs, persona.txt, and job.txt files")
        sys.exit(1)
    
    test_data_dir = sys.argv[1]
    json_format = "--json-format" in sys.argv
    
    if not os.path.exists(test_data_dir):
        print(f"Test data directory not found: {test_data_dir}")
        sys.exit(1)
    
    runner = TestRunner()
    
    if json_format:
        # Scan for JSON challenge files
        for item in os.listdir(test_data_dir):
            if item.endswith('.json'):
                json_path = os.path.join(test_data_dir, item)
                
                # Try to find corresponding PDF directory
                base_name = item.replace('.json', '')
                pdf_dir_candidates = [
                    os.path.join(test_data_dir, base_name),
                    os.path.join(test_data_dir, 'pdfs'),
                    os.path.join(test_data_dir, f"{base_name}_pdfs"),
                    test_data_dir  # PDFs in same directory
                ]
                
                pdf_directory = None
                for candidate in pdf_dir_candidates:
                    if os.path.exists(candidate):
                        # Check if it contains PDF files
                        pdf_files = [f for f in os.listdir(candidate) if f.endswith('.pdf')]
                        if pdf_files:
                            pdf_directory = candidate
                            break
                
                if pdf_directory:
                    runner.add_challenge_test(
                        name=f"Challenge: {base_name}",
                        input_file=json_path,
                        pdf_directory=pdf_directory
                    )
                else:
                    print(f"Warning: No PDF directory found for {item}")
    else:
        # Legacy format - scan for test case directories
        for test_dir in os.listdir(test_data_dir):
            test_path = os.path.join(test_data_dir, test_dir)
            if os.path.isdir(test_path):
                pdf_files = [os.path.join(test_path, f) for f in os.listdir(test_path) if f.endswith('.pdf')]
                persona_file = os.path.join(test_path, 'persona.txt')
                job_file = os.path.join(test_path, 'job.txt')
                
                if pdf_files and os.path.exists(persona_file) and os.path.exists(job_file):
                    with open(persona_file, 'r') as f:
                        persona = f.read().strip()
                    with open(job_file, 'r') as f:
                        job = f.read().strip()
                    
                    runner.add_test_case(test_dir, pdf_files, persona, job)
    
    if not runner.test_cases:
        print("No valid test cases found in the test data directory.")
        if json_format:
            print("For JSON format, ensure you have:")
            print("  - JSON challenge files (*.json)")
            print("  - Corresponding PDF directories with the same base name")
            print("Example structure:")
            print("  test_data/")
            print("  ├── travel_planner.json")
            print("  ├── travel_planner/")
            print("  │   ├── South of France - Cities.pdf")
            print("  │   └── South of France - Cuisine.pdf")
        else:
            print("For legacy format, each test case should be a subdirectory containing:")
            print("  - One or more PDF files")
            print("  - persona.txt (persona description)")
            print("  - job.txt (job description)")
        sys.exit(1)
    
    # Run all tests
    summary = runner.run_all_tests()
    
    # Save results
    results_file = 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if summary['failed'] == 0 else 1)

if __name__ == "__main__":
    main()