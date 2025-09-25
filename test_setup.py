"""
Test script to verify RAG system setup
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pinecone
        print("✅ Pinecone imported successfully")
    except ImportError as e:
        print(f"❌ Pinecone import failed: {e}")
        return False
    
    try:
        import google.generativeai
        print("✅ Google GenerativeAI imported successfully")
    except ImportError as e:
        print(f"❌ Google GenerativeAI import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ Sentence Transformers imported successfully")
    except ImportError as e:
        print(f"❌ Sentence Transformers import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    return True

def test_env_file():
    """Test if .env file exists and has required keys"""
    print("\nTesting environment configuration...")
    
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        return False
    
    load_dotenv()
    
    required_keys = ['PINECONE_API_KEY', 'PINECONE_ENVIRONMENT', 'GEMINI_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        value = os.getenv(key)
        if not value or value.startswith('your_'):
            missing_keys.append(key)
        else:
            print(f"✅ {key} is set")
    
    if missing_keys:
        print(f"❌ Missing or placeholder values for: {', '.join(missing_keys)}")
        return False
    
    return True

def test_modules():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")
    
    try:
        from pdf_processor import PDFProcessor
        print("✅ PDFProcessor imported successfully")
    except ImportError as e:
        print(f"❌ PDFProcessor import failed: {e}")
        return False
    
    try:
        from embeddings import EmbeddingGenerator
        print("✅ EmbeddingGenerator imported successfully")
    except ImportError as e:
        print(f"❌ EmbeddingGenerator import failed: {e}")
        return False
    
    try:
        from vector_store import PineconeVectorStore
        print("✅ PineconeVectorStore imported successfully")
    except ImportError as e:
        print(f"❌ PineconeVectorStore import failed: {e}")
        return False
    
    try:
        from llm_client import GeminiLLMClient
        print("✅ GeminiLLMClient imported successfully")
    except ImportError as e:
        print(f"❌ GeminiLLMClient import failed: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🧪 Testing RAG Document Assistant Setup")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test environment
    env_ok = test_env_file()
    
    # Test custom modules
    modules_ok = test_modules()
    
    print("\n" + "=" * 50)
    
    if imports_ok and env_ok and modules_ok:
        print("✅ All tests passed! Your RAG system is ready to use.")
        print("\nTo start the application, run:")
        print("streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above and fix them.")
        
        if not imports_ok:
            print("\n📦 To fix import issues, run:")
            print("pip install -r requirements.txt")
        
        if not env_ok:
            print("\n🔑 To fix environment issues:")
            print("1. Make sure .env file exists")
            print("2. Add your actual API keys (not placeholder values)")
        
        if not modules_ok:
            print("\n📁 To fix module issues:")
            print("1. Make sure all Python files are in the same directory")
            print("2. Check for syntax errors in the module files")

if __name__ == "__main__":
    main()
