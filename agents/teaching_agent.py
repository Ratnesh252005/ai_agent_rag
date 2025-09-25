# agents/teaching_agent.py
"""
Teaching Mode Agent - Enhanced with Direct Gemini Fallback
"""
from typing import Dict, Any, List, Optional
import logging
import random
import json
import os

logger = logging.getLogger(__name__)


def _call_llm_direct_gemini(prompt: str, max_tokens: int = 512) -> str:
    """Direct Gemini API call as reliable fallback"""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.warning("No GEMINI_API_KEY found in environment")
            return ""
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content with safety settings for educational content
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.3,
            )
        )
        
        return response.text if response.text else ""
        
    except Exception as e:
        logger.error(f"Direct Gemini call failed: {e}")
        return ""


def _find_llm_callable():
    """
    Try to import llm_client and return the first callable we can find.
    """
    try:
        import llm_client
    except Exception:
        return None

    candidates = ['generate_answer', 'generate', 'call_llm', 'generate_text',
                  'generate_response', 'query', 'run', 'chat']
    for name in candidates:
        fn = getattr(llm_client, name, None)
        if callable(fn):
            return fn
    
    # Try GeminiLLMClient class approach
    if hasattr(llm_client, 'GeminiLLMClient'):
        try:
            gemini_key = os.getenv('GEMINI_API_KEY')
            if gemini_key:
                client = llm_client.GeminiLLMClient(gemini_key)
                client.initialize_gemini()
                return lambda prompt, **kwargs: client.generate_text(prompt)
        except Exception as e:
            logger.warning(f"Could not initialize GeminiLLMClient: {e}")
    
    return None


def _call_llm(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """Enhanced LLM calling with Gemini fallback"""
    
    # First try your existing llm_client
    fn = _find_llm_callable()
    if fn is not None:
        try:
            # Try different calling patterns
            result = fn(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            if isinstance(result, dict) and 'text' in result:
                text_result = result['text']
                if text_result and len(text_result.strip()) > 10:
                    return text_result
            elif isinstance(result, str) and len(result.strip()) > 10:
                return result
            else:
                result_str = str(result)
                if len(result_str.strip()) > 10:
                    return result_str
        except TypeError:
            try:
                result = fn(prompt)
                if isinstance(result, dict) and 'text' in result:
                    text_result = result['text']
                    if text_result and len(text_result.strip()) > 10:
                        return text_result
                elif isinstance(result, str) and len(result.strip()) > 10:
                    return result
            except Exception as e:
                logger.debug(f"Secondary LLM call attempt failed: {e}")
        except Exception as e:
            logger.debug(f"Primary LLM call failed: {e}")
    
    # Fallback to direct Gemini call
    logger.info("Using direct Gemini fallback")
    return _call_llm_direct_gemini(prompt, max_tokens)


def _assemble_explain_from_structured(structured: Dict[str, Any]) -> str:
    """Create a structured explanation from the data"""
    parts = []
    topic = structured.get("topic") or structured.get("title") or "Topic"
    parts.append(f"# {topic}\n")
    
    # Extract key information from context
    context = structured.get("context", "")
    if context:
        parts.append("## Overview\n")
        # Take first few sentences as overview
        sentences = context.split('. ')[:3]
        parts.append('. '.join(sentences) + ".\n")
    
    if d := structured.get("definition"):
        parts.append("## Definition\n" + d + "\n")
    
    if steps := structured.get("steps") or structured.get("conditions") or structured.get("subtopics"):
        parts.append("## Key Points\n")
        for i, s in enumerate(steps, 1):
            parts.append(f"{i}. {s}")
        parts.append("\n")
    
    # Extract examples from context
    if context and ("example" in context.lower() or "such as" in context.lower()):
        parts.append("## Examples\n")
        parts.append("Based on the content, here are key examples mentioned:\n")
        # Simple extraction of example-related sentences
        example_sentences = [s for s in context.split('.') if 'example' in s.lower() or 'such as' in s.lower()]
        for ex in example_sentences[:2]:
            parts.append(f"- {ex.strip()}\n")
    
    if summary := structured.get("summary"):
        parts.append("\n## Summary\n" + summary + "\n")
    
    return "\n".join(parts)


def explain_mode(structured: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced explain mode with better content generation"""
    
    # Build a comprehensive prompt
    context = structured.get("context", "")
    topic = structured.get("topic", "the topic")
    
    prompt = f"""You are an expert teacher. Create a clear, educational explanation about "{topic}".

Use this content from the document:
{context[:1500]}

Structure your response as follows:
1. **Definition**: Start with a clear definition
2. **Key Concepts**: Break down the main ideas (use bullet points)
3. **How it Works**: Explain the process or mechanism
4. **Real Example**: Give a practical example
5. **Summary**: Conclude with key takeaways

Write in a clear, educational style suitable for students. Use markdown formatting."""
    
    llm_out = _call_llm(prompt, max_tokens=1000, temperature=0.1)
    if llm_out and len(llm_out.strip()) > 20:
        return {"mode": "explain", "content": llm_out.strip()}
    
    # Enhanced fallback
    fallback_content = _assemble_explain_from_structured(structured)
    return {"mode": "explain", "content": fallback_content}


def summary_mode(structured: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced summary mode"""
    lines = int(options.get("summary_lines", 5))
    context = structured.get("context", "")
    topic = structured.get("topic", "the topic")
    
    prompt = f"""Create a concise {lines}-point summary about "{topic}" using this content:

{context[:1000]}

Format as exactly {lines} bullet points, each being one complete sentence.
Focus on the most important concepts and key information."""
    
    llm_out = _call_llm(prompt, max_tokens=300)
    if llm_out and len(llm_out.strip()) > 10:
        return {"mode": "summary", "content": llm_out.strip()}
    
    # Fallback: extract key sentences from context
    sentences = [s.strip() for s in structured.get("context", "").split('.') if len(s.strip()) > 20]
    key_sentences = sentences[:lines]
    formatted_summary = "\n".join([f"• {sentence}." for sentence in key_sentences])
    
    if not formatted_summary:
        formatted_summary = f"• Summary of {topic} based on available content.\n• Key information extracted from document context.\n• Please refer to the original document for complete details."
    
    return {"mode": "summary", "content": formatted_summary}


def _create_better_quiz_from_context(context: str, topic: str, n: int = 3) -> List[Dict[str, Any]]:
    """Create better quiz questions from context when LLM fails"""
    
    quiz = []
    sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 30]
    
    if not sentences:
        # Ultimate fallback with generic questions
        return [{
            "id": 1,
            "question": f"What is the main topic of this content about {topic}?",
            "options": [
                f"Information about {topic}",
                f"Unrelated content",
                f"Different subject matter",
                f"Technical documentation"
            ],
            "answer": f"Information about {topic}"
        }]
    
    # Question templates
    templates = [
        "What is the main purpose of {topic}?",
        "Which of the following best describes {topic}?",
        "What are the key characteristics of {topic}?"
    ]
    
    for i in range(min(n, len(templates))):
        question_text = templates[i].format(topic=topic)
        
        # Find relevant sentence from context
        correct_answer = sentences[i % len(sentences)] if sentences else f"Key information about {topic}"
        # Truncate long answers
        if len(correct_answer) > 100:
            correct_answer = correct_answer[:97] + "..."
        
        # Create distractors
        distractors = [
            f"This is not related to {topic}",
            f"This contradicts {topic}",
            f"Incomplete information about {topic}"
        ]
        
        # Combine options
        all_options = [correct_answer] + distractors[:3]
        
        quiz.append({
            "id": i + 1,
            "question": question_text,
            "options": all_options,
            "answer": correct_answer
        })
    
    return quiz


def quiz_mode(structured: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced quiz mode with better question generation"""
    
    num_questions = int(options.get("num_questions", 3))  # Reduced default for reliability
    context = structured.get("context", "")
    topic = structured.get("topic", "the topic")
    
    # Enhanced prompt for better quiz generation
    prompt = f"""Create {num_questions} multiple-choice questions about "{topic}" based on this content:

{context[:1200]}

Return ONLY valid JSON in this exact format:
[
    {{
        "question": "Clear question about the topic?",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "answer": "The correct option text"
    }}
]

Make questions test understanding of key concepts from the content provided. Ensure options are distinct and the answer matches one of the options exactly."""
    
    llm_out = _call_llm(prompt, max_tokens=800, temperature=0.3)
    
    # Try to parse LLM output as JSON
    if llm_out:
        try:
            # Clean the output - remove any markdown formatting
            json_text = llm_out.strip()
            
            # Remove code blocks if present
            if json_text.startswith('```'):
                lines = json_text.split('\n')
                json_text = '\n'.join(lines[1:])
            if json_text.endswith('```'):
                json_text = json_text.rsplit('```', 1)[0]
            
            # Try to find JSON array in the text
            start_idx = json_text.find('[')
            end_idx = json_text.rfind(']')
            if start_idx != -1 and end_idx != -1:
                json_text = json_text[start_idx:end_idx+1]
            
            parsed = json.loads(json_text)
            
            # Validate the parsed data
            if isinstance(parsed, list) and len(parsed) >= 1:
                valid_quiz = []
                for item in parsed:
                    if (isinstance(item, dict) and 
                        'question' in item and 
                        'options' in item and 
                        'answer' in item):
                        
                        options = item['options'] if isinstance(item['options'], list) else [str(item['options'])]
                        answer = item['answer']
                        
                        # Ensure answer is in options (basic validation)
                        if answer not in options:
                            # Try to find similar option
                            for opt in options:
                                if answer.lower() in opt.lower() or opt.lower() in answer.lower():
                                    answer = opt
                                    break
                        
                        valid_quiz.append({
                            "id": len(valid_quiz) + 1,
                            "question": item['question'],
                            "options": options[:4],  # Limit to 4 options
                            "answer": answer
                        })
                
                if valid_quiz:
                    logger.info(f"Generated {len(valid_quiz)} quiz questions via LLM")
                    return {"mode": "quiz", "quiz": valid_quiz}
                    
        except json.JSONDecodeError as e:
            logger.debug(f"LLM quiz output not valid JSON: {e}")
        except Exception as e:
            logger.debug(f"Error parsing quiz JSON: {e}")
    
    # Enhanced fallback
    logger.info("Using fallback quiz generation")
    fallback_quiz = _create_better_quiz_from_context(context, topic, num_questions)
    return {"mode": "quiz", "quiz": fallback_quiz}


def teach(structured: Dict[str, Any], mode: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main teaching function with enhanced error handling and logging
    """
    options = options or {}
    mode = (mode or structured.get("intent") or "explain").lower()
    
    logger.info(f"Teaching mode: {mode}, Topic: {structured.get('topic', 'Unknown')}")
    
    try:
        if mode == "explain":
            result = explain_mode(structured, options)
        elif mode == "summary":
            result = summary_mode(structured, options)
        elif mode == "quiz":
            result = quiz_mode(structured, options)
        else:
            # Unknown mode => default to explain
            logger.warning(f"Unknown teaching mode '{mode}', defaulting to explain")
            result = explain_mode(structured, options)
        
        # Validate result has content
        if mode == "quiz":
            quiz_count = len(result.get("quiz", []))
            logger.info(f"Quiz mode returned {quiz_count} questions")
        else:
            content_length = len(result.get("content", ""))
            logger.info(f"{mode} mode returned {content_length} characters")
        
        return result
            
    except Exception as e:
        logger.exception(f"Error in teaching mode {mode}: {e}")
        # Return safe fallback
        context = structured.get('context', 'No content available')
        topic = structured.get('topic', 'the topic')
        
        if mode == "quiz":
            return {
                "mode": mode,
                "quiz": [{
                    "id": 1,
                    "question": f"What can you tell me about {topic}?",
                    "options": [
                        f"Information about {topic}",
                        "No relevant information",
                        "Different topic entirely",
                        "Technical documentation"
                    ],
                    "answer": f"Information about {topic}"
                }]
            }
        else:
            return {
                "mode": mode,
                "content": f"I encountered an issue generating {mode} content. Here's what I can tell you about {topic}:\n\n{context[:500]}..."
            }