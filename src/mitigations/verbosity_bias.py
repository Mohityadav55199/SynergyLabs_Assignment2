import re
from typing import Dict, Any

class VerbosityMitigator:
    """
    Mitigates verbosity/length bias:
    1. Injects explicit anti-verbosity & anti-fluff instructions into judge prompts.
    2. Detects vacuous padding and applies length normalization penalties.
    """

    ANTI_VERBOSITY_SYSTEM_INSTRUCTION = (
        "CRITICAL INSTRUCTION - COMBAT VERBOSITY BIAS:\n"
        "- Do NOT reward an answer simply because it is long, verbose, or formatted with excess headings.\n"
        "- If an answer is concise, direct, and factually complete, award it the MAXIMUM score for clarity.\n"
        "- If an answer contains conversational filler, repetitive restatements, or empty padding, "
        "you MUST penalize the 'tone_and_conciseness' score severely.\n"
        "- Substance and precision always outweigh superficial length."
    )

    @staticmethod
    def calculate_length_penalty(text: str, reference_text: str = None) -> float:
        """
        Calculates a penalty deduction (0.0 to 0.8) if the response exhibits
        excessive padding, repetitive phrases, or empty fluff.
        """
        words = text.split()
        word_count = len(words)
        
        # Check for repetitive filler phrases
        fluff_phrases = [
            r"as an ai language model",
            r"it is important to remember that",
            r"in today's fast-paced world",
            r"delve into the fascinating",
            r"in conclusion.*both are remarkable",
            r"what an incredible and insightful question",
            r"allow me to elaborate extensively"
        ]
        
        fluff_count = 0
        text_lower = text.lower()
        for pattern in fluff_phrases:
            if re.search(pattern, text_lower):
                fluff_count += 1
                
        penalty = 0.0
        if fluff_count > 0:
            penalty += min(0.6, fluff_count * 0.3)
            
        # If response is over 500 words but has high repetition
        if word_count > 400:
            unique_words = len(set(w.lower() for w in words))
            lexical_diversity = unique_words / max(word_count, 1)
            if lexical_diversity < 0.40: # Low lexical diversity means repetitive loop
                penalty += 0.3

        return round(min(penalty, 1.0), 2)
