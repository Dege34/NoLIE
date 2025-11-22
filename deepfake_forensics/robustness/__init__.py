"""
Robustness testing and adversarial defense methods.
"""

from .attacks import FGSMAttack, PGDAttack, CWAttack
from .defenses import AdversarialTraining, TTAEnsemble, FrequencyPreprocessing
from .evaluation import RobustnessEvaluator, CompressionTest, NoiseTest

__all__ = [
    # Attacks
    "FGSMAttack",
    "PGDAttack", 
    "CWAttack",
    # Defenses
    "AdversarialTraining",
    "TTAEnsemble",
    "FrequencyPreprocessing",
    # Evaluation
    "RobustnessEvaluator",
    "CompressionTest",
    "NoiseTest",
]
