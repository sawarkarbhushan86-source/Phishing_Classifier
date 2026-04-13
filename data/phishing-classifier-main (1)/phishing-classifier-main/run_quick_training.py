#!/usr/bin/env python
"""Direct execution wrapper for optimized training"""
import os
import sys

# Navigate to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Add to path
sys.path.insert(0, os.getcwd())

# Import and run
from train_optimized import execute_training_pipeline
import argparse

class Args:
    mode = 'quick'
    skip_deep_learning = True
    skip_shap = False
    verbose = True
    no_cache = False

args = Args()
success = execute_training_pipeline(args)
sys.exit(0 if success else 1)
