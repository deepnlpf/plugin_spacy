#!/bin/bash

echo "Install SpaCy.."
pip install -U spacy

echo "Install Model Lang En.."
python -m spacy download en_core_web_sm
