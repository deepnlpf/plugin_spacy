#!/bin/bash

echo "Install SpaCy.."
pip install -U spacy

echo "Install Model Lang En.."
python -m import stanza stanza.download en
