#!/bin/bash

echo "Install SpaCy.."
pip install -U spacy

echo "Install Model Lang En.."
python && import stanza && stanza.download('en')
