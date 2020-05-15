#!/usr/bin/env python
# -*- coding: utf-8 -*-

import spacy, json

from deepnlpf.core.iplugin import IPlugin
from deepnlpf.core.boost import Boost

class Plugin(IPlugin):

    def __init__(self, id_pool, lang, document, pipeline):
        self._id_pool = id_pool
        self._document = document
        self._pipeline = pipeline

        self.nlp = spacy.load('en_core_web_sm')

    def run(self):
        annotation = Boost().multithreading(
            self.wrapper, self._document['sentences'])
        return self.out_format(annotation)

    def wrapper(self, sentence):
        doc = self.nlp(sentence) #generated document spacy.

        data_tokens_list = []

        # Analisys in nivel token.
        for idx, token in enumerate(doc):
            data_token = {}

            data_token['idx'] = idx
            data_token['text'] = token.text

            if "pos" in self._pipeline:
                data_token['pos'] = token.pos_
            if "tag" in self._pipeline:
                data_token['tag'] = token.tag_
            if "shape" in self._pipeline:
                data_token['shape'] = token.shape_
            if "is_alpha" in self._pipeline:
                data_token['is_alpha'] = token.is_alpha
            if "is_title" in self._pipeline:
                data_token['is_title'] = token.is_title
            if "like_num" in self._pipeline:
                data_token['like_num'] = token.like_num
            
            data_tokens_list.append(data_token)
        
        list_chunks = list()
        if "noun_chunks" in self._pipeline:
            for chunk in doc.noun_chunks:
                data_chunk = {}
                data_chunk['text'] = chunk.text
                data_chunk['root_text'] = chunk.root.text
                data_chunk['root_dep_'] = chunk.root.dep_
                data_chunk['root_head_text'] = chunk.root.head.text
                json_data_chunk = data_chunk
                
                list_chunks.append(json_data_chunk)
                #print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)

        data = {}
        data['sentence'] = sentence
        data['tokens'] = data_tokens_list
        if "noun_chunks" in self._pipeline:
            data['noun_chunks'] = list_chunks

        json_data_result = json.loads(json.dumps(data))

        return json_data_result

    def out_format(self, doc):
        pass
