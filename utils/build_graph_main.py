#!/usr/bin/env python

# reference: athena-decode



from absl import logging
from graph.graph_builder import GraphBuilder
import json
import sys



graph_conf = {
        "graph_type":"LG",
        "sil_symbol":"<space>",
        "input_lexicon_file": "corpus/aishell1/lm/dict/lexicon.txt",
        "input_graphemes_file": "corpus/aishell1/lm/dict/units.txt",  # vocab
        "input_grammar_file": "corpus/aishell1/lm/lm.arpa",
        "output_disambig_graphemes_file":"corpus/aishell1/lm/graph/graphemes_disambig.txt", 
        "output_words_file": "corpus/aishell1/lm/graph/words.txt",
        "output_graph_file": "corpus/aishell1/lm/graph/LG.fst"
}


def build_graph(graph_type, input_lexicon_file, input_graphemes_file, input_grammar_file, sil_symbol,
        output_disambig_graphemes_file='graphemes_disambig.txt',
        output_words_file='words.txt',
        output_graph_file='LG.fst'):
    if graph_type == 'LG':
        logging.info('start to build  LG graph')
    elif graph_type == 'TLG':
        logging.info('start to build  TLG graph')
    graph_builder = GraphBuilder(graph_type)
    graph_builder.make_graph(
            input_lexicon_file,
            input_graphemes_file,
            input_grammar_file,
            sil_symbol,
            output_disambig_graphemes_file,
            output_words_file,
            output_graph_file)

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    # if len(sys.argv) < 2:
    #     logging.warning('Usage: python {} config_json_file'.format(sys.argv[0]))
    #     sys.exit()
    # json_file = sys.argv[1]
    # config = None
    # with open(json_file) as f:
    #     config = json.load(f)
    build_graph(graph_conf['graph_type'],
            graph_conf['input_lexicon_file'],
            graph_conf['input_graphemes_file'],
            graph_conf['input_grammar_file'],
            graph_conf['sil_symbol'],
            graph_conf['output_disambig_graphemes_file'],
            graph_conf['output_words_file'],
            graph_conf['output_graph_file'])
