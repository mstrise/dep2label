from dep2label.encodings import * 
from dep2label.postprocessing import *
import codecs


class Labeler:

    def __init__(self):   
        self.index_of_sentence = 1
        self.index_to_decode = 1

    def encode(self, file2encode, output, enc, mtl):
        sentence = {}
        encoded_sentences ={}
       
        with codecs.open(file2encode) as f2label:
            lines = f2label.readlines()

        for line in lines:
            if not line == '\n':
                dummy_root = {"id": 0, "word": "ROOT",
                         "lemma": "ROOT","pos":"ROOT", "head": 0, "deprel": "root"}
                sentence.update({0: dummy_root})
                field = line.split("\t")
                if "." in field[0] or "-" in field[0]:
                    continue
                elif field[0].isdigit():
                    word_id = int(field[0])
                    if enc == "zero":
                        word = {"id":int(field[0]), "word": field[1], "lemma":field[2], "pos":field[3], "head":field[6], "deprel":field[7]}
                    else:
                        word = {"id":int(field[0]), "word": field[1], "lemma":field[2], "pos":field[3], "head":int(field[6]), "deprel":field[7]}
                    sentence.update({word_id: word})
            else:
                words_with_labels = {}
                if enc == "rel-pos":
                    if mtl == None:
                        task="2-task-combined"
                    else:
                        task = mtl
                    rel = RelPosEncoding()
                    words_with_labels = rel.encode(sentence, task)
                elif enc == "1-planar-brackets":
                    if mtl == None:
                        task="2-task"
                    else:
                        task = mtl
                    brackets = Bracketing1pEncoding()
                    words_with_labels = brackets.encode(sentence, task)
                elif enc =="2-planar-brackets-greedy":
                    if mtl == None:
                        task="3-task"
                    else:
                        task = mtl
                    brackets_2p = Bracketing2pGreedyEncoding()
                    words_with_labels = brackets_2p.encode(sentence, task)
                elif enc == "2-planar-brackets-propagation":
                    if mtl == None:
                        task="3-task"
                    else:
                        task = mtl
                    brackets_2p = Bracketing2pPropagationEncoding()
                    words_with_labels = brackets_2p.encode(sentence, task)
                elif enc == "arc-standard":
                    if mtl == None:
                        task="2-task"
                    else:
                        task = mtl
                    transition = ArcStandardEncoding()
                    words_with_labels = transition.encode(sentence, task)
                elif enc == "arc-eager":
                    if mtl == None:
                        task="2-task"
                    else:
                        task = mtl
                    transition = ArcEagerEncoding()
                    words_with_labels = transition.encode(sentence, task)
                elif enc == "arc-hybrid":
                    if mtl == None:
                        task="2-task"
                    else:
                        task = mtl
                    transition = ArcHybridEncoding()
                    words_with_labels = transition.encode(sentence, task)
                elif enc == "covington":
                    if mtl == None:
                        task="2-task"
                    else:
                        task = mtl
                    transition = CovingtonEncoding()
                    words_with_labels = transition.encode(sentence, task)
                elif enc =="zero":
                    if mtl == None:
                        task="2-task-combined"
                    else:
                        task = mtl
                    z = ZeroEncoding()
                    words_with_labels = z.encode(sentence, task)
                else:
                    print("Invalid encoding")
                
                encoded_sentences.update({self.index_of_sentence: words_with_labels})
                self.index_of_sentence += 1
                sentence = {}        
        f2label.close()
        p = CoNLLPostProcessor()
        p.write_to_file(encoded_sentences, output)


    def decode(self, file2decode, output, enc, conllu):
        p = CoNLLPostProcessor()
        l = LabelPostProcessor()
        sentences = l.separate_labels(file2decode)
        decoded_sentences = {}
        nb_of_sentence = 1
        
        for sent_id in sentences:
            word_index = 1
            sentence = sentences.get(sent_id)
            sentence_with_sep_labels = {}
            sentence_with_sep_labels[0] = ["ROOT", "ROOT", 0, "root", "ROOT"]
            
            for word in sentence:
                sentence_with_sep_labels.update({word_index: word})
                word_index += 1
        
            if enc == "rel-pos":
                rel = RelPosEncoding()
                decoded_words, headless_words = rel.decode(sentence_with_sep_labels)
            elif enc == "1-planar-brackets":
                bracketing = Bracketing1pEncoding()
                decoded_words, headless_words = bracketing.decode(sentence_with_sep_labels)
            elif enc == "2-planar-brackets-greedy":
                brackets_2p = Bracketing2pGreedyEncoding()
                decoded_words, headless_words = brackets_2p.decode(sentence_with_sep_labels)
            elif enc == "2-planar-brackets-propagation":
                brackets_2p = Bracketing2pPropagationEncoding()
                decoded_words, headless_words = brackets_2p.decode(sentence_with_sep_labels)
            elif enc== "arc-standard":
                transition = ArcStandardEncoding()
                decoded_words, headless_words = transition.decode(sentence_with_sep_labels)
            elif enc== "arc-eager":
                transition = ArcEagerEncoding()
                decoded_words, headless_words = transition.decode(sentence_with_sep_labels)
            elif enc== "arc-hybrid":
                transition = ArcHybridEncoding()
                decoded_words, headless_words = transition.decode(sentence_with_sep_labels)
            elif enc== "covington":
                transition = CovingtonEncoding()
                decoded_words, headless_words = transition.decode(sentence_with_sep_labels)

            
            # POSTPROCESSING
            if not l.has_root(decoded_words):
                l.find_candidates_for_root(decoded_words, headless_words)
            if not l.has_root(decoded_words):
                l.assign_root_index_one(decoded_words, headless_words)
            if l.has_multiple_roots(decoded_words):
                l.choose_root_from_multiple_candidates(decoded_words)
            l.assign_headless_words_to_root(decoded_words, headless_words)
            l.find_cycles(decoded_words)
           
            p.convert_labels2conllu(decoded_words)
            decoded_sentences.update({nb_of_sentence: decoded_words})
            nb_of_sentence += 1

        if conllu == None:
            p.write_to_file(decoded_sentences, output)
        else:
            lookup = p.dump_into_lookup(conllu) 
            p.write_to_file(decoded_sentences, output)
            p.merge_lookup(output, lookup)
