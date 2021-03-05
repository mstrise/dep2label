from dep2label.postprocessing import LabelPostProcessor

class RelPosEncoding():

    def encode(self,sentence, task):
        """[summary]
        
        Args:
            sentence [dict]: [sentence represented as {index_word: {"id":int,"word":str,"lemma":str,"pos":str,"head":int }}]

        Returns:
            words_with_labels [dict]: [word with its PoS and label]
        """
        words_with_labels = {}
        l = LabelPostProcessor()
        # combined 2-task label: x@x{}x 
        words_with_labels=l.tag_BOS(task, words_with_labels)

        for index_word in sentence:
            if not index_word == 0:
                word = sentence[index_word]
                head = word["head"]
                if index_word < head:
                    relative_position_head = 1
                    head_word = sentence[head]
                    postag_head = head_word["pos"]
                    for w in range(index_word + 1, head):
                        word_i = sentence[w]
                        postag_word_i = word_i["pos"]
                        if postag_word_i == postag_head:
                            relative_position_head += 1
                    
                    label = str("+" + repr(relative_position_head) + "@" + postag_head + "{}" + word["deprel"])
                    full_label = str(
                        word["word"] + "\t" + word["pos"] + "\t" + label)
                    words_with_labels.update({index_word: full_label})
                elif index_word > head:
                    relative_position_head = 1
                    head_word = sentence[head]
                    postag_head = head_word["pos"]
                    for w in range(head + 1, index_word):
                        word_i = sentence[w]
                        postag_word_i = word_i["pos"]
                        if postag_word_i == postag_head:
                            relative_position_head += 1
                    
                    label = str("-" + repr(relative_position_head) + "@" + postag_head + "{}" + word["deprel"])
                    full_label = str(word["word"] + "\t" + word["pos"] + "\t" + label)
                    words_with_labels.update({index_word: full_label})

        words_with_labels= l.tag_EOS(task, words_with_labels)
        return words_with_labels
       
    def decode(self,sentence):
        """[summary]

        Args:
            sentence [dict]: [int: ["word", "PoS", "rel.position", "deprel","head's PoS"]]

        Returns:
            decoded_words [dict]: [words with assigned head]
            unassigned_word [dict]: [words for which head assignment failed]
        """
        decoded_words = {} # 1 : ['The', 'DT', '+1', 'det', 'NN']
        unassigned_word = {}

        def assignHeadLeft(word_index, info_word, postag_head, decoded, abs_posit):
            count_posit = 0
            # find postag_head with the relative position -1,-2....
            for index in range(word_index - 1, -1, -1):
                word_candidate = decoded[index]
                postag_candidate = word_candidate[1]
                if postag_candidate == postag_head:
                    count_posit += 1
                    if abs_posit == count_posit:
                        head_word = {1: word_index, 2: info_word[0],
                                        3: "_", 4: info_word[1],
                                        5: index, 6: info_word[3]}
                        return head_word

        def assignHeadRight(word_index, info_word, postag_head, decoded, abs_posit):
            count_posit = 0
            # find postag_head with the relative position +1,+2....
            for index in range(word_index + 1, len(decoded)):
                word_candidate = decoded[index]
                postag_candidate = word_candidate[1]
                if postag_candidate == postag_head:
                    count_posit += 1
                    if abs_posit == count_posit:
                        head_word = {1: word_index, 2: info_word[0],
                                        3: "_", 4: info_word[1],
                                        5: index, 6: info_word[3]}
                        return head_word

        for word_index in sentence:
            if not word_index == 0:
                word_line = sentence.get(word_index)
                info_word = word_line
                found_head = False
                if not word_line[2] == "-EOS-" and not word_line[2] == "-BOS-":
                    rel_pos_head = int(word_line[2])
                    pos_head = word_line[4]
                    abs_posit = abs(rel_pos_head)
                    positminus1 = abs_posit - 1
                    positplus1 = abs_posit + 1
                    if rel_pos_head < 0:
                        head_word = assignHeadLeft(word_index, info_word, pos_head,
                                                    sentence, abs_posit)
                        if head_word:
                            decoded_words.update({word_index: head_word})
                            found_head = True

                        elif not positminus1 == 0:
                            head_word = assignHeadLeft(word_index, info_word, pos_head,
                                                        sentence, positminus1)
                            if head_word:
                                decoded_words.update({word_index: head_word})
                                found_head = True
                        else:
                            head_word = assignHeadLeft(word_index, info_word, pos_head,
                                                        sentence, positplus1)
                            if head_word:
                                found_head = True
                                decoded_words.update({word_index: head_word})

                        # find postag_head with the relative position +1,+2....
                    elif rel_pos_head > 0:
                        found_head = False
                        head_word = assignHeadRight(word_index, info_word, pos_head,
                                                    sentence, abs_posit)
                        if head_word:
                            decoded_words.update({word_index: head_word})
                            found_head = True

                        elif not positminus1 == 0:
                            head_word = assignHeadRight(word_index, info_word, pos_head,
                                                        sentence, positminus1)
                            if head_word:
                                decoded_words.update({word_index: head_word})
                                found_head = True
                        else:
                            head_word = assignHeadRight(word_index, info_word, pos_head,
                                                        sentence, positplus1)
                            if head_word:
                                decoded_words.update({word_index: head_word})
                                found_head = True
                    if not found_head:
                        head_word = {1: word_index, 2: info_word[0],
                                        3: "_", 4: info_word[1],
                                        5: -1, 6: info_word[3]}
                        unassigned_word.update({word_index: head_word})
                        
                        decoded_words.update({word_index: head_word})
                else:
                    head_word = {1: word_index, 2: info_word[0],
                                    3: "_", 4: info_word[1],
                                    5: -1, 6: "root"}
                    unassigned_word.update({word_index: head_word})
                    decoded_words.update({word_index: head_word})
        
        return decoded_words, unassigned_word 


class Bracketing1pEncoding(): 
    
    def encode(self,sentence, task):
        
        words_with_labels = {}
        arcs_in_plane = []
        plane =1
        l = LabelPostProcessor()
        l.tag_BOS(task, words_with_labels)

        for index_of_word in range(1, len(sentence), 1):
            info_of_a_word = sentence[index_of_word]
            head_index = info_of_a_word["head"]
            arcs_pair = [head_index, index_of_word]
            if not 0 in arcs_pair:
                arcs_in_plane.append(arcs_pair)
            
        indexed_brackets=index_labels_with_brackets(arcs_in_plane, plane)
        words_with_labels = generate_bracketing_labels(sentence, words_with_labels, indexed_brackets, None, plane)
        l.tag_EOS(task, words_with_labels)

        return words_with_labels
       
    def decode(self,sentence):
        decoded_words = {}
        unassigned_words={}
        stack1 = []
        stack2 = []
        for word_index in range(1, len(sentence), 1):
            word_info = sentence[word_index]
            word = {1: word_index, 2: word_info[0],
                            3: word_info[1],
                            4: word_info[1],
                            5: -1,
                            6: word_info[3]}
            decoded_words.update({word_index: word})
            unassigned_words.update({word_index: word})

        for word_index in range(1, len(sentence), 1):
            processed_word = sentence[word_index]
            if processed_word[3] == "root":
                word = {1: word_index, 2: processed_word[0],
                                3: processed_word[1],
                                4: processed_word[1],
                                5: 0,
                                6: processed_word[3]}
                decoded_words.update({word_index: word})
                if unassigned_words.__contains__(word_index):
                    unassigned_words.pop(word_index)

            char = list(processed_word[2])
            for x in char:
                if x == "<":
                    x += "_" + repr(word_index)
                    stack1.append(x)
                elif x == "R":
                    x += "_" + repr(word_index)
                    stack2.append(x)
                elif x == "L":
                    if not len(stack1) == 0:
                        dependent = stack1[-1]
                        dependent_index = int(dependent.split("_")[1])-1
                        if not dependent_index == 0:
                            info_about_dependent = sentence[dependent_index]
                            word = {1: dependent_index, 2: info_about_dependent[0],
                                            3: info_about_dependent[1],
                                            4: info_about_dependent[1],
                                            5: word_index,
                                            6: info_about_dependent[3]}
                            decoded_words.update({dependent_index: word})
                            if unassigned_words.__contains__(dependent_index):
                                unassigned_words.pop(dependent_index)
                            stack1 = stack1[:-1]
                elif x ==">":
                    if not len(stack2) == 0:
                        head = stack2[-1]
                        head_index = int(head.split("_")[1]) - 1
                        word = {1: word_index, 2: processed_word[0],
                                        3: processed_word[1],
                                        4: processed_word[1],
                                        5: head_index,
                                        6: processed_word[3]}
                        decoded_words.update({word_index: word})
                        if unassigned_words.__contains__(word_index):
                            unassigned_words.pop(word_index)
                        stack2 = stack2[:-1]
        return decoded_words, unassigned_words

class Bracketing2pGreedyEncoding(): 
    
    def encode(self,sentence, task):
        words_with_labels = {}
        plane_1 = []
        plane_2 = []
        l = LabelPostProcessor()
        l.tag_BOS(task, words_with_labels)
        
        for index_word_r in range(1, len(sentence), 1):        
            for index_word_l in range(index_word_r-1,0,-1):
                head_word_l = int(sentence[index_word_l]["head"])
                head_word_r = int(sentence[index_word_r]["head"])
                if head_word_l== index_word_r or head_word_r == index_word_l:
                    if head_word_l== index_word_r:
                        next_arc = (index_word_r,index_word_l)
                    elif head_word_r == index_word_l:
                        next_arc = (index_word_l,index_word_r)
                    p1_U_p2 = plane_1+plane_2
                    c= check_if_crossing_arcs(p1_U_p2, [next_arc])
                    if not list(set(c).intersection(set(plane_1))):     
                        plane_1.append(next_arc)
                    elif not list(set(c).intersection(set(plane_2))):
                        plane_2.append(next_arc)
                    
        indexed_brackets=index_labels_with_brackets(plane_1, 1)
        indexed_brackets_2p =index_labels_with_brackets(plane_2, 2)
        words_with_labels = generate_bracketing_labels(sentence, words_with_labels, indexed_brackets, indexed_brackets_2p, 2)                  
        l.tag_EOS(task, words_with_labels)

        return words_with_labels
       
    def decode(self,sentence):
        decoded_words, unassigned_words= decode_2p_brackets(sentence)
        return decoded_words, unassigned_words

class Bracketing2pPropagationEncoding(): 
   
    def encode(self,sentence, task):
        words_with_labels = {}
        plane_1 = []
        plane_2 = []
        plane_1_compl = []
        plane_2_compl = []
        l = LabelPostProcessor()
        l.tag_BOS(task, words_with_labels)
        
        def get_pair_arcs(sentence):
            t = []
            for index_word_r in range(1, len(sentence), 1):
                for index_word_l in range(index_word_r-1,0,-1):
                    head_word_l = int(sentence[index_word_l]["head"])
                    head_word_r = int(sentence[index_word_r]["head"])
                    if head_word_l== index_word_r or head_word_r == index_word_l:
                        if head_word_l== index_word_r:
                            next_arc = (index_word_r,index_word_l)
                        elif head_word_r == index_word_l:
                            next_arc = (index_word_l,index_word_r)
                        t.append(next_arc)
            return t
    
        def propagate(t, p1_compl, p2_compl, e, plane_i ):
            
            if plane_i == 1:
                p1_compl.append(e)
            elif plane_i == 2:
                p2_compl.append(e)      
            crossing_arcs = check_if_crossing_arcs(t, [e])
        
            if (3-plane_i) ==1:
                active_plane = p1_compl
            elif (3-plane_i) ==2:
                active_plane = p2_compl
            for arc in crossing_arcs:
                if not arc in active_plane:
                    p1_compl, p2_compl = propagate(t, p1_compl,p2_compl, arc, 3-plane_i)        
            return p1_compl,p2_compl


        t = get_pair_arcs(sentence)
        for index_word_r in range(1, len(sentence), 1):
            for index_word_l in range(index_word_r-1,0,-1):
                head_word_l = int(sentence[index_word_l]["head"])
                head_word_r = int(sentence[index_word_r]["head"])
                if head_word_l== index_word_r or head_word_r == index_word_l:
                    if head_word_l== index_word_r:
                        next_arc = (index_word_r,index_word_l)
                    elif head_word_r == index_word_l:
                        next_arc = (index_word_l,index_word_r)
                
                    if not next_arc in plane_1_compl:
                        plane_1.append(next_arc)
                        propagate(t,plane_1_compl,plane_2_compl,next_arc,2)
                    if not next_arc in plane_2_compl:
                        plane_2.append(next_arc)
                        propagate(t,plane_1_compl,plane_2_compl,next_arc,1)

        indexed_brackets=index_labels_with_brackets(plane_1, 1)
        indexed_brackets_2p =index_labels_with_brackets(plane_2, 2)
        words_with_labels = generate_bracketing_labels(sentence, words_with_labels, indexed_brackets, indexed_brackets_2p, 2)                  
        l.tag_EOS(task, words_with_labels)

        return words_with_labels  
       
    def decode(self,sentence):
        decoded_words, unassigned_words= decode_2p_brackets(sentence)
        return decoded_words, unassigned_words


class ArcStandardEncoding():
    
    def encode(self,sentence, task):
        words_with_labels = {}
        buff = sentence
        stack = []
        buffer = []
        tokens = []
        transitions = []
        l = LabelPostProcessor()
        l.tag_BOS(task, words_with_labels)

        for word_index in buff:
            buffer.append(buff[word_index])
            if not word_index == 0:
                tokens.append(buff[word_index])
        #push ROOT to the stack
        stack.append(buffer[0])
        buffer.pop(0)

        #get sequence of transitions for arc-standard
        #read first token
        stack.append(buffer[0])
        buffer.pop(0)
        transitions.append("SH")

        while len(buffer) > 0 or len(stack) > 1:
            #requirement: legal LA and RA if stack has at least two elements
            if len(stack) > 1:
                #requirement: LA if second element on the stack is not ROOT
                if not stack[-2]["id"] == "ROOT" and stack[-2]["head"] == stack[-1]["id"]:
                    transitions.append("LA")       
                    stack.pop(len(stack) - 2)
                elif stack[-1]["head"] == stack[-2]["id"]:
                    if has_descendants(stack[-1]["id"], buffer):
                        transitions.append("SH")
                        stack.append(buffer[0])
                        buffer.pop(0)
                    else:
                        transitions.append("RA")
                        stack.pop(len(stack) - 1)
                else:
                    transitions.append("SH")
                    if not len(buffer) == 0:
                        stack.append(buffer[0])
                        buffer.pop(0)
                    else:
                        stack.pop(len(stack) - 1)
            else:
                transitions.append("SH")
                buffer.pop(0)

        # given a sequence of actions, produce labels
        index = -1
        while len(transitions) > 0:
            if transitions[0] == "LA":
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "LA")
                transitions.pop(0)
            elif transitions[0] == "RA":
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "RA")
                transitions.pop(0)
            else:
                index = index+1
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "SH")
                transitions.pop(0)
        
        l.tag_EOS(task, words_with_labels)
        return words_with_labels
       
    def decode(self,sentence):
        decoded_words = {}
        unassigned_words={}
        stack = []
        buffer = []
        transitions = []
        
        for index in range(1, len(sentence), 1):
        
            word_info = sentence[index]
            word = {1: index, 2: word_info[0],
                            3: word_info[1],
                            4: word_info[1],
                            5: -1,
                            6: word_info[3]}
            decoded_words.update({index: word})
            unassigned_words.update({index: word})
            word_info.append(index)
            buffer.append(word_info)
            labels = word_info[2].split("_")
            for l in labels:
                transitions.append(l)

        stack.append(sentence[0])

        while len(transitions)>0:
            if transitions[0] == "LA":
                transitions.pop(0)
                if len(stack) > 1:
                    if not stack[-2][1] == "ROOT":
                        word_to_remove = stack[-2]
                        index_word_remove= word_to_remove[4]
                        token = word_to_remove[0]
                        pos = word_to_remove[1]
                        head = stack[-1]
                        index_head = head[4]
                        deprel = word_to_remove[3]
                        word = {1: index_word_remove, 2: token,
                                            3: pos,
                                            4: pos,
                                            5: index_head,
                                            6: deprel}
                        stack.pop(len(stack)-2)
                        decoded_words.update({int(index_word_remove): word})
                        if unassigned_words.__contains__(int(index_word_remove)):
                            unassigned_words.pop(int(index_word_remove))    
            elif transitions[0] == "RA":
                transitions.pop(0)
                if len(stack) > 1:
                    word_to_remove = stack[-1]
                    index_word_remove = word_to_remove[4]
                    token = word_to_remove[0]
                    pos = word_to_remove[1]
                    deprel = word_to_remove[3]
                    head = stack[-2]
                    if head[1] =="ROOT":
                        index_head=0
                    else:
                        index_head = head[4]
                    word = {1: index_word_remove, 2: token,
                                        3: pos,
                                        4: pos,
                                        5: index_head,
                                        6: deprel}
                    stack.pop(len(stack)-1)
                    decoded_words.update({int(index_word_remove): word})
                    if unassigned_words.__contains__(int(index_word_remove)):
                        unassigned_words.pop(int(index_word_remove))   
            else:  
                transitions.pop(0)
                if len(buffer) >0:
                    stack.append(buffer[0])
                    buffer.pop(0)
    
        return decoded_words, unassigned_words

class ArcEagerEncoding():
    
    def encode(self,sentence, task):
        words_with_labels = {}
        buff = sentence
        stack = []
        buffer = []
        tokens = []
        l = LabelPostProcessor()
        l.tag_BOS(task, words_with_labels)

        for word_index in buff:
            buffer.append(buff[word_index])
            if not word_index == 0:
                tokens.append(buff[word_index])
        
        #get sequence of transitions for arc-eager
        stack.append(buffer[0])
        buffer.pop(0)
        transitions = []

        while not len(buffer) == 0:
            index_left_most = 200
            index_first_buffer = buffer[0]["id"]
            for i in buff:
                left_most = buff[i]
                head = left_most["head"]
                if head == index_first_buffer:
                    index_left_most = left_most["id"]
                    break

            if not stack[-1]["word"] == "ROOT" and stack[-1]["head"] == buffer[0]["id"]:
                transitions.append("LA")
                stack.pop(len(stack) - 1)
            elif buffer[0]["head"] == stack[-1]["id"]:
                transitions.append("RA")
                stack.append(buffer[0])
                buffer.pop(0)
            elif not index_left_most == 200 and int(index_left_most) < int(stack[-1]["id"]):
                transitions.append("RE")
                stack.pop(len(stack) - 1)
            elif int(buffer[0]["head"]) < int(stack[-1]["id"]):
                transitions.append("RE")
                stack.pop(len(stack) - 1)
            else:
                transitions.append("SH")
                stack.append(buffer[0])
                buffer.pop(0)
        last = stack[-1]
        
        while len(stack) > 1:
            stack.pop(len(stack) - 1)
            transitions.append("RE")
    
        # given a sequence of actions, produce labels
        index = -1
        while len(transitions) > 0:
            if transitions[0] == "LA":
                if index < len(tokens): 
                    create_arc(words_with_labels, tokens[index], "LA")
                transitions.pop(0)
            elif transitions[0] == "RE":
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "RE")
                transitions.pop(0)
            elif transitions[0] == "RA":
                index = index+1
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "RA")
                transitions.pop(0)
            else:
                index = index+1
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "SH")
                transitions.pop(0)

        l.tag_EOS(task, words_with_labels)
        return words_with_labels
       
    def decode(self,sentence):
        decoded_words = {}
        unassigned_words={}
        stack = []
        buffer = []
        transitions = []

        for word_index in range(1, len(sentence), 1):
            word_info = sentence[word_index]
            word = {1: word_index, 2: word_info[0],
                                3: word_info[1],
                                4: word_info[1],
                                5: -1,
                                6: word_info[3]}
            decoded_words.update({word_index: word})
            unassigned_words.update({word_index: word})
            word_info.append(word_index)
            buffer.append(word_info)
            labels = word_info[2].split("_")
            for l in labels:
                transitions.append(l)
        stack.append(sentence[0])
        while len(transitions)>0:
            if transitions[0] == "LA":
                transitions.pop(0)
                if len(stack) > 1 and len(buffer) >0: 
                    #requirement: check that stack[-1] does NOT have parent in decoded words (A)
                    if decoded_words[stack[-1][4]][5] ==-1:
                        word_to_remove = stack[-1]
                        index_word_remove= word_to_remove[4]
                        token = word_to_remove[0]
                        pos = word_to_remove[1]
                        head = buffer[0]
                        index_head = head[4]
                        deprel = word_to_remove[3]
                        word = {1: index_word_remove, 2: token,
                                            3: pos,
                                            4: pos,
                                            5: index_head,
                                            6: deprel}
                        stack.pop(len(stack)-1)
                        decoded_words.update({int(index_word_remove): word})
                        if unassigned_words.__contains__(int(index_word_remove)):
                            unassigned_words.pop(int(index_word_remove))     
            elif transitions[0] == "RA":
                transitions.pop(0)
                if len(stack) > 0:
                    word_to_remove = buffer[0]
                    index_word_remove = word_to_remove[4]
                    token = word_to_remove[0]
                    pos = word_to_remove[1]
                    deprel = word_to_remove[3]
                    head = stack[-1]
                    if head[1] =="ROOT":
                        index_head=0
                    else:
                        index_head = head[4]
                    word = {1: index_word_remove, 2: token,
                                        3: pos,
                                        4: pos,
                                        5: index_head,
                                        6: deprel}
                    stack.append(buffer[0])
                    buffer.pop(0)
                    decoded_words.update({int(index_word_remove): word})
                    if unassigned_words.__contains__(int(index_word_remove)):
                        unassigned_words.pop(int(index_word_remove))
            elif transitions[0] == "SH":
                transitions.pop(0)
                if len(buffer) >0:
                    stack.append(buffer[0])
                    buffer.pop(0)  
            else:
                #requirement: check that stack[-1] does have parent in decoded words
                transitions.pop(0)
                if len(stack)>1:
                    if not decoded_words[stack[-1][4]][5] ==-1:
                        stack.pop(len(stack)-1)

        return decoded_words, unassigned_words


class ArcHybridEncoding():
    
    def encode(self,sentence, task):
        words_with_labels = {}
        stack = []
        buffer = []
        tokens = []
        transitions = []
        buff = sentence
        l = LabelPostProcessor()
        l.tag_BOS(task, words_with_labels)

        for word_index in buff:
            buff[word_index].update({0: str(word_index)})
            buffer.append(buff[word_index])
            if not word_index == 0:
                tokens.append(buff[word_index])
        #push ROOT to the stack
        stack.append(buffer[0])
        buffer.pop(0)

        #get sequence of transitions for arc-hybrid
        #read first token
        stack.append(buffer[0])
        buffer.pop(0)
        transitions.append("SH")

        while len(buffer) > 0 or len(stack) > 1: 
            if len(stack) > 1:
                #LA like in arc-eager
                if len(buffer) >0 and not stack[-1]["word"] == "ROOT" and stack[-1]["head"] == buffer[0]["id"]:
                    transitions.append("LA")
                    stack.pop(len(stack) - 1)
                #RA like in arc-standard 
                elif stack[-1]["head"] == stack[-2]["id"]:
                    if has_descendants(stack[-1]["id"], buffer):
                        transitions.append("SH")
                        stack.append(buffer[0])
                        buffer.pop(0)
                    else:
                        transitions.append("RA")
                        stack.pop(len(stack) - 1)
                else:
                    transitions.append("SH")
                    if not len(buffer) == 0:
                        stack.append(buffer[0])
                        buffer.pop(0)
                    else:
                        stack.pop(len(stack) - 1)
            else:
                transitions.append("SH")
                stack.append(buffer[0])
                buffer.pop(0)
        
        # given a sequence of actions, produce labels
        index = -1
        while len(transitions) > 0:
            if transitions[0] == "LA":
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "LA")
                transitions.pop(0)
            elif transitions[0] == "RA":
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "RA")
                transitions.pop(0)
            else:
                index = index+1
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "SH")
                transitions.pop(0)
        l.tag_EOS(task, words_with_labels)
        return words_with_labels
       
    def decode(self,sentence):
        decoded_words = {}
        unassigned_words={}
        stack = []
        buffer = []
        transitions = []
    
        for index in range(1, len(sentence), 1):
            word_info = sentence[index]
            word = {1: index, 2: word_info[0],
                            3: word_info[1],
                            4: word_info[1],
                            5: -1,
                            6: word_info[3]}
            decoded_words.update({index: word})
            unassigned_words.update({index: word})
            word_info.append(index)
            buffer.append(word_info)
            labels = word_info[2].split("_")
            for l in labels:
                transitions.append(l)
        stack.append(sentence[0])
        while len(transitions)>0:
            #LA like in arc-eager
            if transitions[0] == "LA":
                transitions.pop(0)
                if len(stack) > 1 and len(buffer) >0: 
                    #requirement: check that stack[-1] does NOT have parent in decoded words (A)
                    if decoded_words[stack[-1][4]][5] ==-1:
                        word_to_remove = stack[-1]
                        index_word_remove= word_to_remove[4]
                        token = word_to_remove[0]
                        pos = word_to_remove[1]
                        head = buffer[0]
                        index_head = head[4]
                        deprel = word_to_remove[3]
                        word = {1: index_word_remove, 2: token,
                                            3: pos,
                                            4: pos,
                                            5: index_head,
                                            6: deprel}
                        stack.pop(len(stack)-1)
                        decoded_words.update({int(index_word_remove): word})
                        if unassigned_words.__contains__(int(index_word_remove)):
                            unassigned_words.pop(int(index_word_remove))    
            #RA like in arc-standard
            elif transitions[0] == "RA":
                transitions.pop(0)
                if len(stack) > 1:
                    word_to_remove = stack[-1]
                    index_word_remove = word_to_remove[4]
                    token = word_to_remove[0]
                    pos = word_to_remove[1]
                    deprel = word_to_remove[3]
                    head = stack[-2]
                    if head[1] =="ROOT":
                        index_head=0
                    else:
                        index_head = head[4]
                    word = {1: index_word_remove, 2: token,
                                        3: pos,
                                        4: pos,
                                        5: index_head,
                                        6: deprel}
                    stack.pop(len(stack)-1)
                    decoded_words.update({int(index_word_remove): word})
                    if unassigned_words.__contains__(int(index_word_remove)):
                        unassigned_words.pop(int(index_word_remove))   
            else:  
                transitions.pop(0)
                if len(buffer) >0:
                    stack.append(buffer[0])
                    buffer.pop(0)
        return decoded_words, unassigned_words


class CovingtonEncoding():
   
    def encode(self,sentence, task):
        words_with_labels = {}
        stack = []
        tokens = []
        transitions = []
        lambda1 = []
        lambda2 = []
        buffer = []
        A = []
        l = LabelPostProcessor()
        l.tag_BOS(task, words_with_labels)

        for word_index in sentence:
            if not word_index == 0:
                tokens.append(sentence[word_index])

        #initialize lambda1 with the dummy ROOT
        lambda1.append(0)
        #initialize all position indexes of words in the buffer Beta
        for ind in sentence:
            if not ind == 0:
                buffer.append(ind)

        # add an artificial SHIFT
        transitions.append("SH")

        while not len(buffer) == 0:
            # processed_word1= i
            # processed_word2 = j
            # LA
            left_arc_lambda1 = False
            if len(lambda1)>0:
                index_word1 = lambda1[-1] 
                index_word2 = buffer[0]
                processed_word1= sentence[index_word1]
                processed_word2= sentence[index_word2]
                for i in reversed(lambda1):
                    cand = sentence[i]  
                    if i == int(processed_word2["head"]) or int(cand["head"]) == buffer[0]:
                        left_arc_lambda1=True
                        break 
            # LA
            # <lambda1|i, lambda2, j|B, A> --> <lambda1, i|lambda2, j|B, A>
        
            if len(lambda1)>0 and not index_word1== 0 and int(processed_word1["head"]) == buffer[0]:
                transitions.append("LA")
                lambda2 = [lambda1[index_word1]]+lambda2
                lambda1 = lambda1[:index_word1]     
            # RA
            # <lambda1|i, lambda2, j|B, A> --> <lambda1, i|lambda2, j|B, A>
            elif len(lambda1)>0 and index_word1 == int(processed_word2["head"]):
                transitions.append("RA")  
                lambda2 = [lambda1[index_word1]]+lambda2
                lambda1 = lambda1[:index_word1]     
            # NO-ARC
            # <lambda1|i, lambda2, B, A> --> <lambda1, i|lambda2, B, A>
            elif left_arc_lambda1:
                transitions.append("NOARC")
                lambda2 = [lambda1[index_word1]]+lambda2
                lambda1 = lambda1[:index_word1]
            else:
                # SHIFT 
                # <lambda1, lambda2, j|B, A> --> <lambda1.lambda2|j, [], B, A>
                transitions.append("SH")  
                lambda1 = lambda1+lambda2+[buffer[0]]
                lambda2 = []
                buffer.pop(0)     
        
        # given a sequence of actions, produce labels
        index = -1
        while len(transitions) > 0:
            if transitions[0] == "LA":
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "LA")
                transitions.pop(0)
            elif transitions[0] == "RA":
                if index < len(tokens): 
                    create_arc(words_with_labels, tokens[index], "RA")
                transitions.pop(0)
            elif transitions[0] == "NOARC":
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "NOARC")
                transitions.pop(0)
            else:
                index = index+1
                #omit the last SHIFT
                if index < len(tokens):
                    create_arc(words_with_labels, tokens[index], "SH")
                transitions.pop(0)
        l.tag_EOS(task, words_with_labels)
 
        return words_with_labels
       
    def decode(self,sentence):
        decoded_words = {}
        unassigned_words={}
        lambda1 = []
        lambda2 = []
        buffer = []
        A = []
        transitions = []

        def check_cycles(A, a):
            A_up = A.copy()
            A_up.append(a)
            have_desc=True
            next_arc=-1
            orig_arc= a
            next_arc=a[0]
            found_arc=set()
            found_arc.add(a)
        
            while have_desc:
                for arc in A_up:
                    if next_arc==arc[1]:
                        have_desc=True
                        next_arc=arc[0]
                        if arc in found_arc:
                            return True
                        found_arc.add(arc)
                        break
                    else:
                        have_desc=False
            return False
        
        for index in range(1, len(sentence), 1):
            word_info = sentence[index]
            word = {1: index, 2: word_info[0],
                            3: word_info[1],
                            4: word_info[1],
                            5: -1,
                            6: word_info[3]}
            decoded_words.update({index: word})
            unassigned_words.update({index: word})
            word_info.append(index)
            labels = word_info[2].split("_")
            for l in labels:
                transitions.append(l)
            #initialize all position indexes of words in the buffer Beta
            buffer.append(index)
        #check if the first in elements is SHIFT and skip it as it's an artificial SHIFT
        if transitions[0] == "SH":
            transitions.pop(0)
        #initialize lambda1 with the dummy ROOT
        lambda1.append(0)
        
        while len(transitions)>0:
            # LA
            # <lambda1|i, lambda2, j|B, A> --> <lambda1, i|lambda2, j|B, A>
            if transitions[0]=="LA":
                # single head & acyclicity contraints
                transitions.pop(0)
                if len(lambda1)>0 and len(buffer)>0:
                    index_word1 = lambda1[-1] 
                    if not index_word1 == 0:
                    #single head constraint
                        if decoded_words[index_word1][5] == -1 and check_cycles(A, (buffer[0],index_word1))==False:
                            A.append((buffer[0], index_word1))
                            lambda2 = [lambda1[index_word1]]+lambda2
                            lambda1 = lambda1[:index_word1]    
            # RA
            # <lambda1|i, lambda2, j|B, A> --> <lambda1, i|lambda2, j|B, A>
            elif transitions[0]=="RA":
                # single head & acyclicity contraints
                transitions.pop(0)
                if len(lambda1)>0 and len(buffer)>0:
                    index_word1 = lambda1[-1] 
                    #single head constraint
                    if decoded_words[buffer[0]][5] == -1 and check_cycles(A, (index_word1,buffer[0]))==False:
                        A.append((index_word1, buffer[0]))
                        lambda2 = [lambda1[index_word1]]+lambda2
                        lambda1 = lambda1[:index_word1]  
                
            # NO-ARC
            # <lambda1|i, lambda2, B, A> --> <lambda1, i|lambda2, B, A>
            elif transitions[0]=="NOARC":
                transitions.pop(0)
                if len(lambda1)>0:
                    index_word1 = lambda1[-1] 
                    lambda2 = [lambda1[index_word1]]+lambda2
                    lambda1 = lambda1[:index_word1]

            else:
                # SHIFT 
                # <lambda1, lambda2, j|B, A> --> <lambda1.lambda2|j, [], B, A>
                transitions.pop(0)
                lambda1 = lambda1+lambda2+[buffer[0]]
                lambda2 = []
                buffer.pop(0)  
        
        for arc in A:
            index_head = arc[0]
            index_dependent= arc[1]
            word_to_remove = decoded_words[index_dependent]
            word_to_remove[5] = index_head   
            if unassigned_words.__contains__(int(index_dependent)):
                unassigned_words.pop(int(index_dependent))

        return decoded_words, unassigned_words


class ZeroEncoding():
    def encode(self,sentence, task):
        words_with_labels = {}
        l = LabelPostProcessor()
        l.tag_BOS(task, words_with_labels)

        for word_index in sentence:
            # exclude the dummy root
            if not word_index == 0:
                word_info = sentence[word_index]
                if task == "1-task":
                    label = str(repr(0))
                elif task == "2-task-combined":
                    label = str(
                        repr(0) + "@" + repr(0) + "{}" + repr(0))
                elif task == "2-task":
                    label = str(
                        repr(0) + "{}" + repr(0))
                elif task =="3-task":
                    label = str(
                        repr(0) + "{}" + repr(0) + "{}" + repr(0))
                else:
                    print("Invalid task type")
                    exit(1)
                full_label = str(
                    word_info["word"] + "\t" + word_info["pos"] + "\t" + label)
                words_with_labels.update({word_index: full_label})
        l.tag_EOS(task, words_with_labels)

        return words_with_labels

def decode_2p_brackets(sentence):
    # Brackets in the first plane
    # <L denotes <\
    # R> denotes />
    # Brackets in the second plane
    # [l denotes <*\*
    # r] denotes /*>*
    decoded_words = {}
    unassigned_words = {}
    stack1 = []
    stack2 = []
    stack_secondPlane_1 = []
    stack_secondPlane_2 = []
    for word_index in range(1, len(sentence), 1):
        word_info = sentence[word_index]
        word = {1: word_index, 2: word_info[0],
                           3: word_info[1],
                           4: word_info[1],
                           5: -1,
                           6: word_info[4]}
        decoded_words.update({word_index: word})
        unassigned_words.update({word_index: word})

    for word_index in range(1, len(sentence), 1):
        processed_word = sentence[word_index]
        if processed_word[4] == "root":
            word = {1: word_index, 2: processed_word[0],
                               3: processed_word[1],
                               4: processed_word[1],
                               5: 0,
                               6: processed_word[4]}
            decoded_words.update({word_index: word})
            if unassigned_words.__contains__(word_index):
                unassigned_words.pop(word_index)
        char = list(processed_word[2])
        for x in char:
            if x == "<":
                x += "_" + repr(word_index)
                stack1.append(x)
            elif x == "R":
                x += "_" + repr(word_index)
                stack2.append(x)
            elif x == "L":
                if not len(stack1) == 0:
                    dependent = stack1[-1]
                    dependent_index = int(dependent.split("_")[1]) - 1
                    if not dependent_index == 0:
                        dependent_info = sentence[dependent_index]
                        word = {1: dependent_index, 2: dependent_info[0],
                                           3: dependent_info[1],
                                           4: dependent_info[1],
                                           5: word_index,
                                           6: dependent_info[4]}

                        decoded_words.update({dependent_index: word})
                        if unassigned_words.__contains__(dependent_index):
                            unassigned_words.pop(dependent_index)
                        stack1 = stack1[:-1]
            elif x == ">":
                if not len(stack2) == 0:
                    head = stack2[-1]
                    head_index = int(head.split("_")[1]) - 1
                    word = {1: word_index, 2: processed_word[0],
                                       3: processed_word[1],
                                       4: processed_word[1],
                                       5: head_index,
                                       6: processed_word[4]}
                    decoded_words.update({word_index: word})
                    if unassigned_words.__contains__(word_index):
                        unassigned_words.pop(word_index)
                    stack2 = stack2[:-1]
        char = list(processed_word[3])
        for x in char:
            if x == "[":
                x += "_" + repr(word_index)
                stack_secondPlane_1.append(x)
            elif x == "r":
                x += "_" + repr(word_index)
                stack_secondPlane_2.append(x)
            elif x == "l":
                if not len(stack_secondPlane_1) == 0:
                    dependent = stack_secondPlane_1[-1]
                    dependent_index = int(dependent.split("_")[1]) - 1
                    if not dependent_index == 0:
                        dependent_info = sentence[dependent_index]
                        word = {1: dependent_index, 2: dependent_info[0],
                                           3: dependent_info[1],
                                           4: dependent_info[1],
                                           5: word_index,
                                           6: dependent_info[4]}
                        decoded_words.update({dependent_index: word})
                        if unassigned_words.__contains__(dependent_index):
                            unassigned_words.pop(dependent_index)
                        stack_secondPlane_1 = stack_secondPlane_1[:-1]
            elif x == "]":
                if not len(stack_secondPlane_2) == 0:
                    head = stack_secondPlane_2[-1]
                    head_index = int(head.split("_")[1]) - 1
                    word = {1: word_index, 2: processed_word[0],
                                       3: processed_word[1],
                                       4: processed_word[1],
                                       5: head_index,
                                       6: processed_word[4]}
                    decoded_words.update({word_index: word})
                    if unassigned_words.__contains__(word_index):
                        unassigned_words.pop(word_index)
                    stack_secondPlane_2 = stack_secondPlane_2[:-1]

    return decoded_words, unassigned_words

def create_arc(words_with_labels, top, label):
    index_top = int(top["id"])
    if index_top in words_with_labels:
        line = words_with_labels[index_top]
        column = line.split("\t")
        token = column[0]
        pos = column[1]
        old_head_label = column[2].split("{}")[0]
        old_deprel_label = column[2].split("{}")[1]
        transitions = old_head_label + "_" + label
        label = token + "\t" + pos + "\t" + transitions + "{}" + old_deprel_label
        words_with_labels.update({index_top: label})
    else:
        token = top["word"]
        pos = top["pos"]
        deprel = top["deprel"]
        label = token + "\t" + pos + "\t" + label + "{}" + deprel
        words_with_labels.update({index_top: label})

def has_descendants(word, buffer):
    for b in buffer:
        if b["head"] == word:
            return True
    return False

def check_if_crossing_arcs(plane, arc):
    crossing_arc_from_plane = []
    for (i, j) in plane:
        for (k, l) in arc:
            #if min(i, j) < min(k, l) < max(i, j) < max(k, l):
            if (min(i, j) < min(k, l) < max(i, j) < max(k, l)) or (min(k,l) < min(i,j) < max(k,l) < max(i,j)):
                crossing_arc_from_plane.append((i,j))
    return crossing_arc_from_plane

def index_labels_with_brackets(arcs_in_plane, plane):

    indexed_brackets= {}
    for arc in arcs_in_plane:
        head = arc[0]
        dependent = arc[1]
        if head > dependent:
            if not dependent + 1 in indexed_brackets:
                if plane==1:
                    bracket = "<"
                else:
                    bracket = "["
                indexed_brackets.update({dependent + 1: bracket})
            else:
                bracket = indexed_brackets[dependent + 1]
                if plane==1:
                    bracket += "<"
                else:
                    bbracket += "["
                indexed_brackets.update({dependent + 1: bracket})
    for arc in arcs_in_plane:
        head = arc[0]
        dependent = arc[1]
        if not head == 0:
            if head < dependent:
                if not head + 1 in indexed_brackets:
                    if plane==1:
                        bracket = "R"
                    else:
                        bracket = "r"
                    indexed_brackets.update({head + 1: bracket})
                else:
                    bracket = indexed_brackets[head + 1]
                    if plane==1:
                        bracket += "R"
                    else:
                        bracket += "r"
                    indexed_brackets.update({head + 1: bracket})
    for arc in arcs_in_plane:
        head = arc[0]
        dependent = arc[1]
        if head > dependent:
            if head not in indexed_brackets:
                if plane==1:
                    bracket = "L"
                else:
                    bracket = "l"
                indexed_brackets.update({head: bracket})
            else:
                bracket = indexed_brackets[head]
                if plane==1:
                    bracket += "L"
                else:
                    bracket += "l"
                indexed_brackets.update({head: bracket})
    for arc in arcs_in_plane:
        head = arc[0]
        dependent = arc[1]
        if not head == 0:
            if head < dependent:
                if dependent not in indexed_brackets:
                    if plane==1:
                        bracket = ">"
                    else:
                        bracket = "]"
                    indexed_brackets.update({dependent: bracket})
                else:
                    bracket = indexed_brackets[dependent]
                    if plane==1:
                        bracket += ">"
                    else:
                        bracket += "]"
                    indexed_brackets.update({dependent: bracket})
    return indexed_brackets

def generate_bracketing_labels(sentence, words_with_labels, indexed_brackets_1p, indexed_brackets_2p, plane):
    split = "{}"
    for word_index in range(1, len(sentence), 1):
        info_of_a_word = sentence[word_index]
        if word_index == 1:
            if plane ==1:
                label = str("." + split + info_of_a_word["deprel"])
            else:
                label = str("." + split + "." + split + info_of_a_word["deprel"])
            full_label = str(info_of_a_word["word"] +"\t" +info_of_a_word["pos"] +"\t" +label)
            words_with_labels.update({word_index: full_label})
        else:
            if plane ==1:
                if word_index in indexed_brackets_1p:
                    brackets = indexed_brackets_1p[word_index]
                    label = str(brackets + split + info_of_a_word["deprel"])
                else:
                    label = str("." + split + info_of_a_word["deprel"])

                full_label = str(
                    info_of_a_word["word"] +"\t" +info_of_a_word["pos"] +"\t" +label)
                words_with_labels.update({word_index: full_label})
            else:
                brackets_2p = "."
                if word_index in indexed_brackets_2p:
                    brackets_2p = indexed_brackets_2p[word_index]

                if word_index in indexed_brackets_1p:
                    brackets = indexed_brackets_1p[word_index]

                    label = str(brackets + split + brackets_2p + split + info_of_a_word["deprel"])
                    full_label = str(info_of_a_word["word"] +"\t" +info_of_a_word["pos"] +"\t" +label)
                    words_with_labels.update({word_index: full_label})
                else:

                    label = str("." + split + brackets_2p + split + info_of_a_word["deprel"])
                    full_label = str(info_of_a_word["word"] +"\t" +info_of_a_word["pos"] +"\t" +label)
                    words_with_labels.update({word_index: full_label})
    return words_with_labels