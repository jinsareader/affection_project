import re
import numpy
from tqdm import tqdm

def text_preprocess(text : str, end_mark : bool = False) :
    target = r"[^0-9a-zA-Z"
    if end_mark :
        target += r"!\?\."
    target += r"]"
    text = re.sub(target,repl=" ",string=text.lower().replace("n't"," not"))
    text = re.sub(r"[0-9]+",repl="N",string=text)
    text = re.sub(r"\s+",repl=" ",string=text)
    if end_mark :
        text = re.sub(r"!+",repl=r"!",string=text)
        text = re.sub(r"\?+",repl=r"?",string=text)
        text = re.sub(r"\.+",repl=r".",string=text)
    return text

def text_preprocess_kor(text : str, end_mark : bool = False, chosung : bool = False) :
    target = r"[^가-힣"
    if end_mark :
        target += r"!\?\."
    if chosung :
        target += r"ㄱ-ㅎ"
    target += r"]"
    text = re.sub(target,repl=" ",string=text)
    text = re.sub(r"[0-9]+",repl="N",string=text)
    text = re.sub(r"\s+",repl=" ",string=text)
    if end_mark :
        text = re.sub(r"!+",repl=r"!",string=text)
        text = re.sub(r"\?+",repl=r"?",string=text)
        text = re.sub(r"\.+",repl=r".",string=text)
    return text 

def del_stopword(text : str, stopword : list) :
    text = text.split()
    for i in range(len(text)) :
        for w in stopword :
            if text[i] == w:
                text[i] = ""
    text = " ".join(text)
    text = re.sub(r"\s+",repl=" ",string=text)
    return text

def make_dict(sentences : list, word_dict : dict = None) :
    data = " ".join(sentences)
    data = text_preprocess(data).split()
    if word_dict is None :
        word_dict = {}
        word_dict["<pad>"] = 0 #패딩
        word_dict["<unk>"] = 1 #없는 단어
    for w in data :
        if w not in word_dict :
            word_dict[w] = len(word_dict)
    number_dict = {i : w for w, i in word_dict.items()}
    return word_dict, number_dict

def word_num_encoding(sentences : list, word_dict : dict, unk : str = "<unk>") :
    word_size = len(word_dict)
    corpus = []
    max_len = 0
    for s in sentences :
        s = text_preprocess(s).split()
        max_len = max(max_len, len(s))
    for s in sentences :
        s_array = []
        s = text_preprocess(s).split()
        for i in range(max_len) :
            if len(s) <= i :
                s_array.append(0)
                continue
            try :
                s_array.append(word_dict[s[i]])
            except :
                s_array.append(word_dict[unk])
        corpus.append(s_array)
    corpus = numpy.array(corpus)    
    return corpus

def make_comatrix(corpus, word_size, window_size = 1, pad_idx : int = 0) :
    comatrix = numpy.zeros(shape = (word_size, word_size))
    for s in corpus :
        for w in range(len(s)) :
            if s[w] == pad_idx :
                continue
            for i in range(1,window_size+1) :
                if w-i >= 0 :
                    if s[w-i] == pad_idx  :
                        continue
                    comatrix[s[w], s[w-i]] += 1
                if w+i < len(s) :
                    if s[w+i] == pad_idx  :
                        continue
                    comatrix[s[w], s[w+i]] += 1
    return comatrix

def cos_similarity(x, y) :
    eps = 1e-15
    return numpy.dot(x,y) / (numpy.linalg.norm(x)*numpy.linalg.norm(y) + eps)

def euc_distance(x, y) :
    return numpy.sqrt(numpy.sum((x - y) ** 2))

def most_similiar(query, word_dict, number_dict, vector_array, top = 5, mode : str = "euc") :
    if query not in word_dict :
        print("{}(이)가 사전에 존재하지 않습니다.".format(query))
        return

    word_size = len(word_dict)
    similiar = numpy.zeros(shape = (word_size))
    if mode.lower() == "euc" :
        for i in range(word_size) :
            similiar[i] = euc_distance(vector_array[word_dict[query]], vector_array[i])
    elif mode.lower() == "cos" :
        for i in range(word_size) :
            similiar[i] = cos_similarity(vector_array[word_dict[query]], vector_array[i])
    else :
        print("{}는 잘못된 모드입니다. 모드 종류 : 'euc', 'cos'".format(mode))
        return
    
    print("검색어 ||",query)
    cnt = 0
    if mode.lower() == "euc" :
        argsort = similiar.argsort()
    elif mode.lower() == "cos" :
        argsort = (-1 * similiar).argsort()
    for i in argsort :
        if number_dict[i] == query :
            continue
        print("{} : {}".format(number_dict[i], similiar[i]))
        cnt += 1
        if cnt >= top :
            break
    print("")

def make_pmi(comatrix, verdose = False) :
    P = numpy.zeros_like(comatrix)
    N = numpy.sum(comatrix)
    S = numpy.sum(comatrix, axis = 0)
    eps = 1e-15
    
    if verdose :
        li = tqdm(range(P.shape[0]))
    else :
        li = range(P.shape[0])
        
    for i in li :
        for j in range(P.shape[1]) :
            pmi = numpy.log2(comatrix[i, j] * N / (S[i]*S[j] + eps) + eps)
            P[i,j] = max(0, pmi)
    return P   

def make_word_pair(comatrix) :
    word_pair = []
    rows = comatrix.shape[0]
    cols = comatrix.shape[1]
    for r in range(rows) :
        for c in range(cols) :
            if comatrix[r][c] > 0 :
                word_pair.append([r,c])
        
    return numpy.array(word_pair)

def word_vectorize(sentence : str | list, vec_dict : dict, word_len : int | None = None, padding_front = True, pad_word : str = "<pad>", unk_word : str = "<unk>") :
    temp = []
    
    if type(sentence) == str : 
        words = str(sentence).split()
    else :
        words = sentence
    if word_len is None :
        word_len = len(words)

    if padding_front :
        for i in range(word_len - len(words)) :
            temp.append(vec_dict[pad_word])
            
    for i in range(min(word_len,len(words))) :
        if words[i] not in vec_dict :
            temp.append(vec_dict[unk_word])
            continue
        temp.append(vec_dict[words[i]])
        
    if not padding_front :
        for i in range(word_len - len(words)) :
            temp.append(vec_dict[pad_word])

    return temp

def get_unk_words(sentence : str | list, vec_dict : dict) :
    unk_list = []
    if type(sentence) == str : 
        words = str(sentence).split()
    else :
        words = sentence
    for w in words :
        if w not in vec_dict :
            unk_list.append(w)
    
    return unk_list