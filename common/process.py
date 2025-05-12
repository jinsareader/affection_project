import re
from konlpy.tag import Kkma as word_div
import threading
import pickle
import numpy
import onnxruntime
import os
dir = os.path.dirname(os.path.abspath(__file__)) + "\\"

def text_preprocess(text : str) :
    text = re.sub(r"[^0-9a-zA-Z]",repl=" ",string=text.lower().replace("n't"," not"))
    text = re.sub(r"[0-9]+",repl="N",string=text)
    text = re.sub(r"\s+",repl=" ",string=text)
    return text

def text_preprocess_kor(text : str, chosung : bool = False) :
    if chosung :
        text = re.sub(r"[^가-힣ㄱ-ㅎ]",repl=" ",string=text)
    else :
        text = re.sub(r"[^가-힣]",repl=" ",string=text)
    text = re.sub(r"[0-9]+",repl="N",string=text)
    text = re.sub(r"\s+",repl=" ",string=text)
    return text 

def word_vectorize(sentence : str | list, vec_dict : dict, word_len : int | None = None) :
    temp = []
    
    if type(sentence) == str : 
        words = str(sentence).split()
    else :
        words = sentence
    if word_len is None :
        word_len = len(sentence)
        
    for i in range(word_len - len(words)) :
        temp.append(vec_dict["<pad>"])
    for i in range(len(words)) :
        if words[i] not in vec_dict :
            temp.append(vec_dict["<unk>"])
            continue
        temp.append(vec_dict[words[i]])

    return temp


class Preprocessor() :
    def __init__(self) :
        self._result = None
        self._error = None
    def _process_with_timeout(self, s : str, timeout : int) :
        def work() :
            try :
                self._result = word_div().morphs(s)
            except Exception as e :
                self._error = e        
        self._error = None
        thread = threading.Thread(target=work, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive() :
            raise Exception(f"Error : Time Exceeded")
        if self._error is not None :
            raise Exception(str(self._error))

        return self._result
    def process(self, s : str, timeout : int = 5) :
        result = self._process_with_timeout(s, timeout)
        return result


class Process() :    
    def __init__(self, vector_file, nn_file, max_word_len : int | None = None) :
        with open(dir+"target.txt", mode = "r", encoding="UTF8") as f:
            self.targets = f.readlines()
            for i in range(len(self.targets)) :
                self.targets[i] = self.targets[i].strip()
        with open(dir+vector_file, mode = "rb") as f :
            self.vector_dict = pickle.load(f)
        self.F = onnxruntime.InferenceSession(dir+nn_file, providers=["CPUExecutionProvider"])
        self.preprocessor = Preprocessor()
        self.max_word_len = max_word_len
        word_div().morphs("더미")

    def query_preprocess(self, s : str) :
        unk_list = []
        s = text_preprocess_kor(s, True)
        try :
            word_list = self.preprocessor.process(s)
        except Exception as e :
            word_list = ["<unk>"]
            unk_list.append(str(e))
        vector_list = word_vectorize(word_list, self.vector_dict, self.max_word_len)
        for w in word_list :
            if w not in self.vector_dict :
                unk_list.append(w)
        return vector_list, unk_list
    
    def cal(self, vector) :
        vector = numpy.array(vector).astype(numpy.float32)

        if vector.ndim == 2 :
            vector = numpy.expand_dims(vector, 0)

        input = {self.F.get_inputs()[0].name : vector}
        output = self.F.run(None, input)

        return output[0]

    def get_softmax(self, y) :
        y = y - y.max()
        result = numpy.exp(y) / numpy.exp(y).sum()
        result = (result * 100).astype(numpy.int32)
        return result.squeeze()

    def get_softmax_text(self, y) :
        text = ""
        target = self.targets
        percent = self.get_softmax(y)
        text += "--Percentage--\n"
        for i in range(len(target)) :
            text += f"\t{target[i]} : {percent[i]}\n"
        return text

    def print_softmax(self, y) :
        print(self.get_softmax_text(y), end="")

    def get_argmax(self, y) :
        return y.argmax()

    def get_argmax_text(self, y) :
        text = ""
        target = self.targets
        argmax = self.get_argmax(y)
        text = f"Mood : {target[argmax]}\n"
        return text

    def print_argmax(self, y) :
        print(self.get_argmax_text(y), end = "")

