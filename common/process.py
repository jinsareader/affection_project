import common.custom as custom
from kiwipiepy import Kiwi
import pickle
import numpy
import onnxruntime
import os
dir = os.path.dirname(os.path.abspath(__file__)) + "\\"


class Process() :    
    def __init__(self, vector_file, nn_file, max_word_len : int | None = None) :
        self.targets = ['공포','놀람','분노','슬픔','중립','행복','혐오']
        with open(dir+vector_file, mode = "rb") as f :
            self.vector_dict = pickle.load(f)
        self.F = onnxruntime.InferenceSession(dir+nn_file, providers=["CPUExecutionProvider"])
        self.kiwi = Kiwi()
        self.max_word_len = max_word_len
        self.kiwi.tokenize('더미')

    def query_preprocess(self, s : str) :
        unk_list = []
        word_list = []
        s = custom.text_preprocess_kor(s, True, True)
        try :
            temp = self.kiwi.tokenize(s)
            for w in temp :
                word_list.append(w.form)
        except Exception as e :
            word_list = ["<unk>"]
            unk_list.append(str(e))
        vector_list = custom.word_vectorize(word_list, self.vector_dict, self.max_word_len)
        unk_list = custom.get_unk_words(word_list, self.vector_dict)
        return vector_list, unk_list
    
    def cal(self, vector) :
        vector = numpy.array(vector).astype(numpy.int64)

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

