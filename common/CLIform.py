import common.process as common_process
import os
dir = os.path.dirname(os.path.abspath(__file__)) + "\\"

separator = "###############################################\n"

def activate(vector_file, nn_file, max_word_len : int | None = None) :

    process = common_process.Process(vector_file,nn_file,max_word_len)
    print(f"NN : {nn_file.split('.')[0]}\n")
    print(separator)
    
    while True :
        query = input("문장을 입력하세요 (종료를 입력하면 종료합니다) : \n")
        if len(query) < 1 :
            continue
        if query.strip() == "종료" :
            break
        x, unk = process.query_preprocess(query)
        y = process.cal(x)
        process.print_argmax(y)
        process.print_softmax(y)
        print("모르는 단어 : ", unk)
        print(separator)