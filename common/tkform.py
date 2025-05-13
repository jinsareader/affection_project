import tkinter
import common.process as common_process
import os
dir = os.path.dirname(os.path.abspath(__file__)) + "\\"

separator = "\n###############################################\n\n"

class Mainform() :
    def __init__(self,vector_file,nn_file,max_word_len : int | None = None):
        self.process = common_process.Process(vector_file,nn_file,max_word_len)

        self.window = tkinter.Tk()
        self.leftpanel = tkinter.PanedWindow(master= self.window, width = 70)
        self.leftpanel.grid(row=0, column=0)
        self.middlepanel = tkinter.PanedWindow(master = self.window, width = 70)
        self.middlepanel.grid(row=1, column=0)
        self.rightpanel = tkinter.PanedWindow(master= self.window, width = 70)
        self.rightpanel.grid(row=2, column=0)

        self.entry = tkinter.Entry(master= self.leftpanel, width = 50, )
        self.entry.grid(row=0, column=0)
        self.button = tkinter.Button(master= self.leftpanel, width = 20, text="입력")
        self.button.config(command = self.enter_query)
        self.button.grid(row=0, column=1)

        NN_name = nn_file.split(".")[0]
        self.label = tkinter.Label(master = self.middlepanel, background="yellow", width = 70, text = str(NN_name))
        self.label.pack()

        self.text = tkinter.Text(master = self.rightpanel, width=70, height=30, state="disabled")
        self.text.grid(row=0, column=0)
        self.clear_button = tkinter.Button(master=self.rightpanel, width=70, text = "삭제")
        self.clear_button.config(command = self.text_clear)
        self.clear_button.grid(row=1, column=0)

        self.window.mainloop()

    def enter_query(self) :
        self.button.config(command = None)

        end = tkinter.END
        query = self.entry.get()
        if len(query) < 1 :
            self.button.config(command = self.enter_query)
            return
        x, unk = self.process.query_preprocess(query)
        y = self.process.cal(x)
        mood = self.process.get_argmax_text(y)
        percent = self.process.get_softmax_text(y)

        self.text.config(state="normal")
        self.text.insert((end), query+"\n")
        self.text.insert((end), mood)
        self.text.insert((end), percent)
        self.text.insert((end), "모르는 단어 : "+str(unk)+"\n")
        self.text.insert((end), separator)
        self.text.config(state="disable")

        self.entry.delete(first = 0, last = tkinter.END)
        self.button.config(command = self.enter_query)

    def text_clear(self) :
        end = tkinter.END
        self.text.config(state="normal")
        self.text.delete("1.0", end)
        self.text.config(state="disable")


if __name__ == "__main__" :
    Mainform("korean_vector.pkl","LSTM.pt");