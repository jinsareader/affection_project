# korean mood AI
VER 0.2

# github link
https://github.com/jinsareader/affection_project


# recent update
- Merge word vector into Neural Networks files(*.onnx)
- Change word vector
- Tweak Neural Networks Structures
- Change text preprocessing module to kiwi. now konlpy and java are not needed. Install kiwipiepy instead. 


# vector url
https://github.com/Kyubyong/wordvectors

# train data url
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=271
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=270

# python version
3.12

# must be installed module
numpy
kiwipiepy
onnxruntime
tqdm




#how to use
First, install those four modules before using this program.
There are four versions of .py files in main directory.

CLI_ : program that has command line interface
tk_ : program that has gui that made with tk module

_LSTM : program that operating with LSTM NN
_transformer : program that operating with transformer NN



# Update 0.1
- AI now can recognize end marks (! ? .)
- Tweak Neural Networks.
- Fix some minor bugs