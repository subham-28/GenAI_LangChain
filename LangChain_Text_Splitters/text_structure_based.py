from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """ Long Short-Term Memory (LSTM, Hochreiter & Schmidhuber, 1997) can solve numerous
tasks not solvable by previous learning algorithms for recurrent neural networks (RNNs).
We identify a weakness of LSTM networks processing continual input streams that are not
a priori segmented into subsequences with explicitly marked ends at which the network's
internal state could be reset. Without resets, the state may grow inde nitely and eventually
cause the network to break down. Our remedy is a novel, adaptive \forget gate" that enables
an LSTM cell to learn to reset itself at appropriate times, thus releasing internal resources.
Wereview illustrative benchmark problems on which standard LSTM outperforms other RNN
algorithms. All algorithms (including LSTM) fail to solve continual versions of these problems.
LSTM with forget gates, however, easily solves them in an elegant way.
"""

''' these technique split in the order: 
    \n\n(para) , \n(sentence) , words, letters'''
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)