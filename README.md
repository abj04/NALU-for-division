The repository contains various implementations for Neural Arithmetic Logic Unit by Google's Deep Mind.
The paper on the same can be found here: https://arxiv.org/abs/1808.00508

Description of various files:

1. nac.py : Implementation of Neural Accumulator to be trained on addition task. Mean square eror of 0 achieved on extrapolation task.

2. nalu.py : Implementation of NALU and can be trained on addition or multiplication task. Training samples consist of only positive float values. Mean square error of 0.02 achieved on extrapolated  multipication task.

3. nalu_div.py: Implementation of NALU for dividion task. Very poor result so far. MSE reducing with increase in number of training samples.

4. nalu_neg.py: Task of nalu.py for both negative and positive flaot values. Very poor results on both addition and multiplication, despite increasing no. of training samples. The MSE doesn't decrease after a certain value. Results on extrapolation are even worse.
