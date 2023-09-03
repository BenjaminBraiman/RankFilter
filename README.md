# RankFilter
Rank filtering in Pytorch, similar in functionality to Scipy. 

For example, given the tensor [1,2,3,4,5], a width 3 stride 1 rank 2 filter would yield the array [2, 3, 4]. The operation works by taking the array [1,2,3,4,5] and unfolding it into 3x3 patches:
1. [1,2,3,4,5]
2. Unfolding: [[1,2,3], [2,3,4], [3,4,5]]
3. Taking the 2nd smallest element of each "patch": [2,3,4].

This is formalized by a torch.unfold operation.

In image processing, if X is a NxN gray-scale image, then a 3x3 stride 1 rank 5 filter with 1 pixel border padding is the equivalent of a median filter, a common and useful way to denoise images.

TODO:
* Continue testing
* Code cleanup
* Integrate into Torch
