# 13/07/18 Daily Report


## Denoising using MRF 

The code has originated from [andredung's code](https://github.com/andreydung/MRF) which is based on *MRF* and 
*SNR(Signal to Noise Ratio)* algorithm.

### Code
```python

# Image denoising using MRF model
from PIL import Image
import numpy
from pylab import *

def main():
	# Read in image
	im=Image.open('lena512.bmp')
	im=numpy.array(im)
	im=where (im>100, 1, 0) #convert to binary image
	(M,N)=im.shape

	# Add noise
	noisy_img=im.copy()
	noise=numpy.random.rand(M,N)	# generate M x N size with random value 0 ~ 1
	ind=where(noise<0.2)	# get index list which satisfies the condition
	noisy_img[ind]=1-noisy_img[ind]	# manipulate noise image

	# show noisy image
	gray()
	title('Noisy Image')
	imshow(noisy_img)

	# process by MRF de-noise
	out=MRF_denoise(noisy_img)

	# show de-noised image
	figure()		
	gray()
	title('Denoised Image')
	imshow(out)
	show()

def MRF_denoise(noisy_img):
	# Start MRF	
	(M,N) = noisy_img.shape
	y_old = noisy_img
	y = zeros((M,N))

	while SNR(y_old, y)>0.01:
		print(SNR(y_old,y))
		for i in range(M):
			for j in range(N):
				index=neighbor(i,j,M,N)
				
				a=cost(1, noisy_img[i, j], y_old, index)
				b=cost(0, noisy_img[i, j], y_old, index)

				if a>b:
					y[i,j]=1
				else:
					y[i,j]=0
		y_old=y
	print(SNR(y_old,y))
	return y


def SNR(A,B):
	if A.shape==B.shape:
		return numpy.sum(numpy.abs(A-B))/A.size
	else:	# Exception case
		raise Exception("Two matrices must have the same size!")


def delta(a,b):
	return 1 if a== b else 0


def neighbor(i,j,M,N):
	"""
		i : row index
		j : col index
		M : image's row size
		N : image's col size
		:return  neighboring points
	"""

	# find correct neighbors
	if i==0 and j==0:	# top-left corner
		neighbor=[(0,1), (1,0)]
	elif i==0 and j==N-1:	# top-right corner
		neighbor=[(0,N-2), (1,N-1)]
	elif i==M-1 and j==0:	# bottom-left corner
		neighbor=[(M-1,1), (M-2,0)]
	elif i==M-1 and j==N-1:		# right-bottom corner
		neighbor=[(M-1,N-2), (M-2,N-1)]
	elif i==0:		# first row
		neighbor=[(0,j-1), (0,j+1), (1,j)]
	elif i==M-1:	# last row
		neighbor=[(M-1,j-1), (M-1,j+1), (M-2,j)]
	elif j==0:		# first column
		neighbor=[(i-1,0), (i+1,0), (i,1)]
	elif j==N-1:	# last column
		neighbor=[(i-1,N-1), (i+1,N-1), (i,N-2)]
	else:			# inside boundary
		neighbor=[(i-1,j), (i+1,j), (i,j-1), (i,j+1),\
				  (i-1,j-1), (i-1,j+1), (i+1,j-1), (i+1,j+1)]

	return neighbor


def cost(y,x,y_old,index):
	alpha=1
	beta=10
	return alpha*delta(y,x)+\
		beta*sum(delta(y,y_old[i]) for i in index)


if __name__=="__main__":
	main()

```

### Result
<img src="https://github.com/jwcse/DeepLearning/blob/master/img/mrf_denoise_ex_result.PNG" width="700" height="420">
