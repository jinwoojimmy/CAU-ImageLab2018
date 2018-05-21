import torch
from torch.autograd import Variable


# 1) Declaration

x_tensor =torch.Tensor(3, 4)    # FloatTensor of size 3x4

x_variable = Variable(x_tensor)

# 2) Variables of a Variable

# .data -> wrapped tensor
x_variable.data

# .grad -> gradient of the variable
x_variable.grad

# .requires_grad -> whether variable requires gradient
x_variable.requires_grad

# .volatile -> inference mode with minimal memory usage
x_variable.volatile


# 3) Graph & Variables
x = Variable(torch.FloatTensor(3, 4), requires_grad=True)
y = x**2 + 4*x  # requires_grad = True
z = 2*y + 3     # requires_grad = True

# .backward(gradient, retain_graph, create_graph, retain_variables)
# compute gradient of current variable w.r.t. graph leaves

gradient = torch.FloatTensor(3, 4)
z.backward(gradient)
# x.grad, y.grad, z.grad
