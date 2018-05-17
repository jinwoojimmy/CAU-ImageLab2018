"""
    This is for understanding the concept "Gradient Descent"
    We assume that 
        hypothesis function :
            y = w * x
    
    We have to find out 'w' which minimizes the loss,  
    so that we can get appropriate function
    
    In real training, process of gradient descent is executed until convergence.
    But in this practice, we iterate just until enough to watch almost converged.
    
"""

# just random value
w = 3.0
# learning rate - just randomly selected
LR = 0.01

x_data = [1.0, 2.0, 3.0]    # x1, x2, x3
y_data = [2.0, 4.0, 6.0]    # y1, y2, y3


# our model's forward pass
def forward(x):
    return x * w


# loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)      # (w * x - y)^2


# compute gradient on loss function
def gradient(x, y):
    return 2 * (w * x - y) * w


# run program
def run():
    # use global variable w and update in the function
    global w

    print("predict (before training)", 4, forward(4))

    # Training loop ; train 15 times
    for epoch in range(15):
        # train once with given set of data
        for x_val, y_val in zip(x_data, y_data):    # (xi, yi)
            grad = gradient(x_val, y_val)
            w = w - LR * grad
            print("\tgrad: ", x_val, y_val, round(grad, 2))
            l = loss(x_val, y_val)

        print("progress{num}:".format(num=epoch), " w=", round(w, 2), "loss=", round(l, 2))

    # After training
    print("predict (after training)",  "4 hours", forward(4))


if __name__ == "__main__":
    run()
