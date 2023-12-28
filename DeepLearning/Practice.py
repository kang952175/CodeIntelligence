import torch
import torch.nn.functional as F


x = torch.ones(7)
y = torch.zeros(3)

# Generate w1, b1, w2, b2
w1 = torch.randn(7, 5, requires_grad= True)
b1 = torch.randn(5, requires_grad= True)
w2 = torch.randn(5, 3, requires_grad= True)
b2 = torch.randn(3, requires_grad=True)

# Compute
z1 = x @ w1 + b1

z2 = F.relu(z1)

z3 = z2 @ w2 + b2

# Calculate loss
loss = F.binary_cross_entropy_with_logits(z3, y)

# Backpropagation
loss.backward()

# Print the gradient of w and b
print(w1.grad)
print(b1.grad)
print(w2.grad)
print(b2.grad)

