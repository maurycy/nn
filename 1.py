import torch
import torch.nn as nn
import torch.nn.functional as F

def hyperoperation_power(base, exponent):
    return torch.pow(base, exponent)

def hyperoperation_tetration(base, n):
    result = base
    for _ in range(n - 1):
        result = torch.pow(base, result)
    return result

class HypertensorLayer(nn.Module):
    def __init__(self, input_shape, output_size, hyperoperation_order=2, activation=None):
        super(HypertensorLayer, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.hyperoperation_order = hyperoperation_order
        self.activation = activation

        weight_shape = (output_size,) + input_shape
        self.weight = nn.Parameter(torch.randn(weight_shape))

        self.bias = nn.Parameter(torch.randn(output_size))

    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        weight_expanded = self.weight.unsqueeze(0)

        hypertensor_product = x_expanded * weight_expanded

        sum_dims = tuple(range(2, 2 + len(self.input_shape)))
        z = hypertensor_product.sum(dim=sum_dims)

        z = z + self.bias

        if self.hyperoperation_order == 2:
            z = hyperoperation_power(z, 2)
        elif self.hyperoperation_order == 3:
            z = hyperoperation_tetration(z, 2)
        else:
            raise ValueError("Only 2 or 3!")

        if self.activation is not None:
            z = self.activation(z)

        return z

class HypertensorNetwork(nn.Module):
    def __init__(self, input_shape, output_size):
        super(HypertensorNetwork, self).__init__()
        self.hypertensor_layer = HypertensorLayer(
            input_shape=input_shape,
            output_size=output_size,
            hyperoperation_order=2,
            activation=F.relu
        )
        self.linear_layer = nn.Linear(output_size, 1)

    def forward(self, x):
        x = self.hypertensor_layer(x)
        x = self.linear_layer(x)
        return x

batch_size = 10
n = 3
input_shape = (n, n, n)
output_size = 5

# meh
input_data = torch.randn(batch_size, *input_shape)
target_data = torch.randn(batch_size, 1)

model = HypertensorNetwork(input_shape=input_shape, output_size=output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000

for epoch in range(num_epochs):
    outputs = model(input_data)
    loss = criterion(outputs, target_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

