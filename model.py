import torch
from layers.BatchNorm import RegularizedBatchNorm
from layers.TiedDropoutLayer import TiedDropoutLayer
from layers.TiedLinear import TiedLinear

class TiedBlock(torch.nn.Module):
    def __init__(self, D_in, D_out, p = 0.5, paired_block = None, activation_function = torch.nn.ReLU):
        self.D_in = D_in
        self.D_out = D_out
        self.bn = RegularizedBatchNorm(D_out)
        if paired_block is None:
            # Create all the layers fresh
            self.linear = TiedLinear(D_in, D_out, None)
            self.dropout = TiedDropoutLayer(p, None)
        else:
            # Create layers but tie them
            self.linear = TiedLinear(D_in, D_out, paired_block.linear)
            self.dropout = TiedDropoutLayer(p, paired_block.dropout)
        self.activation = activation_function()

    def forward(self, x):
        return self.dropout(self.bn(self.activation(self.linear(x))))

class TwoWayNet(torch.nn.Module):
    def __init__(self, input_x_shape, input_y_shape, layer_sizes, hidden_output_layer, dropout_prob):
        super(TwoWayNet, self).__init__()

        self.input_x_shape = input_x_shape
        self.input_y_shape = input_y_shape
        self.layer_sizes = layer_sizes       
        self.hidden_output_layer = hidden_output_layer
        self.dropout_prob = dropout_prob

        # Note: Tieing occurs in reverse (the first layer of x is tied with the last layer of y)
        # Create x -> y channel
        self.x_encoder = []
        D_in = self.input_x_shape
        for s in self.layer_sizes:
            self.x_encoder.append(TiedBlock(D_in, s, self.dropout_prob))
            D_in = s
        self.x_encoder.append(TiedBlock(D_in, self.input_y_shape, self.dropout_prob))

        # Create y -> x channel
        self.y_encoder = []
        D_in = self.input_y_shape
        for i, s in enumerate(self.layer_sizes[::-1]):
            self.y_encoder.append(TiedBlock(D_in, s, self.dropout_prob, self.x_encoder[-i - 1]))
            D_in = s
        self.y_encoder.append(TiedBlock(D_in, self.input_x_shape, self.x_encoder[0]))

    def get_summed_gammas():
        summed = torch.zeros((1, 1))
        for l in self.x_encoder:
            summed += torch.reciprocal(l.bn.gamma).sum()
        for l in self.y_encoder:
            summed += torch.reciprocal(l.bn.gamma).sum()
        return summed

    def get_summed_l2_weights():
        summed = torch.zeros((1, 1))
        for l in self.x_encoder:
            summed += torch.norm(l.linear.weight) ** 2
        # Weights are tied, so don't need to loop over y
        return summed

    def forward(self, data):
        """
            Args:
                data["x"] = x input
                data["y"] = y input
            Returns:
                Reconstructed x, Reconstructed y, hidden outputs for x, hidden outputs for y, 
        """
        hidden_xs = []
        hidden_ys = []
        x, y = data["x"], data["y"]
        for l in self.x_encoder[:-1]:
            x = l(x)
            hidden_xs.append(x)
        reconstructed_y = self.x_encoder[-1](x)
        for l in self.y_encoder[-1]:
            y = l(y)
            hidden_ys.append(y)
        reconstructed_x = self.y_encoder[-1](y)
        # Now y = reconstructed x, x = reconstructed y
        return reconstructed_x, reconstructed_y, hidden_xs, hidden_ys 


def train(dataset, model, epochs, reconstructed_weight, hidden_weight, cov_weight, gamma_weight, lr):
    optim = torch.optim.SGD(model.parameters(), lr, nesterov=True)
    mse_loss = torch.nn.MSELoss()

    losses = []
    model.train()
    for epoch in range(epochs):
        avg_loss = 0
        for data in dataset:
            optim.zero_grad()
            # Forward pass
            xprime, yprime, hidden_xs, hidden_ys = model(data)

            # Compute losses
            # reconstruction losses
            loss_x = mse_loss(xprime, data["x"])
            loss_y = mse_loss(yprime, data["y"])

            # hidden loss
            loss_hidden = mse_loss(hidden_xs[model.hidden_output_layer], hidden_ys[model.hidden_output_layer])

            # covariance loss
            hidden_xs_tensor, hidden_ys_tensor = torch.stack(hidden_xs), torch.stack(hidden_ys)
            cov_x = torch.dot(hidden_xs_tensor.T, hidden_xs_tensor) / hidden_xs_tensor.shape[0]
            cov_y = torch.dot(hidden_ys_tensor.T, hidden_ys_tensor) / hidden_ys_tensor.shape[0]

            cov_loss_x = torch.sqrt(torch.sum(cov_x ** 2)) - torch.sqrt(torch.sum(torch.diag(cov_x) ** 2))
            cov_loss_y = torch.sqrt(torch.sum(cov_y ** 2)) - torch.sqrt(torch.sum(torch.diag(cov_y) ** 2))

            # Weight Decay loss
            loss_weight_decay = model.get_summed_l2_weights()
            
            # Gamma loss
            loss_gamma = model.get_summed_gammas()

            loss = (reconstructed_weight * loss_x + 
                    reconstructed_weight * loss_y + 
                    hidden_weight * loss_hidden + 
                    cov_weight * cov_loss_x + 
                    cov_weight * cov_loss_y + 
                    gamma_weight * loss_gamma)
            
            # Backward step
            loss.backward()
            avg_loss += loss.item()

            # Update step
            optim.step()
        losses.append(avg_loss / len(dataset))
        print(f"Epoch {epoch+1}, loss: {losses[-1]}")

if __name__ == "__main__":
    




