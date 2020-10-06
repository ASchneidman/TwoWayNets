import torch
from layers.BatchNorm import RegularizedBatchNorm
from layers.TiedDropoutLayer import TiedDropoutLayer
from layers.TiedLinear import TiedLinear

class ListModule(torch.nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class TiedBlock(torch.nn.Module):
    def __init__(self, 
                    D_in, 
                    D_out, 
                    p = 0.5, 
                    paired_linear = None,
                    paired_dropout = None,
                    activation_function = torch.nn.ReLU, 
                    is_linear = False):
        super(TiedBlock, self).__init__()
        self.D_in = D_in
        self.D_out = D_out
        self.is_linear = is_linear
        if not is_linear:
            self.bn = RegularizedBatchNorm(D_out)
            self.dropout = TiedDropoutLayer(p, None) if paired_dropout is None else TiedDropoutLayer(p, paired_dropout)
            self.activation = activation_function()
        self.linear = TiedLinear(D_in, D_out, None) if paired_linear is None else TiedLinear(D_in, D_out, paired_linear)

    def forward(self, x):
        if not self.is_linear:
            return self.dropout(self.bn(self.activation(self.linear(x))))
        else:
            return self.linear(x)
        #return self.activation(self.linear(self.bn(self.dropout(x))))

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
        self.x_enc = []
        D_in = self.input_x_shape
        for s in self.layer_sizes:
            self.x_enc.append(TiedBlock(D_in, s, self.dropout_prob))
            D_in = s
        # Last layer is just linear
        self.x_enc.append(TiedBlock(D_in, self.input_y_shape, self.dropout_prob, is_linear=True))
        # Register layers
        self.x_encoder = ListModule(*self.x_enc)

        # Create y -> x channel
        self.y_enc = []
        D_in = self.input_y_shape
        for i, s in enumerate(self.layer_sizes[::-1]):
            self.y_enc.append(TiedBlock(D_in, 
                                        s,
                                        paired_dropout=self.x_enc[-(i+2)].dropout, 
                                        paired_linear=self.x_enc[-(i+1)].linear))
            D_in = s
        self.y_enc.append(TiedBlock(D_in, self.input_x_shape, paired_linear=self.x_enc[0].linear, is_linear=True))
        self.y_encoder = ListModule(*self.y_enc)

        print(f"X Encoder shapes:")
        for l in self.x_encoder:
            print(f"D_in: {l.D_in}, D_out: {l.D_out}")

        print(f"Y Encoder shapes:")
        for l in self.y_encoder:
            print(f"D_in: {l.D_in}, D_out: {l.D_out}")

    def get_summed_gammas():
        summed = torch.zeros((1, 1))
        summed.requires_grad = True
        for l in self.x_encoder:
            summed += torch.reciprocal(l.bn.gamma).sum()
        for l in self.y_encoder:
            summed += torch.reciprocal(l.bn.gamma).sum()
        return summed

    def get_summed_l2_weights():
        summed = torch.zeros((1, 1))
        summed.requires_grad = True
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
        for l in self.x_enc[:len(self.x_encoder) - 1]:
            x = l(x)
            hidden_xs.append(x)
        reconstructed_y = self.x_enc[len(self.x_encoder) - 1](x)
        for l in self.y_enc[:len(self.y_encoder) - 1]:
            y = l(y)
            hidden_ys.append(y)
        reconstructed_x = self.y_enc[len(self.y_encoder) - 1](y)
        # Now y = reconstructed x, x = reconstructed y
        return reconstructed_x, reconstructed_y, hidden_xs, list(reversed(hidden_ys))


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





