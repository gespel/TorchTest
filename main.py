import torch
import torch.nn as nn
import torch.optim as optim
from sympy.physics.units import yards


class AddNet(nn.Module):
    def __init__(self):
        super(AddNet, self).__init__()
        self.fc1 = nn.Linear(1, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = AddNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


x_train = torch.tensor([[1.0],
                        [2.0],
                        [3.0],
                        [4.0],
                        [5.0],
                        [6.0],
                        [7.0],
                        [8.0],
                        [9.0],
                        [10.0]])

y_temp = []
for i in x_train:
    y_temp.append([i+1, i+2])
print(y_temp)
y_train = torch.tensor(y_temp)

# Training
epochs = 50000
for epoch in range(epochs):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Testen des Modells mit neuen Eingabewerten
x_test = torch.tensor([[2.0], [4.0], [7.0]])
y_test_pred = model(x_test)
print("Testeingaben:\n", x_test)
print("Vorhergesagte Summen:\n", y_test_pred)
torch.save(model, "model.pt")