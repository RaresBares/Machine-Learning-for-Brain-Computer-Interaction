import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader#


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = torch.randn(10_000, 256)
y = torch.randint(0, 4, (10_000,))
D = X.shape[1]
C = int(y.max().item()+1)

ds = TensorDataset(X, y)
dl = DataLoader(ds, batch_size=128, shuffle=True)

class MLP(nn.Module):
    def __init__(self, input_size=8*128, output_size=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

model = MLP(D, 256, C).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()

model.eval()
with torch.no_grad():
    xb = torch.randn(5, D).to(device)
    logits = model(xb)
    pred = logits.argmax(dim=1)
    print(pred.cpu())