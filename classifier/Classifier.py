import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

NUM_CHANNELS = 12
SPECTRUM_SIZE = 128
SCALAR_SIZE = 5
FEATURES_PER_CHANNEL = SPECTRUM_SIZE + SCALAR_SIZE
NUM_CLASSES = 3

class BCIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PerChannelEncoder(nn.Module):
    def __init__(self, in_dim=FEATURES_PER_CHANNEL, hidden=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class BCINet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = PerChannelEncoder()
        self.fuse = nn.Sequential(
            nn.Linear(NUM_CHANNELS * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )
    def forward(self, x):
        b, ch, feat = x.shape
        x = x.view(b * ch, feat)
        x = self.enc(x)
        x = x.view(b, ch * 32)
        logits = self.fuse(x)
        return logits

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += float(loss.detach()) * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss) * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            n += xb.size(0)
    return total_loss / n, correct / n

def infer_intent(model, feature_tensor, device):
    model.eval()
    with torch.no_grad():
        x = feature_tensor.to(device)
        logits = model(x.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()
    return probs.cpu().numpy(), pred

if __name__ == "__main__":
    torch.manual_seed(0)

    N_train = 200
    X_train = torch.randn(N_train, NUM_CHANNELS, FEATURES_PER_CHANNEL).numpy()
    y_train = torch.randint(low=0, high=NUM_CLASSES, size=(N_train,)).numpy()

    N_val = 50
    X_val = torch.randn(N_val, NUM_CHANNELS, FEATURES_PER_CHANNEL).numpy()
    y_val = torch.randint(low=0, high=NUM_CLASSES, size=(N_val,)).numpy()

    train_ds = BCIDataset(X_train, y_train)
    val_ds = BCIDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCINet().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, loss_fn, device)
        va_loss, va_acc = eval_epoch(model, val_loader, loss_fn, device)
        print(epoch, tr_loss, tr_acc, va_loss, va_acc)

    dummy_sample = torch.randn(NUM_CHANNELS, FEATURES_PER_CHANNEL)
    probs, pred = infer_intent(model, dummy_sample, device)
    print("probs:", probs)
    print("pred class:", pred)