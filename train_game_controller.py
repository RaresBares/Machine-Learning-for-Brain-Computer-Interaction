#!/usr/bin/env python3
"""
BCI Game Controller Training Pipeline
8 channels, lowpass filter @ 40Hz, 20251121 data only
Commands: Rest (0), Cmd2, Cmd3, Cmd4, Cmd5
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.interpolate import interp1d
from scipy.signal import detrend
import os
from load_data import convert_data_biowolf_GUI_2c
from softmax_model import SoftmaxRegression
from sklearn.metrics import classification_report, confusion_matrix


# =============================================================================
# PREPROCESSING WITH LOWPASS FILTER
# =============================================================================

def preprocess_time_to_freq(input_time_array, dt, fc=40):
    """Convert time-domain EEG to frequency domain with lowpass filter."""

    def getFourier(array, dt):
        N = array.size
        f = np.fft.rfftfreq(N, d=dt)
        X = np.fft.rfft(array)
        df = 1/(dt*N)
        A = np.abs(X) / N
        if N % 2 == 0:
            A[1:-1] *= 2.0
        else:
            A[1:] *= 2.0
        return A, f, df

    def normalize(array):
        x = np.asarray(array, dtype=float)
        m = np.max(np.abs(x))
        y = x / m if m != 0 else np.zeros_like(x)
        return y

    def deTrend(array):
        return detrend(array, type='constant')

    def lowPass(fourier, freqs, fc):
        fourier[freqs > fc] = 0
        return fourier

    def cutOff(fourier, freqs, fc):
        mask = freqs <= fc
        return fourier[mask], freqs[mask]

    def interpolate(fourier, freqs, fc):
        fd = float(freqs[0])
        f_new = np.linspace(fd, fc, 128)
        y_128 = interp1d(freqs, fourier, kind='cubic', bounds_error=False,
                        fill_value='extrapolate')(f_new)
        df_new = (f_new[-1] - f_new[0]) / (len(f_new) - 1) if len(f_new) > 1 else 0.0
        return y_128, f_new, df_new

    def deMean(y):
        return y - np.mean(y)

    # Preprocessing pipeline
    input_time_array = deMean(input_time_array)
    input_time_array = deTrend(input_time_array)
    input_time_array, freqs, df = getFourier(input_time_array, dt)
    input_time_array = normalize(input_time_array)

    # Apply lowpass filter at 40 Hz
    input_time_array = lowPass(input_time_array, freqs, fc)

    cut_fourier, cut_freqs = cutOff(input_time_array, freqs, fc)
    result_fourier, result_freqs, result_df = interpolate(cut_fourier, cut_freqs, fc)
    result_fourier = normalize(result_fourier)

    return result_fourier, result_freqs, result_df


# =============================================================================
# DATA PREPARATION
# =============================================================================

def segment_by_trigger(eeg_data, triggers, min_samples=500):
    """Segment continuous EEG data by trigger values."""
    segments = {}
    trigger_changes = np.where(np.diff(triggers) != 0)[0] + 1
    trigger_starts = np.concatenate([[0], trigger_changes])
    trigger_ends = np.concatenate([trigger_changes, [len(triggers)]])

    for start, end in zip(trigger_starts, trigger_ends):
        trigger_val = triggers[start]
        segment_length = end - start

        if segment_length >= min_samples:
            if trigger_val not in segments:
                segments[trigger_val] = []
            segments[trigger_val].append(eeg_data[start:end])

    return segments


def create_dataset(bin_folder='data-raw/', output_file='game_controller_dataset.npz'):
    """Create dataset from 20251121 files only."""

    files_20251121 = [
        'Data_20251121_084219Samuel.bin',
        'Data_20251121_091105Rares.bin',
        'Data_20251121_093700Bhavya.bin'
    ]

    command_triggers = [2, 3, 4, 5]
    samples_per_trigger = 100

    all_features = []
    all_labels = []

    print("="*70)
    print("CREATING DATASET - 8 Channels with Lowpass @ 40Hz")
    print("="*70)
    print("Files: 20251121 only (Samuel, Rares, Bhavya)")
    print("Commands: 2, 3, 4, 5")
    print("Rest: Everything else (0, 12, 81)")
    print("="*70)

    for filename in files_20251121:
        filepath = os.path.join(bin_folder, filename)
        print(f"\nProcessing {filename}...")

        try:
            data = convert_data_biowolf_GUI_2c(filepath, VoltageScale='uV', TimeStampScale='ms')
            eeg_data = data['Data']
            triggers = data['Trigger']
            sample_rate = data.get('SampleRate', 500)
            dt = 1.0 / sample_rate

            segments = segment_by_trigger(eeg_data, triggers, min_samples=500)

            for trigger_val, trigger_segments in segments.items():
                trigger_val = int(trigger_val)

                # Label: commands stay as-is, everything else → 0
                if trigger_val in command_triggers:
                    label = trigger_val
                else:
                    label = 0

                num_to_use = min(samples_per_trigger, len(trigger_segments))

                for segment in trigger_segments[:num_to_use]:
                    channel_features = []

                    # Process all 8 channels
                    for ch_idx in range(8):
                        channel_signal = segment[:, ch_idx]
                        fourier, freqs, df = preprocess_time_to_freq(channel_signal, dt)
                        channel_features.append(fourier)

                    # Concatenate to 1024-dim vector
                    feature_vector = np.concatenate(channel_features)
                    all_features.append(feature_vector)
                    all_labels.append(label)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    print(f"\n{'='*70}")
    print(f"Dataset: {X.shape}")
    print(f"Labels: {y.shape}")

    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for label, count in zip(unique, counts):
        label_name = "Rest" if label == 0 else f"Command {label}"
        print(f"  {label_name}: {count} samples")

    np.savez(output_file, X=X, y=y)
    print(f"\nDataset saved to {output_file}")
    print("="*70)

    return X, y


# =============================================================================
# TRAINING
# =============================================================================

def train_model(X, y, epochs=50, lr=0.01, batch_size=32):
    """Train softmax regression model."""

    print("\n" + "="*70)
    print("TRAINING SOFTMAX REGRESSION MODEL")
    print("="*70)

    # Map labels to consecutive integers
    label_map = {}
    unique_labels = sorted(np.unique(y))
    for idx, label in enumerate(unique_labels):
        label_map[int(label)] = idx

    y_mapped = torch.tensor([label_map[int(label)] for label in y])
    X_tensor = torch.from_numpy(X)
    reverse_map = {v: k for k, v in label_map.items()}

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {len(label_map)}")
    print(f"Label mapping: {label_map}")

    # Split dataset
    dataset = TensorDataset(X_tensor, y_mapped)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SoftmaxRegression(input_size=1024, num_classes=len(label_map)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    print(f"Device: {device}")
    print("="*70)

    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=1)
            train_correct += (predictions == y_batch).sum().item()
            train_total += y_batch.size(0)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                logits = model(X_batch)
                predictions = logits.argmax(dim=1)
                val_correct += (predictions == y_batch).sum().item()
                val_total += y_batch.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print("="*70)

    return model, label_map, reverse_map


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X, y, label_map, reverse_map):
    """Evaluate model and display results."""

    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    # Map labels
    y_mapped = torch.tensor([label_map[int(label)] for label in y])
    X_tensor = torch.from_numpy(X)

    dataset = TensorDataset(X_tensor, y_mapped)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Evaluate
    device = next(model.parameters()).device
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            predictions = logits.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Overall accuracy
    overall_acc = (all_predictions == all_labels).mean()
    print(f"Overall Accuracy: {overall_acc:.4f}")

    # Per-class accuracy
    print("\n" + "-"*70)
    print("Per-Class Performance:")
    print("-"*70)

    cmd_accs = []
    for class_idx in range(len(label_map)):
        mask = all_labels == class_idx
        if mask.sum() > 0:
            correct = (all_predictions[mask] == class_idx).sum()
            total = mask.sum()
            accuracy = correct / total
            orig_trigger = reverse_map[class_idx]

            if orig_trigger == 0:
                label_name = "Rest (No Command)"
            else:
                label_name = f"Command {orig_trigger}"
                cmd_accs.append(accuracy)

            print(f"  {label_name:20s}: {correct:3d}/{total:3d} = {accuracy:.4f}")

    avg_cmd = np.mean(cmd_accs) if cmd_accs else 0
    print(f"\n  Average Command Accuracy: {avg_cmd:.4f}")

    # Confusion matrix
    print("\n" + "-"*70)
    print("Confusion Matrix:")
    print("-"*70)
    cm = confusion_matrix(all_labels, all_predictions)

    print("     Predicted →")
    print("True ↓   ", end="")
    for i in range(len(label_map)):
        print(f"{reverse_map[i]:4d} ", end="")
    print()

    for i in range(len(label_map)):
        print(f"T{reverse_map[i]:3d}  ", end="")
        for j in range(len(label_map)):
            print(f"{cm[i][j]:4d} ", end="")
        print()

    print("="*70)

    return overall_acc, avg_cmd


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Complete training and evaluation pipeline."""

    print("\n" + "="*70)
    print("BCI GAME CONTROLLER - TRAINING PIPELINE")
    print("="*70)
    print("Configuration:")
    print("  - 8 EEG channels")
    print("  - Lowpass filter @ 40 Hz")
    print("  - 20251121 data only")
    print("  - Commands: 2, 3, 4, 5 + Rest")
    print("="*70)

    # Step 1: Create dataset
    X, y = create_dataset(bin_folder='data-raw/',
                         output_file='game_controller_dataset.npz')

    # Step 2: Train model
    model, label_map, reverse_map = train_model(X, y, epochs=50, lr=0.01)

    # Step 3: Evaluate
    overall_acc, avg_cmd = evaluate_model(model, X, y, label_map, reverse_map)

    # Step 4: Save model
    model_path = 'game_controller_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_map': label_map,
        'reverse_label_map': reverse_map,
        'num_classes': len(label_map),
        'input_size': 1024,
        'overall_accuracy': overall_acc,
        'avg_command_accuracy': avg_cmd
    }, model_path)

    print(f"\n✓ Model saved to {model_path}")
    print("="*70)
    print("\nTRAINING COMPLETE!")
    print(f"  Overall Accuracy: {overall_acc:.1%}")
    print(f"  Command Accuracy: {avg_cmd:.1%}")
    print("="*70)


if __name__ == "__main__":
    main()
