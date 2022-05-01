import os
import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytesseract
import shapely.geometry
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import auc, precision_recall_curve
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model


def OCR(image_path: str) -> List[Dict[str, Any]]:
    image = Image.open(image_path)
    image_data = pytesseract.image_to_data(image, output_type='data.frame')
    image_data = image_data.loc[
        image_data.text.apply(lambda x: pd.notnull(x) and x != '')
    ]
    image_data['position'] = image_data.apply(
        lambda row: [
            int(row['left']),
            int(row['top']),
            int(row['left']) + int(row['width']),
            int(row['top']) + int(row['height']),
        ],
        axis=1,
    )
    return image_data[['text', 'position']].to_dict(orient='record')


def display_doc(doc: Dict[str, Any], predicted_tokens: Optional[List[int]] = None):
    image = Image.open(doc['image_path'])
    draw = ImageDraw.Draw(image)
    if predicted_tokens is None:
        subset_of_tokens = range(0, len(doc['OCR']))
    else:
        # -1 to account for the stop token
        subset_of_tokens = [idx - 1 for idx in predicted_tokens if idx != 0]

    for i in subset_of_tokens:
        token = doc['OCR'][i]
        draw.rectangle(token['position'], outline='blue')
    draw.rectangle(doc['ground_truth'], outline='red', width=3)
    return image


def ground_truth_match(
    ocr_doc: List[Dict[str, Any]], ground_truth: List[float], threshold: float = 0.5
) -> List[int]:
    ground_truth = shapely.geometry.box(*ground_truth)

    labels = []
    for (i, token) in enumerate(ocr_doc):
        box = shapely.geometry.box(*token['position'])
        match_score = ground_truth.intersection(box).area / box.area
        if match_score > threshold:
            labels.append(i + 1)  # 0 is reserved for the padding / stop token

    return labels


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loaders(datasets, batch_size):
    train_loader = DataLoader(
        datasets['train'], batch_size=batch_size, worker_init_fn=seed_worker
    )

    val_loader = DataLoader(
        datasets['validation'], batch_size=batch_size, worker_init_fn=seed_worker
    )

    test_loader = DataLoader(
        datasets['test'], batch_size=batch_size, worker_init_fn=seed_worker
    )

    return train_loader, val_loader, test_loader


def text_pre_processing(text: str) -> str:
    text = text.strip().lower()
    text = ''.join([c for c in text if c in string.ascii_lowercase + string.digits])
    return text


def pad_tensor(
    tensor: torch.tensor, max_shape_0: int, max_shape_1: int
) -> torch.tensor:
    new_tensor = torch.zeros(max_shape_0, max_shape_1)
    a, b = tensor.shape
    new_tensor[:a, :b] = tensor
    return new_tensor


@dataclass
class Tensors:
    keys: torch.tensor
    words: torch.tensor
    positions: torch.tensor
    target: torch.tensor


def make_tensors(
    dataset: List[Tuple[str, List[Tuple[List[int], List[float]]], List[int]]]
) -> Tensors:

    list_keys, list_words, list_positions, list_targets = [], [], [], []
    for key, input_data, target in dataset:
        words = pad_sequence(
            [torch.tensor(chars) for chars, position in input_data], batch_first=True
        )
        positions = torch.tensor([position for chars, position in input_data])
        list_keys.append(int(key))
        list_words.append(words)
        list_positions.append(positions)
        list_targets.append(torch.tensor(target + [0]))

    shapes = [words.shape for words in list_words]
    max_shape_0 = (
        max([shape[0] for shape in shapes]) + 1
    )  # Adding a row so that the last row is
    # always zero for consistency
    max_shape_1 = max([shape[1] for shape in shapes]) + 1  # idem

    tensor_words = torch.cat(
        [
            pad_tensor(words, max_shape_0, max_shape_1).unsqueeze(0)
            for words in list_words
        ]
    )

    tensor_positions = torch.cat(
        [
            pad_tensor(positions, max_shape_0, 4).unsqueeze(0)
            for positions in list_positions
        ]
    )

    tensor_target = pad_sequence(list_targets, batch_first=True)
    return Tensors(
        keys=torch.tensor(list_keys),
        words=tensor_words,
        positions=tensor_positions,
        target=tensor_target,
    )


def loss_function(
    overall_probabilities: torch.tensor, target: torch.tensor
) -> torch.tensor:
    batch_size, max_seq_len, n_tokens = overall_probabilities.shape
    flat_target = target.reshape(-1)
    flat_probabilities = overall_probabilities.reshape(-1, n_tokens)
    loss = torch.nn.functional.cross_entropy(
        flat_probabilities, flat_target, reduction='mean'
    )
    return loss


def get_val_loss(
    model: Model, optimizer: torch.optim.Optimizer, val_loader: DataLoader
) -> float:
    epoch_losses = []
    for _, words, positions, target in val_loader:
        overall_probabilities, peak_indices = model.forward(words, positions)
        loss = loss_function(overall_probabilities, target)
        optimizer.zero_grad()
        epoch_losses.append(loss.item())
    val_loss = np.mean(epoch_losses)
    return val_loss


def train_model(
    n_epochs: int,
    model: Model,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> Tuple[List[float], List[float], List[float]]:

    if not os.path.exists('models'):
        os.mkdir('models')

    train_losses, val_losses, validation_metrics = [], [], []
    for epoch in range(n_epochs):

        # Train
        model.train()
        epoch_losses = []
        for _, words, positions, target in tqdm(train_loader):
            overall_probabilities, peak_indices = model.forward(words, positions)
            loss = loss_function(overall_probabilities, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # Validation
        model.eval()

        val_loss = get_val_loss(model, optimizer, val_loader)
        val_losses.append(val_loss)

        val_threshold_data = get_threshold_data(model, optimizer, val_loader)
        val_metrics = get_metrics(val_threshold_data)
        validation_metrics.append(val_metrics)

        print(
            f'Epoch {epoch}, train_loss={train_loss}, val_loss={val_loss} \n val_metrics={val_metrics}'
        )
        torch.save(model, f'models/model_{epoch}.torch')

    return train_losses, val_losses, validation_metrics


def get_threshold_data(
    model: Model, optimizer: torch.optim.Optimizer, loader: DataLoader
) -> pd.DataFrame:
    model.eval()
    confidence_and_is_correct = []
    for _, words, positions, target in loader:
        overall_probabilities, peak_indices = model.forward(words, positions)
        optimizer.zero_grad()
        predicted_tokens = torch.argmax(overall_probabilities, 2)
        prediction_confidence = (
            overall_probabilities.exp().max(axis=2).values.min(axis=1).values.tolist()
        )

        prediction = np.array(
            [
                set(
                    single_prediction
                )  # We don't care about the ordering or repetitions
                for single_prediction in predicted_tokens.tolist()
            ]
        )
        target = np.array(list(map(set, target.tolist())))
        prediction_correct = prediction == target
        confidence_and_is_correct += list(
            zip(prediction_confidence, prediction_correct)
        )
    threshold_data = pd.DataFrame(confidence_and_is_correct)
    threshold_data.columns = ['confidence', 'is_correct']
    return threshold_data


def get_metrics(threshold_data: pd.DataFrame) -> Dict[str, float]:
    accuracy = len(threshold_data.loc[threshold_data.is_correct]) / len(threshold_data)
    precision, recall, thresholds = precision_recall_curve(
        1 * threshold_data.is_correct.values, threshold_data.confidence.values
    )
    precision_recall_auc = auc(recall, precision)
    return {'accuracy': accuracy, 'PR-AUC': precision_recall_auc}


def find_threshold(
    target_accuracy: float, val_threshold_data: pd.DataFrame
) -> Tuple[List[float], List[float], float]:
    thresholds = np.linspace(val_threshold_data.confidence.min(), 1, 100)
    accuracies, automations = [], []
    for th in thresholds:
        tmp = val_threshold_data.loc[val_threshold_data.confidence >= th]
        accuracy = tmp.is_correct.mean()
        automation = len(tmp) / len(val_threshold_data)
        accuracies.append(accuracy)
        automations.append(automation)

    threshold_99acc = min(
        [th for th, acc in zip(thresholds, accuracies) if acc >= target_accuracy]
    )
    return accuracies, automations, threshold_99acc


def display_prediction(doc: Dict[str, Any], peaks: List[int]) -> Image.Image:
    return display_doc(doc, predicted_tokens=peaks)


def set_seed(seed: int) -> None:  # To ensure reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.set_deterministic(True)
    torch.set_num_threads(1)
