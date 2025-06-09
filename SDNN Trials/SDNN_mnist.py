#trial code where i am using the particular onnx model for the relu. if not relu we change the activations to *_sigma_input.copy()

import os
import random
from pathlib import Path
from PIL import Image
import onnx
from onnx import numpy_helper, shape_inference
import numpy as np

def read_onnx(onnx_path: str):
    model = onnx.load(onnx_path)
    inferred = shape_inference.infer_shapes(model)
    graph = inferred.graph

    # Determine total input dimension by multiplying all positive dims
    full_input_size = None
    for input_tensor in graph.input:
        dims = input_tensor.type.tensor_type.shape.dim
        shape_list = [d.dim_value for d in dims if d.HasField("dim_value") and d.dim_value > 0]
        if not shape_list:
            continue
        full_input_size = int(np.prod(shape_list))
        break
    if full_input_size is None:
        raise RuntimeError("Cannot infer input dimension from ONNX model.")

    # Load weight initializers
    weights = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # Automatically detect linear layer weight matrices by looking at node ops
    linear_nodes = [node for node in graph.node if node.op_type == "Gemm"]
    weight_names = []
    for node in linear_nodes:
        if len(node.input) >= 2:
            weight_names.append(node.input[1])
    print(f"Found {len(weight_names)} linear layers with weights: {weight_names}")
    W_in_h = weights[weight_names[0]]
    W_h_out = weights[weight_names[1]]

    hidden_size = W_in_h.shape[0]
    output_size = W_h_out.shape[0]

    print("\n-- ONNX Model Info --")
    print(f" Full input dimension = {full_input_size}")
    print(f" Hidden layer size     = {hidden_size}")
    print(f" Output size           = {output_size}\n")
    return full_input_size, hidden_size, output_size, W_in_h, W_h_out

def round_vector(v: np.ndarray) -> np.ndarray:
    rounded = np.floor(v + 0.5)  # Add 0.5 and floor to round half up
    return rounded.astype(int)


def temporal_sdnn(onnx_path: str):
    # Load model and weights
    full_input_size, hidden_size, output_size, W_in_h, W_h_out = read_onnx(onnx_path)

    # Prompt only for number of time frames
    T = int(input("Enter the number of time frames (T): ").strip())
    if T < 1:
        raise ValueError("T must be at least 1.")

    # Prompt for MNIST directory and select random images
    mnist_dir = input("Enter path to MNIST image directory: ").strip()
    # Ensure we have a Path object
    mnist_dir = Path(mnist_dir)
    files = [f for f in mnist_dir.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    if len(files) < T:
        raise ValueError(f"Directory contains only {len(files)} images, but T={T} requested.")
    chosen = random.sample(files, T)

    # Build the input matrix: flatten each 28×28 image into one column
    user_input = np.zeros((full_input_size, T), dtype=float)
    print("\nSelected frames:")
    for t, img_path in enumerate(chosen):
        img = Image.open(img_path).convert('L').resize((28,28))
        arr = np.array(img, dtype=float) / 255.0
        user_input[:, t] = arr.flatten()
        print(f" Frame {t+1}: {img_path.name}")

    # Initialize SDNN states
    prev_input_rounded  = np.zeros(full_input_size, dtype=int)
    prev_hidden_rounded = np.zeros(hidden_size, dtype=int)
    prev_hidden_sigma   = np.zeros(hidden_size, dtype=float)
    prev_output_sigma   = np.zeros(output_size, dtype=float)

    outputs = []  # will store one label per frame
    print("\n-- Beginning temporal propagation --\n")

    for t in range(T):
        print(f"=== Time frame {t+1}/{T} ===")

        #rounding and delta computation
        curr = user_input[:, t]
        print(f"  Input float vector: {curr.tolist()}")
        x_rounded = round_vector(curr)
        print(f"  Rounded input x^(t): {x_rounded.tolist()}")
        delta_x = x_rounded - prev_input_rounded
        print(f"  delta_input: {delta_x.tolist()}")

        #propagating to hidden: del → sig → ReLU → round → del
        h_delta_in = W_in_h.dot(delta_x)
        print(f"  Hidden delta_in (W·delta_input): {h_delta_in.tolist()}")
        h_sigma = prev_hidden_sigma + h_delta_in
        print(f"  sigma_hidden before activation: {h_sigma.tolist()}")
        h_act = np.maximum(0, h_sigma)
        print(f"  Hidden activation (ReLU sigma): {h_act.tolist()}")
        h_rounded = round_vector(h_act)
        print(f"  Hidden rounded: {h_rounded.tolist()}")
        delta_h = h_rounded - prev_hidden_rounded
        print(f"  delta_hidden: {delta_h.tolist()}")

        #propagating to output: del → sig → ReLU → round
        y_delta_in = W_h_out.dot(delta_h)
        print(f"  Output delta_in (W·delta_hidden): {y_delta_in.tolist()}")
        y_sigma = prev_output_sigma + y_delta_in
        print(f"  sigma_output before activation: {y_sigma.tolist()}")
        y_act = np.maximum(0, y_sigma)
        print(f"  Output activation (ReLU sigma): {y_act.tolist()}")
        y_rounded = round_vector(y_act)
        print(f"  Output rounded: {y_rounded.tolist()}")

        label = int(np.argmax(y_act))
        outputs.append(label)
        print(f" Predicted label: {label}\n")

        # previous states updation
        prev_input_rounded  = x_rounded
        prev_hidden_rounded = h_rounded
        prev_hidden_sigma   = h_sigma
        prev_output_sigma   = y_sigma

    # Final output sequence
    result = np.array(outputs).reshape(1, T)
    print("Final 1×T output labels:")
    print(result)

if __name__ == '__main__':
    onnx_path = r"C:\Users\Deborshi Chakrabarti\Desktop\ISI\SDNN\SDNN Trials\two_layer_dnn.onnx"
    temporal_sdnn(onnx_path)
