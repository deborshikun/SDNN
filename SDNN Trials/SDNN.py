import onnx
from onnx import numpy_helper, shape_inference
import numpy as np


def read_onnx(onnx_path):
    # Load and infer shapes in ONNX model
    model = onnx.load(onnx_path)
    inferred_model = shape_inference.infer_shapes(model)
    graph = inferred_model.graph

    input_shapes = {}
    output_shapes = {}
    hidden_shapes = {}

    # Model Inputs
    print("\n-- Inputs --")
    for input_tensor in graph.input:
        shape = [dim.dim_value if dim.HasField("dim_value") else "?" for dim in input_tensor.type.tensor_type.shape.dim]
        input_shapes[input_tensor.name] = shape
        print(f"Name: {input_tensor.name}, Shape: {shape}")

    # Model Outputs
    print("\n-- Outputs --")
    for output_tensor in graph.output:
        shape = [dim.dim_value if dim.HasField("dim_value") else "?" for dim in output_tensor.type.tensor_type.shape.dim]
        output_shapes[output_tensor.name] = shape
        print(f"Name: {output_tensor.name}, Shape: {shape}")

    # Initializers (weights/biases)
    print("\n-- Initializers (Weights/Biases) --")
    weights = {}
    for initializer in graph.initializer:
        arr = numpy_helper.to_array(initializer)
        weights[initializer.name] = arr
        print(f"Name: {initializer.name}, Shape: {arr.shape}, Dtype: {arr.dtype}")

    # Nodes (operations)
    print("\n-- Nodes (Operations) --")
    for node in graph.node:
        print(f"Op Type: {node.op_type}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")
        if node.attribute:
            print("  Attributes:")
            for attr in node.attribute:
                print(f"    - {attr.name}: {onnx.helper.get_attribute_value(attr)}")
        # Store hidden layer shapes from value_info
        for out in node.output:
            for vi in graph.value_info:
                if vi.name == out:
                    dims = vi.type.tensor_type.shape.dim
                    shape = [dim.dim_value if dim.HasField("dim_value") else "?" for dim in dims]
                    hidden_shapes[out] = shape

    return model, input_shapes, output_shapes, hidden_shapes, weights


def temporal_propagation(model_path, T):
    model, input_shapes, output_shapes, hidden_shapes, weights = read_onnx(model_path)

    # Flatten input size (e.g., 28x28 = 784)
    input_tensor_name = list(input_shapes.keys())[0]
    input_size = np.prod([s for s in input_shapes[input_tensor_name] if isinstance(s, int)])

    # Accept n x T input
    print(f"\nEnter temporal input as {input_size} x {T} binary matrix (0/1):")
    temporal_input = []
    for i in range(input_size):
        row = input(f"Input for node {i} (space-separated {T} values): ").strip().split()
        temporal_input.append([int(x) for x in row])
    temporal_input = np.array(temporal_input)  # shape: (n, T)

    # Extract weights and biases once
    fc1_weight = weights['fc1.weight']  # shape: (128, 784)
    fc1_bias = weights['fc1.bias']      # shape: (128,)
    fc2_weight = weights['fc2.weight']  # shape: (10, 128)
    fc2_bias = weights['fc2.bias']      # shape: (10,)

    # No onnxruntime needed – simulate manually
    print("\n-- Temporal Inference --")
    outputs = []
    for t in range(T):
        input_t = temporal_input[:, t]           # shape: (784,)
        x = input_t.reshape(-1)                  # (784,)

        # fc1: input → hidden
        h1 = np.dot(fc1_weight, x) + fc1_bias    # (128,)
        h1 = np.maximum(0, h1)                   # ReLU

        # fc2: hidden → output
        out = np.dot(fc2_weight, h1) + fc2_bias  # (10,)
        label = np.argmax(out)                   # predicted class index
        outputs.append(label)

        print(f"Time Frame {t+1}/{T}: Output Vector Shape: {out.shape}, Predicted Label: {label}")

    print("\nFinal Output (1 x T):")
    print(np.array(outputs).reshape(1, T))


# Run the function with your model path and T frames
onnx_path = r"C:\Users\Deborshi Chakrabarti\Desktop\ISI\SDNN\two_layer_dnn.onnx"
T = int(input("Enter the number of time frames (T): "))
temporal_propagation(onnx_path, T)
