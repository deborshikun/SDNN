import onnx
from onnx import numpy_helper, shape_inference
import numpy as np


def read_onnx(onnx_path):
    model = onnx.load(onnx_path)
    inferred_model = shape_inference.infer_shapes(model)
    graph = inferred_model.graph

    input_shapes = {}
    output_shapes = {}
    hidden_shapes = {}

    print("\n-- Inputs --")
    for input_tensor in graph.input:
        shape = [dim.dim_value if dim.HasField("dim_value") else "?" for dim in input_tensor.type.tensor_type.shape.dim]
        input_shapes[input_tensor.name] = shape
        print(f"Name: {input_tensor.name}, Shape: {shape}")

    print("\n-- Outputs --")
    for output_tensor in graph.output:
        shape = [dim.dim_value if dim.HasField("dim_value") else "?" for dim in output_tensor.type.tensor_type.shape.dim]
        output_shapes[output_tensor.name] = shape
        print(f"Name: {output_tensor.name}, Shape: {shape}")

    print("\n-- Initializers (Weights/Biases) --")
    weights = {}
    for initializer in graph.initializer:
        arr = numpy_helper.to_array(initializer)
        weights[initializer.name] = arr
        print(f"Name: {initializer.name}, Shape: {arr.shape}, Dtype: {arr.dtype}")

    print("\n-- Nodes (Operations) --")
    for node in graph.node:
        print(f"Op Type: {node.op_type}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")
        if node.attribute:
            print("  Attributes:")
            for attr in node.attribute:
                print(f"    - {attr.name}: {onnx.helper.get_attribute_value(attr)}")
        for out in node.output:
            for vi in graph.value_info:
                if vi.name == out:
                    dims = vi.type.tensor_type.shape.dim
                    shape = [dim.dim_value if dim.HasField("dim_value") else "?" for dim in dims]
                    hidden_shapes[out] = shape

    print("\n-- Hidden Layers --")
    for name, shape in hidden_shapes.items():
        print(f"Name: {name}, Shape: {shape}")

    return model, input_shapes, output_shapes, hidden_shapes, weights


def round_tensor(x):
    # Round each element: >=0.5 up, else down
    return np.array([int(np.floor(val + 0.5)) for val in x])


def temporal_propagation(model_path, T):
    model, input_shapes, output_shapes, hidden_shapes, weights = read_onnx(model_path)

    input_tensor_name = list(input_shapes.keys())[0]
    input_dims = input_shapes[input_tensor_name]
    input_size = int(np.prod([d for d in input_dims if isinstance(d, int)]))

    print(f"\nEnter temporal input as {input_size} x {T} matrix of floats:")
    temporal_input = []
    for i in range(input_size):
        row = input(f"Input feature {i} (space-separated {T} float values): ").strip().split()
        temporal_input.append([float(x) for x in row])
    temporal_input = np.array(temporal_input)  # shape: (input_size, T)

    # Extract weight matrices
    # Assume keys: 'fc1.weight' shape (hidden_size, input_size); 'fc2.weight' shape (output_size, hidden_size)
    W_input_hidden = weights['fc1.weight']
    W_hidden_output = weights['fc2.weight']

    # Initialize states
    prev_input_rounded = np.zeros(input_size, dtype=int)
    prev_hidden_rounded = np.zeros(W_input_hidden.shape[0], dtype=int)
    prev_hidden_sigma = np.zeros(W_input_hidden.shape[0])
    prev_output_sigma = np.zeros(W_hidden_output.shape[0])

    outputs = []
    print("\n-- Temporal Inference with Δ and Σ modules --")
    for t in range(T):
        # Step 1: Round input
        curr_input_float = temporal_input[:, t]
        curr_input_rounded = round_tensor(curr_input_float)

        # Step 2: Δ at input
        delta_input = curr_input_rounded - prev_input_rounded

        # Step 3: Propagate Δ to hidden layer
        hidden_delta_input = W_input_hidden.dot(delta_input)

        # Step 4: Σ at hidden: accumulate
        hidden_sigma_input = prev_hidden_sigma + hidden_delta_input

        # Activation (identity) and rounding at hidden
        hidden_activation = hidden_sigma_input.copy()
        hidden_rounded = round_tensor(hidden_activation)

        # Δ at hidden
        hidden_delta = hidden_rounded - prev_hidden_rounded

        # Step 5: Propagate Δ hidden to output
        output_delta_input = W_hidden_output.dot(hidden_delta)

        # Σ at output: accumulate
        output_sigma_input = prev_output_sigma + output_delta_input

        # Activation (identity) and rounding at output
        output_activation = output_sigma_input.copy()
        output_rounded = round_tensor(output_activation)

        # Store result (assuming single output dimension or take all)
        outputs.append(output_rounded.tolist())

        # Print debug info
        print(f"Time {t+1}: Input Rounded: {curr_input_rounded.tolist()}")
        print(f"  Δ Input: {delta_input.tolist()}")
        print(f"  Hidden Σ Input: {hidden_sigma_input.tolist()}")
        print(f"  Hidden Rounded: {hidden_rounded.tolist()}")
        print(f"  Δ Hidden: {hidden_delta.tolist()}")
        print(f"  Output Σ Input: {output_sigma_input.tolist()}")
        print(f"  Output Rounded: {output_rounded.tolist()}\n")

        # Update previous states
        prev_input_rounded = curr_input_rounded.copy()
        prev_hidden_rounded = hidden_rounded.copy()
        prev_hidden_sigma = hidden_sigma_input.copy()
        prev_output_sigma = output_sigma_input.copy()

    # Final result: for each time frame, output_rounded vector
    print("\nFinal Outputs for all T frames:")
    print(np.array(outputs).T)  # shape: (output_size, T)


# Run
onnx_path = r"C:\Users\Deborshi Chakrabarti\Desktop\ISI\SDNN\two_layer_dnn.onnx"
T = int(input("Enter the number of time frames (T): "))
temporal_propagation(onnx_path, T)
