import torch
import matplotlib.pyplot as plt
import seaborn as sns
  


def generate_data(response_fn, N):
    data = []
    while len(data) < N:
        x = torch.randn(3)
        x[0] = 1.0
        y = response_fn(x)
        if y is not None:
            datum = (x, y)
            data.append(datum)
    return data


def compute_linear_response(x):
    parameter_vector = torch.tensor([0.5, 0.8, -0.4])
    result = x @ parameter_vector.t()
    if result < -0.1:
        return 0
    elif result > 0.1:
        return 1
    else:
        return None

def compute_nonlinear_response(x):
    value = x[1]**2 + x[2]**3
    if value < 1.25:
        return 1
    elif value > 1.75:
        return 0
    else:
        return None


def visualize(data):
    xs = [datum[0] for datum in data]
    ys = [datum[1] for datum in data]
    positives = [xs[i] for i in range(len(xs)) if ys[i] == 1]
    negatives = [xs[i] for i in range(len(xs)) if ys[i] == 0]
    sns.scatterplot(
        x=[x[1].item() for x in positives],
        y=[x[2].item() for x in positives],
        color="yellow",
    )
    sns.scatterplot(
        x=[x[1].item() for x in negatives],
        y=[x[2].item() for x in negatives],
        color="blue",
    )
    plt.ion()
    plt.show()
    plt.pause(0.2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate artificial data for machine learning.")
    parser.add_argument("--type", type=str, required=True, choices=["zeros", "ones", "linear", "nonlinear"], help="Type of data to generate: 'trivial', 'linear', or 'nonlinear'")
    args = parser.parse_args()
    
    if args.type == "linear":
        response_fn = compute_linear_response
    elif args.type == "nonlinear":
        response_fn = compute_nonlinear_response
    elif args.type == "zeros":
        response_fn = lambda x: 0
    else:
        response_fn = lambda x: 1
    
    data = generate_data(response_fn, 5000)
    visualize(data)
    plt.show(block=True)
